"""
Intent Router - Classifies queries and routes to appropriate retrieval strategy.

Routes:
- Wage queries -> Structured JSON lookup
- Contract queries -> Vector search
- High-stakes queries -> Vector search + escalation flag
"""

import re
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import (
    WAGES_DIR, HIGH_STAKES_TOPICS,
    HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT,
    CAG_ENABLE_HYPOTHESIS_LAYER, CAG_ENABLE_FULL_ARTICLE_EXPANSION,
    CAG_ENABLE_TITLE_BOOSTING, FULL_ARTICLE_MAX_CHUNKS,
    FULL_ARTICLE_MIN_TOP_K_MATCH, CHUNKS_DIR,
    CAG_ENABLE_QUERY_INTERPRETER, MULTI_QUERY_MAX_SEARCHES,
    MULTI_QUERY_RESULTS_PER_SEARCH, MULTI_QUERY_TOTAL_RESULTS,
    CAG_ENABLE_RERANKER, MANIFESTS_DIR, CONTRACT_ID
)
from backend.retrieval.vector_store import ContractVectorStore
from backend.retrieval.hypothesis import (
    get_hypothesis_generator,
    apply_title_boosting,
    HypothesisResult
)
from backend.retrieval.query_interpreter import (
    get_interpreter,
    QueryInterpretation
)


# ============================================================================
# QUERY EXPANSION - Maps worker slang to contract language
# ============================================================================

# Universal labor language mappings (work for any CBA)
UNIVERSAL_SLANG = {
    # Abbreviations
    "ot": "overtime",
    "pto": "vacation personal holiday time off",
    "fmla": "family medical leave",
    "loa": "leave of absence",

    # Float days / floating holidays -> Personal Holidays
    "float": "personal holiday",
    "float day": "personal holiday",
    "float days": "personal holidays",
    "floater": "personal holiday",
    "floaters": "personal holidays",
    "floating holiday": "personal holiday",

    # Common worker terms
    "fired": "discharge termination",
    "canned": "discharge termination",
    "let go": "discharge termination layoff",
    "pink slip": "discharge termination layoff",
    "written up": "discipline warning",
    "write up": "discipline warning",
    "writeup": "discipline warning",

    # Scheduling
    "my schedule": "work schedule hours",
    "when do i work": "schedule hours",
    "shift change": "schedule change",
    "called in": "call in reporting pay",
    "call out": "call in sick absence",
    "no call no show": "absence discipline",

    # Benefits
    "health insurance": "health benefits health trust",
    "medical": "health benefits",
    "dental": "health benefits",
    "vision": "health benefits",
    "retirement": "pension",

    # Pay-related
    "raise": "wage increase progression",
    "bump": "wage increase step",
    "time and a half": "overtime premium",
    "double time": "overtime premium",
    "sunday pay": "sunday premium",
    "night pay": "night premium",
    "holiday pay": "holiday premium",

    # Leave types
    "bereavement": "funeral leave",
    "maternity": "family care leave",
    "paternity": "family care leave",
    "jury duty": "jury service",
    "sick time": "sick leave",
    "sick days": "sick leave",

    # Roles (universal)
    "cashier": "all purpose clerk",

    # Union stuff
    "steward": "union steward union representative",
    "rep": "union representative steward",
    "dues": "union dues",
    "union meeting": "union business leave",

    # Misc
    "dress code": "uniform appearance dress",
    "uniform": "dress code appearance",
    "break": "rest period",
    "lunch": "meal period",
    "tardiness": "attendance discipline",
    "late": "attendance tardiness",
    "late policy": "attendance discipline",
}


# ============================================================================
# MANIFEST-BASED ROUTING - Per-contract config loaded from JSON
# ============================================================================

@lru_cache(maxsize=16)
def load_manifest_routing(contract_id: str = CONTRACT_ID) -> dict:
    """Load contract-specific routing config from manifest. Cached per contract_id."""
    manifest_path = MANIFESTS_DIR / f"{contract_id}.json"
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        return manifest.get("query_routing", {})
    return {}


def get_slang_map(contract_id: str = CONTRACT_ID) -> dict:
    """Get merged slang map: universal defaults + contract-specific overrides."""
    routing = load_manifest_routing(contract_id)
    return {**UNIVERSAL_SLANG, **routing.get("slang_to_contract", {})}


def get_topic_article_map(contract_id: str = CONTRACT_ID) -> dict:
    """Get topic-to-articles mapping from manifest (always contract-specific)."""
    return load_manifest_routing(contract_id).get("topic_to_articles", {})


def get_topic_patterns(contract_id: str = CONTRACT_ID) -> dict:
    """Get merged topic patterns: universal defaults + manifest additions."""
    routing = load_manifest_routing(contract_id)
    return {**_UNIVERSAL_TOPIC_PATTERNS, **routing.get("topic_patterns", {})}


def get_classification_article_map(contract_id: str = CONTRACT_ID) -> dict:
    """Get classification-to-articles mapping from manifest."""
    return load_manifest_routing(contract_id).get("classification_to_articles", {})

def expand_query(query: str, contract_id: str = CONTRACT_ID) -> Tuple[str, list]:
    """
    Expand query by replacing worker slang with contract terminology.

    Args:
        query: The user's question
        contract_id: Contract ID for loading contract-specific slang

    Returns:
        Tuple of (expanded_query, list of expansions applied)
    """
    slang_map = get_slang_map(contract_id)
    query_lower = query.lower()
    expanded = query
    expansions_applied = []

    # Sort by length (longest first) to avoid partial replacements
    sorted_slang = sorted(slang_map.items(), key=lambda x: len(x[0]), reverse=True)

    for slang, contract_term in sorted_slang:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(slang) + r'\b'
        if re.search(pattern, query_lower):
            # Append contract terms to the query rather than replacing
            # This preserves the original query while adding searchable terms
            if contract_term not in expanded.lower():
                expanded = f"{expanded} ({contract_term})"
                expansions_applied.append(f"{slang} -> {contract_term}")

    return expanded, expansions_applied


@dataclass
class QueryIntent:
    """Classified intent of a user query."""
    intent_type: str  # 'wage', 'contract', 'high_stakes'
    confidence: float
    classification: Optional[str]  # For wage queries
    topic: Optional[str]
    requires_escalation: bool
    keywords_matched: list
    relevant_articles: list = None  # Articles relevant to detected topic
    
    def __post_init__(self):
        if self.relevant_articles is None:
            self.relevant_articles = []


# Wage-related keywords (specific phrases to avoid false positives)
WAGE_KEYWORDS = [
    "my pay", "my wage", "my rate", "my salary", "my hourly",
    "wage rate", "pay rate", "hourly rate", "rate of pay",
    "what do i make", "what's my pay", "what am i making",
    "how much do i make", "how much should i make", "how much will i make",
    "how much should i be making", "what should i be making", "what should i make",
    "compensation", "starting pay", "experience pay", "step", "progression",
    "appendix a"
]

# Exclude these from wage detection (contain "pay" but aren't wage queries)
WAGE_EXCLUDE_PATTERNS = [
    "vacation", "holiday", "sick", "time off", "pto", "personal day",
    "pay stub", "pay period", "pay check"
]

# Classification extraction patterns
CLASSIFICATION_PATTERNS = {
    "courtesy_clerk": r"courtesy\s*clerk|bagger",
    "head_clerk": r"head\s*clerk",
    "all_purpose_clerk": r"all\s*purpose\s*clerk|clerk",
    "produce_manager": r"produce\s*(department)?\s*manager",
    "bakery_manager": r"bakery\s*manager",
    "cake_decorator": r"cake\s*decorator",
    "pharmacy_tech": r"pharmacy\s*tech",
    "non_foods_clerk": r"non.?food|gm\s*clerk|general\s*merchandise",
}

# Universal topic patterns for routing (language patterns, not article numbers)
_UNIVERSAL_TOPIC_PATTERNS = {
    "overtime": r"overtime|over\s*time|ot|time\s*and\s*a\s*half",
    "scheduling": r"schedul|shift|hours|when do i work",
    "seniority": r"seniority|senior|how long|years of service",
    "layoff": r"layoff|lay\s*off|bumping|displacement|reduction",
    "personal_holiday": r"personal\s*holiday|float\s*(day|days)?|floater|pto",
    "vacation": r"vacation|time\s*off|holiday|personal day",
    "sick_leave": r"sick\s*leave|sick\s*day|illness|call\s*in\s*sick",
    "discipline": r"disciplin|warning|write\s*up|written up|tardiness|tardy|late|attendance",
    "grievance": r"grievance|arbitration|file\s*a\s*complaint",
    "breaks": r"break|lunch|meal\s*period|relief|rest\s*period",
    "premiums": r"premium|night\s*pay|sunday\s*pay|sunday\s*premium",
    "weingarten": r"weingarten|right\s*to\s*representation|union\s*rep",
    "health_benefits": r"health\s*(benefit|insurance|coverage|care)|medical\s*benefit|eligible.*(health|benefit)|benefit.*eligible",
    "promotion": r"promot|advance|move up|basket.*hours|credit.*hours",
    "drive_up_go": r"drive\s*up|dug|personal\s*shopper|clicklist",
    "probation": r"probation|probationary|trial\s*period|new\s*employee.*hours",
    "term": r"term\s*of|contract\s*term|agreement\s*term|expir|effective\s*date",
    "minimum_wage": r"minimum\s*wage|colorado.*wage|\$15",
    "joint_committee": r"joint.*committee|labor.*management\s*committee",
}


# NOTE: TOPIC_ARTICLE_MAP and CLASSIFICATION_ARTICLE_MAP are now loaded from
# the contract manifest via get_topic_article_map() and get_classification_article_map().
# Article numbers are always contract-specific and should not be hardcoded.


def extract_classification(query: str) -> Optional[str]:
    """Extract job classification from query."""
    query_lower = query.lower()
    for class_name, pattern in CLASSIFICATION_PATTERNS.items():
        if re.search(pattern, query_lower):
            return class_name
    return None


def extract_topic(query: str, contract_id: str = CONTRACT_ID) -> Optional[str]:
    """
    Extract main topic from query.

    Uses priority ordering to prefer more specific topics over generic ones.
    Merges universal patterns with contract-specific patterns from manifest.
    """
    topic_patterns = get_topic_patterns(contract_id)
    query_lower = query.lower()

    # Priority order: specific topics first, generic topics last
    TOPIC_PRIORITY = [
        "retirement_savings",
        "weingarten",
        "health_benefits",
        "drive_up_go",
        "personal_holiday",  # Check before vacation since it's more specific
        "promotion",
        "layoff",
        "sick_leave",
        "vacation",
        "overtime",
        "grievance",
        "discipline",
        "seniority",
        "premiums",
        "breaks",
        "scheduling",  # Generic - matches "hours" so put last
    ]

    # First pass: check topics in priority order
    for topic in TOPIC_PRIORITY:
        if topic in topic_patterns:
            pattern = topic_patterns[topic]
            if re.search(pattern, query_lower):
                return topic

    # Second pass: check any remaining topics
    for topic, pattern in topic_patterns.items():
        if topic not in TOPIC_PRIORITY:
            if re.search(pattern, query_lower):
                return topic

    return None


def is_wage_query(query: str) -> tuple[bool, list]:
    """Check if query is asking about wages/pay."""
    query_lower = query.lower()

    # First check if this is actually about time off/benefits (not wages)
    for exclude in WAGE_EXCLUDE_PATTERNS:
        if exclude in query_lower:
            return False, []

    matched = []
    for keyword in WAGE_KEYWORDS:
        if keyword in query_lower:
            matched.append(keyword)

    # Also check for specific patterns
    wage_patterns = [
        r"how much (do|does|will|would|should) .+ (make|earn|get paid|be making|be earning)",
        r"what (is|are|should) (my|the) (pay|wage|rate)",
        r"what (is|are|'s) the .+ rate of pay",
        r"what should i (make|be making|earn|be earning)",
        r"\$\d+.*hour",  # Dollar amounts with hour
    ]

    for pattern in wage_patterns:
        if re.search(pattern, query_lower):
            matched.append(f"pattern:{pattern}")

    return len(matched) > 0, matched


def is_high_stakes(query: str) -> tuple[bool, list, bool]:
    """
    Check if query involves high-stakes topics.

    Returns:
        tuple: (is_high_stakes, matched_patterns, is_active_situation)
        - is_high_stakes: True if topic relates to discipline/termination/etc
        - matched_patterns: List of what matched
        - is_active_situation: True only if user is CURRENTLY facing this (needs escalation UI)
    """
    query_lower = query.lower()
    matched = []
    is_active = False

    # Patterns that indicate ACTIVE/URGENT situation (currently happening)
    active_patterns = [
        r"(i'?m|i am|was|been|being|getting) (just\s+)?(fired|terminated|discharged)",
        r"(i'?m|i am|was|been|being|getting) (disciplined|written up|suspended)",
        r"(i'?m|i am) being (harass|discriminat)",
        r"(harassing|discriminating\s+against|retaliating\s+against)\s+me",
        r"(my\s+)?(manager|boss|supervisor|coworker).*(harass|discriminat|retaliat)",
        r"(called|summoned) (into|to) (a\s+)?(meeting|office)",
        r"just (got|been|was) (terminated|fired|discharged|written up|suspended)",
        r"(manager|boss|supervisor).*(wants|asked|told|called).*(meeting|office|talk)",
        r"(right now|today|yesterday|just happened)",
    ]

    # Check for active situation first
    for pattern in active_patterns:
        if re.search(pattern, query_lower):
            matched.append(f"active:{pattern}")
            is_active = True

    # General high-stakes topic patterns (informational, not active)
    general_patterns = [
        r"(harass|harassment|harassing)",
        r"(discriminat|discrimination)",
        r"unsafe|dangerous|injury|injured",
        r"investigation",
        r"weingarten",
        r"(my rights|what rights).*(if|when|during).*(disciplin|fired|terminat)",
    ]

    for pattern in general_patterns:
        if re.search(pattern, query_lower):
            matched.append(f"general:{pattern}")

    # Check for exact keyword matches
    for topic in HIGH_STAKES_TOPICS:
        if topic in query_lower:
            matched.append(topic)

    return len(matched) > 0, matched, is_active


def classify_intent(query: str, user_classification: str = None, contract_id: str = CONTRACT_ID) -> QueryIntent:
    """
    Classify the intent of a user query.

    Args:
        query: The user's question
        user_classification: Optional classification from user profile (e.g., from dropdown)
        contract_id: Contract ID for loading contract-specific routing config

    Returns:
        QueryIntent with type, confidence, and metadata
    """
    # Use provided classification or try to extract from query
    classification = user_classification or extract_classification(query)
    topic = extract_topic(query, contract_id)

    # Get relevant articles from manifest (contract-specific)
    topic_article_map = get_topic_article_map(contract_id)
    relevant_articles = topic_article_map.get(topic, []) if topic else []

    # Add classification-specific articles for role-based boosting
    if classification:
        classification_article_map = get_classification_article_map(contract_id)
        classification_articles = classification_article_map.get(classification, [])
        relevant_articles = list(set(relevant_articles + classification_articles))
    
    is_wage, wage_matches = is_wage_query(query)
    is_hs, hs_matches, is_active_situation = is_high_stakes(query)

    # Determine primary intent
    if is_hs:
        # Add discipline/grievance articles for high-stakes
        relevant_articles = list(set(relevant_articles + [43, 45, 46]))
        return QueryIntent(
            intent_type="high_stakes",
            confidence=0.9 if len(hs_matches) > 1 else 0.7,
            classification=classification,
            topic=topic,
            # Only escalate (show steward contact UI) for ACTIVE situations
            requires_escalation=is_active_situation,
            keywords_matched=hs_matches,
            relevant_articles=relevant_articles
        )
    
    if is_wage:
        # Check if we have enough info for a wage lookup
        confidence = 0.8 if classification else 0.6
        relevant_articles = list(set(relevant_articles + [8, 9]))  # Wages articles
        return QueryIntent(
            intent_type="wage",
            confidence=confidence,
            classification=classification,
            topic="wages",
            requires_escalation=False,
            keywords_matched=wage_matches,
            relevant_articles=relevant_articles
        )
    
    # Default to contract query
    return QueryIntent(
        intent_type="contract",
        confidence=0.7,
        classification=classification,
        topic=topic,
        requires_escalation=False,
        keywords_matched=[],
        relevant_articles=relevant_articles
    )


class HybridRetriever:
    """
    Combines hybrid search (vector + BM25) with structured wage lookups.
    
    Uses Reciprocal Rank Fusion to combine semantic and keyword search
    for better retrieval across union contract terminology.
    """
    
    def __init__(self, vector_store: ContractVectorStore = None):
        """Initialize the hybrid retriever."""
        self.vector_store = vector_store
        self.hybrid_searcher = None
        self.wages_data = None
        self._load_wages()
    
    def _load_wages(self):
        """Load wage data from JSON."""
        wages_file = WAGES_DIR / "wage_tables.json"
        if wages_file.exists():
            with open(wages_file, 'r', encoding='utf-8') as f:
                self.wages_data = json.load(f)
    
    def _ensure_hybrid_searcher(self):
        """Lazy-initialize the hybrid searcher."""
        if self.hybrid_searcher is None:
            # Create vector store if not provided
            if self.vector_store is None:
                self.vector_store = ContractVectorStore()
            from backend.retrieval.hybrid_search import HybridSearcher
            self.hybrid_searcher = HybridSearcher(vector_store=self.vector_store)
    
    def lookup_wage(
        self, 
        classification: str,
        hours_worked: int = 0,
        months_employed: int = 0,
        effective_date: str = None
    ) -> Optional[dict]:
        """Look up wage from structured data."""
        if not self.wages_data:
            return None
        
        # Import here to avoid circular dependency
        from backend.ingest.extract_wages import lookup_wage
        return lookup_wage(self.wages_data, classification, hours_worked, months_employed, effective_date)
    
    def _expand_with_related_sections(self, chunks: list, max_total: int = 8) -> list:
        """
        Expand retrieved chunks with related sections from the same articles.
        
        This enables cross-section synthesis by ensuring that if we retrieve
        Section 49, we also include nearby sections like 44, 45, 46 that may
        contain definitions or related provisions.
        """
        if not chunks:
            return chunks
        
        # Load all chunks if not cached - prefer enriched version
        if not hasattr(self, '_all_chunks'):
            from backend.config import CHUNKS_DIR
            # Try enriched chunks first, fall back to smart chunks, then original
            chunks_file = CHUNKS_DIR / "contract_chunks_enriched.json"
            if not chunks_file.exists():
                chunks_file = CHUNKS_DIR / "contract_chunks_smart.json"
            if not chunks_file.exists():
                chunks_file = CHUNKS_DIR / "contract_chunks.json"
            
            if chunks_file.exists():
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    self._all_chunks = json.load(f)
            else:
                self._all_chunks = []
        
        # Get article numbers from retrieved chunks
        retrieved_articles = set()
        retrieved_ids = set()
        for chunk in chunks:
            article_num = chunk.get('article_num')
            if article_num:
                retrieved_articles.add(article_num)
            retrieved_ids.add(chunk.get('chunk_id', chunk.get('citation', '')))
        
        # Find related sections from same articles
        related_chunks = []
        for article_num in retrieved_articles:
            # Get all sections from this article
            article_chunks = [
                c for c in self._all_chunks 
                if c.get('article_num') == article_num
                and c.get('chunk_id', c.get('citation', '')) not in retrieved_ids
            ]
            
            # Add up to 2 related sections per article (prioritize earlier sections which often contain definitions)
            article_chunks.sort(key=lambda x: x.get('section_num', 999) or 999)
            for c in article_chunks[:2]:
                if len(chunks) + len(related_chunks) < max_total:
                    # Mark as related context
                    c_copy = dict(c)
                    c_copy['similarity'] = 0.5  # Lower score to indicate it's supplemental
                    c_copy['is_related'] = True
                    related_chunks.append(c_copy)
        
        # Combine: original chunks first, then related
        return chunks + related_chunks

    def _expand_to_full_article(
        self,
        chunks: list,
        n_results: int = 5
    ) -> list:
        """
        Expand retrieval to include ALL chunks from the "winning" article.

        The "winning" article is defined as the article that appears most
        frequently in the top results. This ensures the LLM gets complete
        context for synthesizing an accurate answer.

        This is the "Breadcrumb Retrieval" technique: follow the trail
        to the full source.

        Args:
            chunks: Initial retrieved chunks
            n_results: Number of top results to analyze for winning article

        Returns:
            Original chunks + all chunks from winning article (up to max limit)
        """
        if not CAG_ENABLE_FULL_ARTICLE_EXPANSION or not chunks:
            return chunks

        # Load all chunks if not cached
        if not hasattr(self, '_all_chunks') or not self._all_chunks:
            chunks_file = CHUNKS_DIR / "contract_chunks_enriched.json"
            if not chunks_file.exists():
                chunks_file = CHUNKS_DIR / "contract_chunks_smart.json"
            if not chunks_file.exists():
                chunks_file = CHUNKS_DIR / "contract_chunks.json"

            if chunks_file.exists():
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    self._all_chunks = json.load(f)
            else:
                self._all_chunks = []

        if not self._all_chunks:
            return chunks

        # Count article occurrences in top-N results
        from collections import Counter
        top_chunks = chunks[:n_results]
        article_counts = Counter()

        for chunk in top_chunks:
            article_num = chunk.get('article_num')
            if article_num:
                article_counts[article_num] += 1

        if not article_counts:
            return chunks

        # Find the "winning" article (most frequent in top results)
        winning_article, count = article_counts.most_common(1)[0]

        # Only expand if the winning article appears enough times
        if count < FULL_ARTICLE_MIN_TOP_K_MATCH:
            return chunks

        # Get chunk IDs already in results
        existing_ids = {c.get('chunk_id', c.get('citation', '')) for c in chunks}

        # Fetch ALL chunks from the winning article
        article_chunks = [
            c for c in self._all_chunks
            if c.get('article_num') == winning_article
            and c.get('chunk_id', c.get('citation', '')) not in existing_ids
        ]

        # Sort by section number for logical ordering
        article_chunks.sort(key=lambda x: (
            x.get('section_num') or 999,
            x.get('subsection') or ''
        ))

        # Limit total chunks to avoid context overflow
        available_slots = FULL_ARTICLE_MAX_CHUNKS - len(chunks)
        if available_slots <= 0:
            return chunks

        # Mark as full-article context and add
        for chunk in article_chunks[:available_slots]:
            chunk_copy = dict(chunk)
            chunk_copy['similarity'] = 0.4  # Lower score to indicate supplemental
            chunk_copy['is_full_article_context'] = True
            chunk_copy['winning_article'] = winning_article
            chunks.append(chunk_copy)

        return chunks

    def retrieve(
        self,
        query: str,
        intent: QueryIntent = None,
        n_results: int = 5,
        hours_worked: int = 0,
        months_employed: int = 0,
        use_hybrid: bool = True,
    ) -> dict:
        """
        Retrieve relevant context for a query.

        Uses the "Rosetta Stone" CAG pipeline:
        1. Hypothesis Layer - LLM predicts section titles (bridges vocabulary gap)
        2. Hybrid Search - Vector + BM25 with RRF fusion
        3. Title Boosting - Boost chunks matching hypothesized titles
        4. Full Article Expansion - Fetch complete article for context

        Returns:
            dict with:
            - chunks: List of relevant contract chunks
            - wage_info: Wage lookup result (if applicable)
            - intent: Query intent classification
            - escalation_required: Whether to add escalation language
            - query_expansions: List of slang->contract term expansions applied
            - hypothesis_result: HypothesisResult from pre-retrieval reasoning (NEW)
        """
        # Expand query with contract terminology (static mappings)
        expanded_query, expansions = expand_query(query)

        if intent is None:
            # Use expanded query for intent classification
            intent = classify_intent(expanded_query)

        # ===== PHASE 2: HYPOTHESIS LAYER (Rosetta Stone Brain) =====
        # Use LLM to predict likely section titles before searching
        hypothesis_result = None
        if CAG_ENABLE_HYPOTHESIS_LAYER:
            hypothesis_generator = get_hypothesis_generator()
            hypothesis_result = hypothesis_generator.generate_sync(query)

            if hypothesis_result.success and hypothesis_result.hypothesized_titles:
                # Append hypothesized titles to the query for better matching
                # e.g., "When do I get a break?" -> "When do I get a break? (Relief Periods Rest Intervals Meal Periods)"
                expanded_query = hypothesis_result.query_expansion
        # ===== END HYPOTHESIS LAYER =====

        result = {
            "chunks": [],
            "wage_info": None,
            "intent": intent,
            "escalation_required": intent.requires_escalation,
            "query_expansions": expansions,  # Track what slang was expanded
            "hypothesis_result": hypothesis_result  # Track hypothesis for metrics
        }

        # Use hybrid search (vector + BM25 with RRF)
        # Weights configured in config.py (default: equal 1.0/1.0 for balanced fusion)
        if use_hybrid:
            self._ensure_hybrid_searcher()
            chunks = self.hybrid_searcher.search_to_chunks(
                query=expanded_query,  # Use expanded query with hypotheses
                n_results=n_results,
                use_expansion=True,
                vector_weight=HYBRID_VECTOR_WEIGHT,
                keyword_weight=HYBRID_KEYWORD_WEIGHT,
                boost_articles=intent.relevant_articles,
                concept_query=query,  # Use original query for stable concept matching
            )
        else:
            # Fallback to vector-only search
            if self.vector_store is None:
                self._ensure_hybrid_searcher()
            chunks = self.vector_store.search(
                query=expanded_query,  # Use expanded query with hypotheses
                n_results=n_results,
                classification=intent.classification,
                topic=intent.topic,  # Pass detected topic for boosting
                boost_articles=intent.relevant_articles
            )

        # ===== PHASE 2: TITLE BOOSTING =====
        # Boost chunks whose article_title matches hypothesized titles
        if hypothesis_result and hypothesis_result.success:
            chunks = apply_title_boosting(
                chunks,
                hypothesis_result.hypothesized_titles
            )
        # ===== END TITLE BOOSTING =====

        # ===== PHASE 3: FULL ARTICLE EXPANSION =====
        # Fetch all chunks from the "winning" article for complete context
        chunks = self._expand_to_full_article(chunks, n_results)
        # ===== END FULL ARTICLE EXPANSION =====

        # Expand context: fetch related sections from the same articles
        chunks = self._expand_with_related_sections(chunks, n_results)

        result["chunks"] = chunks

        # If wage query and we have classification, also do wage lookup
        if intent.intent_type == "wage" and intent.classification:
            wage_info = self.lookup_wage(
                classification=intent.classification,
                hours_worked=hours_worked,
                months_employed=months_employed
            )
            result["wage_info"] = wage_info

        return result

    def multi_angle_retrieve(
        self,
        query: str,
        intent: QueryIntent = None,
        n_results: int = 5,
        hours_worked: int = 0,
        months_employed: int = 0,
    ) -> dict:
        """
        Retrieve using multiple search angles from query interpretation.

        This is the Phase 4 enhancement that:
        1. Uses LLM to deeply interpret the query
        2. Generates multiple search queries (original, hypothetical answers, alternatives)
        3. Runs retrieval for each angle
        4. Merges and deduplicates results using score fusion
        5. Optionally fetches explicit article references directly

        Args:
            query: User's question
            intent: Pre-classified intent (optional)
            n_results: Number of results to return
            hours_worked: For wage lookups
            months_employed: For wage lookups

        Returns:
            dict with chunks, interpretation, and metadata
        """
        if not CAG_ENABLE_QUERY_INTERPRETER:
            # Fall back to standard retrieval
            return self.retrieve(
                query=query,
                intent=intent,
                n_results=n_results,
                hours_worked=hours_worked,
                months_employed=months_employed
            )

        # ===== PHASE 4: QUERY INTERPRETATION =====
        interpreter = get_interpreter()
        interpretation = interpreter.interpret(query)

        # Get all search queries to try
        search_queries = interpreter.get_all_search_queries(interpretation)

        # Limit to configured max
        search_queries = search_queries[:MULTI_QUERY_MAX_SEARCHES]

        # Ensure vector store is initialized for direct HyDE searches
        self._ensure_hybrid_searcher()

        # ===== MULTI-ANGLE RETRIEVAL =====
        all_chunks = []
        chunk_scores = {}  # chunk_id -> best score

        # Add explicit article lookups first (highest priority)
        if interpretation.explicit_articles:
            self._load_all_chunks()
            for article_num in interpretation.explicit_articles:
                article_chunks = [
                    c for c in self._all_chunks
                    if c.get('article_num') == article_num
                ]
                # Sort by section number
                article_chunks.sort(key=lambda x: (
                    x.get('section_num') or 0,
                    x.get('subsection') or ''
                ))
                # Add with high score
                for chunk in article_chunks[:MULTI_QUERY_RESULTS_PER_SEARCH]:
                    chunk_id = chunk.get('chunk_id', chunk.get('citation', ''))
                    chunk_copy = dict(chunk)
                    chunk_copy['similarity'] = 0.95  # High score for explicit reference
                    chunk_copy['search_angle'] = f"explicit_article_{article_num}"

                    if chunk_id not in chunk_scores:
                        chunk_scores[chunk_id] = chunk_copy
                        all_chunks.append(chunk_copy)
                    elif chunk_copy['similarity'] > chunk_scores[chunk_id].get('similarity', 0):
                        # Update with better score
                        idx = next(i for i, c in enumerate(all_chunks)
                                   if c.get('chunk_id', c.get('citation', '')) == chunk_id)
                        all_chunks[idx] = chunk_copy
                        chunk_scores[chunk_id] = chunk_copy

        # Run retrieval for each search angle
        for i, search_query in enumerate(search_queries):
            # For hypothetical answers (HyDE), use direct vector search
            # This avoids score distortion from hybrid fusion
            is_hypothetical = i > 0 and i <= len(interpretation.hypothetical_answers)

            if is_hypothetical and self.vector_store:
                # Direct vector search for hypothetical answers
                angle_chunks = self.vector_store.search(
                    query=search_query,
                    n_results=MULTI_QUERY_RESULTS_PER_SEARCH
                )
                angle_result = {"chunks": angle_chunks}
            else:
                # Use standard retrieval for original query and search queries
                angle_result = self.retrieve(
                    query=search_query,
                    intent=intent,
                    n_results=MULTI_QUERY_RESULTS_PER_SEARCH,
                    hours_worked=hours_worked,
                    months_employed=months_employed,
                    use_hybrid=True
                )

            # Merge chunks with score tracking
            for chunk in angle_result.get('chunks', []):
                chunk_id = chunk.get('chunk_id', chunk.get('citation', ''))
                chunk_copy = dict(chunk)
                chunk_copy['search_angle'] = f"angle_{i}_{search_query[:30]}"

                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = chunk_copy
                    all_chunks.append(chunk_copy)
                elif chunk_copy.get('similarity', 0) > chunk_scores[chunk_id].get('similarity', 0):
                    # Update with better score
                    idx = next((i for i, c in enumerate(all_chunks)
                                if c.get('chunk_id', c.get('citation', '')) == chunk_id), None)
                    if idx is not None:
                        all_chunks[idx] = chunk_copy
                        chunk_scores[chunk_id] = chunk_copy

        # Sort by similarity and limit results
        all_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        final_chunks = all_chunks[:MULTI_QUERY_TOTAL_RESULTS]

        # ===== PHASE 5: LLM RERANKING =====
        # Reorder chunks by semantic relevance before expansion
        reranker_result = None
        if CAG_ENABLE_RERANKER:
            from backend.retrieval.reranker import get_reranker
            reranker_result = get_reranker().rerank(
                query=query,
                chunks=final_chunks,
                interpretation=interpretation
            )
            if reranker_result.success:
                final_chunks = reranker_result.chunks
        # ===== END RERANKING =====

        # Apply full article expansion on merged results
        final_chunks = self._expand_to_full_article(final_chunks, n_results)
        final_chunks = self._expand_with_related_sections(final_chunks)

        # Get intent if not provided (use expanded query from interpretation)
        if intent is None:
            expanded_query = " ".join([query] + interpretation.key_concepts)
            intent = classify_intent(expanded_query)

        # Build result
        result = {
            "chunks": final_chunks,
            "wage_info": None,
            "intent": intent,
            "escalation_required": intent.requires_escalation,
            "query_expansions": [],
            "interpretation": interpretation,  # Include interpretation for debugging
            "search_angles_used": len(search_queries),
            "explicit_articles_fetched": interpretation.explicit_articles,
            "reranker_result": reranker_result,  # Include reranker metrics
        }

        # If wage query and we have classification, also do wage lookup
        if intent.intent_type == "wage" and intent.classification:
            wage_info = self.lookup_wage(
                classification=intent.classification,
                hours_worked=hours_worked,
                months_employed=months_employed
            )
            result["wage_info"] = wage_info

        return result

    def _load_all_chunks(self):
        """Load all chunks for direct article lookup."""
        if not hasattr(self, '_all_chunks') or not self._all_chunks:
            chunks_file = CHUNKS_DIR / "contract_chunks_enriched.json"
            if not chunks_file.exists():
                chunks_file = CHUNKS_DIR / "contract_chunks_smart.json"
            if not chunks_file.exists():
                chunks_file = CHUNKS_DIR / "contract_chunks.json"

            if chunks_file.exists():
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    self._all_chunks = json.load(f)
            else:
                self._all_chunks = []


def main():
    """Test the router and retriever."""
    
    # Test query expansion first
    print("=" * 60)
    print("Testing Query Expansion (Slang -> Contract Terms)")
    print("=" * 60)
    
    expansion_tests = [
        "do i get float days?",
        "what about DUG employees?",
        "can i get written up for being late?",
        "what's the policy on tardiness?",
        "how many floaters do i get?",
        "what are my pto options?",
        "when do i get a raise?",
    ]
    
    for query in expansion_tests:
        expanded, expansions = expand_query(query)
        print(f"\nOriginal: {query}")
        if expansions:
            print(f"Expanded: {expanded}")
            print(f"  Mapped: {expansions}")
        else:
            print("  (no expansion needed)")
    
    print("\n" + "=" * 60)
    print("Testing Intent Router")
    print("=" * 60)
    
    test_queries = [
        "What is my pay rate as a courtesy clerk?",
        "How much do I make after 3000 hours?",
        "I'm being called into a meeting with my manager about my performance",
        "How does seniority work for scheduling?",
        "What are my Weingarten rights?",
        "When can I take vacation?",
        "My manager is harassing me",
        "What are the duties of a head clerk?",
    ]
    
    for query in test_queries:
        intent = classify_intent(query)
        print(f"\nQuery: {query}")
        print(f"  Intent: {intent.intent_type} (conf: {intent.confidence:.2f})")
        print(f"  Classification: {intent.classification}")
        print(f"  Topic: {intent.topic}")
        print(f"  Escalation: {intent.requires_escalation}")
        if intent.keywords_matched:
            print(f"  Matched: {intent.keywords_matched}")
    
    # Test hybrid retrieval
    print("\n\n--- Testing Hybrid Retrieval ---")
    retriever = HybridRetriever()
    
    test_cases = [
        ("What is the starting pay for a courtesy clerk?", 0, 0),
        ("How much does an all purpose clerk make after 5000 hours?", 5000, 0),
        ("What are my rights during a disciplinary meeting?", 0, 0),
    ]
    
    for query, hours, months in test_cases:
        print(f"\nQuery: {query}")
        result = retriever.retrieve(query, hours_worked=hours, months_employed=months)
        
        if result["wage_info"]:
            wi = result["wage_info"]
            print(f"  Wage: ${wi['rate']:.2f}/hr ({wi['step']}) - {wi['citation']}")
        
        print(f"  Escalation: {result['escalation_required']}")
        print(f"  Top chunks:")
        for i, chunk in enumerate(result["chunks"][:2]):
            print(f"    {i+1}. [{chunk['citation']}] (sim: {chunk['similarity']:.3f})")


if __name__ == "__main__":
    main()

