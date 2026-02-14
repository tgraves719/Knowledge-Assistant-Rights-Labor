"""
Comprehensive Evaluation Script - Tests RAG retrieval against 55-question test suite.

Evaluates:
- Retrieval accuracy (finding correct articles)
- Level-by-level performance
- Identification of limitations for Level 10 questions
- Ablation modes for component analysis
- Bucket-filtered evaluation for meaningful metrics

Usage:
    # Standard evaluation
    python -m backend.evaluate_comprehensive

    # Ablation mode
    python -m backend.evaluate_comprehensive --ablation-mode no_retrieval

    # Bucket-filtered
    python -m backend.evaluate_comprehensive --bucket-filter contract_only

    # Combined
    python -m backend.evaluate_comprehensive --ablation-mode no_hypothesis --bucket-filter exact_numeric
"""

import json
import sys
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

from backend.config import DATA_DIR, CONTRACT_ID
from backend.retrieval.router import HybridRetriever, classify_intent
from backend.retrieval.vector_store import ContractVectorStore
from backend.contracts import get_contract_catalog_entry, resolve_default_contract_id


# =============================================================================
# ABLATION MODE DEFINITIONS
# =============================================================================

ABLATION_MODES = {
    "normal": "Standard retrieval with all features enabled",
    "no_retrieval": "n_results=0 — tests LLM memorization without any retrieved context",
    "random": "Random chunk selection — tests if retriever contributes vs random baseline",
    "top1": "n_results=1, no article expansion — tests chunk fusion dependency",
    "no_hypothesis": "Disables hypothesis layer (CAG_ENABLE_HYPOTHESIS_LAYER=False)",
    "bm25_only": "BM25 keyword search only (vector_weight=0)",
    "vector_only": "Vector/semantic search only (keyword_weight=0)",
    "no_expansion": "Disables full article expansion (CAG_ENABLE_FULL_ARTICLE_EXPANSION=False)",
}

VALID_BUCKETS = ["world_knowledge", "contract_only", "multi_hop", "exact_numeric"]


@dataclass
class TestResult:
    test_id: int
    level: int
    category: str
    bucket: str
    question: str
    expected_articles: list
    retrieved_articles: list
    retrieval_passed: bool
    notes: str


@dataclass
class AblationConfig:
    """Configuration overrides for an ablation mode."""
    mode: str = "normal"
    n_results: int = 5
    use_hybrid: bool = True
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    enable_hypothesis: bool = True
    enable_expansion: bool = True
    random_chunks: bool = False

    @staticmethod
    def from_mode(mode: str) -> "AblationConfig":
        """Create config from ablation mode name."""
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation mode '{mode}'. Valid: {list(ABLATION_MODES.keys())}")

        config = AblationConfig(mode=mode)

        if mode == "no_retrieval":
            config.n_results = 0
        elif mode == "random":
            config.random_chunks = True
        elif mode == "top1":
            config.n_results = 1
            config.enable_expansion = False
        elif mode == "no_hypothesis":
            config.enable_hypothesis = False
        elif mode == "bm25_only":
            config.vector_weight = 0.0
            config.keyword_weight = 1.0
        elif mode == "vector_only":
            config.vector_weight = 1.0
            config.keyword_weight = 0.0
        elif mode == "no_expansion":
            config.enable_expansion = False
        # "normal" uses all defaults

        return config


def check_retrieval(expected: list, chunks: list) -> tuple:
    """Check if expected articles were retrieved."""
    import re

    if not expected:
        return True, []  # No expectation = pass

    retrieved_articles = []
    for chunk in chunks:
        citation = (chunk.get('citation') or '').lower()

        # Extract article references
        article_matches = re.findall(r'article\s*(\d+)', citation)
        for match in article_matches:
            ref = f"Article {match}"
            if ref not in retrieved_articles:
                retrieved_articles.append(ref)

        if 'letter of understanding' in citation:
            if 'Letter of Understanding' not in retrieved_articles:
                retrieved_articles.append('Letter of Understanding')

        if 'appendix' in citation:
            if 'Appendix A' not in retrieved_articles:
                retrieved_articles.append('Appendix A')

    # Check if any expected article was found
    found = False
    for expected_article in expected:
        expected_lower = expected_article.lower()
        for retrieved in retrieved_articles:
            if expected_lower in retrieved.lower() or retrieved.lower() in expected_lower:
                found = True
                break
        if found:
            break

    return found, retrieved_articles


def _get_random_chunks(retriever: HybridRetriever, n: int = 5) -> list:
    """Get n random chunks from the corpus for the random ablation baseline."""
    retriever._ensure_hybrid_searcher()
    all_chunk_ids = list(retriever.hybrid_searcher.chunks_by_id.keys())
    if not all_chunk_ids:
        return []
    selected_ids = random.sample(all_chunk_ids, min(n, len(all_chunk_ids)))
    return [retriever.hybrid_searcher.chunks_by_id[cid] for cid in selected_ids]


def run_comprehensive_eval(
    ablation_mode: str = "normal",
    bucket_filter: str = None,
    seed: int = 42,
):
    """Run the comprehensive evaluation with optional ablation and bucket filtering."""
    import backend.config as config

    # Parse ablation config
    ablation = AblationConfig.from_mode(ablation_mode)

    # Set random seed for reproducibility (important for "random" mode)
    random.seed(seed)

    # Apply config overrides
    original_hypothesis = config.CAG_ENABLE_HYPOTHESIS_LAYER
    original_expansion = config.CAG_ENABLE_FULL_ARTICLE_EXPANSION
    original_vector_weight = config.HYBRID_VECTOR_WEIGHT
    original_keyword_weight = config.HYBRID_KEYWORD_WEIGHT

    try:
        config.CAG_ENABLE_HYPOTHESIS_LAYER = ablation.enable_hypothesis
        config.CAG_ENABLE_FULL_ARTICLE_EXPANSION = ablation.enable_expansion
        config.HYBRID_VECTOR_WEIGHT = ablation.vector_weight
        config.HYBRID_KEYWORD_WEIGHT = ablation.keyword_weight

        configured_contract_id = CONTRACT_ID
        catalog_entry = get_contract_catalog_entry(configured_contract_id)
        if catalog_entry:
            eval_contract_id = catalog_entry["contract_id"]
        else:
            eval_contract_id = resolve_default_contract_id() or configured_contract_id

        print(f"Loading comprehensive test set...")
        test_file = DATA_DIR / "test_set" / "comprehensive_test.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_cases = data['test_cases']

        # Filter by bucket if requested
        if bucket_filter:
            if bucket_filter not in VALID_BUCKETS:
                raise ValueError(f"Unknown bucket '{bucket_filter}'. Valid: {VALID_BUCKETS}")
            test_cases = [tc for tc in test_cases if tc.get('bucket') == bucket_filter]
            if not test_cases:
                print(f"No test cases found for bucket '{bucket_filter}'")
                return None

        print(f"Ablation mode: {ablation_mode} — {ABLATION_MODES[ablation_mode]}")
        if bucket_filter:
            print(f"Bucket filter: {bucket_filter}")
        print(f"Config overrides: n_results={ablation.n_results}, use_hybrid={ablation.use_hybrid}, "
              f"vector_w={ablation.vector_weight}, keyword_w={ablation.keyword_weight}, "
              f"hypothesis={ablation.enable_hypothesis}, expansion={ablation.enable_expansion}, "
              f"random={ablation.random_chunks}")
        print(f"Eval contract_id: {eval_contract_id} (configured: {configured_contract_id})")

        print("\nInitializing RAG system...")
        vs = ContractVectorStore()
        retriever = HybridRetriever(vs)

        print(f"\nRunning {len(test_cases)} test cases...")
        print("=" * 80)

        results = []
        level_stats = defaultdict(lambda: {'passed': 0, 'total': 0})
        bucket_stats = defaultdict(lambda: {'passed': 0, 'total': 0})

        for i, tc in enumerate(test_cases, 1):
            question = tc['question']
            expected = tc.get('expected_articles', [])
            level = tc['level']
            category = tc['category']
            bucket = tc.get('bucket', 'unknown')

            # ---- Ablation-aware retrieval ----
            if ablation.n_results == 0:
                # No retrieval mode
                chunks = []
                wage_info = None
            elif ablation.random_chunks:
                # Random chunks mode
                chunks = _get_random_chunks(retriever, n=5)
                wage_info = None
            else:
                # Standard or modified retrieval (runtime path parity)
                intent = classify_intent(question, contract_id=eval_contract_id)
                result = retriever.multi_angle_retrieve(
                    query=question,
                    intent=intent,
                    n_results=ablation.n_results,
                    contract_id=eval_contract_id,
                )
                chunks = result['chunks']
                wage_info = result.get('wage_info')

                # Check if wage lookup is needed
                if tc.get('requires_wage_lookup') and 'Appendix A' in expected:
                    if wage_info:
                        chunks = chunks + [{'citation': 'Appendix A'}]

            # ---- Evaluate ----
            passed, retrieved = check_retrieval(expected, chunks)

            # For Level 10 "impossible" questions
            if tc.get('should_identify_limitation'):
                if not expected and len(retrieved) == 0:
                    passed = True
                    notes = "Correctly found no relevant content"
                elif not expected:
                    passed = False
                    notes = f"Should identify limitation, got: {retrieved[:2]}"
                else:
                    notes = ""
            else:
                notes = "" if passed else f"Expected {expected}, got {retrieved[:3]}"

            test_result = TestResult(
                test_id=tc['id'],
                level=level,
                category=category,
                bucket=bucket,
                question=question,
                expected_articles=expected,
                retrieved_articles=retrieved,
                retrieval_passed=passed,
                notes=notes
            )
            results.append(test_result)

            # Track stats
            level_stats[level]['total'] += 1
            if passed:
                level_stats[level]['passed'] += 1
            bucket_stats[bucket]['total'] += 1
            if passed:
                bucket_stats[bucket]['passed'] += 1

            # Print progress
            status_emoji = "[OK]" if passed else "[XX]"
            print(f"  {status_emoji} L{level} Q{tc['id']} [{bucket}]: {category}")
            if not passed and notes:
                print(f"       {notes[:70]}...")

        # ================================================================
        # SUMMARY
        # ================================================================
        print("\n" + "=" * 80)
        print(f"COMPREHENSIVE EVALUATION SUMMARY — Ablation: {ablation_mode}")
        print("=" * 80)

        total_passed = sum(1 for r in results if r.retrieval_passed)
        total = len(results)

        print(f"\nOverall Retrieval Accuracy: {total_passed}/{total} ({100 * total_passed / total:.1f}%)")

        # --- By Level ---
        print("\nBy Level:")
        level_names = {
            1: "Direct Retrieval (Basic)",
            2: "Calculation (Intermediate)",
            3: "Multi-Article (Advanced)",
            4: "Edge Cases (Expert)",
            5: "Ambiguity (Expert+)",
            6: "Temporal (Expert++)",
            7: "Procedural (Expert++)",
            8: "External Law (Expert+++)",
            9: "Systemic (Expert+++)",
            10: "Impossible/Boundary"
        }

        for level in sorted(level_stats.keys()):
            stats = level_stats[level]
            rate = 100 * stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            bar_len = int(rate / 5)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            name = level_names.get(level, f"Level {level}")
            print(f"  L{level:2d} [{bar}] {stats['passed']}/{stats['total']} ({rate:.0f}%) - {name}")

        # --- By Bucket (NEW) ---
        print("\nBy Bucket:")
        bucket_labels = {
            "world_knowledge": "World Knowledge",
            "contract_only": "Contract-Only",
            "multi_hop": "Multi-Hop",
            "exact_numeric": "Exact Numeric",
        }
        for bucket in VALID_BUCKETS:
            if bucket in bucket_stats:
                stats = bucket_stats[bucket]
                rate = 100 * stats['passed'] / stats['total'] if stats['total'] > 0 else 0
                bar_len = int(rate / 5)
                bar = "#" * bar_len + "-" * (20 - bar_len)
                label = bucket_labels.get(bucket, bucket)
                print(f"  {label:20s} [{bar}] {stats['passed']}/{stats['total']} ({rate:.0f}%)")

        # --- Failures ---
        print("\n" + "-" * 80)
        print("Failed Tests by Bucket:")
        for bucket in VALID_BUCKETS:
            failures = [r for r in results if r.bucket == bucket and not r.retrieval_passed]
            if failures:
                print(f"\n  {bucket_labels.get(bucket, bucket)} ({len(failures)} failures):")
                for f in failures[:5]:
                    print(f"    Q{f.test_id}: {f.question[:55]}...")
                    print(f"         Expected: {f.expected_articles}")
                    print(f"         Got: {f.retrieved_articles[:3]}")

        # --- Save Results ---
        mode_suffix = f"_{ablation_mode}" if ablation_mode != "normal" else ""
        bucket_suffix = f"_{bucket_filter}" if bucket_filter else ""
        output_file = DATA_DIR / "test_set" / f"comprehensive_results{mode_suffix}{bucket_suffix}.json"

        results_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'ablation_mode': ablation_mode,
                'ablation_description': ABLATION_MODES[ablation_mode],
                'bucket_filter': bucket_filter,
                'seed': seed,
                'config_overrides': {
                    'n_results': ablation.n_results,
                    'use_hybrid': ablation.use_hybrid,
                    'vector_weight': ablation.vector_weight,
                    'keyword_weight': ablation.keyword_weight,
                    'enable_hypothesis': ablation.enable_hypothesis,
                    'enable_expansion': ablation.enable_expansion,
                    'random_chunks': ablation.random_chunks,
                }
            },
            'summary': {
                'total': total,
                'passed': total_passed,
                'accuracy': total_passed / total if total > 0 else 0,
                'by_level': {str(k): v for k, v in level_stats.items()},
                'by_bucket': {k: v for k, v in bucket_stats.items()},
            },
            'results': [
                {
                    'test_id': r.test_id,
                    'level': r.level,
                    'category': r.category,
                    'bucket': r.bucket,
                    'question': r.question,
                    'expected': r.expected_articles,
                    'retrieved': r.retrieved_articles,
                    'passed': r.retrieval_passed,
                    'notes': r.notes
                }
                for r in results
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)

        print(f"\nResults saved to {output_file}")

        return results_data

    finally:
        # Restore original config values
        config.CAG_ENABLE_HYPOTHESIS_LAYER = original_hypothesis
        config.CAG_ENABLE_FULL_ARTICLE_EXPANSION = original_expansion
        config.HYBRID_VECTOR_WEIGHT = original_vector_weight
        config.HYBRID_KEYWORD_WEIGHT = original_keyword_weight


def compare_ablation_results(baseline_file: str, ablation_file: str):
    """
    Compare ablation results against baseline and compute deltas per bucket.

    Args:
        baseline_file: Path to baseline results JSON (normal mode)
        ablation_file: Path to ablation results JSON
    """
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    with open(ablation_file, 'r', encoding='utf-8') as f:
        ablation = json.load(f)

    base_mode = baseline.get('metadata', {}).get('ablation_mode', 'normal')
    abl_mode = ablation.get('metadata', {}).get('ablation_mode', 'unknown')

    print("=" * 80)
    print(f"ABLATION COMPARISON: {base_mode} vs {abl_mode}")
    print("=" * 80)

    base_buckets = baseline.get('summary', {}).get('by_bucket', {})
    abl_buckets = ablation.get('summary', {}).get('by_bucket', {})

    bucket_labels = {
        "world_knowledge": "World Knowledge",
        "contract_only": "Contract-Only",
        "multi_hop": "Multi-Hop",
        "exact_numeric": "Exact Numeric",
    }

    print(f"\n{'Bucket':<20s} {'Baseline':>10s} {'Ablation':>10s} {'Delta':>10s} {'Drop%':>8s}")
    print("-" * 60)

    for bucket in VALID_BUCKETS:
        b_stats = base_buckets.get(bucket, {'passed': 0, 'total': 0})
        a_stats = abl_buckets.get(bucket, {'passed': 0, 'total': 0})

        b_rate = b_stats['passed'] / b_stats['total'] * 100 if b_stats['total'] > 0 else 0
        a_rate = a_stats['passed'] / a_stats['total'] * 100 if a_stats['total'] > 0 else 0
        delta = a_rate - b_rate
        drop_pct = abs(delta) if delta < 0 else 0

        label = bucket_labels.get(bucket, bucket)
        delta_str = f"{delta:+.1f}%"
        print(f"  {label:<20s} {b_rate:>8.1f}% {a_rate:>8.1f}% {delta_str:>10s} {drop_pct:>6.1f}%")

    # Overall
    b_total = baseline['summary']
    a_total = ablation['summary']
    b_rate = b_total['accuracy'] * 100
    a_rate = a_total['accuracy'] * 100
    delta = a_rate - b_rate
    print("-" * 60)
    print(f"  {'OVERALL':<20s} {b_rate:>8.1f}% {a_rate:>8.1f}% {delta:+.1f}%")

    # Success criteria checks
    print("\n" + "-" * 80)
    print("SUCCESS CRITERIA CHECKS:")
    co_base = base_buckets.get('contract_only', {'passed': 0, 'total': 0})
    co_abl = abl_buckets.get('contract_only', {'passed': 0, 'total': 0})
    co_base_rate = co_base['passed'] / co_base['total'] * 100 if co_base['total'] > 0 else 0
    co_abl_rate = co_abl['passed'] / co_abl['total'] * 100 if co_abl['total'] > 0 else 0
    co_drop = co_base_rate - co_abl_rate

    if abl_mode == "no_retrieval":
        passed = co_drop > 50
        print(f"  [{'OK' if passed else 'XX'}] Retrieval-OFF drop (Contract-Only): {co_drop:.1f}% (threshold: >50%)")
    elif abl_mode == "random":
        passed = co_abl_rate < 25
        print(f"  [{'OK' if passed else 'XX'}] Random-retrieval accuracy (Contract-Only): {co_abl_rate:.1f}% (threshold: <25%)")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Karl RAG Comprehensive Evaluation with Ablation Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation Modes:
  normal          Standard retrieval (baseline)
  no_retrieval    No chunks retrieved (tests memorization)
  random          Random chunk selection (tests retriever contribution)
  top1            Single chunk, no expansion (tests fusion)
  no_hypothesis   Hypothesis layer disabled
  bm25_only       BM25 keyword search only (no vector)
  vector_only     Vector/semantic search only (no BM25)
  no_expansion    Full article expansion disabled

Buckets:
  world_knowledge   Answerable from general labor law knowledge
  contract_only     Requires specific contract text
  multi_hop         Requires synthesizing 2+ sections
  exact_numeric     Requires precise number/date from contract

Examples:
  python -m backend.evaluate_comprehensive
  python -m backend.evaluate_comprehensive --ablation-mode no_retrieval
  python -m backend.evaluate_comprehensive --bucket-filter contract_only
  python -m backend.evaluate_comprehensive --ablation-mode no_hypothesis --bucket-filter exact_numeric
  python -m backend.evaluate_comprehensive --compare baseline.json ablation.json
"""
    )

    parser.add_argument(
        "--ablation-mode",
        type=str,
        default="normal",
        choices=list(ABLATION_MODES.keys()),
        help="Ablation mode to run (default: normal)"
    )
    parser.add_argument(
        "--bucket-filter",
        type=str,
        default=None,
        choices=VALID_BUCKETS,
        help="Run only on questions in this bucket"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "ABLATION"),
        help="Compare two result files: baseline vs ablation"
    )

    args = parser.parse_args()

    if args.compare:
        compare_ablation_results(args.compare[0], args.compare[1])
    else:
        run_comprehensive_eval(
            ablation_mode=args.ablation_mode,
            bucket_filter=args.bucket_filter,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
