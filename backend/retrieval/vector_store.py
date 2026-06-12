"""
Vector Store abstraction.

Supports PostgreSQL/pgvector when available and falls back to ChromaDB for
prototype or no-database environments.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import (
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL, COLLECTION_NAME,
    TOP_K_RESULTS, SIMILARITY_THRESHOLD, CONTRACT_ID
)
from backend.chunk_files import resolve_chunk_file
from backend.contracts import resolve_contract_region_id
from backend.platform.settings import get_platform_settings

# Lazy imports for optional dependencies
chromadb = None
SentenceTransformer = None
sqlalchemy_create_engine = None
sqlalchemy_sessionmaker = None
sqlalchemy_select = None
sqlalchemy_func = None
ChunkEmbedding = None


def _load_dependencies():
    """Lazy load heavy dependencies."""
    global chromadb, SentenceTransformer
    if chromadb is None:
        import chromadb as _chromadb
        chromadb = _chromadb
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST


def _load_sqlalchemy():
    global sqlalchemy_create_engine, sqlalchemy_sessionmaker, sqlalchemy_select, sqlalchemy_func, ChunkEmbedding
    if sqlalchemy_create_engine is None:
        from sqlalchemy import create_engine as _create_engine, func as _func, select as _select
        from sqlalchemy.orm import sessionmaker as _sessionmaker

        from backend.platform.models import ChunkEmbedding as _ChunkEmbedding

        sqlalchemy_create_engine = _create_engine
        sqlalchemy_sessionmaker = _sessionmaker
        sqlalchemy_select = _select
        sqlalchemy_func = _func
        ChunkEmbedding = _ChunkEmbedding


@dataclass
class SearchFilters:
    contract_id: str | None = None
    region_id: str | None = None
    classification: str | None = None
    topic: str | None = None
    urgency_tier: str | None = None
    doc_type: str | None = None
    boost_articles: list | None = None

def _apply_result_boosts(
    chunks: list[dict],
    *,
    query: str,
    contract_id: str | None = None,
    region_id: str | None = None,
    classification: str | None = None,
    topic: str | None = None,
    urgency_tier: str | None = None,
    boost_articles: list | None = None,
    n_results: int = 5,
) -> list[dict]:
    import re

    article_refs = re.findall(r'article\s*(\d+)', query.lower())
    section_refs = re.findall(r'section\s*(\d+)', query.lower())
    effective_region_id = str(region_id or resolve_contract_region_id(contract_id)) if contract_id else None
    ranked = []
    for chunk in chunks:
        if contract_id and str(chunk.get("contract_id")) != str(contract_id):
            continue
        if effective_region_id:
            chunk_region = chunk.get("region_id") or resolve_contract_region_id(str(chunk.get("contract_id") or contract_id))
            if str(chunk_region) != str(effective_region_id):
                continue
        similarity = float(chunk.get("similarity", 0.0))
        if article_refs and str(chunk.get("article_num", 0)) in article_refs:
            similarity += 0.3
        if section_refs and str(chunk.get("section_num", 0)) in section_refs:
            similarity += 0.1
        if boost_articles and chunk.get("article_num", 0) in boost_articles:
            similarity += 0.2
        if classification:
            applies_to = str(chunk.get("applies_to") or "")
            if classification in applies_to:
                similarity += 0.15
            elif "all" not in applies_to:
                similarity -= 0.05
        if topic and topic in str(chunk.get("topics") or ""):
            similarity += 0.15
        if urgency_tier == "high_stakes" and chunk.get("is_high_stakes"):
            similarity += 0.1
        chunk["similarity"] = similarity
        ranked.append(chunk)
    ranked.sort(key=lambda item: item["similarity"], reverse=True)
    return ranked[:n_results]


class PgVectorContractVectorStore:
    def __init__(self, postgres_url: str):
        _load_dependencies()
        _load_sqlalchemy()
        self.engine = sqlalchemy_create_engine(postgres_url, future=True, pool_pre_ping=True)
        self.session_factory = sqlalchemy_sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def reset_collection(self):
        with self.session_factory() as db:
            db.query(ChunkEmbedding).delete()
            db.commit()

    def add_chunks(self, chunks: list[dict], batch_size: int = 50) -> int:
        if not chunks:
            return 0
        added = 0
        with self.session_factory() as db:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                for chunk in batch:
                    embedding = self.embedder.encode(chunk["content"]).tolist()
                    db.add(
                        ChunkEmbedding(
                            id=str(chunk.get("chunk_id") or ""),
                            union_id=str(chunk.get("union_id")) if chunk.get("union_id") else None,
                            document_id=str(chunk.get("document_id")) if chunk.get("document_id") else None,
                            chunk_index=int(chunk.get("chunk_index") or chunk.get("section_num") or 0),
                            chunk_text=chunk.get("content_with_tables", chunk["content"]),
                            metadata_json={
                                "contract_id": chunk.get("contract_id", ""),
                                "region_id": chunk.get("region_id", ""),
                                "article_num": chunk.get("article_num") or 0,
                                "article_title": chunk.get("article_title", ""),
                                "section_num": chunk.get("section_num") or 0,
                                "subsection": chunk.get("subsection") or "",
                                "citation": chunk.get("citation", ""),
                                "parent_context": chunk.get("parent_context", ""),
                                "doc_type": chunk.get("doc_type", "cba"),
                                "applies_to": ",".join(chunk.get("applies_to", ["all"])) if isinstance(chunk.get("applies_to"), list) else str(chunk.get("applies_to") or "all"),
                                "topics": ",".join(chunk.get("topics", chunk.get("topic_tags", []))) if isinstance(chunk.get("topics", chunk.get("topic_tags", [])), list) else str(chunk.get("topics", chunk.get("topic_tags", "")) or ""),
                                "summary": chunk.get("summary") or "",
                                "is_definition": chunk.get("is_definition", False),
                                "is_exception": chunk.get("is_exception", False),
                                "hire_date_sensitive": chunk.get("hire_date_sensitive", False),
                                "is_high_stakes": chunk.get("is_high_stakes", False),
                                "worker_questions": "|".join(chunk.get("worker_questions", [])) if isinstance(chunk.get("worker_questions"), list) else str(chunk.get("worker_questions") or ""),
                                "alternative_names": "|".join(chunk.get("alternative_names", [])) if isinstance(chunk.get("alternative_names"), list) else str(chunk.get("alternative_names") or ""),
                            },
                            embedding=embedding,
                        )
                    )
                db.commit()
                added += len(batch)
        return added

    def search(
        self,
        query: str,
        n_results: int = None,
        contract_id: str = None,
        region_id: str = None,
        classification: str = None,
        topic: str = None,
        urgency_tier: str = None,
        doc_type: str = None,
        boost_articles: list = None,
    ) -> list[dict]:
        n_results = n_results or TOP_K_RESULTS
        query_embedding = self.embedder.encode(query).tolist()
        with self.session_factory() as db:
            stmt = sqlalchemy_select(ChunkEmbedding)
            if contract_id:
                stmt = stmt.where(ChunkEmbedding.metadata_json["contract_id"].astext == str(contract_id))
            if region_id:
                stmt = stmt.where(ChunkEmbedding.metadata_json["region_id"].astext == str(region_id))
            if doc_type:
                stmt = stmt.where(ChunkEmbedding.metadata_json["doc_type"].astext == str(doc_type))
            if topic:
                stmt = stmt.where(ChunkEmbedding.metadata_json["topics"].astext.contains(str(topic)))

            if hasattr(ChunkEmbedding.embedding, "cosine_distance"):
                stmt = stmt.order_by(ChunkEmbedding.embedding.cosine_distance(query_embedding)).limit(max(n_results * 3, 15))
                rows = db.execute(stmt).scalars().all()
                chunks = []
                for row in rows:
                    metadata = dict(row.metadata_json or {})
                    if not row.embedding:
                        continue
                    dot = sum(a * b for a, b in zip(query_embedding, row.embedding))
                    query_norm = sum(a * a for a in query_embedding) ** 0.5
                    row_norm = sum(a * a for a in row.embedding) ** 0.5
                    similarity = dot / (query_norm * row_norm) if query_norm and row_norm else 0.0
                    if similarity < SIMILARITY_THRESHOLD:
                        continue
                    chunks.append(
                        {
                            "chunk_id": row.id,
                            "content": row.chunk_text,
                            "content_with_tables": row.chunk_text,
                            "similarity": similarity,
                            **metadata,
                        }
                    )
                return _apply_result_boosts(
                    chunks,
                    query=query,
                    contract_id=contract_id,
                    region_id=region_id,
                    classification=classification,
                    topic=topic,
                    urgency_tier=urgency_tier,
                    boost_articles=boost_articles,
                    n_results=n_results,
                )

            rows = db.execute(stmt.limit(max(n_results * 5, 25))).scalars().all()
            chunks = []
            for row in rows:
                if not row.embedding:
                    continue
                dot = sum(a * b for a, b in zip(query_embedding, row.embedding))
                query_norm = sum(a * a for a in query_embedding) ** 0.5
                row_norm = sum(a * a for a in row.embedding) ** 0.5
                similarity = dot / (query_norm * row_norm) if query_norm and row_norm else 0.0
                if similarity < SIMILARITY_THRESHOLD:
                    continue
                chunks.append(
                    {
                        "chunk_id": row.id,
                        "content": row.chunk_text,
                        "content_with_tables": row.chunk_text,
                        "similarity": similarity,
                        **dict(row.metadata_json or {}),
                    }
                )
            return _apply_result_boosts(
                chunks,
                query=query,
                contract_id=contract_id,
                region_id=region_id,
                classification=classification,
                topic=topic,
                urgency_tier=urgency_tier,
                boost_articles=boost_articles,
                n_results=n_results,
            )

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        with self.session_factory() as db:
            row = db.get(ChunkEmbedding, chunk_id)
            if row is None:
                return None
            return {
                "chunk_id": row.id,
                "content": row.chunk_text,
                **dict(row.metadata_json or {}),
            }

    def clear(self):
        self.reset_collection()

    def count(self) -> int:
        with self.session_factory() as db:
            return int(db.scalar(sqlalchemy_select(sqlalchemy_func.count()).select_from(ChunkEmbedding)) or 0)


class ContractVectorStore:
    """Vector store facade for PostgreSQL/pgvector or ChromaDB."""
    
    def __init__(self, persist_dir: Path = None, collection_name: str = None):
        settings = get_platform_settings()
        self._backend = None
        if settings.db_enabled:
            try:
                self._backend = PgVectorContractVectorStore(settings.postgres_url)
                print(f"Vector store initialized with PostgreSQL. Indexed chunks: {self._backend.count()}")
                return
            except Exception as exc:
                print(f"Warning: pgvector backend unavailable, falling back to ChromaDB: {exc}")
        self._backend = _ChromaContractVectorStore(persist_dir=persist_dir, collection_name=collection_name)
    
    def reset_collection(self):
        return self._backend.reset_collection()
    
    def add_chunks(self, chunks: list[dict], batch_size: int = 50) -> int:
        return self._backend.add_chunks(chunks, batch_size=batch_size)
    
    def search(
        self,
        query: str,
        n_results: int = None,
        contract_id: str = None,
        region_id: str = None,
        classification: str = None,
        topic: str = None,
        urgency_tier: str = None,
        doc_type: str = None,
        boost_articles: list = None,
    ) -> list[dict]:
        return self._backend.search(
            query=query,
            n_results=n_results,
            contract_id=contract_id,
            region_id=region_id,
            classification=classification,
            topic=topic,
            urgency_tier=urgency_tier,
            doc_type=doc_type,
            boost_articles=boost_articles,
        )

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        return self._backend.get_chunk(chunk_id)

    def clear(self):
        return self._backend.clear()

    def count(self) -> int:
        return self._backend.count()


class _ChromaContractVectorStore:
    """Chroma-backed fallback implementation."""
    
    def __init__(self, persist_dir: Path = None, collection_name: str = None):
        """Initialize the vector store."""
        _load_dependencies()
        
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.collection_name = collection_name or COLLECTION_NAME
        
        # Create persist directory
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Initialize embedding model
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Vector store initialized. Collection '{self.collection_name}' has {self.collection.count()} documents.")
    
    def reset_collection(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection '{self.collection_name}' reset. Now has {self.collection.count()} documents.")
    
    def add_chunks(self, chunks: list[dict], batch_size: int = 50) -> int:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_id', 'content', and metadata
            batch_size: Number of chunks to process at once
        
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Deduplicate chunks by ID (keep first occurrence)
        seen_ids = set()
        unique_chunks = []
        for chunk in chunks:
            chunk_id = chunk['chunk_id']
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)
            else:
                # Make ID unique by appending index
                new_id = f"{chunk_id}_{len([c for c in unique_chunks if c['chunk_id'].startswith(chunk_id)])}"
                chunk['chunk_id'] = new_id
                unique_chunks.append(chunk)
        
        chunks = unique_chunks
        print(f"Processing {len(chunks)} unique chunks...")
        
        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for chunk in batch:
                chunk_id = chunk['chunk_id']
                embed_text = chunk['content']  # Clean text for embedding (no HTML, no pipe-tables)
                display_text = chunk.get('content_with_tables', chunk['content'])  # Rich text for LLM

                # Create embedding from clean text
                embedding = self.embedder.encode(embed_text).tolist()

                # Prepare metadata (ChromaDB only supports str, int, float, bool)
                # Handle both old format (topic_tags) and new format (topics)
                topics = chunk.get('topics', chunk.get('topic_tags', []))
                if isinstance(topics, str):
                    topics = topics.split(',') if topics else []
                
                applies_to = chunk.get('applies_to', ['all'])
                if isinstance(applies_to, str):
                    applies_to = applies_to.split(',') if applies_to else ['all']
                
                # Phase 4: Handle concept-indexed fields
                worker_questions = chunk.get('worker_questions', [])
                if isinstance(worker_questions, str):
                    worker_questions = [worker_questions] if worker_questions else []

                alternative_names = chunk.get('alternative_names', [])
                if isinstance(alternative_names, str):
                    alternative_names = [alternative_names] if alternative_names else []

                metadata = {
                    'contract_id': chunk.get('contract_id', ''),
                    'region_id': chunk.get('region_id', ''),
                    'article_num': chunk.get('article_num') or 0,
                    'article_title': chunk.get('article_title', ''),
                    'section_num': chunk.get('section_num') or 0,
                    'subsection': chunk.get('subsection') or '',
                    'citation': chunk.get('citation', ''),
                    'parent_context': chunk.get('parent_context', ''),
                    'doc_type': chunk.get('doc_type', 'cba'),
                    # Enriched metadata
                    'applies_to': ','.join(applies_to),
                    'topics': ','.join(topics),
                    'summary': chunk.get('summary') or '',
                    'is_definition': chunk.get('is_definition', False),
                    'is_exception': chunk.get('is_exception', False),
                    'hire_date_sensitive': chunk.get('hire_date_sensitive', False),
                    'is_high_stakes': chunk.get('is_high_stakes', False),
                    # Phase 4: Concept-indexed fields for vocabulary bridging
                    'worker_questions': '|'.join(worker_questions),
                    'alternative_names': '|'.join(alternative_names),
                }
                
                ids.append(chunk_id)
                documents.append(display_text)  # Store rich text (with pipe-tables) for LLM
                metadatas.append(metadata)
                embeddings.append(embedding)
            
            # Upsert to collection
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            added += len(batch)
            print(f"  Added {added}/{len(chunks)} chunks...")
        
        return added
    
    def search(
        self,
        query: str,
        n_results: int = None,
        contract_id: str = None,
        region_id: str = None,
        classification: str = None,
        topic: str = None,
        urgency_tier: str = None,
        doc_type: str = None,
        boost_articles: list = None,
    ) -> list[dict]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            contract_id: Filter by contract ID
            classification: Filter by job classification
            topic: Filter by topic tag
            urgency_tier: Filter by urgency (standard/high_stakes)
            doc_type: Filter by document type (cba/lou/appendix)
        
        Returns:
            List of matching chunks with scores
        """
        import re
        
        if n_results is None:
            n_results = TOP_K_RESULTS
        
        # Check for explicit article/section references in query
        article_refs = re.findall(r'article\s*(\d+)', query.lower())
        section_refs = re.findall(r'section\s*(\d+)', query.lower())
        
        # Build where clause
        where = None
        where_clauses = []
        
        effective_region_id = None
        if contract_id:
            effective_region_id = str(region_id or resolve_contract_region_id(contract_id))
            where_clauses.append({"contract_id": contract_id})
            where_clauses.append({"region_id": effective_region_id})
        if urgency_tier:
            where_clauses.append({"urgency_tier": urgency_tier})
        if doc_type:
            where_clauses.append({"doc_type": doc_type})
        
        if where_clauses:
            if len(where_clauses) == 1:
                where = where_clauses[0]
            else:
                where = {"$and": where_clauses}
        
        # Create query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Request more results initially to allow for boosting
        search_n = max(n_results * 2, 15) if article_refs else n_results
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_n,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        chunks = []
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity < SIMILARITY_THRESHOLD:
                    continue
                
                document_text = results['documents'][0][i] if results['documents'] else ''
                chunk = {
                    'chunk_id': chunk_id,
                    'content': document_text,
                    'content_with_tables': document_text,  # Consistent with JSON-loaded chunks
                    'similarity': similarity,
                    **results['metadatas'][0][i]
                }

                # Defense-in-depth tenancy guard.
                if contract_id and str(chunk.get("contract_id")) != str(contract_id):
                    continue
                if effective_region_id:
                    chunk_region = chunk.get("region_id") or resolve_contract_region_id(str(chunk.get("contract_id") or contract_id))
                    if str(chunk_region) != str(effective_region_id):
                        continue
                
                # Boost score if chunk matches explicit article reference in query
                if article_refs:
                    chunk_article = str(chunk.get('article_num', 0))
                    for ref in article_refs:
                        if chunk_article == ref:
                            chunk['similarity'] += 0.3  # Significant boost for exact match
                            break
                
                if section_refs:
                    chunk_section = str(chunk.get('section_num', 0))
                    for ref in section_refs:
                        if chunk_section == ref:
                            chunk['similarity'] += 0.1  # Smaller boost for section match
                            break
                
                # Boost score if chunk matches topic-relevant articles
                if boost_articles:
                    chunk_article_num = chunk.get('article_num', 0)
                    if chunk_article_num in boost_articles:
                        chunk['similarity'] += 0.2  # Moderate boost for topic relevance
                
                # Boost if classification matches (but don't filter)
                if classification:
                    applies_to = chunk.get('applies_to', '')
                    if classification in applies_to:
                        chunk['similarity'] += 0.15  # Boost for exact classification match
                    elif 'all' in applies_to:
                        pass  # No boost or penalty for "all"
                    else:
                        chunk['similarity'] -= 0.05  # Slight penalty for non-matching classification
                
                # Boost if topic matches query topic
                if topic:
                    chunk_topics = chunk.get('topics', '')
                    if topic in chunk_topics:
                        chunk['similarity'] += 0.15  # Boost for topic match
                
                # Boost high-stakes chunks for high-stakes queries
                if urgency_tier == 'high_stakes':
                    if chunk.get('is_high_stakes'):
                        chunk['similarity'] += 0.1
                
                # NOTE: We don't filter, only boost/penalize
                # Semantic search handles relevance, metadata just re-ranks
                
                chunks.append(chunk)
        
        # Re-sort by boosted similarity and limit to n_results
        chunks.sort(key=lambda x: x['similarity'], reverse=True)
        return chunks[:n_results]
    
    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Get a specific chunk by ID."""
        result = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        
        if result['ids']:
            return {
                'chunk_id': result['ids'][0],
                'content': result['documents'][0] if result['documents'] else '',
                **result['metadatas'][0]
            }
        return None
    
    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Cleared collection '{self.collection_name}'")
    
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()


def load_chunks_from_file(chunks_file: Path = None, contract_id: str = CONTRACT_ID) -> list[dict]:
    """Load chunks from JSON file."""
    if chunks_file is None:
        chunks_file = resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)
    if chunks_file is None:
        raise FileNotFoundError("No chunk artifact found for vector index build")
    with open(chunks_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_index(clear_existing: bool = False, contract_id: str = CONTRACT_ID) -> ContractVectorStore:
    """Build or rebuild the vector index from chunks."""
    store = ContractVectorStore()
    
    if clear_existing:
        store.clear()
    
    if store.count() > 0:
        print(f"Index already contains {store.count()} documents. Use clear_existing=True to rebuild.")
        return store
    
    # Load chunks
    chunks_file = resolve_chunk_file(contract_id=contract_id, allow_shared_fallback=True)
    if not chunks_file or not chunks_file.exists():
        print(f"Error: Chunks file not found: {chunks_file}")
        print("Run parse_contract.py first to generate chunks.")
        return store
    
    chunks = load_chunks_from_file(chunks_file, contract_id=contract_id)
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")
    
    # Add to index
    added = store.add_chunks(chunks)
    print(f"Indexed {added} chunks successfully.")
    
    return store


def main():
    """Build the vector index and test search."""
    print("Building vector index...")
    store = build_index(clear_existing=True)
    
    # Test searches
    print("\n--- Test Searches ---")
    
    test_queries = [
        "What is the overtime rate?",
        "How does seniority work for layoffs?",
        "When can I take vacation?",
        "What are my rights during a disciplinary meeting?",
        "Courtesy clerk duties",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = store.search(query, n_results=3)
        for i, result in enumerate(results):
            print(f"  {i+1}. [{result['citation']}] (sim: {result['similarity']:.3f})")
            print(f"     {result['content'][:100]}...")


if __name__ == "__main__":
    main()
