"""
Vector Store - ChromaDB wrapper for contract chunk storage and retrieval.

Uses sentence-transformers for local embeddings (no API key needed).
Supports metadata filtering for contract, classification, and topic.
"""

import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import (
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL, COLLECTION_NAME,
    CHUNKS_DIR, TOP_K_RESULTS, SIMILARITY_THRESHOLD
)

# Lazy imports for optional dependencies
chromadb = None
SentenceTransformer = None


def _load_dependencies():
    """Lazy load heavy dependencies."""
    global chromadb, SentenceTransformer
    if chromadb is None:
        import chromadb as _chromadb
        chromadb = _chromadb
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST


class ContractVectorStore:
    """Vector store for union contract chunks using ChromaDB."""
    
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
                content = chunk['content']
                
                # Create embedding
                embedding = self.embedder.encode(content).tolist()
                
                # Prepare metadata (ChromaDB only supports str, int, float, bool)
                # Handle both old format (topic_tags) and new format (topics)
                topics = chunk.get('topics', chunk.get('topic_tags', []))
                if isinstance(topics, str):
                    topics = topics.split(',') if topics else []
                
                applies_to = chunk.get('applies_to', ['all'])
                if isinstance(applies_to, str):
                    applies_to = applies_to.split(',') if applies_to else ['all']
                
                metadata = {
                    'contract_id': chunk.get('contract_id', ''),
                    'article_num': chunk.get('article_num') or 0,
                    'article_title': chunk.get('article_title', ''),
                    'section_num': chunk.get('section_num') or 0,
                    'subsection': chunk.get('subsection') or '',
                    'citation': chunk.get('citation', ''),
                    'parent_context': chunk.get('parent_context', ''),
                    'doc_type': chunk.get('doc_type', 'cba'),
                    # New enriched metadata
                    'applies_to': ','.join(applies_to),
                    'topics': ','.join(topics),
                    'summary': chunk.get('summary') or '',
                    'is_definition': chunk.get('is_definition', False),
                    'is_exception': chunk.get('is_exception', False),
                    'hire_date_sensitive': chunk.get('hire_date_sensitive', False),
                    'is_high_stakes': chunk.get('is_high_stakes', False),
                }
                
                ids.append(chunk_id)
                documents.append(content)
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
        
        if contract_id:
            where_clauses.append({"contract_id": contract_id})
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
                
                chunk = {
                    'chunk_id': chunk_id,
                    'content': results['documents'][0][i] if results['documents'] else '',
                    'similarity': similarity,
                    **results['metadatas'][0][i]
                }
                
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


def load_chunks_from_file(chunks_file: Path = None) -> list[dict]:
    """Load chunks from JSON file."""
    if chunks_file is None:
        chunks_file = CHUNKS_DIR / "contract_chunks.json"
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_index(clear_existing: bool = False) -> ContractVectorStore:
    """Build or rebuild the vector index from chunks."""
    store = ContractVectorStore()
    
    if clear_existing:
        store.clear()
    
    if store.count() > 0:
        print(f"Index already contains {store.count()} documents. Use clear_existing=True to rebuild.")
        return store
    
    # Load chunks
    chunks_file = CHUNKS_DIR / "contract_chunks.json"
    if not chunks_file.exists():
        print(f"Error: Chunks file not found: {chunks_file}")
        print("Run parse_contract.py first to generate chunks.")
        return store
    
    chunks = load_chunks_from_file(chunks_file)
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

