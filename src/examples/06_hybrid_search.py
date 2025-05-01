"""
Hybrid search in Pinecone combining dense and sparse vectors.
- Creating separate dense and sparse indexes
- Generating both vector types
- Performing hybrid search
- Comparing results with pure dense and pure sparse search
"""

import time
import uuid
import numpy as np
from typing import List, Dict, Tuple

from src.config import DEFAULT_CLOUD, DEFAULT_REGION, DEFAULT_DIMENSION, DEFAULT_METRIC
from src.utils import get_pinecone_client, create_random_vectors, wait_for_index_ready, clean_up_index
from pinecone import ServerlessSpec

def create_indexes(sparse_suffix: str = None, dense_suffix: str = None) -> Tuple[str, str]:
    """Create both sparse and dense indexes."""
    pc = get_pinecone_client()
    
    # Create unique index names
    unique_id = str(uuid.uuid4())[:8]
    sparse_index_name = f"hybrid-sparse-{sparse_suffix or unique_id}"
    dense_index_name = f"hybrid-dense-{dense_suffix or unique_id}"
    
    # Create sparse index if it doesn't exist
    if pc.has_index(sparse_index_name):
        print(f"Sparse index {sparse_index_name} already exists, skipping creation")
    else:
        print(f"Creating sparse index: {sparse_index_name}")
        pc.create_index(
            name=sparse_index_name,
            vector_type="sparse",
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud=DEFAULT_CLOUD,
                region=DEFAULT_REGION
            )
        )
        wait_for_index_ready(pc, sparse_index_name)
    
    # Create dense index if it doesn't exist
    if pc.has_index(dense_index_name):
        print(f"Dense index {dense_index_name} already exists, skipping creation")
    else:
        print(f"Creating dense index: {dense_index_name}")
        pc.create_index(
            name=dense_index_name,
            vector_type="dense",
            dimension=DEFAULT_DIMENSION,
            metric=DEFAULT_METRIC,
            spec=ServerlessSpec(
                cloud=DEFAULT_CLOUD,
                region=DEFAULT_REGION
            )
        )
        wait_for_index_ready(pc, dense_index_name)
    
    return sparse_index_name, dense_index_name

def create_sample_data() -> List[Dict]:
    """Create sample document data for hybrid search."""
    sample_documents = [
        "Pinecone is a vector database for machine learning applications.",
        "Vector databases store and query high-dimensional vectors efficiently.",
        "Machine learning models often use vector embeddings to represent data.",
        "Sparse vectors are useful for keyword and lexical search.",
        "Dense vectors are good for semantic search and similarity matching.",
        "Hybrid search combines both sparse and dense vectors for better results.",
        "Pinecone supports serverless indexes for both sparse and dense vectors.",
        "Vector similarity search is faster than traditional database queries.",
        "Embeddings convert text, images, or audio into numerical vectors.",
        "Cosine similarity measures the angle between two vectors."
    ]
    
    return [{"id": f"doc-{i}", "text": doc} for i, doc in enumerate(sample_documents)]

def create_sparse_vectors(documents: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """Create sparse vectors for documents using a simple TF approach."""
    print("Creating sparse vectors for sample documents")
    
    # Create a vocabulary from all documents
    all_words = set()
    for doc in documents:
        words = [word.strip(".,?!()[]{}\"':;").lower() for word in doc["text"].split()]
        all_words.update(words)
    
    # Sort vocabulary for consistent indices
    vocabulary = sorted(list(all_words))
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    
    print(f"Vocabulary size: {len(vocabulary)}")
    
    # Create sparse vectors
    sparse_vectors = []
    
    for doc in documents:
        # Count word occurrences
        word_counts = {}
        words = [word.strip(".,?!()[]{}\"':;").lower() for word in doc["text"].split()]
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Convert to sparse representation
        indices = []
        values = []
        for word, count in word_counts.items():
            indices.append(word_to_index[word])
            values.append(float(count))
        
        # Create vector record
        sparse_vectors.append({
            "id": doc["id"],
            "sparse_values": {
                "indices": indices,
                "values": values
            },
            "metadata": {
                "text": doc["text"],
                "word_count": len(words)
            }
        })
    
    return sparse_vectors, vocabulary

def create_dense_vectors(documents: List[Dict], dimension: int = DEFAULT_DIMENSION) -> List[Dict]:
    """
    Create mock dense vectors for documents.
    
    In a real application, these would come from a text embedding model.
    Here we just create random vectors for demonstration purposes.
    """
    print("Creating dense vectors for sample documents")
    
    # Create random vectors (these would normally come from an embedding model)
    # We ensure vectors are deterministic by setting a seed based on document content
    dense_vectors = []
    
    for doc in documents:
        # Create a seed based on the document text
        seed = sum(ord(c) for c in doc["text"]) % 10000
        np.random.seed(seed)
        
        # Create a random normalized vector
        vector = np.random.rand(dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        # Create vector record
        dense_vectors.append({
            "id": doc["id"],
            "values": vector.tolist(),
            "metadata": {
                "text": doc["text"],
                "word_count": len(doc["text"].split())
            }
        })
    
    return dense_vectors

def upsert_vectors(sparse_index_name: str, dense_index_name: str, 
                  sparse_vectors: List[Dict], dense_vectors: List[Dict]):
    """Upsert vectors to their respective indexes."""
    pc = get_pinecone_client()
    sparse_index = pc.index(sparse_index_name)
    dense_index = pc.index(dense_index_name)
    
    # Upsert sparse vectors
    print(f"Upserting {len(sparse_vectors)} sparse vectors")
    sparse_index.upsert(vectors=sparse_vectors)
    
    # Upsert dense vectors
    print(f"Upserting {len(dense_vectors)} dense vectors")
    dense_index.upsert(vectors=dense_vectors)
    
    # Allow time for indexing
    time.sleep(2)
    
    # Check vector counts
    sparse_stats = sparse_index.describe_index_stats()
    dense_stats = dense_index.describe_index_stats()
    
    print(f"Sparse index vector count: {sparse_stats.namespaces.get('', {}).get('vector_count', 0)}")
    print(f"Dense index vector count: {dense_stats.namespaces.get('', {}).get('vector_count', 0)}")

def create_query_vectors(query_text: str, vocabulary: List[str], dimension: int = DEFAULT_DIMENSION) -> Tuple[Dict, List[float]]:
    """Create both sparse and dense vectors for a query."""
    print(f"\nPreparing vectors for query: '{query_text}'")
    
    # Create sparse vector for query
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    
    # Count word occurrences in query
    word_counts = {}
    words = [word.strip(".,?!()[]{}\"':;").lower() for word in query_text.split()]
    for word in words:
        if word in word_counts and word in word_to_index:
            word_counts[word] += 1
        elif word in word_to_index:
            word_counts[word] = 1
    
    # Convert to sparse representation
    indices = []
    values = []
    for word, count in word_counts.items():
        indices.append(word_to_index[word])
        values.append(float(count))
    
    query_sparse_vector = {
        "indices": indices,
        "values": values
    }
    
    # Create dense vector for query (mock for demonstration)
    # In a real app, this would come from the same embedding model used for documents
    seed = sum(ord(c) for c in query_text) % 10000
    np.random.seed(seed)
    vector = np.random.rand(dimension).astype(np.float32)
    query_dense_vector = (vector / np.linalg.norm(vector)).tolist()
    
    print(f"Query sparse vector has {len(indices)} non-zero dimensions")
    print(f"Query dense vector has {dimension} dimensions")
    
    return query_sparse_vector, query_dense_vector

def search_sparse(sparse_index_name: str, query_sparse_vector: Dict, top_k: int = 3):
    """Perform a sparse vector search."""
    pc = get_pinecone_client()
    sparse_index = pc.index(sparse_index_name)
    
    print("\n--- Sparse Search Results ---")
    sparse_results = sparse_index.query(
        sparse_vector=query_sparse_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    for match in sparse_results.matches:
        print(f"ID: {match.id}, Score: {match.score:.4f}")
        print(f"Text: {match.metadata['text']}")
        print()
    
    return sparse_results

def search_dense(dense_index_name: str, query_dense_vector: List[float], top_k: int = 3):
    """Perform a dense vector search."""
    pc = get_pinecone_client()
    dense_index = pc.index(dense_index_name)
    
    print("\n--- Dense Search Results ---")
    dense_results = dense_index.query(
        vector=query_dense_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    for match in dense_results.matches:
        print(f"ID: {match.id}, Score: {match.score:.4f}")
        print(f"Text: {match.metadata['text']}")
        print()
    
    return dense_results

def perform_hybrid_search(sparse_results, dense_results, alpha: float = 0.5):
    """
    Combine sparse and dense results for hybrid search.
    
    In a real application, this would be done automatically by Pinecone's hybrid search.
    We're simulating it here for learning purposes.
    
    alpha: Weight for sparse results (0.0 to 1.0), with (1-alpha) weight for dense results
    """
    print(f"\n--- Hybrid Search Results (alpha={alpha}) ---")
    
    # Combine results from both searches
    combined_scores = {}
    
    # Process sparse results
    for match in sparse_results.matches:
        combined_scores[match.id] = {
            "sparse_score": match.score,
            "dense_score": 0.0,
            "metadata": match.metadata
        }
    
    # Process dense results
    for match in dense_results.matches:
        if match.id in combined_scores:
            combined_scores[match.id]["dense_score"] = match.score
        else:
            combined_scores[match.id] = {
                "sparse_score": 0.0,
                "dense_score": match.score,
                "metadata": match.metadata
            }
    
    # Calculate weighted scores
    weighted_scores = []
    for doc_id, scores in combined_scores.items():
        hybrid_score = alpha * scores["sparse_score"] + (1 - alpha) * scores["dense_score"]
        weighted_scores.append({
            "id": doc_id,
            "hybrid_score": hybrid_score,
            "sparse_score": scores["sparse_score"],
            "dense_score": scores["dense_score"],
            "metadata": scores["metadata"]
        })
    
    # Sort by hybrid score
    weighted_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    # Print top results
    for i, result in enumerate(weighted_scores[:5]):
        print(f"ID: {result['id']}, Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"  Sparse Score: {result['sparse_score']:.4f}, Dense Score: {result['dense_score']:.4f}")
        print(f"  Text: {result['metadata']['text']}")
        print()
    
    return weighted_scores

def run_demo():
    """Run the complete hybrid search demo."""
    try:
        # Create indexes
        sparse_index_name, dense_index_name = create_indexes()
        
        # Create sample data
        documents = create_sample_data()
        
        # Create vectors
        sparse_vectors, vocabulary = create_sparse_vectors(documents)
        dense_vectors = create_dense_vectors(documents)
        
        # Upsert vectors
        upsert_vectors(sparse_index_name, dense_index_name, sparse_vectors, dense_vectors)
        
        # Example queries
        queries = [
            "vector database search",
            "machine learning embeddings",
            "Pinecone similarity"
        ]
        
        for query in queries:
            # Create query vectors
            query_sparse, query_dense = create_query_vectors(query, vocabulary)
            
            # Perform searches
            sparse_results = search_sparse(sparse_index_name, query_sparse)
            dense_results = search_dense(dense_index_name, query_dense)
            
            # Perform hybrid search with different weightings
            hybrid_results_1 = perform_hybrid_search(sparse_results, dense_results, alpha=0.3)
            hybrid_results_2 = perform_hybrid_search(sparse_results, dense_results, alpha=0.7)
            
            print("\n" + "="*50 + "\n")
        
        # Clean up - uncomment to delete the indexes when done
        # clean_up_index(get_pinecone_client(), sparse_index_name)
        # clean_up_index(get_pinecone_client(), dense_index_name)
        
        print("\nHybrid search demo completed successfully!")
        return sparse_index_name, dense_index_name
    
    except Exception as e:
        print(f"Error in demo: {e}")
        raise

if __name__ == "__main__":
    run_demo()