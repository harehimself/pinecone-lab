"""
Working with sparse vectors in Pinecone for lexical search.
- Creating a sparse index
- Upserting sparse vectors
- Querying sparse vectors
- Comparing to dense vectors
"""

import time
import uuid
from typing import Dict, List, Tuple

import numpy as np

from pinecone import ServerlessSpec
from src.config import DEFAULT_CLOUD, DEFAULT_REGION
from src.utils import clean_up_index, get_pinecone_client, wait_for_index_ready


def create_sparse_index(name_suffix: str = None) -> str:
    """Create a sparse vector index."""
    pc = get_pinecone_client()
    
    # Create a unique index name
    unique_id = str(uuid.uuid4())[:8] if not name_suffix else name_suffix
    index_name = f"sparse-demo-{unique_id}"
    
    # Check if index already exists
    if pc.has_index(index_name):
        print(f"Index {index_name} already exists, skipping creation")
        return index_name
    
    # Create a new serverless sparse index
    print(f"Creating sparse index: {index_name}")
    pc.create_index(
        name=index_name,
        vector_type="sparse",  # This is the key difference for sparse indexes
        metric="dotproduct",   # Sparse indexes only support dotproduct
        spec=ServerlessSpec(
            cloud=DEFAULT_CLOUD,
            region=DEFAULT_REGION
        )
    )
    
    # Wait for the index to be ready
    wait_for_index_ready(pc, index_name)
    
    return index_name

def create_sparse_vectors(documents: List[str]) -> List[Dict]:
    """
    Create sparse vectors from text documents using a simple TF (Term Frequency) approach.
    
    This is a very simplified example. In a real application, you would use
    better techniques like BM25 or a pre-trained sparse embedding model.
    """
    print("Creating sparse vectors for sample documents")
    
    # Create a vocabulary from all documents
    all_words = set()
    for doc in documents:
        # Tokenize by splitting on whitespace and removing punctuation
        words = [word.strip(".,?!()[]{}\"':;").lower() for word in doc.split()]
        all_words.update(words)
    
    # Sort vocabulary for consistent indices
    vocabulary = sorted(list(all_words))
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    
    print(f"Vocabulary size: {len(vocabulary)}")
    
    # Create sparse vectors
    sparse_vectors = []
    
    for i, doc in enumerate(documents):
        # Count word occurrences
        word_counts = {}
        words = [word.strip(".,?!()[]{}\"':;").lower() for word in doc.split()]
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
            values.append(float(count))  # TF (simplified)
        
        # Create vector record
        sparse_vectors.append({
            "id": f"doc-{i}",
            "sparse_values": {
                "indices": indices,
                "values": values
            },
            "metadata": {
                "text": doc,
                "word_count": len(words)
            }
        })
    
    return sparse_vectors, vocabulary

def upsert_sparse_vectors(index_name: str, sparse_vectors: List[Dict]):
    """Upsert sparse vectors to the index."""
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    print(f"Upserting {len(sparse_vectors)} sparse vectors")
    
    # Upsert all at once (usually small number in this example)
    index.upsert(vectors=sparse_vectors)
    
    # Allow time for indexing
    time.sleep(2)
    
    # Check vector count
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.namespaces.get('', {}).get('vector_count', 0)}")

def query_sparse_vector(index_name: str, query_text: str, vocabulary: List[str], top_k: int = 3):
    """Query the sparse index with a text query."""
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    print(f"\nQuerying with text: '{query_text}'")
    
    # Create sparse vector for the query
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
    
    # Check if query has any matching terms
    if not indices:
        print("Query contains no terms from the vocabulary")
        return None
    
    print(f"Query sparse vector has {len(indices)} non-zero dimensions")
    
    # Perform the query
    query_sparse_vector = {
        "indices": indices,
        "values": values
    }
    
    results = index.query(
        sparse_vector=query_sparse_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # Print results
    print("\nResults:")
    for match in results.matches:
        print(f"ID: {match.id}, Score: {match.score:.4f}")
        print(f"Text: {match.metadata['text']}")
        print()
    
    return results

def run_demo():
    """Run the complete sparse vectors demo."""
    try:
        # Sample documents (simplified for demonstration)
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
        
        # Create sparse index
        index_name = create_sparse_index()
        
        # Create and upsert sparse vectors
        sparse_vectors, vocabulary = create_sparse_vectors(sample_documents)
        upsert_sparse_vectors(index_name, sparse_vectors)
        
        # Query examples
        query_sparse_vector(index_name, "vector database", vocabulary)
        query_sparse_vector(index_name, "machine learning embeddings", vocabulary)
        query_sparse_vector(index_name, "sparse vectors for search", vocabulary)
        
        # Clean up - uncomment to delete the index when done
        # clean_up_index(get_pinecone_client(), index_name)
        
        print("\nSparse vectors demo completed successfully!")
        return index_name, vocabulary
    
    except Exception as e:
        print(f"Error in demo: {e}")
        raise

if __name__ == "__main__":
    run_demo()