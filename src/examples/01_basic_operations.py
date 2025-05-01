"""
Basic Pinecone operations including:
- Creating an index
- Upserting vectors
- Querying vectors
- Deleting vectors
- Deleting the index
"""

import time
import uuid
from typing import List

import numpy as np

from pinecone import ServerlessSpec
from src.config import (DEFAULT_CLOUD, DEFAULT_DIMENSION, DEFAULT_METRIC,
                        DEFAULT_REGION)
from src.utils import (clean_up_index, create_random_vectors,
                       get_pinecone_client, wait_for_index_ready)


def create_sample_index(name_suffix: str = None) -> str:
    """Create a sample index with a unique name."""
    pc = get_pinecone_client()
    
    # Create a unique index name
    unique_id = str(uuid.uuid4())[:8] if not name_suffix else name_suffix
    index_name = f"basic-demo-{unique_id}"
    
    # Check if index already exists
    if pc.has_index(index_name):
        print(f"Index {index_name} already exists, skipping creation")
        return index_name
    
    # Create a new serverless index
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=DEFAULT_DIMENSION,
        metric=DEFAULT_METRIC,
        spec=ServerlessSpec(
            cloud=DEFAULT_CLOUD,
            region=DEFAULT_REGION
        )
    )
    
    # Wait for the index to be ready
    wait_for_index_ready(pc, index_name)
    
    return index_name

def upsert_sample_vectors(index_name: str, vector_count: int = 100):
    """Upsert random vectors to the index."""
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    # Create random vectors
    vectors = create_random_vectors(vector_count, DEFAULT_DIMENSION)
    
    # Prepare vector data with ids and metadata
    vector_data = [
        {
            "id": f"vec-{i}",
            "values": vectors[i],
            "metadata": {
                "category": f"category-{i % 5}",
                "score": np.random.random(),
                "is_valid": bool(i % 2)
            }
        }
        for i in range(vector_count)
    ]
    
    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vector_data), batch_size):
        batch = vector_data[i:i+batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted vectors {i} to {i+len(batch)-1}")
    
    # Allow time for indexing
    time.sleep(5)
    
    # Check vector count
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.namespaces.get('', {}).get('vector_count', 0)}")
    
    return vector_data

def query_vectors(index_name: str, vector_data: List, top_k: int = 5):
    """Query the index with one of the vectors."""
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    # Pick a random vector to query with
    query_idx = np.random.randint(0, len(vector_data))
    query_vector = vector_data[query_idx]["values"]
    query_id = vector_data[query_idx]["id"]
    
    print(f"Querying with vector {query_id}")
    
    # Perform the query
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    # Print results
    print(f"Query results for {query_id}:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        if match.metadata:
            print(f"  Metadata: {match.metadata}")
    
    return results

def delete_vectors(index_name: str, vector_data: List, count: int = 5):
    """Delete a subset of vectors from the index."""
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    # Select random vectors to delete
    delete_indices = np.random.choice(len(vector_data), count, replace=False)
    ids_to_delete = [vector_data[i]["id"] for i in delete_indices]
    
    print(f"Deleting vectors: {ids_to_delete}")
    index.delete(ids=ids_to_delete)
    
    # Check vector count after deletion
    time.sleep(2)  # Allow time for deletion to process
    stats = index.describe_index_stats()
    print(f"Total vectors after deletion: {stats.namespaces.get('', {}).get('vector_count', 0)}")

def run_demo():
    """Run the complete demo."""
    try:
        # Create index
        index_name = create_sample_index()
        
        # Upsert vectors
        vector_data = upsert_sample_vectors(index_name, vector_count=100)
        
        # Query vectors
        query_vectors(index_name, vector_data)
        
        # Delete vectors
        delete_vectors(index_name, vector_data)
        
        # Clean up - uncomment to delete the index when done
        # clean_up_index(get_pinecone_client(), index_name)
        
        print("Basic operations demo completed successfully!")
        return index_name
    
    except Exception as e:
        print(f"Error in demo: {e}")
        raise

if __name__ == "__main__":
    run_demo()