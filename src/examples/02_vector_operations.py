"""
Vector operations in Pinecone including:
- Batch operations
- Vector fetching
- Update operations
- Working with high-dimensional vectors
"""

import time
import uuid
import numpy as np
from typing import List, Dict

from src.config import DEFAULT_CLOUD, DEFAULT_REGION, DEFAULT_DIMENSION, DEFAULT_METRIC
from src.utils import get_pinecone_client, create_random_vectors, wait_for_index_ready, clean_up_index
from pinecone import ServerlessSpec

def create_test_index(name_suffix: str = None) -> str:
    """Create a test index with a unique name."""
    pc = get_pinecone_client()
    
    # Create a unique index name
    unique_id = str(uuid.uuid4())[:8] if not name_suffix else name_suffix
    index_name = f"vector-ops-{unique_id}"
    
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

def batch_upsert_demo(index_name: str, batch_size: int = 50, total_vectors: int = 200):
    """Demonstrate efficient batch upserting."""
    pc = get_pinecone_client()
    index = pc.index(index_name)
    
    print(f"Performing batch upsert of {total_vectors} vectors in batches of {batch_size}")
    
    # Create all vectors at once
    all_vectors = create_random_vectors(total_vectors, DEFAULT_DIMENSION)
    
    # Batch upsert
    start_time = time.time()
    
    for i in range(0, total_vectors, batch_size):
        end_idx = min(i + batch_size, total_vectors)
        batch = [
            {
                "id": f"batch-vec-{j}",
                "values": all_vectors[j],
                "metadata": {
                    "batch": i // batch_size,
                    "position": j % batch_size,
                    "value": float(j) / total_vectors
                }
            }
            for j in range(i, end_idx)
        ]
        
        index.upsert(vectors=batch)
        print(f"Upserted batch {i//batch_size + 1}/{(total_vectors + batch_size - 1)//batch_size}")
    
    elapsed = time.time() - start_time
    print(f"Batch upsert completed in {elapsed:.2f} seconds")
    
    # Verify vector count
    time.sleep(2)  # Allow time for indexing
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.namespaces.get('', {}).get('vector_count', 0)}")

def fetch_vectors_demo(index_name: str, count: int = 5):
    """Demonstrate vector fetching operations."""
    pc = get_pinecone_client()
    index = pc.index(index_name)
    
    # Get some vector ids to fetch
    vector_ids = [f"batch-vec-{i}" for i in range(count)]
    print(f"Fetching {len(vector_ids)} vectors by ID")
    
    # Fetch the vectors
    fetch_response = index.fetch(ids=vector_ids)
    
    # Print results
    print("Fetched vectors:")
    for vec_id, vector_data in fetch_response.vectors.items():
        print(f"  ID: {vec_id}")
        print(f"  Metadata: {vector_data.metadata}")
        print(f"  Vector dimensions: {len(vector_data.values)}")
        print(f"  First 5 values: {vector_data.values[:5]}")
        print()

def update_vectors_demo(index_name: str, count: int = 5):
    """Demonstrate vector updates."""
    pc = get_pinecone_client()
    index = pc.index(index_name)
    
    # Get some vector ids to update
    vector_ids = [f"batch-vec-{i}" for i in range(count)]
    print(f"Updating {len(vector_ids)} vectors")
    
    # Fetch the current vectors
    current_vectors = index.fetch(ids=vector_ids)
    
    # Prepare updates - only updating metadata, keeping the same vectors
    updates = [
        {
            "id": vec_id,
            "values": current_vectors.vectors[vec_id].values,
            "metadata": {
                **current_vectors.vectors[vec_id].metadata,
                "updated": True,
                "update_time": time.time()
            }
        }
        for vec_id in vector_ids
    ]
    
    # Perform the updates
    index.upsert(vectors=updates)
    print("Vectors updated")
    
    # Verify updates
    updated_vectors = index.fetch(ids=vector_ids)
    print("Updated vector metadata:")
    for vec_id, vector_data in updated_vectors.vectors.items():
        print(f"  ID: {vec_id}")
        print(f"  Updated metadata: {vector_data.metadata}")
        print()

def run_demo():
    """Run the complete vector operations demo."""
    try:
        # Create index
        index_name = create_test_index()
        
        # Batch upsert
        batch_upsert_demo(index_name)
        
        # Fetch vectors
        fetch_vectors_demo(index_name)
        
        # Update vectors
        update_vectors_demo(index_name)
        
        # Clean up - uncomment to delete the index when done
        # clean_up_index(get_pinecone_client(), index_name)
        
        print("Vector operations demo completed successfully!")
        return index_name
    
    except Exception as e:
        print(f"Error in demo: {e}")
        raise

if __name__ == "__main__":
    run_demo()