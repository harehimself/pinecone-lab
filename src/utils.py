from typing import Dict, List, Optional, Union, Any
import time
import numpy as np
from pinecone import Pinecone, ServerlessSpec

from src.config import PINECONE_API_KEY

def get_pinecone_client():
    """Initialize and return a Pinecone client."""
    return Pinecone(api_key=PINECONE_API_KEY)

def create_random_vectors(count: int, dim: int) -> List[List[float]]:
    """Create random normalized vectors for testing."""
    vectors = np.random.rand(count, dim).astype(np.float32)
    # Normalize vectors
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.tolist()

def wait_for_index_ready(pc, index_name: str, timeout: int = 300):
    """Wait until the index is ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            status = pc.describe_index(index_name).status
            if status['ready']:
                print(f"Index {index_name} is ready")
                return True
            else:
                print(f"Waiting for index to be ready: {status}")
                time.sleep(10)
        except Exception as e:
            print(f"Error checking index status: {e}")
            time.sleep(5)
    
    raise TimeoutError(f"Index {index_name} did not become ready within {timeout} seconds")

def clean_up_index(pc, index_name: str):
    """Safely delete an index if it exists."""
    if pc.has_index(index_name):
        print(f"Deleting index: {index_name}")
        pc.delete_index(index_name)
        print(f"Index {index_name} deleted")
    else:
        print(f"Index {index_name} does not exist")