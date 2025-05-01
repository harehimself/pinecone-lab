"""
Working with Pinecone namespaces to isolate vector collections
- Creating and using multiple namespaces
- Querying across namespaces
- Namespace statistics
- Isolating data between tenants
"""

import time
import uuid
from typing import Dict, List

import numpy as np

from pinecone import ServerlessSpec
from src.config import (DEFAULT_CLOUD, DEFAULT_DIMENSION, DEFAULT_METRIC,
                        DEFAULT_REGION)
from src.utils import (clean_up_index, create_random_vectors,
                       get_pinecone_client, wait_for_index_ready)


def create_namespace_index(name_suffix: str = None) -> str:
    """Create a test index with a unique name."""
    pc = get_pinecone_client()
    
    # Create a unique index name
    unique_id = str(uuid.uuid4())[:8] if not name_suffix else name_suffix
    index_name = f"namespace-demo-{unique_id}"
    
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

def populate_namespaces(index_name: str, namespaces: List[str], vectors_per_namespace: int = 50):
    """Populate multiple namespaces with vectors."""
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    print(f"Populating {len(namespaces)} namespaces with {vectors_per_namespace} vectors each")
    
    for ns in namespaces:
        # Create random vectors for this namespace
        vectors = create_random_vectors(vectors_per_namespace, DEFAULT_DIMENSION)
        
        # Prepare vector data with ids and metadata
        vector_data = [
            {
                "id": f"{ns}-vec-{i}",
                "values": vectors[i],
                "metadata": {
                    "namespace": ns,
                    "category": f"category-{i % 3}",
                    "value": float(i) / vectors_per_namespace
                }
            }
            for i in range(vectors_per_namespace)
        ]
        
        # Upsert vectors to the specific namespace
        index.upsert(vectors=vector_data, namespace=ns)
        print(f"Upserted {vectors_per_namespace} vectors to namespace '{ns}'")
    
    # Allow time for indexing
    time.sleep(3)

def namespace_stats(index_name: str):
    """Display statistics about each namespace."""
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    # Get index stats
    stats = index.describe_index_stats()
    
    print("\nNamespace Statistics:")
    if not stats.namespaces:
        print("  No namespaces found")
        return
    
    # Print statistics for each namespace
    for ns_name, ns_stats in stats.namespaces.items():
        ns_display = ns_name if ns_name else "default"
        print(f"  Namespace: {ns_display}")
        print(f"    Vector count: {ns_stats.vector_count}")
        
        # Some indexes might not return dimension info
        if hasattr(ns_stats, 'dimension'):
            print(f"    Dimension: {ns_stats.dimension}")
            
        # Print index fullness if available
        if hasattr(ns_stats, 'index_fullness'):
            print(f"    Index fullness: {ns_stats.Index_fullness:.2f}")
        
        print()

def query_across_namespaces(index_name: str, namespaces: List[str], top_k: int = 3):
    """Query vectors across multiple namespaces."""
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    # Get a random vector from the first namespace to use as a query
    query_ns = namespaces[0]
    query_id = f"{query_ns}-vec-{np.random.randint(0, 10)}"
    
    # Fetch the vector to use as query
    fetch_result = index.fetch(ids=[query_id], namespace=query_ns)
    if query_id not in fetch_result.vectors:
        print(f"Could not find vector {query_id} in namespace {query_ns}")
        return
    
    query_vector = fetch_result.vectors[query_id].values
    
    print(f"\nQuerying with vector {query_id} from namespace '{query_ns}'")
    
    # Query each namespace individually
    for ns in namespaces:
        print(f"\nResults from namespace '{ns}':")
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=ns,
            include_metadata=True
        )
        
        for match in results.matches:
            print(f"  ID: {match.id}, Score: {match.score:.4f}")
            if match.metadata:
                print(f"  Metadata: {match.metadata}")

def delete_namespace_content(index_name: str, namespace: str):
    """Delete all vectors in a namespace."""
    pc = get_pinecone_client()
    index = pc.Index(index_name)
    
    print(f"\nDeleting all vectors in namespace '{namespace}'")
    
    # Delete all vectors in the namespace
    index.delete(delete_all=True, namespace=namespace)
    
    # Allow time for deletion to process
    time.sleep(2)
    
    # Verify deletion
    stats = index.describe_index_stats()
    ns_count = stats.namespaces.get(namespace, {}).get('vector_count', 0)
    print(f"Vectors in namespace '{namespace}' after deletion: {ns_count}")

def run_demo():
    """Run the complete namespaces demo."""
    try:
        # Create index
        index_name = create_namespace_index()
        
        # Define namespaces to use
        namespaces = ["customer1", "customer2", "internal"]
        
        # Populate namespaces
        populate_namespaces(index_name, namespaces)
        
        # Show namespace statistics
        namespace_stats(index_name)
        
        # Query across namespaces
        query_across_namespaces(index_name, namespaces)
        
        # Delete a namespace's content
        delete_namespace_content(index_name, "customer2")
        
        # Show updated statistics
        namespace_stats(index_name)
        
        # Clean up - uncomment to delete the index when done
        # clean_up_index(get_pinecone_client(), index_name)
        
        print("Namespaces demo completed successfully!")
        return index_name
    
    except Exception as e:
        print(f"Error in demo: {e}")
        raise

if __name__ == "__main__":
    run_demo()