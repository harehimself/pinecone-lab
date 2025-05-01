"""
Metadata filtering in Pinecone including:
- Basic metadata filters
- Complex filters with logical operators
- Range filters
- Regex filters
- Performance considerations
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


def create_metadata_index(name_suffix: str = None) -> str:
    """Create a test index with a unique name."""
    pc = get_pinecone_client()
    
    # Create a unique index name
    unique_id = str(uuid.uuid4())[:8] if not name_suffix else name_suffix
    index_name = f"metadata-demo-{unique_id}"
    
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

def populate_index_with_rich_metadata(index_name: str, vector_count: int = 1000):
    """Populate index with vectors that have rich metadata."""
    pc = get_pinecone_client()
    index = pc.index(index_name)
    
    print(f"Populating index with {vector_count} vectors containing rich metadata")
    
    # Create random vectors
    vectors = create_random_vectors(vector_count, DEFAULT_DIMENSION)
    
    # Define categories and tags for variety
    categories = ["electronics", "clothing", "furniture", "food", "toys"]
    tags = ["sale", "new", "featured", "limited", "seasonal", "clearance"]
    regions = ["north", "south", "east", "west", "central"]
    
    # Prepare vector data with rich metadata
    current_date = time.time()
    vector_data = []
    
    for i in range(vector_count):
        # Create rich metadata
        category = categories[i % len(categories)]
        num_tags = np.random.randint(1, 4)  # 1-3 tags per vector
        selected_tags = np.random.choice(tags, num_tags, replace=False).tolist()
        region = regions[i % len(regions)]
        
        # Generate a price between $5 and $500
        price = round(5 + np.random.random() * 495, 2)
        
        # Generate a date within the last year (in seconds)
        days_ago = np.random.randint(0, 365)
        timestamp = current_date - (days_ago * 86400)  # 86400 seconds in a day
        
        # Generate in_stock boolean (80% in stock)
        in_stock = np.random.random() < 0.8
        
        # Generate stock_count (0-100)
        stock_count = 0 if not in_stock else np.random.randint(1, 101)
        
        # Create a vector record
        vector_data.append({
            "id": f"item-{i}",
            "values": vectors[i],
            "metadata": {
                "category": category,
                "tags": selected_tags,
                "region": region,
                "price": price,
                "timestamp": timestamp,
                "created_date": time.strftime("%Y-%m-%d", time.localtime(timestamp)),
                "in_stock": in_stock,
                "stock_count": stock_count,
                "rating": round(np.random.random() * 5, 1),  # 0-5 rating with one decimal
                "reviews": np.random.randint(0, 1000),  # 0-999 reviews
                "product_id": f"PROD-{np.random.randint(10000, 99999)}"
            }
        })
    
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

def run_basic_filters(index_name: str, query_vector):
    """Demonstrate basic metadata filters."""
    pc = get_pinecone_client()
    index = pc.index(index_name)
    
    print("\n--- Basic Metadata Filters ---")
    
    # Example 1: Equality filter
    print("\nFilter: category = 'electronics'")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={"category": "electronics"}
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")
    
    # Example 2: Equality filter with $eq operator (same as above)
    print("\nFilter: category $eq 'electronics'")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={"category": {"$eq": "electronics"}}
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")
    
    # Example 3: Inequality filter
    print("\nFilter: category $ne 'electronics'")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={"category": {"$ne": "electronics"}}
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")
    
    # Example 4: Boolean filter
    print("\nFilter: in_stock = true")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={"in_stock": True}
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")
    
    # Example 5: Numerical comparison
    print("\nFilter: price < 50")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={"price": {"$lt": 50}}
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")

def run_complex_filters(index_name: str, query_vector):
    """Demonstrate complex metadata filters with logical operators."""
    pc = get_pinecone_client()
    index = pc.index(index_name)
    
    print("\n--- Complex Metadata Filters with Logical Operators ---")
    
    # Example 1: AND operator
    print("\nFilter: category = 'electronics' AND price < 100")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={
            "$and": [
                {"category": "electronics"},
                {"price": {"$lt": 100}}
            ]
        }
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")
    
    # Example 2: OR operator
    print("\nFilter: category = 'electronics' OR category = 'toys'")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={
            "$or": [
                {"category": "electronics"},
                {"category": "toys"}
            ]
        }
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")
    
    # Example 3: Nested logical operators
    print("\nFilter: (category = 'electronics' AND price < 100) OR (category = 'toys' AND rating > 4.5)")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={
            "$or": [
                {
                    "$and": [
                        {"category": "electronics"},
                        {"price": {"$lt": 100}}
                    ]
                },
                {
                    "$and": [
                        {"category": "toys"},
                        {"rating": {"$gt": 4.5}}
                    ]
                }
            ]
        }
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")
    
    # Example 4: NOT operator
    print("\nFilter: NOT (category = 'electronics' OR category = 'toys')")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={
            "$not": {
                "$or": [
                    {"category": "electronics"},
                    {"category": "toys"}
                ]
            }
        }
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")

def run_range_filters(index_name: str, query_vector):
    """Demonstrate range filters for numerical values and dates."""
    pc = get_pinecone_client()
    index = pc.index(index_name)
    
    print("\n--- Range Filters ---")
    
    # Example 1: Price range
    print("\nFilter: 50 <= price <= 150")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={
            "price": {
                "$gte": 50,
                "$lte": 150
            }
        }
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")
    
    # Example 2: Date range (using timestamps)
    # Find items from last 90 days
    current_time = time.time()
    ninety_days_ago = current_time - (90 * 86400)
    
    print("\nFilter: Created in the last 90 days")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={
            "timestamp": {
                "$gte": ninety_days_ago
            }
        }
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")
    
    # Example 3: Multiple ranges
    print("\nFilter: 4 <= rating <= 5 AND reviews >= 100")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={
            "$and": [
                {
                    "rating": {
                        "$gte": 4.0,
                        "$lte": 5.0
                    }
                },
                {
                    "reviews": {
                        "$gte": 100
                    }
                }
            ]
        }
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")

def run_array_contains_filters(index_name: str, query_vector):
    """Demonstrate filters for array field contains."""
    pc = get_pinecone_client()
    index = pc.index(index_name)
    
    print("\n--- Array Contains Filters ---")
    
    # Example 1: Array contains specific value
    print("\nFilter: tags contains 'sale'")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={
            "tags": {
                "$in": ["sale"]
            }
        }
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")
    
    # Example 2: Array contains any of multiple values
    print("\nFilter: tags contains 'sale' OR 'new'")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={
            "tags": {
                "$in": ["sale", "new"]
            }
        }
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")
    
    # Example 3: Array contains all of multiple values
    print("\nFilter: tags contains both 'sale' AND 'new'")
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        filter={
            "$and": [
                {"tags": {"$in": ["sale"]}},
                {"tags": {"$in": ["new"]}}
            ]
        }
    )
    
    print("Results:")
    for match in results.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}")
        print(f"  Metadata: {match.metadata}")

def run_demo():
    """Run the complete metadata filtering demo."""
    try:
        # Create index
        index_name = create_metadata_index()
        
        # Populate with rich metadata
        vector_data = populate_index_with_rich_metadata(index_name, vector_count=1000)
        
        # Use a random vector for queries
        query_idx = np.random.randint(0, len(vector_data))
        query_vector = vector_data[query_idx]["values"]
        print(f"Using vector from ID: {vector_data[query_idx]['id']} for all queries")
        
        # Run different filter types
        run_basic_filters(index_name, query_vector)
        run_complex_filters(index_name, query_vector)
        run_range_filters(index_name, query_vector)
        run_array_contains_filters(index_name, query_vector)
        
        # Clean up - uncomment to delete the index when done
        # clean_up_index(get_pinecone_client(), index_name)
        
        print("\nMetadata filtering demo completed successfully!")
        return index_name
    
    except Exception as e:
        print(f"Error in demo: {e}")
        raise

if __name__ == "__main__":
    run_demo()