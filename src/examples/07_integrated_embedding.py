"""
Using Pinecone's integrated embedding capabilities.
- Creating an index with integrated embedding model
- Upserting and searching with raw text
- Comparing with bring-your-own-vectors approach
"""

import time
import uuid
from typing import Dict, List

from pinecone import ServerlessSpec
from src.config import DEFAULT_CLOUD, DEFAULT_REGION
from src.utils import clean_up_index, get_pinecone_client, wait_for_index_ready


def create_integrated_embedding_index(name_suffix: str = None) -> str:
    """Create an index with an integrated embedding model."""
    pc = get_pinecone_client()
    
    # Create a unique index name
    unique_id = str(uuid.uuid4())[:8] if not name_suffix else name_suffix
    index_name = f"integrated-embed-{unique_id}"
    
    # Check if index already exists
    if pc.has_index(index_name):
        print(f"Index {index_name} already exists, skipping creation")
        return index_name
    
    # Create a new index with integrated embedding
    print(f"Creating index with integrated embedding: {index_name}")
    
    pc.create_index_for_model(
        name=index_name,
        cloud=DEFAULT_CLOUD,
        region=DEFAULT_REGION,
        embed={
            "model": "multilingual-e5-large",  # Pinecone's hosted embedding model
            "field_map": {"text": "chunk_text"}  # Maps "text" field to "chunk_text" in the model
        }
    )
    
    # Wait for the index to be ready
    wait_for_index_ready(pc, index_name)
    
    return index_name

def upsert_text_documents(index_name: str, documents: List[Dict]):
    """Upsert text documents directly to the index with integrated embedding."""
    pc = get_pinecone_client()
    index = pc.index(index_name)
    
    print(f"Upserting {len(documents)} text documents")
    
    # Prepare records with the expected text field
    records = [
        {
            "id": doc["id"],
            "text": doc["text"],  # This will be automatically embedded using the integrated model
            "metadata": {
                "category": doc.get("category", "general"),
                "length": len(doc["text"])
            }
        }
        for doc in documents
    ]
    
    # Upsert all at once
    index.upsert(vectors=records)
    
    # Allow time for processing
    time.sleep(3)
    
    # Check vector count
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.namespaces.get('', {}).get('vector_count', 0)}")

def query_with_text(index_name: str, query_text: str, top_k: int = 3):
    """Query the index directly with text using the integrated embedding."""
    pc = get_pinecone_client()
    index = pc.index(index_name)
    
    print(f"\nQuerying with text: '{query_text}'")
    
    # Perform the query with text directly
    results = index.query(
        text=query_text,
        top_k=top_k,
        include_metadata=True
    )
    
    # Print results
    print("\nResults:")
    for match in results.matches:
        print(f"ID: {match.id}, Score: {match.score:.4f}")
        if match.metadata:
            print(f"Text: {match.metadata.get('text', 'No text available')}")
        print()
    
    return results

def run_demo():
    """Run the complete integrated embedding demo."""
    try:
        # Create an index with integrated embedding
        index_name = create_integrated_embedding_index()
        
        # Sample documents
        sample_documents = [
            {
                "id": "doc-1",
                "text": "Pinecone provides vector databases for machine learning applications.",
                "category": "product"
            },
            {
                "id": "doc-2",
                "text": "Vector databases enable efficient similarity search across millions of vectors.",
                "category": "technical"
            },
            {
                "id": "doc-3",
                "text": "Embeddings transform data like text, images, or audio into vectors for machine learning.",
                "category": "concept"
            },
            {
                "id": "doc-4",
                "text": "Serverless deployment simplifies infrastructure management for vector databases.",
                "category": "deployment"
            },
            {
                "id": "doc-5",
                "text": "Semantic search uses vector similarity to find related content rather than keyword matching.",
                "category": "concept"
            },
            {
                "id": "doc-6",
                "text": "Metadata filtering lets you combine vector search with traditional database queries.",
                "category": "feature"
            },
            {
                "id": "doc-7",
                "text": "Pinecone's integrated embedding feature simplifies the vector database workflow.",
                "category": "feature"
            },
            {
                "id": "doc-8",
                "text": "Machine learning models are increasingly used for natural language understanding tasks.",
                "category": "technical"
            },
            {
                "id": "doc-9",
                "text": "Cloud-based vector databases scale automatically to handle growing data volumes.",
                "category": "deployment"
            },
            {
                "id": "doc-10",
                "text": "Recommendation systems use vector similarity to suggest relevant items to users.",
                "category": "application"
            }
        ]
        
        # Upsert documents
        upsert_text_documents(index_name, sample_documents)
        
        # Query examples
        query_with_text(index_name, "How do vector databases work?")
        query_with_text(index_name, "What is semantic search?")
        query_with_text(index_name, "Tell me about Pinecone features")
        
        # Clean up - uncomment to delete the index when done
        # clean_up_index(get_pinecone_client(), index_name)
        
        print("\nIntegrated embedding demo completed successfully!")
        return index_name
    
    except Exception as e:
        print(f"Error in demo: {e}")
        raise

if __name__ == "__main__":
    run_demo()