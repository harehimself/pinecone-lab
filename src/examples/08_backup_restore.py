"""
Working with Pinecone backup and restore operations.
- Creating a source index and adding data
- Creating a backup
- Restoring from a backup to a new index
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


def create_source_index(name_suffix: str = None) -> str:
    """Create a source index that will be backed up."""
    pc = get_pinecone_client()
    
    # Create a unique index name
    unique_id = str(uuid.uuid4())[:8] if not name_suffix else name_suffix
    index_name = f"backup-source-{unique_id}"
    
    # Check if index already exists
    if pc.has_index(index_name):
        print(f"Index {index_name} already exists, skipping creation")
        return index_name
    
    # Create a new serverless index
    print(f"Creating source index: {index_name}")
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

def populate_source_index(index_name: str, vector_count: int = 100):
    """Populate the source index with sample vectors."""
    pc = get_pinecone_client()
    index = pc.index(index_name)
    
    print(f"Populating source index with {vector_count} vectors")
    
    # Create random vectors
    vectors = create_random_vectors(vector_count, DEFAULT_DIMENSION)
    
    # Prepare vector data with ids and metadata
    vector_data = [
        {
            "id": f"backup-vec-{i}",
            "values": vectors[i],
            "metadata": {
                "category": f"category-{i % 5}",
                "value": round(np.random.random(), 2),
                "is_backup_demo": True
            }
        }
        for i in range(vector_count)
    ]
    
    # Upsert vectors
    batch_size = 100
    for i in range(0, len(vector_data), batch_size):
        batch = vector_data[i:i+batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted vectors {i} to {i+len(batch)-1}")
    
    # Allow time for indexing
    time.sleep(3)
    
    # Check vector count
    stats = index.describe_index_stats()
    print(f"Total vectors in source index: {stats.namespaces.get('', {}).get('vector_count', 0)}")
    
    return vector_data

def create_backup(index_name: str) -> str:
    """Create a backup of the source index."""
    pc = get_pinecone_client()
    
    # Generate a unique backup name
    backup_name = f"backup-{str(uuid.uuid4())[:8]}"
    
    print(f"Creating backup: {backup_name} from index: {index_name}")
    
    # Create the backup
    pc.create_backup(
        source_index=index_name,
        backup_name=backup_name
    )
    
    # Wait for backup completion
    print("Waiting for backup to complete...")
    
    while True:
        try:
            # Check backup status
            backup_info = pc.describe_backup(
                source_index=index_name,
                backup_name=backup_name
            )
            
            state = backup_info.state
            print(f"Backup state: {state}")
            
            if state == "Completed":
                print(f"Backup completed successfully: {backup_name}")
                break
            elif state in ["Failed", "Cancelled"]:
                raise Exception(f"Backup failed with state: {state}")
            
            # Wait before checking again
            time.sleep(5)
            
        except Exception as e:
            print(f"Error checking backup status: {e}")
            time.sleep(5)
    
    return backup_name

def restore_from_backup(source_index: str, backup_name: str) -> str:
    """Restore a new index from a backup."""
    pc = get_pinecone_client()
    
    # Generate a unique target index name
    target_index = f"restore-{str(uuid.uuid4())[:8]}"
    
    print(f"Restoring from backup: {backup_name} to new index: {target_index}")
    
    # Start the restore operation
    pc.restore_backup(
        source_index=source_index,
        backup_name=backup_name,
        target_index=target_index
    )
    
    # Wait for restore completion
    print("Waiting for restore to complete...")
    
    while True:
        try:
            # Check restore status
            restore_info = pc.describe_restore(
                source_index=source_index,
                backup_name=backup_name,
                target_index=target_index
            )
            
            state = restore_info.state
            print(f"Restore state: {state}")
            
            if state == "Completed":
                print(f"Restore completed successfully to index: {target_index}")
                break
            elif state in ["Failed", "Cancelled"]:
                raise Exception(f"Restore failed with state: {state}")
            
            # Wait before checking again
            time.sleep(5)
            
        except Exception as e:
            print(f"Error checking restore status: {e}")
            time.sleep(5)
    
    return target_index

def verify_restore(source_index: str, target_index: str):
    """Verify that the restored index matches the source index."""
    pc = get_pinecone_client()
    source = pc.index(source_index)
    target = pc.index(target_index)
    
    # Get index stats
    source_stats = source.describe_index_stats()
    target_stats = target.describe_index_stats()
    
    # Compare vector counts
    source_count = source_stats.namespaces.get('', {}).get('vector_count', 0)
    target_count = target_stats.namespaces.get('', {}).get('vector_count', 0)
    
    print(f"Source index vector count: {source_count}")
    print(f"Restored index vector count: {target_count}")
    
    if source_count == target_count:
        print("✅ Vector counts match!")
    else:
        print("❌ Vector counts do not match!")
    
    # Sample a few vectors to compare
    print("\nSampling vectors to compare content...")
    
    # Get some sample IDs from source
    vector_ids = [f"backup-vec-{i}" for i in range(5)]
    
    # Fetch vectors from both indexes
    source_vectors = source.fetch(ids=vector_ids)
    target_vectors = target.fetch(ids=vector_ids)
    
    # Compare the vectors
    all_match = True
    for vec_id in vector_ids:
        if vec_id not in source_vectors.vectors or vec_id not in target_vectors.vectors:
            print(f"❌ Vector {vec_id} missing in one of the indexes!")
            all_match = False
            continue
        
        source_vec = source_vectors.vectors[vec_id]
        target_vec = target_vectors.vectors[vec_id]
        
        # Compare metadata
        metadata_match = source_vec.metadata == target_vec.metadata
        
        # Compare vector values (close enough)
        values_match = np.allclose(
            np.array(source_vec.values),
            np.array(target_vec.values),
            rtol=1e-5
        )
        
        if metadata_match and values_match:
            print(f"✅ Vector {vec_id} matches in both indexes")
        else:
            print(f"❌ Vector {vec_id} differs between indexes!")
            if not metadata_match:
                print("  Metadata does not match")
            if not values_match:
                print("  Vector values do not match")
            all_match = False
    
    if all_match:
        print("\nAll sampled vectors match exactly!")
    else:
        print("\nSome vectors do not match between indexes!")

def run_demo():
    """Run the complete backup and restore demo."""
    try:
        # Create and populate source index
        source_index = create_source_index()
        vector_data = populate_source_index(source_index)
        
        # Create a backup
        backup_name = create_backup(source_index)
        
        # Restore from backup to a new index
        target_index = restore_from_backup(source_index, backup_name)
        
        # Verify the restoration
        verify_restore(source_index, target_index)
        
        # Clean up - uncomment to delete indexes and backup when done
        # clean_up_index(get_pinecone_client(), source_index)
        # clean_up_index(get_pinecone_client(), target_index)
        # pc.delete_backup(source_index=source_index, backup_name=backup_name)
        
        print("\nBackup and restore demo completed successfully!")
        return source_index, target_index, backup_name
    
    except Exception as e:
        print(f"Error in demo: {e}")
        raise

if __name__ == "__main__":
    run_demo()