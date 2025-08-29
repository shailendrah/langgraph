#!/usr/bin/env python3
"""Test the full Oracle checkpointer with real connection."""

import asyncio
import json
from langgraph.checkpoint.oracle import OracleCheckpointer
from langgraph.checkpoint.oracle.aio import AsyncOracleCheckpointer
from langgraph.checkpoint.base import empty_checkpoint, create_checkpoint, CheckpointMetadata

# Connection details from user's working program
username = "skmishra"
password = "skmishra"
dsn = "shaunaq/FREEPDB1"

def test_sync_full_checkpointer():
    """Test the synchronous full Oracle checkpointer."""
    print("Testing synchronous full Oracle checkpointer...")
    print("=" * 60)
    
    try:
        # Test connection and setup
        with OracleCheckpointer.from_conn_string(f"{username}/{password}@{dsn}") as checkpointer:
            print("✓ Successfully connected to Oracle")
            
            # Setup tables
            print("Setting up tables...")
            checkpointer.setup()
            print("✓ Tables setup completed")
            
            # Test basic operations
            config = {
                "configurable": {
                    "thread_id": "test_thread_full",
                    "checkpoint_ns": "test_namespace",
                    "thread_ts": "1"
                }
            }
            
            # Create test checkpoint
            checkpoint = empty_checkpoint()
            checkpoint = create_checkpoint(checkpoint, {"test_channel": "test_value"}, 1)
            
            metadata: CheckpointMetadata = {
                "source": "test",
                "step": 1
            }
            
            # Put checkpoint
            print("Putting checkpoint...")
            next_config = checkpointer.put(config, checkpoint, metadata)
            print(f"✓ Checkpoint put successfully, next_config: {next_config}")
            
            # Get checkpoint
            print("Getting checkpoint...")
            retrieved_checkpoint, retrieved_metadata = checkpointer.get_tuple(config)
            print(f"✓ Checkpoint retrieved: {retrieved_checkpoint}")
            print(f"✓ Metadata retrieved: {retrieved_metadata}")
            
            # List checkpoints
            print("Listing checkpoints...")
            checkpoints = list(checkpointer.list(config))
            print(f"✓ Found {len(checkpoints)} checkpoints")
            for cp in checkpoints:
                print(f"  - {cp}")
                
    except Exception as e:
        print(f"✗ Error in sync test: {e}")
        import traceback
        traceback.print_exc()

async def test_async_full_checkpointer():
    """Test the asynchronous full Oracle checkpointer."""
    print("\nTesting asynchronous full Oracle checkpointer...")
    print("=" * 60)
    
    try:
        # Test connection and setup
        async with AsyncOracleCheckpointer.from_conn_string(f"{username}/{password}@{dsn}") as checkpointer:
            print("✓ Successfully connected to Oracle (async)")
            
            # Setup tables
            print("Setting up tables...")
            await checkpointer.asetup()
            print("✓ Tables setup completed")
            
            # Test basic operations
            config = {
                "configurable": {
                    "thread_id": "test_thread_full_async",
                    "checkpoint_ns": "test_namespace_async",
                    "thread_ts": "1"
                }
            }
            
            # Create test checkpoint
            checkpoint = empty_checkpoint()
            checkpoint = create_checkpoint(checkpoint, {"test_channel_async": "test_value_async"}, 1)
            
            metadata: CheckpointMetadata = {
                "source": "test_async",
                "step": 1
            }
            
            # Put checkpoint
            print("Putting checkpoint...")
            next_config = await checkpointer.aput(config, checkpoint, metadata)
            print(f"✓ Checkpoint put successfully, next_config: {next_config}")
            
            # Get checkpoint
            print("Getting checkpoint...")
            retrieved_checkpoint, retrieved_metadata = await checkpointer.aget_tuple(config)
            print(f"✓ Checkpoint retrieved: {retrieved_checkpoint}")
            print(f"✓ Metadata retrieved: {retrieved_metadata}")
            
            # List checkpoints
            print("Listing checkpoints...")
            checkpoints = [cp async for cp in checkpointer.alist(config)]
            print(f"✓ Found {len(checkpoints)} checkpoints")
            for cp in checkpoints:
                print(f"  - {cp}")
                
    except Exception as e:
        print(f"✗ Error in async test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("Testing Oracle Full Checkpointer with Real Connection")
    print("=" * 80)
    
    # Test sync version
    test_sync_full_checkpointer()
    
    # Test async version
    asyncio.run(test_async_full_checkpointer())
    
    print("\n" + "=" * 80)
    print("Testing completed!")

if __name__ == "__main__":
    main()
