#!/usr/bin/env python3
"""Test the Oracle shallow checkpointers with real database connection."""

import asyncio
import oracledb
from langgraph.checkpoint.oracle.shallow import ShallowOracleCheckpointer, AsyncShallowOracleCheckpointer

# Connection details from user
username = "skmishra"
password = "skmishra"
dsn = "shaunaq/FREEPDB1"  # host/container name + service

def test_sync_shallow_checkpointer():
    """Test synchronous shallow checkpointer."""
    print("Testing ShallowOracleCheckpointer...")
    
    try:
        # First test the basic connection like the user's working program
        print("Testing basic Oracle connection...")
        conn = oracledb.connect(user=username, password=password, dsn=dsn)
        print("✓ Basic Oracle connection successful")
        conn.close()
        
        # Now test the shallow checkpointer
        print("Testing shallow checkpointer...")
        with ShallowOracleCheckpointer.from_conn_string(f"{username}/{password}@{dsn}") as checkpointer:
            print("✓ Shallow checkpointer connection established")
            
            # Setup tables
            checkpointer.setup()
            print("✓ Tables created/setup completed")
            
            # Test basic operations
            config = {"configurable": {"thread_id": "test_thread_1"}}
            checkpoint = {
                "v": 1,
                "ts": "2024-01-01T00:00:00+00:00",
                "id": "test_checkpoint_1",
                "channel_values": {"test_key": "test_value"},
                "channel_versions": {"test_key": 1},
                "versions_seen": {"test_key": {"test_key": 1}}
            }
            
            # Put checkpoint
            result = checkpointer.put(config, checkpoint, {}, {})
            print(f"✓ Checkpoint stored: {result}")
            
            # Get checkpoint
            retrieved = checkpointer.get_tuple(config)
            print(f"✓ Checkpoint retrieved: {retrieved is not None}")
            
            # List checkpoints
            checkpoints = list(checkpointer.list(config))
            print(f"✓ Found {len(checkpoints)} checkpoints")
            
            print("✓ All sync tests passed!")
            
    except Exception as e:
        print(f"✗ Sync test failed: {e}")
        raise

async def test_async_shallow_checkpointer():
    """Test asynchronous shallow checkpointer."""
    print("\nTesting AsyncShallowOracleCheckpointer...")
    
    try:
        # First test the basic async connection
        print("Testing basic async Oracle connection...")
        conn = await oracledb.connect_async(user=username, password=password, dsn=dsn)
        print("✓ Basic async Oracle connection successful")
        await conn.close()
        
        # Now test the async shallow checkpointer
        print("Testing async shallow checkpointer...")
        async with AsyncShallowOracleCheckpointer.from_conn_string(f"{username}/{password}@{dsn}") as checkpointer:
            print("✓ Async shallow checkpointer connection established")
            
            # Setup tables
            await checkpointer.setup()
            print("✓ Async tables created/setup completed")
            
            # Test basic operations
            config = {"configurable": {"thread_id": "test_thread_2"}}
            checkpoint = {
                "v": 1,
                "ts": "2024-01-01T00:00:00+00:00",
                "id": "test_checkpoint_2",
                "channel_values": {"test_key": "test_value"},
                "channel_versions": {"test_key": 1},
                "versions_seen": {"test_key": {"test_key": 1}}
            }
            
            # Put checkpoint
            result = await checkpointer.aput(config, checkpoint, {}, {})
            print(f"✓ Async checkpoint stored: {result}")
            
            # Get checkpoint
            retrieved = await checkpointer.aget_tuple(config)
            print(f"✓ Async checkpoint retrieved: {retrieved is not None}")
            
            # List checkpoints
            checkpoints = []
            async for c in checkpointer.alist(config):
                checkpoints.append(c)
            print(f"✓ Found {len(checkpoints)} async checkpoints")
            
            print("✓ All async tests passed!")
            
    except Exception as e:
        print(f"✗ Async test failed: {e}")
        raise

def main():
    """Run all tests."""
    print("Testing Oracle Shallow Checkpointers with real database connection")
    print("=" * 70)
    print(f"Username: {username}")
    print(f"DSN: {dsn}")
    print("=" * 70)
    
    # Test sync version
    test_sync_shallow_checkpointer()
    
    # Test async version
    asyncio.run(test_async_shallow_checkpointer())
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")

if __name__ == "__main__":
    main()
