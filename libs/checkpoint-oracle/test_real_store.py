#!/usr/bin/env python3
"""Test the Oracle store with real connection."""

import asyncio
import json
from langgraph.store.oracle.base import OracleStore
from langgraph.store.oracle.aio import AsyncOracleStore

# Connection details from user's working program
username = "skmishra"
password = "skmishra"
dsn = "shaunaq/FREEPDB1"

def test_sync_store():
    """Test the synchronous Oracle store."""
    print("Testing synchronous Oracle store...")
    print("=" * 60)
    
    try:
        # Test connection and setup
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            print("✓ Successfully connected to Oracle")
            
            # Setup tables
            print("Setting up tables...")
            store.setup()
            print("✓ Tables setup completed")
            
            # Test basic operations
            namespace = ("test", "namespace")
            key = "test_key"
            value = {"data": "test_value", "number": 42}
            
            # Put value
            print("Putting value...")
            store.mset([(namespace, key, value)])
            print("✓ Value put successfully")
            
            # Get value
            print("Getting value...")
            retrieved_value = store.mget([(namespace, key)])
            print(f"✓ Value retrieved: {retrieved_value}")
            
            # List keys
            print("Listing keys...")
            keys = list(store.list_keys(namespace))
            print(f"✓ Found keys: {keys}")
            
            # Delete value
            print("Deleting value...")
            store.mdelete([(namespace, key)])
            print("✓ Value deleted successfully")
            
            # Verify deletion
            retrieved_value_after_delete = store.mget([(namespace, key)])
            print(f"✓ Value after delete: {retrieved_value_after_delete}")
            
    except Exception as e:
        print(f"✗ Error in sync test: {e}")
        import traceback
        traceback.print_exc()

async def test_async_store():
    """Test the asynchronous Oracle store."""
    print("\nTesting asynchronous Oracle store...")
    print("=" * 60)
    
    try:
        # Test connection and setup
        async with AsyncOracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            print("✓ Successfully connected to Oracle (async)")
            
            # Setup tables
            print("Setting up tables...")
            await store.asetup()
            print("✓ Tables setup completed")
            
            # Test basic operations
            namespace = ("test", "namespace_async")
            key = "test_key_async"
            value = {"data": "test_value_async", "number": 100}
            
            # Put value
            print("Putting value...")
            await store.amset([(namespace, key, value)])
            print("✓ Value put successfully")
            
            # Get value
            print("Getting value...")
            retrieved_value = await store.amget([(namespace, key)])
            print(f"✓ Value retrieved: {retrieved_value}")
            
            # List keys
            print("Listing keys...")
            keys = [k async for k in store.alist_keys(namespace)]
            print(f"✓ Found keys: {keys}")
            
            # Delete value
            print("Deleting value...")
            await store.amdelete([(namespace, key)])
            print("✓ Value deleted successfully")
            
            # Verify deletion
            retrieved_value_after_delete = await store.amget([(namespace, key)])
            print(f"✓ Value after delete: {retrieved_value_after_delete}")
            
    except Exception as e:
        print(f"✗ Error in async test: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("Testing Oracle Store with Real Connection")
    print("=" * 80)
    
    # Test sync version
    test_sync_store()
    
    # Test async version
    asyncio.run(test_async_store())
    
    print("\n" + "=" * 80)
    print("Testing completed!")

if __name__ == "__main__":
    main()
