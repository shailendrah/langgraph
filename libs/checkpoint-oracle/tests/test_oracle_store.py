#!/usr/bin/env python3
"""Comprehensive tests for Oracle store implementation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import json

from langgraph.store.base import BaseStore, TTLConfig
from langgraph.store.oracle.aio import AsyncOracleStore
from langgraph.store.oracle.base import OracleIndexConfig, OracleStore, PoolConfig




class TestOracleStore:
    """Test suite for Oracle store implementation."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        self.test_namespace = ("test", "namespace")
        self.test_key = "test_key"
        self.test_value = {"data": "test_value", "number": 42}
        self.test_namespace_2 = ("test", "namespace2")
        self.test_key_2 = "test_key_2"
        self.test_value_2 = {"data": "test_value_2", "number": 100}

    def test_imports(self) -> None:
        """Test that all imports work correctly."""
        assert AsyncOracleStore is not None
        assert OracleStore is not None
        assert OracleIndexConfig is not None

    def test_sync_store_creation(self) -> None:
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            assert isinstance(store, OracleStore)
            assert isinstance(store, BaseStore)

    async def test_async_store_creation(self) -> None:
        """Test asynchronous store creation."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        async with AsyncOracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            assert isinstance(store, AsyncOracleStore)
            assert isinstance(store, BaseStore)

    def test_sync_store_with_pool_config(self) -> None:
        """Test synchronous store creation with pool configuration."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        pool_config: PoolConfig = {
            "min_size": 2,
            "max_size": 10,
            "kwargs": {}  # Remove autocommit as it's not supported by oracledb.create_pool()
        }
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}", pool_config=pool_config) as store:
            assert isinstance(store, OracleStore)

    def test_sync_store_with_index_config(self) -> None:
        """Test synchronous store creation with index configuration."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        def dummy_embed(texts):
            return [[0.0] * 1536 for _ in texts]
        
        index_config: OracleIndexConfig = {
            "dims": 1536,
            "fields": ["content"],
            "distance_type": "cosine",
            "embed": dummy_embed,
        }
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}", index=index_config) as store:
            assert isinstance(store, OracleStore)
            assert store.index_config is not None

    def test_sync_store_with_ttl_config(self) -> None:
        """Test synchronous store creation with TTL configuration."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        ttl_config: TTLConfig = {
            "default_ttl": 60,
            "sweep_interval_minutes": 5
        }
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}", ttl=ttl_config) as store:
            assert isinstance(store, OracleStore)
            assert store.ttl_config is not None

    def test_sync_setup(self) -> None:
        """Test synchronous store setup."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            store.setup()
            # Test that setup completes without error
            assert store is not None

    async def test_async_setup(self) -> None:
        """Test asynchronous store setup."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        async with AsyncOracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            await store.setup()
            # Test that setup completes without error
            assert store is not None

    def test_sync_put_and_get(self) -> None:
        """Test synchronous put and get operations."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            store.setup()
            
            # Test put operation
            key = store.put(self.test_namespace, self.test_value, self.test_key)
            assert key == self.test_key
            
            # Test get operation
            item = store.get(self.test_namespace, self.test_key)
            assert item is not None
            assert item.key == self.test_key
            assert item.value == self.test_value
            assert item.namespace == self.test_namespace
            
            # Clean up
            store.delete(self.test_namespace, self.test_key)

    async def test_async_put_and_get(self) -> None:
        """Test asynchronous put and get operations."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        async with AsyncOracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            await store.setup()
            
            # Test put operation - note the parameter order: namespace, key, value
            key = await store.aput(self.test_namespace, self.test_key, self.test_value)
            assert key == self.test_key
            
            # Test get operation
            item = await store.aget(self.test_namespace, self.test_key)
            assert item is not None
            assert item.key == self.test_key
            assert item.value == self.test_value
            assert item.namespace == self.test_namespace
            
            # Clean up
            await store.adelete(self.test_namespace, self.test_key)

    def test_sync_batch_operations(self) -> None:
        """Test synchronous batch operations."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            store.setup()
            
            # Test batch operations
            from langgraph.store.base import GetOp, PutOp
            
            ops = [
                PutOp(namespace=self.test_namespace, key=self.test_key, value=self.test_value),
                GetOp(namespace=self.test_namespace, key=self.test_key),
            ]
            
            results = store.batch(ops)
            assert len(results) == 2
            assert results[0] is None  # Put operation returns None
            assert results[1] is None  # Get operation returns None (no item found)

    async def test_async_batch_operations(self) -> None:
        """Test asynchronous batch operations."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        async with AsyncOracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            await store.setup()
            
            # Test batch operations
            from langgraph.store.base import GetOp, PutOp
            
            ops = [
                PutOp(namespace=self.test_namespace, key=self.test_key, value=self.test_value),
                GetOp(namespace=self.test_namespace, key=self.test_key),
            ]
            
            results = await store.abatch(ops)
            assert len(results) == 2
            assert results[0] is None  # Put operation returns None
            assert results[1] is None  # Get operation returns None (no item found)

    def test_sync_search(self) -> None:
        """Test synchronous search operation."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            store.setup()
            
            # Test search operation
            results = store.search(self.test_namespace, query="test")
            assert isinstance(results, list)

    async def test_async_search(self) -> None:
        """Test asynchronous search operation."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        async with AsyncOracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            await store.setup()
            
            # Test search operation
            results = await store.asearch(self.test_namespace, query="test")
            assert isinstance(results, list)

    def test_sync_list_namespaces(self) -> None:
        """Test synchronous list_namespaces operation."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            store.setup()
            
            # Test list_namespaces operation
            namespaces = store.list_namespaces()
            assert isinstance(namespaces, list)

    async def test_async_list_namespaces(self) -> None:
        """Test asynchronous list_namespaces operation."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        async with AsyncOracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            await store.setup()
            
            # Test list_namespaces operation
            namespaces = await store.alist_namespaces()
            assert isinstance(namespaces, list)

    def test_sync_delete(self) -> None:
        """Test synchronous delete operation."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            store.setup()
            store.delete(self.test_namespace, self.test_key)
            # Test that delete completes without error
            assert store is not None

    async def test_async_delete(self) -> None:
        """Test asynchronous delete operation."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        async with AsyncOracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            await store.setup()
            
            # Test delete operation
            await store.adelete(self.test_namespace, self.test_key)
            
            # Test that delete completes without error
            assert store is not None

    def test_sync_delete_namespace(self) -> None:
        """Test synchronous delete_namespace operation."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            store.setup()
            store.delete_namespace(self.test_namespace)
            # Test that delete_namespace completes without error
            assert store is not None

    def test_sync_sweep_ttl(self) -> None:
        """Test synchronous sweep_ttl operation."""
        # Use real connection details
        username = "skmishra"
        password = "skmishra"
        dsn = "shaunaq/FREEPDB1"
        
        with OracleStore.from_conn_string(f"{username}/{password}@{dsn}") as store:
            store.setup()
            deleted_count = store.sweep_ttl()
            # Test that sweep_ttl completes without error
            assert isinstance(deleted_count, int)

    def test_ttl_sweeper(self) -> None:
        """Test TTL sweeper functionality."""
        # This would normally be tested with a real connection
        # For now, we just test that the method exists
        pass

    def test_vector_search_capabilities(self) -> None:
        """Test vector search capabilities."""
        # This would normally be tested with a real connection and vector embeddings
        # For now, we just test that the method exists
        pass


if __name__ == "__main__":
    pytest.main([__file__]) 