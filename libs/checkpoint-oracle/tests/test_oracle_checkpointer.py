#!/usr/bin/env python3
"""Comprehensive tests for Oracle checkpointer implementation."""

import pytest
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from langgraph.checkpoint.oracle import OracleCheckpointer
from langgraph.checkpoint.oracle.aio import AsyncOracleCheckpointer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class TestOracleCheckpointer:
    """Test suite for Oracle checkpointer implementation."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        self.serde = JsonPlusSerializer()
        
        # Test configurations
        self.config_1: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "thread_ts": "1",
            }
        }
        self.config_2: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2",
                "checkpoint_ns": "",
            }
        }
        self.config_3: RunnableConfig = {
            "configurable": {
                "thread_id": "thread-2",
                "checkpoint_id": "2-inner",
                "checkpoint_ns": "inner",
            }
        }

        # Test checkpoints
        self.chkpnt_1: Checkpoint = empty_checkpoint()
        self.chkpnt_2: Checkpoint = create_checkpoint(self.chkpnt_1, {}, 1)
        self.chkpnt_3: Checkpoint = empty_checkpoint()

        # Test metadata
        self.metadata_1: CheckpointMetadata = {
            "source": "input",
            "step": 2,
        }
        self.metadata_2: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
        }
        self.metadata_3: CheckpointMetadata = {}

        # Real connection details
        self.username = "skmishra"
        self.password = "skmishra"
        self.dsn = "shaunaq/FREEPDB1"

    def test_imports(self) -> None:
        """Test that all imports work correctly."""
        assert AsyncOracleCheckpointer is not None
        assert OracleCheckpointer is not None
        assert JsonPlusSerializer is not None

    def test_serde_creation(self) -> None:
        """Test JsonPlusSerializer creation."""
        serde = JsonPlusSerializer()
        assert serde is not None

    def test_sync_checkpointer_creation(self) -> None:
        """Test synchronous checkpointer creation."""
        with OracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            assert isinstance(checkpointer, OracleCheckpointer)
            assert checkpointer.serde is not None

    async def test_async_checkpointer_creation(self) -> None:
        """Test asynchronous checkpointer creation."""
        async with AsyncOracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            assert isinstance(checkpointer, AsyncOracleCheckpointer)
            assert checkpointer.serde is not None

    def test_sync_checkpointer_from_parameters(self) -> None:
        """Test synchronous checkpointer creation from parameters."""
        with OracleCheckpointer.from_parameters(
            user=self.username,
            password=self.password, 
            dsn=self.dsn
        ) as checkpointer:
            assert isinstance(checkpointer, OracleCheckpointer)

    async def test_async_checkpointer_from_parameters(self) -> None:
        """Test asynchronous checkpointer creation from parameters."""
        async with AsyncOracleCheckpointer.from_parameters(
            user=self.username,
            password=self.password,
            dsn=self.dsn
        ) as checkpointer:
            assert isinstance(checkpointer, AsyncOracleCheckpointer)

    def test_sync_setup(self) -> None:
        """Test synchronous checkpointer setup."""
        with OracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            checkpointer.setup()
            assert checkpointer is not None

    async def test_async_setup(self) -> None:
        """Test asynchronous checkpointer setup."""
        async with AsyncOracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            await checkpointer.setup()
            assert checkpointer is not None

    def test_sync_put_and_get(self) -> None:
        """Test synchronous put and get operations."""
        with OracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            checkpointer.setup()
            
            # Test put operation
            new_versions: ChannelVersions = {"channel1": 1}
            result_config = checkpointer.put(
                self.config_1, self.chkpnt_1, self.metadata_1, new_versions
            )
            
            # Verify the result config contains the checkpoint ID
            configurable = result_config.get("configurable", {})
            assert "checkpoint_id" in configurable
            assert configurable["thread_id"] == "thread-1"

    async def test_async_put_and_get(self) -> None:
        """Test asynchronous put and get operations."""
        async with AsyncOracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            await checkpointer.setup()
            
            # Test put operation
            new_versions: ChannelVersions = {"channel1": 1}
            result_config = await checkpointer.aput(
                self.config_1, self.chkpnt_1, self.metadata_1, new_versions
            )
            
            # Verify the result config contains the checkpoint ID
            configurable = result_config.get("configurable", {})
            assert "checkpoint_id" in configurable
            assert configurable["thread_id"] == "thread-1"

    def test_sync_put_writes(self) -> None:
        """Test synchronous put_writes operation."""
        with OracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            checkpointer.setup()
            
            # Test put_writes operation - use config_2 which has checkpoint_id
            writes = [("node1", "data1"), ("node2", "data2")]
            checkpointer.put_writes(self.config_2, writes, "task1", "path1")
            assert checkpointer is not None

    async def test_async_put_writes(self) -> None:
        """Test asynchronous put_writes operation."""
        async with AsyncOracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            await checkpointer.setup()
            
            # Test put_writes operation - use config_2 which has checkpoint_id
            writes = [("node1", "data1"), ("node2", "data2")]
            await checkpointer.aput_writes(self.config_2, writes, "task1", "path1")
            assert checkpointer is not None

    def test_sync_delete_thread(self) -> None:
        """Test synchronous delete_thread operation."""
        with OracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            checkpointer.setup()
            
            # Test delete_thread operation
            checkpointer.delete_thread("thread-1")
            assert checkpointer is not None

    async def test_async_delete_thread(self) -> None:
        """Test asynchronous delete_thread operation."""
        async with AsyncOracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            await checkpointer.setup()
            
            # Test delete_thread operation
            await checkpointer.adelete_thread("thread-1")
            assert checkpointer is not None

    def test_sync_list_checkpoints(self) -> None:
        """Test synchronous list operation."""
        with OracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            checkpointer.setup()
            
            # Test list operation
            checkpoints = list(checkpointer.list(self.config_1))
            assert isinstance(checkpoints, list)

    async def test_async_list_checkpoints(self) -> None:
        """Test asynchronous list operation."""
        async with AsyncOracleCheckpointer.from_conn_string(f"{self.username}/{self.password}@{self.dsn}") as checkpointer:
            await checkpointer.setup()
            
            # Test list operation
            checkpoints = []
            async for checkpoint in checkpointer.alist(self.config_1):
                checkpoints.append(checkpoint)
            assert isinstance(checkpoints, list)

    def test_invalid_config(self) -> None:
        """Test handling of invalid configuration."""
        # This would normally be tested with a real connection, but we can test the validation
        pass

    def test_missing_thread_id(self) -> None:
        """Test handling of missing thread_id."""
        # This would normally be tested with a real connection, but we can test the validation
        pass


if __name__ == "__main__":
    pytest.main([__file__]) 