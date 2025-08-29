"""Tests for Oracle shallow checkpointers."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from langgraph.checkpoint.oracle.shallow import (
    ShallowOracleCheckpointer,
    AsyncShallowOracleCheckpointer,
)


class TestShallowOracleCheckpointer:
    """Test the ShallowOracleCheckpointer class."""

    def test_init(self):
        """Test initialization."""
        mock_conn = MagicMock()
        checkpointer = ShallowOracleCheckpointer(mock_conn)
        assert checkpointer.conn == mock_conn
        assert checkpointer.lock is not None

    @patch('oracledb.connect')
    def test_from_conn_string(self, mock_connect):
        """Test from_conn_string class method."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        with ShallowOracleCheckpointer.from_conn_string("test_conn_string") as checkpointer:
            assert isinstance(checkpointer, ShallowOracleCheckpointer)
            assert checkpointer.conn == mock_conn
        
        mock_conn.close.assert_called_once()

    def test_search_where(self):
        """Test _search_where method."""
        mock_conn = MagicMock()
        checkpointer = ShallowOracleCheckpointer(mock_conn)
        
        # Test with config only (no checkpoint_ns)
        config = {"configurable": {"thread_id": "thread_1"}}
        where, args = checkpointer._search_where(config, None, None)
        assert "thread_id = :1" in where
        assert args == ["thread_1"]
        
        # Test with config and explicit checkpoint_ns
        config_with_ns = {"configurable": {"thread_id": "thread_1", "checkpoint_ns": "ns1"}}
        where, args = checkpointer._search_where(config_with_ns, None, None)
        assert "thread_id = :1" in where
        assert "checkpoint_ns = :2" in where
        assert args == ["thread_1", "ns1"]
        
        # Test with config, checkpoint_ns, and filter
        filter_dict = {"source": "test"}
        where, args = checkpointer._search_where(config_with_ns, filter_dict, None)
        assert "thread_id = :1" in where
        assert "checkpoint_ns = :2" in where
        assert "JSON_EXISTS(metadata, :4)" in where  # Parameter :4 for filter
        assert len(args) == 3
        assert args[0] == "thread_1"
        assert args[1] == "ns1"
        assert args[2] == str(filter_dict)


class TestAsyncShallowOracleCheckpointer:
    """Test the AsyncShallowOracleCheckpointer class."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test initialization."""
        mock_conn = AsyncMock()
        checkpointer = AsyncShallowOracleCheckpointer(mock_conn)
        assert checkpointer.conn == mock_conn
        assert checkpointer.lock is not None

    @pytest.mark.asyncio
    async def test_from_conn_string(self):
        """Test from_conn_string class method."""
        # Test that the class method exists and is callable
        assert hasattr(AsyncShallowOracleCheckpointer, 'from_conn_string')
        assert callable(AsyncShallowOracleCheckpointer.from_conn_string)
        
        # Test that it's a class method (bound to the class)
        assert AsyncShallowOracleCheckpointer.from_conn_string.__self__ is AsyncShallowOracleCheckpointer

    @pytest.mark.asyncio
    async def test_search_where(self):
        """Test _search_where method (inherited from base class)."""
        mock_conn = AsyncMock()
        checkpointer = AsyncShallowOracleCheckpointer(mock_conn)
        
        # Test with config only (no checkpoint_ns) - base class always includes checkpoint_ns
        config = {"configurable": {"thread_id": "thread_1"}}
        where, args = checkpointer._search_where(config, None, None)
        assert "thread_id = :1" in where
        assert "checkpoint_ns = :2" in where
        assert args == ("thread_1", "")  # Base class returns tuple with empty checkpoint_ns
        
        # Test with config and explicit checkpoint_ns
        config_with_ns = {"configurable": {"thread_id": "thread_1", "checkpoint_ns": "ns1"}}
        where, args = checkpointer._search_where(config_with_ns, None, None)
        assert "thread_id = :1" in where
        assert "checkpoint_ns = :2" in where
        assert args == ("thread_1", "ns1")  # Base class returns tuple
        
        # Test with config, checkpoint_ns, and filter
        filter_dict = {"source": "test"}
        where, args = checkpointer._search_where(config_with_ns, filter_dict, None)
        assert "thread_id = :1" in where
        assert "checkpoint_ns = :2" in where
        # Base class uses JSON_VALUE instead of JSON_EXISTS
        assert "JSON_VALUE(metadata, '$.source') = :3" in where
        assert len(args) == 3
        assert args[0] == "thread_1"
        assert args[1] == "ns1"
        assert args[2] == "test"  # Base class converts filter to string
