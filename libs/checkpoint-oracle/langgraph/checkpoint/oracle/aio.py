"""Implementation of asynchronous LangGraph checkpoint saver using Oracle Database."""
import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Optional, cast

import oracledb
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from . import _ainternal

Conn = _ainternal.Conn


class AsyncOracleCheckpointer(BaseCheckpointSaver):
    """Asynchronous checkpointer that stores checkpoints in an Oracle database.

    This checkpoint saver stores checkpoints in an Oracle database using the
    oracledb Python driver. It is compatible with Oracle Database 19c and later.

    Args:
        conn: The Oracle connection object, connection pool, or async connection/pool.
        serde: The serializer to use for serialization/deserialization.
            Defaults to JsonPlusSerializer.
        table_prefix: Prefix to use for the tables. Defaults to empty string.

    Note:
        You should call the setup() method once to create the necessary tables
        before using this checkpointer.

    """

    conn: _ainternal.Conn
    lock: asyncio.Lock
    _table_prefix: str
    _serde: SerializerProtocol

    def __init__(
        self,
        conn: _ainternal.Conn,
        *,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "",
    ) -> None:
        """Initialize the AsyncOracleCheckpointer.

        Args:
            conn: Oracle connection object, connection pool, or async connection/pool
            serde: Serializer to use (defaults to JsonPlusSerializer)
            table_prefix: Prefix for the tables (defaults to "")
        """
        super().__init__(serde=serde or JsonPlusSerializer())
        self.conn = conn
        self._table_prefix = table_prefix
        self._serde = serde or JsonPlusSerializer()
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

    @property
    def _checkpoints_table(self) -> str:
        """Get the fully qualified checkpoints table name."""
        return f"{self._table_prefix}CHECKPOINTS"

    @property
    def _writes_table(self) -> str:
        """Get the fully qualified writes table name."""
        return f"{self._table_prefix}WRITES"

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "",
    ) -> AsyncIterator["AsyncOracleCheckpointer"]:
        """Create an AsyncOracleCheckpointer from a connection string.

        The connection will be closed when the context manager exits.

        Args:
            conn_string: Oracle connection string (username/password@host/service)
            serde: Serializer to use (defaults to JsonPlusSerializer)
            table_prefix: Prefix for the tables (defaults to "")

        Yields:
            AsyncOracleCheckpointer: The checkpointer instance
        """
        conn = None
        try:
            conn = await oracledb.connect_async(conn_string)
            yield cls(
                conn,
                serde=serde,
                table_prefix=table_prefix,
            )
        finally:
            if conn is not None:
                await conn.close()

    @classmethod
    @asynccontextmanager
    async def from_parameters(
        cls,
        *,
        user: str,
        password: str,
        dsn: str,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "",
    ) -> AsyncIterator["AsyncOracleCheckpointer"]:
        """Create an AsyncOracleCheckpointer from connection parameters.

        The connection will be closed when the context manager exits.

        Args:
            user: Database username
            password: Database password
            dsn: Database connection string (host/service)
            serde: Serializer to use (defaults to JsonPlusSerializer)
            table_prefix: Prefix for the tables (defaults to "")

        Yields:
            AsyncOracleCheckpointer: The checkpointer instance
        """
        conn = None
        try:
            conn = await oracledb.connect_async(user=user, password=password, dsn=dsn)
            yield cls(
                conn,
                serde=serde,
                table_prefix=table_prefix,
            )
        finally:
            if conn is not None:
                await conn.close()

    async def setup(self) -> None:
        """Create the necessary tables for the checkpointer.

        This method should be called once before using the checkpointer.
        It creates the tables needed to store the checkpoints and writes.
        """
        async with self.lock:
            async with _ainternal.get_connection(self.conn) as conn:
                async with conn.cursor() as cursor:
                    try:
                        # Create checkpoints table if it doesn't exist
                        await cursor.execute(
                            f"""
                            BEGIN
                              EXECUTE IMMEDIATE '
                                CREATE TABLE {self._table_prefix}CHECKPOINTS (
                                  thread_id VARCHAR2(512) NOT NULL,
                                  thread_ts VARCHAR2(64),
                                  checkpoint_id VARCHAR2(64) NOT NULL,
                                  checkpoint CLOB NOT NULL,
                                  metadata CLOB NOT NULL,
                                  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                  PRIMARY KEY (thread_id, checkpoint_id)
                                )';
                            EXCEPTION
                              WHEN OTHERS THEN
                                IF SQLCODE = -955 THEN NULL;
                                ELSE RAISE;
                              END IF;
                            END;
                          """
                        )

                        # Create writes table if it doesn't exist
                        await cursor.execute(
                            f"""
                            BEGIN
                              EXECUTE IMMEDIATE '
                                CREATE TABLE {self._table_prefix}WRITES (
                                  thread_id VARCHAR2(512) NOT NULL,
                                  checkpoint_id VARCHAR2(64) NOT NULL,
                                  node_name VARCHAR2(512) NOT NULL,
                                  write_idx NUMBER NOT NULL,
                                  write_data CLOB NOT NULL,
                                  task_id VARCHAR2(64) NOT NULL,
                                  task_path VARCHAR2(512) DEFAULT NULL,
                                  created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                  PRIMARY KEY (thread_id, checkpoint_id, node_name, write_idx)
                                )';
                            EXCEPTION
                              WHEN OTHERS THEN
                                IF SQLCODE = -955 THEN NULL;
                                ELSE RAISE;
                              END IF;
                            END;
                          """
                        )

                        # Create an index on the checkpoints table for faster lookups
                        await cursor.execute(
                            f"""
                            BEGIN
                              EXECUTE IMMEDIATE '
                                CREATE INDEX {self._table_prefix}CK_THREAD_IDX ON
                                  {self._table_prefix}CHECKPOINTS (thread_id, created_at)';
                            EXCEPTION
                              WHEN OTHERS THEN
                                IF SQLCODE = -955 THEN NULL;
                                ELSE RAISE;
                              END IF;
                            END;
                          """
                        )

                        # Create indexes on the writes table
                        await cursor.execute(
                            f"""
                            BEGIN
                              EXECUTE IMMEDIATE '
                                CREATE INDEX {self._table_prefix}WR_THREAD_CK_IDX ON
                                  {self._table_prefix}WRITES (thread_id, checkpoint_id)';
                            EXCEPTION
                              WHEN OTHERS THEN
                                IF SQLCODE = -955 THEN NULL;
                                ELSE RAISE;
                              END IF;
                            END;
                          """
                        )
                    
                    except Exception as e:
                        # Table already exists or other error, continue
                        pass
                    
                    await conn.commit()

    async def alist(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints for a thread asynchronously.

        Args:
            config: A RunnableConfig containing the thread_id.
            filter: Optional filter criteria for metadata.
            before: Optional config to get checkpoints before a specific checkpoint.
            limit: Optional limit on the number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of checkpoint tuples.
        """
        filter = filter or {}
        # Type-safe access to configurable
        thread_id = None
        if config is not None:
            configurable = config.get("configurable", {})
            if configurable:
                thread_id = configurable["thread_id"]

        async with self.lock:
            async with _ainternal.get_connection(self.conn) as conn:
                async with conn.cursor() as cursor:
                    try:
                        # Build the query based on the parameters
                        query_parts = [f"SELECT checkpoint, metadata, checkpoint_id, thread_id FROM {self._checkpoints_table}"]
                        params = {}

                        where_clauses = []
                        if thread_id is not None:
                            where_clauses.append("thread_id = :thread_id")
                            params["thread_id"] = thread_id

                        # Add before condition if specified
                        if before is not None:
                            before_configurable = before.get("configurable", {})
                            if before_configurable:
                                before_ts = before_configurable.get("thread_ts")
                                if before_ts is not None:
                                    where_clauses.append(
                                        "created_at < (SELECT created_at FROM {0} WHERE thread_ts = :before_ts)".format(
                                            self._checkpoints_table))
                                    params["before_ts"] = before_ts

                        if where_clauses:
                            query_parts.append("WHERE " + " AND ".join(where_clauses))

                        query_parts.append("ORDER BY created_at DESC")

                        if limit is not None:
                            query_parts.append(f"FETCH FIRST {limit} ROWS ONLY")

                        query = " ".join(query_parts)
                        await cursor.execute(query, **params)

                        # Fetch all matching checkpoints
                        rows = await cursor.fetchall()
                        for checkpoint_json, metadata_json, checkpoint_id, row_thread_id in rows:
                            checkpoint = self._serde.loads(checkpoint_json)
                            metadata = self._serde.loads(metadata_json)

                            # Filter based on metadata if filter is provided
                            if filter and not all(
                                metadata.get(k) == v for k, v in filter.items()
                            ):
                                continue

                            # Fetch pending writes for this checkpoint
                            writes_query = f"""
                                SELECT node_name, write_idx, write_data
                                FROM {self._writes_table}
                                WHERE thread_id = :thread_id AND checkpoint_id = :checkpoint_id
                                ORDER BY node_name, write_idx
                            """
                            await cursor.execute(
                                writes_query,
                                thread_id=row_thread_id,
                                checkpoint_id=checkpoint_id)

                            # Convert to the expected format for CheckpointTuple
                            pending_writes = []
                            async for node_name, write_idx, write_data in cursor:
                                write = self._serde.loads(write_data)
                                pending_writes.append((node_name, write_idx, write))

                            yield CheckpointTuple(
                                config={
                                    "configurable": {
                                        "thread_id": row_thread_id,
                                        "thread_ts": checkpoint["ts"],
                                        "checkpoint_id": checkpoint_id,
                                    }
                                },
                                checkpoint=checkpoint,
                                metadata=metadata,
                                parent=None,  # Simple implementation doesn't track parent
                                pending_writes=pending_writes,
                            )
                    finally:
                        pass

    async def aget_tuple(
            self,
            config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the Oracle database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching
            checkpoint was found.
        """
        # Type-safe access to configurable
        configurable = config.get("configurable", {})
        if not configurable:
            return None

        thread_id = configurable["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        if checkpoint_id:
            args: tuple[Any, ...] = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = :1 AND checkpoint_ns = :2 AND checkpoint_id = :3"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = :1 AND checkpoint_ns = :2 ORDER BY checkpoint_id DESC FETCH FIRST 1 ROWS ONLY"

        async with self._cursor() as cur:
            await cur.execute(self.SELECT_SQL + where, args)
            row = await cur.fetchone()
            if row:
                value_dict = {
                    k.lower(): v for k, v in zip(
                        cur.description, row)}
                return CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": value_dict["checkpoint_id"],
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint,
                        value_dict["checkpoint"],
                        value_dict["channel_values"],
                        value_dict["pending_sends"],
                    ),
                    self._load_metadata(value_dict["metadata"]),
                    (
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": value_dict["parent_checkpoint_id"],
                            }
                        }
                        if value_dict["parent_checkpoint_id"]
                        else None
                    ),
                    await asyncio.to_thread(self._load_writes, value_dict["pending_writes"]),
                )
            return None

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint asynchronously.

        Args:
            config: A RunnableConfig containing the thread_id.
            checkpoint: The checkpoint data to store.
            metadata: Metadata associated with the checkpoint.
            new_versions: New channel versions.

        Returns:
            RunnableConfig: Updated config with thread_ts and checkpoint_id.
        """
        # Type-safe access to configurable
        configurable = config.get("configurable", {})
        if not configurable:
            raise ValueError("Config must contain 'configurable' key")

        configurable = configurable.copy()
        thread_id = configurable.get("thread_id")
        if thread_id is None:
            raise ValueError("thread_id is required")

        # Get or generate checkpoint ID
        checkpoint_id = configurable.get("checkpoint_id") or get_checkpoint_id(config)

        async with self.lock:
            async with _ainternal.get_connection(self.conn) as conn:
                async with conn.cursor() as cursor:
                    try:
                        # Serialize checkpoint and metadata
                        checkpoint_json = self._serde.dumps(checkpoint)
                        metadata_json = self._serde.dumps(metadata)

                        # Insert or update the checkpoint
                        query = f"""
                                MERGE INTO {self._checkpoints_table} t
                                USING (SELECT :thread_id AS thread_id, :checkpoint_id AS checkpoint_id FROM dual) s
                                ON (t.thread_id = s.thread_id AND t.checkpoint_id = s.checkpoint_id)
                                WHEN MATCHED THEN
                                    UPDATE SET
                                        thread_ts = :thread_ts,
                                        checkpoint = :checkpoint,
                                        metadata = :metadata,
                                        created_at = CURRENT_TIMESTAMP
                                WHEN NOT MATCHED THEN
                                    INSERT (thread_id, thread_ts, checkpoint_id, checkpoint, metadata)
                                    VALUES (:thread_id, :thread_ts, :checkpoint_id, :checkpoint, :metadata)
                                """
                        await cursor.execute(
                            query,
                            thread_id=thread_id,
                            thread_ts=checkpoint["ts"],
                            checkpoint_id=checkpoint_id,
                            checkpoint=checkpoint_json,
                            metadata=metadata_json,
                        )
                        await conn.commit()
                    finally:
                        pass

        # Return updated config
        configurable["thread_ts"] = checkpoint["ts"]
        configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable}

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store writes for a checkpoint asynchronously.

        Args:
            config: A RunnableConfig containing the thread_id, thread_ts, and checkpoint_id.
            writes: A sequence of tuples (node_name, data) to store.
            task_id: Task ID associated with the writes.
            task_path: Optional task path.
        """
        # Type-safe access to configurable
        configurable = config.get("configurable", {})
        if not configurable:
            raise ValueError("Config must contain 'configurable' key")

        thread_id = configurable["thread_id"]
        checkpoint_id = configurable["checkpoint_id"]

        if not writes:
            return

        async with self.lock:
            async with _ainternal.get_connection(self.conn) as conn:
                async with conn.cursor() as cursor:
                    try:
                        # Group writes by node
                        writes_by_node: dict[str, list] = {}
                        for node_name, data in writes:
                            if node_name not in writes_by_node:
                                writes_by_node[node_name] = []
                            writes_by_node[node_name].append(data)

                        # Insert writes to database
                        for node_name, data_list in writes_by_node.items():
                            # Get the current write index for this node/checkpoint
                            query = f"""
                                SELECT NVL(MAX(write_idx) + 1, 0)
                                FROM {self._writes_table}
                                WHERE thread_id = :thread_id
                                  AND checkpoint_id = :checkpoint_id
                                  AND node_name = :node_name
                            """
                            await cursor.execute(
                                query,
                                thread_id=thread_id,
                                checkpoint_id=checkpoint_id,
                                node_name=node_name,
                            )
                            row = await cursor.fetchone()
                            write_idx = row[0] if row else 0

                            # Insert each write
                            for i, data in enumerate(data_list):
                                # Serialize the write data
                                write_data = self._serde.dumps(data)

                                query = f"""
                                    INSERT INTO {self._writes_table} (
                                        thread_id, checkpoint_id, node_name, write_idx,
                                        write_data, task_id, task_path
                                    )                                     VALUES (
                                        :thread_id, :checkpoint_id, :node_name, :write_idx,
                                        :write_data, :task_id, :task_path
                                    )
                                """
                                await cursor.execute(
                                    query,
                                    thread_id=thread_id,
                                    checkpoint_id=checkpoint_id,
                                    node_name=node_name,
                                    write_idx=write_idx + i,
                                    write_data=write_data,
                                    task_id=task_id,
                                    task_path=task_path,
                                )

                        await conn.commit()
                    finally:
                        pass

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.
        """
        async with self.lock:
            async with _ainternal.get_connection(self.conn) as conn:
                async with conn.cursor() as cursor:
                    try:
                        # Delete writes first (due to foreign key constraints)
                        await cursor.execute(
                            f"DELETE FROM {self._writes_table} WHERE thread_id = :thread_id",
                            thread_id=str(thread_id),
                        )
                        
                        # Delete checkpoints
                        await cursor.execute(
                            f"DELETE FROM {self._checkpoints_table} WHERE thread_id = :thread_id",
                            thread_id=str(thread_id),
                        )
                        
                        await conn.commit()
                    finally:
                        pass

    @asynccontextmanager
    async def _cursor(
            self, *, transaction: bool = False) -> AsyncIterator[oracledb.AsyncCursor]:
        """Create a database cursor as a context manager.

        Args:
            transaction: Whether to use a transaction for the operations inside the context manager.
        """
        async with self.lock:
            async with _ainternal.get_connection(self.conn) as conn:
                async with conn.cursor() as cur:
                    try:
                        yield cur
                        if transaction:
                            await conn.commit()
                    except Exception:
                        if transaction:
                            await conn.rollback()
                        raise

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the Oracle database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending
        order (newest first).

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria for metadata.
            before: If provided, only checkpoints before the specified checkpoint ID are returned.
            limit: Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncOracleCheckpointer are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `checkpointer.alist(...)` or `await "
                    "graph.ainvoke(...)`.")
        except RuntimeError:
            pass

        # Use a different approach to handle async iteration
        async def collect_all():
            results = []
            async for item in self.alist(config, filter=filter, before=before, limit=limit):
                results.append(item)
            return results

        results = asyncio.run_coroutine_threadsafe(
            collect_all(), self.loop).result()
        for item in results:
            yield item

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the Oracle database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config: The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching
            checkpoint was found.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncOracleCheckpointer are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.aget_tuple(...)` or `await "
                    "graph.ainvoke(...)`.")
        except RuntimeError:
            pass

        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the Oracle database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store, each as (channel, value) pair.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self.loop
        ).result()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes associated with a thread ID.

        Args:
            thread_id: The thread ID to delete.
        """
        try:
            # check if we are in the main thread, only bg threads can block
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to AsyncOracleCheckpointer are only allowed from a "
                    "different thread. From the main thread, use the async interface. "
                    "For example, use `await checkpointer.adelete_thread(...)` or `await "
                    "graph.ainvoke(...)`.")
        except RuntimeError:
            pass

        return asyncio.run_coroutine_threadsafe(
            self.adelete_thread(thread_id), self.loop
        ).result()


__all__ = ["AsyncOracleCheckpointer", "Conn"]
