"""Implementation of shallow LangGraph checkpoint saver using Oracle Database.

This checkpointer ONLY stores the most recent checkpoint and does NOT retain any history.
It is meant to be a light-weight drop-in replacement for the OracleCheckpointer that
supports most of the LangGraph persistence functionality with the exception of time travel.
"""

import asyncio
import json
import threading
import warnings
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Optional, Tuple, List

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.oracle import _ainternal, _internal
from langgraph.checkpoint.oracle.base import BaseOracleCheckpointer
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import TASKS

"""
To add a new migration, add a new string to the MIGRATIONS list.
The position of the migration in the list is the version number.
"""
MIGRATIONS = [
    """CREATE TABLE checkpoint_migrations (
    v NUMBER PRIMARY KEY
)""",
    """CREATE TABLE checkpoints (
    thread_id VARCHAR2(512) NOT NULL,
    checkpoint_ns VARCHAR2(512) NOT NULL,
    type VARCHAR2(100),
    checkpoint CLOB NOT NULL,
    metadata CLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns)
)""",
    """CREATE TABLE checkpoint_blobs (
    thread_id VARCHAR2(512) NOT NULL,
    checkpoint_ns VARCHAR2(512) NOT NULL,
    channel VARCHAR2(512) NOT NULL,
    type VARCHAR2(100) NOT NULL,
    blob BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel)
)""",
    """CREATE TABLE checkpoint_writes (
    thread_id VARCHAR2(512) NOT NULL,
    checkpoint_ns VARCHAR2(512) NOT NULL,
    checkpoint_id VARCHAR2(512) NOT NULL,
    task_id VARCHAR2(512) NOT NULL,
    idx NUMBER NOT NULL,
    channel VARCHAR2(512) NOT NULL,
    type VARCHAR2(100),
    blob BLOB NOT NULL,
    task_path VARCHAR2(512),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
)""",
    """CREATE INDEX checkpoints_thread_id_idx ON checkpoints(thread_id)""",
    """CREATE INDEX checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id)""",
    """CREATE INDEX checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id)""",
]

SELECT_SQL = """
SELECT
    thread_id,
    checkpoint,
    checkpoint_ns,
    metadata,
    (
        SELECT JSON_ARRAYAGG(
            JSON_ARRAY(
                UTL_RAW.CAST_TO_VARCHAR2(bl.channel),
                UTL_RAW.CAST_TO_VARCHAR2(bl.type),
                bl.blob
            )
        )
        FROM JSON_TABLE(
            checkpoint,
            '$.channel_versions.*' COLUMNS (
                key VARCHAR2(512) PATH '$.key',
                value VARCHAR2(512) PATH '$.value'
            )
        ) jt
        INNER JOIN checkpoint_blobs bl
            ON bl.thread_id = checkpoints.thread_id
            AND bl.checkpoint_ns = checkpoints.checkpoint_ns
            AND bl.channel = jt.key
            AND bl.type = jt.value
    ) AS channel_values,
    (
        SELECT JSON_ARRAYAGG(
            JSON_ARRAY(
                UTL_RAW.CAST_TO_VARCHAR2(cw.task_id),
                UTL_RAW.CAST_TO_VARCHAR2(cw.channel),
                UTL_RAW.CAST_TO_VARCHAR2(cw.type),
                cw.blob
            ) ORDER BY cw.task_id, cw.idx
        )
        FROM checkpoint_writes cw
        WHERE cw.thread_id = checkpoints.thread_id
            AND cw.checkpoint_ns = checkpoints.checkpoint_ns
            AND cw.checkpoint_id = JSON_VALUE(checkpoint, '$.id')
    ) AS pending_writes,
    (
        SELECT JSON_ARRAYAGG(
            JSON_ARRAY(
                UTL_RAW.CAST_TO_VARCHAR2(cw.type),
                cw.blob
            ) ORDER BY cw.task_path, cw.task_id, cw.idx
        )
        FROM checkpoint_writes cw
        WHERE cw.thread_id = checkpoints.thread_id
            AND cw.checkpoint_ns = checkpoints.checkpoint_ns
            AND cw.channel = :tasks_channel
    ) AS pending_sends
FROM checkpoints"""

UPSERT_CHECKPOINT_BLOBS_SQL = """
    MERGE INTO checkpoint_blobs cb
    USING (SELECT :thread_id AS thread_id, :checkpoint_ns AS checkpoint_ns, :channel AS channel FROM dual) src
    ON (cb.thread_id = src.thread_id AND cb.checkpoint_ns = src.checkpoint_ns AND cb.channel = src.channel)
    WHEN MATCHED THEN
        UPDATE SET type = :type, blob = :blob
    WHEN NOT MATCHED THEN
        INSERT (thread_id, checkpoint_ns, channel, type, blob)
        VALUES (:thread_id, :checkpoint_ns, :channel, :type, :blob)
"""

UPSERT_CHECKPOINTS_SQL = """
    MERGE INTO checkpoints c
    USING (SELECT :thread_id AS thread_id, :checkpoint_ns AS checkpoint_ns FROM dual) src
    ON (c.thread_id = src.thread_id AND c.checkpoint_ns = src.checkpoint_ns)
    WHEN MATCHED THEN
        UPDATE SET checkpoint = :checkpoint, metadata = :metadata
    WHEN NOT MATCHED THEN
        INSERT (thread_id, checkpoint_ns, checkpoint, metadata)
        VALUES (:thread_id, :checkpoint_ns, :checkpoint, :metadata)
"""

UPSERT_CHECKPOINT_WRITES_SQL = """
    MERGE INTO checkpoint_writes cw
    USING (SELECT :thread_id AS thread_id, :checkpoint_ns AS checkpoint_ns, :checkpoint_id AS checkpoint_id, :task_id AS task_id, :idx AS idx FROM dual) src
    ON (cw.thread_id = src.thread_id AND cw.checkpoint_ns = src.checkpoint_ns AND cw.checkpoint_id = src.checkpoint_id AND cw.task_id = src.task_id AND cw.idx = src.idx)
    WHEN MATCHED THEN
        UPDATE SET channel = :channel, type = :type, blob = :blob
    WHEN NOT MATCHED THEN
        INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
        VALUES (:thread_id, :checkpoint_ns, :checkpoint_id, :task_id, :task_path, :idx, :channel, :type, :blob)
"""

INSERT_CHECKPOINT_WRITES_SQL = """
    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob)
    SELECT :thread_id, :checkpoint_ns, :checkpoint_id, :task_id, :task_path, :idx, :channel, :type, :blob
    FROM dual
    WHERE NOT EXISTS (
        SELECT 1 FROM checkpoint_writes
        WHERE thread_id = :thread_id
            AND checkpoint_ns = :checkpoint_ns
            AND checkpoint_id = :checkpoint_id
            AND task_id = :task_id
            AND idx = :idx
    )
"""


def _dump_blobs(
    serde: SerializerProtocol,
    thread_id: str,
    checkpoint_ns: str,
    values: dict[str, Any],
    versions: ChannelVersions,
) -> list[tuple[str, str, str, str, Optional[bytes]]]:
    if not versions:
        return []

    return [
        (
            thread_id,
            checkpoint_ns,
            k,
            *(serde.dumps_typed(values[k]) if k in values else ("empty", None)),
        )
        for k in versions
    ]


class ShallowOracleCheckpointer(BaseOracleCheckpointer):
    """A checkpoint saver that uses Oracle to store checkpoints.

    This checkpointer ONLY stores the most recent checkpoint and does NOT retain any history.
    It is meant to be a light-weight drop-in replacement for the OracleCheckpointer that
    supports most of the LangGraph persistence functionality with the exception of time travel.
    """

    SELECT_SQL = SELECT_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    lock: threading.Lock

    def __init__(
        self,
        conn: _internal.Conn,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "",
    ) -> None:
        warnings.warn(
            "ShallowOracleCheckpointer is deprecated as of version 2.0.20 and will be removed in 3.0.0. "
            "Use OracleCheckpointer instead, and invoke the graph with `graph.invoke(..., checkpoint_during=False)`.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(serde=serde, table_prefix=table_prefix)
        self.conn = conn
        self.lock = threading.Lock()

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "",
    ) -> Iterator["ShallowOracleCheckpointer"]:
        """Create a new ShallowOracleCheckpointer instance from a connection string.

        Args:
            conn_string: The Oracle connection string (username/password@host/service)
            serde: The serializer to use for serialization/deserialization
            table_prefix: Prefix to use for the tables

        Yields:
            ShallowOracleCheckpointer: A new ShallowOracleCheckpointer instance
        """
        conn = None
        try:
            import oracledb
            conn = oracledb.connect(conn_string)
            yield cls(conn, serde=serde, table_prefix=table_prefix)
        finally:
            if conn is not None:
                conn.close()

    def setup(self) -> None:
        """Set up the checkpoint database.

        This method creates the necessary tables in the Oracle database.
        It MUST be called directly by the user the first time checkpointer is used.
        """
        with self.lock:
            cursor = self.conn.cursor()
            try:
                # Create tables directly (no migrations needed for new implementation)
                for table_sql in self.MIGRATIONS:
                    try:
                        cursor.execute(table_sql)
                    except Exception:
                        # Table might already exist, continue
                        pass
                
                self.conn.commit()
            finally:
                cursor.close()

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
        on the provided config. For ShallowOracleCheckpointer, this method returns a list with
        ONLY the most recent checkpoint.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where
        if limit:
            query += f" FETCH FIRST {limit} ROWS ONLY"
        
        with self.lock:
            cursor = self.conn.cursor()
            try:
                # Replace the tasks channel placeholder
                query = query.replace(":tasks_channel", f"'{TASKS}'")
                
                cursor.execute(query, args)
                for row in cursor:
                    checkpoint: Checkpoint = {
                        **row[1],  # checkpoint
                        "channel_values": self._load_blobs(row[4]),  # channel_values
                        "pending_sends": [
                            self.serde.loads_typed((t.decode() if isinstance(t, bytes) else t, v))
                            for t, v in (row[6] or [])  # pending_sends
                        ]
                        if row[6]
                        else [],
                    }
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": row[0],  # thread_id
                                "checkpoint_ns": row[2],  # checkpoint_ns
                                "checkpoint_id": checkpoint["id"],
                            }
                        },
                        checkpoint=checkpoint,
                        metadata=row[3],  # metadata
                        pending_writes=self._load_writes(row[5] or []),  # pending_writes
                    )
            finally:
                cursor.close()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the Oracle database. The checkpoint is associated
        with the provided config. For ShallowOracleCheckpointer, this method saves ONLY the most recent
        checkpoint and overwrites a previous checkpoint, if it exists.

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")

        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        with self.lock:
            cursor = self.conn.cursor()
            try:
                # Delete old checkpoint data for this thread/namespace
                cursor.execute(
                    """DELETE FROM checkpoint_writes
                    WHERE thread_id = :1 AND checkpoint_ns = :2 AND checkpoint_id NOT IN (:3, :4)""",
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint["id"],
                        configurable.get("checkpoint_id", ""),
                    ),
                )
                
                # Insert checkpoint blobs
                blobs_data = self._dump_blobs(
                    thread_id,
                    checkpoint_ns,
                    copy.pop("channel_values", {}),
                    new_versions,
                )
                for blob_data in blobs_data:
                    cursor.execute(self.UPSERT_CHECKPOINT_BLOBS_SQL, blob_data)
                
                # Insert checkpoint
                cursor.execute(
                    self.UPSERT_CHECKPOINTS_SQL,
                    (
                        thread_id,
                        checkpoint_ns,
                        json.dumps(copy),
                        json.dumps(get_checkpoint_metadata(config, metadata)),
                    ),
                )
                
                self.conn.commit()
            finally:
                cursor.close()
        
        return next_config

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        channel_values: dict[str, Any],
        new_versions: ChannelVersions,
    ) -> List[Tuple]:
        """Dump channel values as blobs for storage.

        Args:
            thread_id: The thread ID.
            checkpoint_ns: The checkpoint namespace.
            channel_values: The channel values to dump.
            new_versions: The new channel versions.

        Returns:
            List of tuples containing blob data for database insertion.
        """
        blobs = []
        for channel, value in channel_values.items():
            if channel in new_versions:
                blob_data = self.serde.dumps_typed(value)
                blobs.append((
                    thread_id,
                    checkpoint_ns,
                    channel,
                    new_versions[channel],
                    type(value).__name__,
                    blob_data,
                ))
        return blobs

    def _search_where(
        self,
        config: Optional[RunnableConfig],
        filter: Optional[dict[str, Any]],
        before: Optional[RunnableConfig] = None,
    ) -> Tuple[str, List[Any]]:
        """Return WHERE clause predicates for list() given config, filter, before.

        This method returns a tuple of a string and a list of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = :1 AND column2 = :2". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = :1")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = :2")
                param_values.append(checkpoint_ns)

            if checkpoint_id := get_checkpoint_id(config):
                wheres.append("checkpoint_id = :3")
                param_values.append(checkpoint_id)

        # construct predicate for metadata filter
        if filter:
            wheres.append("JSON_EXISTS(metadata, :4)")
            param_values.append(str(filter))

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < :5")
            param_values.append(get_checkpoint_id(before))

        return (
            " WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )


class AsyncShallowOracleCheckpointer(BaseOracleCheckpointer):
    """An asynchronous checkpoint saver that uses Oracle to store checkpoints.

    This checkpointer ONLY stores the most recent checkpoint and does NOT retain any history.
    It is meant to be a light-weight drop-in replacement for the AsyncOracleCheckpointer that
    supports most of the LangGraph persistence functionality with the exception of time travel.
    """

    SELECT_SQL = SELECT_SQL
    MIGRATIONS = MIGRATIONS
    UPSERT_CHECKPOINT_BLOBS_SQL = UPSERT_CHECKPOINT_BLOBS_SQL
    UPSERT_CHECKPOINTS_SQL = UPSERT_CHECKPOINTS_SQL
    UPSERT_CHECKPOINT_WRITES_SQL = UPSERT_CHECKPOINT_WRITES_SQL
    INSERT_CHECKPOINT_WRITES_SQL = INSERT_CHECKPOINT_WRITES_SQL

    lock: asyncio.Lock

    def __init__(
        self,
        conn: _ainternal.Conn,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "",
    ) -> None:
        warnings.warn(
            "AsyncShallowOracleCheckpointer is deprecated as of version 2.0.20 and will be removed in 3.0.0. "
            "Use AsyncOracleCheckpointer instead, and invoke the graph with `await graph.ainvoke(..., checkpoint_during=False)`.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(serde=serde, table_prefix=table_prefix)
        self.conn = conn
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_running_loop()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: Optional[SerializerProtocol] = None,
        table_prefix: str = "",
    ) -> AsyncIterator["AsyncShallowOracleCheckpointer"]:
        """Create a new AsyncShallowOracleCheckpointer instance from a connection string.

        Args:
            conn_string: The Oracle connection string (username/password@host/service)
            serde: The serializer to use for serialization/deserialization
            table_prefix: Prefix to use for the tables

        Yields:
            AsyncShallowOracleCheckpointer: A new AsyncShallowOracleCheckpointer instance
        """
        conn = None
        try:
            import oracledb
            conn = await oracledb.connect_async(conn_string)
            yield cls(conn, serde=serde, table_prefix=table_prefix)
        finally:
            if conn is not None:
                await conn.close()

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously.

        This method creates the necessary tables in the Oracle database.
        It MUST be called directly by the user the first time checkpointer is used.
        """
        async with self.lock:
            async with _ainternal.get_connection(self.conn) as conn:
                async with conn.cursor() as cursor:
                    # Create tables directly (no migrations needed for new implementation)
                    for table_sql in self.MIGRATIONS:
                        try:
                            await cursor.execute(table_sql)
                        except Exception:
                            # Table might already exist, continue
                            pass

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the Oracle database based
        on the provided config. For AsyncShallowOracleCheckpointer, this method returns a list with
        ONLY the most recent checkpoint.
        """
        where, args = self._search_where(config, filter, before)
        query = self.SELECT_SQL + where
        if limit:
            query += f" FETCH FIRST {limit} ROWS ONLY"
        
        async with self.lock:
            async with _ainternal.get_connection(self.conn) as conn:
                async with conn.cursor() as cursor:
                    # Replace the tasks channel placeholder
                    query = query.replace(":tasks_channel", f"'{TASKS}'")
                    
                    await cursor.execute(query, args)
                    async for row in cursor:
                        checkpoint: Checkpoint = {
                            **row[1],  # checkpoint
                            "channel_values": self._load_blobs(row[4]),  # channel_values
                            "pending_sends": [
                                self.serde.loads_typed((t.decode() if isinstance(t, bytes) else t, v))
                                for t, v in (row[6] or [])  # pending_sends
                            ]
                            if row[6]
                            else [],
                        }
                        yield CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": row[0],  # thread_id
                                    "checkpoint_ns": row[2],  # checkpoint_ns
                                    "checkpoint_id": checkpoint["id"],
                                }
                            },
                            checkpoint=checkpoint,
                            metadata=row[3],  # metadata
                            pending_writes=self._load_writes(row[5] or []),  # pending_writes
                        )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the Oracle database. The checkpoint is associated
        with the provided config. For AsyncShallowOracleCheckpointer, this method saves ONLY the most recent
        checkpoint and overwrites a previous checkpoint, if it exists.

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to save.
            metadata: Additional metadata to save with the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")

        copy = checkpoint.copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

        async with self.lock:
            async with _ainternal.get_connection(self.conn) as conn:
                async with conn.cursor() as cursor:
                    # Delete old checkpoint data for this thread/namespace
                    await cursor.execute(
                        """DELETE FROM checkpoint_writes
                        WHERE thread_id = :1 AND checkpoint_ns = :2 AND checkpoint_id NOT IN (:3, :4)""",
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint["id"],
                            configurable.get("checkpoint_id", ""),
                        ),
                    )
                    
                    # Insert checkpoint blobs
                    blobs_data = self._dump_blobs(
                        thread_id,
                        checkpoint_ns,
                        copy.pop("channel_values", {}),
                        new_versions,
                    )
                    for blob_data in blobs_data:
                        await cursor.execute(self.UPSERT_CHECKPOINT_BLOBS_SQL, blob_data)
                    
                    # Insert checkpoint
                    await cursor.execute(
                        self.UPSERT_CHECKPOINTS_SQL,
                        (
                            thread_id,
                            checkpoint_ns,
                            json.dumps(copy),
                            json.dumps(get_checkpoint_metadata(config, metadata)),
                        ),
                    )
        
        return next_config
