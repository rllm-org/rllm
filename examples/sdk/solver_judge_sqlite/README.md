# SQLite Session Storage Example

This example demonstrates how to use SQLite-based session storage with the RLLM SDK instead of the default in-memory storage. SQLite storage is useful for:

- **Multi-process scenarios**: Share traces across different processes
- **Persistence**: Traces survive process restarts
- **Debugging**: Inspect traces after execution completes
- **Large-scale training**: Handle large volumes of traces efficiently

## Overview

The example implements a solver-judge workflow similar to `examples/sdk/solver_judge_workflow`, but uses `SqliteSessionStorage` instead of the default `InMemoryStorage`.

### Key Differences from In-Memory Storage

1. **Storage Initialization**:
   ```python
   # In-memory (default)
   with session(agent="solver") as sess:
       # Traces stored in memory

   # SQLite
   storage = SqliteSessionStorage(db_path="./traces.db")
   with ContextVarSession(agent="solver", storage=storage) as sess:
       # Traces stored in SQLite database
   ```

2. **Persistence**: Traces are persisted to disk and can be accessed after the process exits

3. **Multi-process**: Multiple processes can share the same SQLite database

## Files

- `solver_judge_flow_sqlite.py`: Solver-judge workflow with SQLite storage
- `train_sqlite.py`: Training script using the SQLite workflow
- `train_sqlite.sh`: Shell script to run training with full configuration
- `test_sqlite_storage.py`: Simple end-to-end test to verify SQLite storage
- `README.md`: This file

## Quick Start

### 1. Simple Test (Recommended First)

Test the SQLite storage without running full training:

```bash
# Make sure the proxy is running on localhost:4000
# Then run the test script
python3 -m examples.sdk.solver_judge_sqlite.test_sqlite_storage
```

This will:
- Create a test SQLite database (`test_traces.db`)
- Run a simple solver-judge workflow
- Verify traces are stored and retrievable
- Test cross-process storage simulation

### 2. Full Training

Run the complete training pipeline with SQLite storage:

```bash
# Ensure you have the countdown dataset prepared
# Then run the training script
./examples/sdk/solver_judge_sqlite/train_sqlite.sh
```

Or run with custom parameters:

```bash
python3 -m examples.sdk.solver_judge_sqlite.train_sqlite \
    trainer.total_epochs=10 \
    rllm.sdk.store.path=./my_traces.db
```

## Configuration

### SQLite Database Path

You can configure the database path in several ways:

1. **In code** (recommended for testing):
   ```python
   storage = SqliteSessionStorage(db_path="./my_traces.db")
   ```

2. **Via configuration** (for training):
   ```bash
   python3 -m examples.sdk.solver_judge_sqlite.train_sqlite \
       rllm.sdk.store.path=/path/to/traces.db
   ```

3. **Default location**: If not specified, defaults to `~/.rllm/traces.db`

### Storage Parameters

The `SqliteSessionStorage` class accepts:

- `db_path` (str | None): Path to SQLite database file
  - If `None`, uses `~/.rllm/traces.db`
  - Can use relative or absolute paths
  - Parent directories are created automatically

## Usage Patterns

### Basic Usage

```python
from rllm.sdk import ContextVarSession, SqliteSessionStorage, get_chat_client_async

# Create storage instance
storage = SqliteSessionStorage(db_path="./traces.db")

# Create client
client = get_chat_client_async(base_url="http://localhost:4000/v1", api_key="EMPTY")

# Use in session
with ContextVarSession(agent="my_agent", storage=storage) as sess:
    response = await client.chat.completions.create(
        model="Qwen/Qwen3-4B-Instruct-2507",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Traces are automatically stored in SQLite
    print(f"Traces: {len(sess.llm_calls)}")
```

### Sharing Storage Across Agents

```python
# Create one storage instance
storage = SqliteSessionStorage(db_path="./traces.db")

# Share across multiple agents
solver = Solver(storage=storage)
judge = Judge(storage=storage)

# All traces go to the same database
```

### Reading Traces from Database

```python
# Create storage instance pointing to existing database
storage = SqliteSessionStorage(db_path="./traces.db")

# Create session with known UID
sess = ContextVarSession(storage=storage)
sess._uid = "known_session_uid"

# Retrieve traces
traces = sess.llm_calls
```

## Architecture

### Storage Hierarchy

The SQLite storage uses a junction table pattern for efficient session-based queries:

```
traces table:
- id (primary key)
- data (JSON)
- metadata
- created_at

trace_sessions junction table:
- trace_id (foreign key to traces)
- session_uid
- created_at (denormalized for performance)
```

This allows:
- One trace to belong to multiple sessions (nested sessions)
- Fast queries by session UID using composite indexes
- Minimal data duplication

### Session UID Chain

Sessions maintain a UID chain for nested session support:

```python
# Outer session
with ContextVarSession(storage=storage) as outer:
    # outer._session_uid_chain = ["ctx_abc123"]

    # Nested session
    with ContextVarSession(storage=storage) as inner:
        # inner._session_uid_chain = ["ctx_abc123", "ctx_def456"]
        # Traces stored under both UIDs
```

## Performance Considerations

1. **Database Location**:
   - Local disk: Best performance
   - Network filesystem: May have locking issues
   - Use `PRAGMA journal_mode=DELETE` (default) for better network FS compatibility

2. **Concurrent Access**:
   - SQLite handles concurrent reads efficiently
   - Writes are serialized with `PRAGMA busy_timeout`
   - For high concurrency, consider database per process

3. **Database Size**:
   - Monitor database size with large-scale training
   - Consider periodic cleanup of old traces
   - Use `VACUUM` to reclaim space

## Troubleshooting

### Database Locked Errors

If you see "database is locked" errors:

1. Check that no other process is holding a write lock
2. Increase busy timeout (default: 5000ms)
3. Avoid network filesystems if possible
4. Use separate databases per process for high concurrency

### Missing Traces

If traces are not appearing:

1. Verify the storage instance is shared across sessions
2. Check that sessions have the correct UID chain
3. Ensure async operations complete before reading (use `await` properly)
4. Check database file permissions

### Performance Issues

If queries are slow:

1. Check that indexes are created (happens automatically)
2. Monitor database size and vacuum if needed
3. Consider using filters (session_uid, since, limit) in queries
4. Use local disk instead of network filesystem

## Comparison: In-Memory vs SQLite Storage

| Feature | InMemoryStorage | SqliteSessionStorage |
|---------|-----------------|---------------------|
| Speed | Fastest | Fast (with indexes) |
| Persistence | No | Yes |
| Multi-process | No | Yes |
| Memory usage | Higher | Lower (disk-backed) |
| Setup complexity | None | Minimal |
| Use case | Single-process, ephemeral | Multi-process, persistent |

## Advanced Topics

### Custom Database Schema

The SQLite store automatically creates tables and indexes. Schema is managed by `SqliteTraceStore` in `rllm/sdk/store/sqlite_store.py`.

### Querying Traces

Use the store's query methods for advanced filtering:

```python
from rllm.sdk.store import SqliteTraceStore

store = SqliteTraceStore(db_path="./traces.db")

# Query by session UID
traces = await store.get_by_session_uid(
    session_uid="ctx_abc123",
    since=time.time() - 3600,  # Last hour
    limit=100
)

# Advanced queries
traces = await store.query(
    session_uids=["ctx_abc123", "ctx_def456"],
    context_types=["llm_trace"],
    since=timestamp,
    limit=1000
)
```

### Migration from In-Memory

To migrate existing code from in-memory to SQLite:

1. Create a storage instance:
   ```python
   storage = SqliteSessionStorage(db_path="./traces.db")
   ```

2. Replace `session()` calls with `ContextVarSession(storage=storage)`:
   ```python
   # Before
   with session(agent="my_agent") as sess:
       ...

   # After
   with ContextVarSession(agent="my_agent", storage=storage) as sess:
       ...
   ```

3. Share the storage instance across all agents

## See Also

- `examples/sdk/solver_judge_workflow`: Original in-memory version
- `rllm/sdk/session/storage.py`: Storage implementations
- `rllm/sdk/store/sqlite_store.py`: SQLite store details
- `docs/sqlite_store_optimization.md`: Performance optimization guide
