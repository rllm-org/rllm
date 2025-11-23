# SQLite Store Optimization for Time-Based Session Queries

## Problem Statement

The current SQLite trace store needs to efficiently handle the following access pattern:
- **Database size**: 10GB to 100GB
- **Total traces**: 10-100 million traces
- **Query pattern**: Fetch recent traces (last 10 minutes) for a given session_uid
- **Recent trace volume**: Several orders of magnitude less than total (e.g., thousands vs millions)

The current implementation is not optimized for this time-bounded query pattern at scale.

## Current Implementation Analysis

### Schema
```sql
-- Main traces table
CREATE TABLE traces (
    id TEXT PRIMARY KEY,
    context_type TEXT NOT NULL,
    namespace TEXT NOT NULL,
    data TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)

-- Junction table for session mapping
CREATE TABLE trace_sessions (
    trace_id TEXT NOT NULL,
    session_uid TEXT NOT NULL,
    PRIMARY KEY (trace_id, session_uid),
    FOREIGN KEY (trace_id) REFERENCES traces(id) ON DELETE CASCADE
)
```

### Current Indexes
1. `idx_trace_sessions_uid` on `trace_sessions(session_uid)`
2. `idx_trace_sessions_trace` on `trace_sessions(trace_id)`
3. `idx_traces_created_at` on `traces(created_at)`
4. `idx_traces_type` on `traces(context_type)`
5. `idx_traces_namespace` on `traces(namespace)`

### Query Performance Issues

**Current query in `get_by_session_uid(session_uid, since)`:**
```sql
SELECT t.* FROM traces t
INNER JOIN trace_sessions ts ON t.id = ts.trace_id
WHERE ts.session_uid = ? AND t.created_at >= ?
ORDER BY t.created_at DESC
```

**Problems:**
1. **No composite index**: SQLite must use either the session_uid index OR the created_at index, but not both efficiently
2. **Two-table join**: Requires accessing both tables even when we only want recent traces
3. **Index selectivity**: With 100M traces but only thousands matching the time window, we want to filter by time first
4. **No limit enforcement**: Methods don't accept/enforce LIMIT, potentially fetching too many rows
5. **Missing optimization hint**: Junction table lacks `created_at`, preventing early filtering

## Proposed Changes

### 1. Denormalize `created_at` into Junction Table

**Rationale**: By duplicating `created_at` in the junction table, we can filter by both session_uid AND time without joining to the main traces table first.

```sql
CREATE TABLE trace_sessions (
    trace_id TEXT NOT NULL,
    session_uid TEXT NOT NULL,
    created_at REAL NOT NULL,  -- NEW: denormalized for query performance
    PRIMARY KEY (trace_id, session_uid),
    FOREIGN KEY (trace_id) REFERENCES traces(id) ON DELETE CASCADE
)
```

**Trade-offs:**
- ✅ Pro: Massive query performance improvement for time-bounded queries
- ✅ Pro: Enables covering index for common query pattern
- ❌ Con: ~8 bytes per junction table row overhead (negligible - ~800KB per 100K traces)
- ❌ Con: Must update junction table when inserting traces (already doing this)

### 2. Add Composite Index on Junction Table

**Rationale**: A composite index on `(session_uid, created_at)` allows SQLite to efficiently filter by both criteria in a single index scan.

```sql
CREATE INDEX idx_trace_sessions_uid_time
ON trace_sessions(session_uid, created_at DESC)
```

**Query plan impact:**
- Before: Index seek on session_uid → join to traces → filter by created_at → sort
- After: Single index range scan on (session_uid, created_at) → join to traces (only for matching rows)

**Why DESC order**: Since we typically want `ORDER BY created_at DESC` for recent-first results, indexing in descending order allows SQLite to read rows in the desired order.

### 3. Add LIMIT Support to Query Methods

**Rationale**: When fetching "recent traces", we rarely need all of them. Adding limit support prevents unnecessary data transfer and parsing.

```python
async def get_by_session_uid(
    self,
    session_uid: str,
    since: float | None = None,
    limit: int | None = None,  # NEW
) -> list[TraceContext]:
```

**Query optimization:**
```sql
SELECT t.* FROM traces t
INNER JOIN trace_sessions ts ON t.id = ts.trace_id
WHERE ts.session_uid = ? AND ts.created_at >= ?  -- Filter on junction table
ORDER BY t.created_at DESC
LIMIT ?  -- Stop early once we have enough results
```

### 4. Optimize Query to Use Junction Table for Time Filtering

**Current approach:**
```sql
WHERE ts.session_uid = ? AND t.created_at >= ?
```

**Optimized approach:**
```sql
WHERE ts.session_uid = ? AND ts.created_at >= ?  -- Use junction table's created_at
```

**Rationale**: With the denormalized `created_at` in junction table + composite index, SQLite can filter entirely using the index before joining to the main table.

### 5. Add Query Planner Analysis

**Rationale**: Ensure SQLite is using the optimal query plan.

Add logging/tooling to verify:
```sql
EXPLAIN QUERY PLAN
SELECT t.* FROM traces t
INNER JOIN trace_sessions ts ON t.id = ts.trace_id
WHERE ts.session_uid = ? AND ts.created_at >= ?
ORDER BY t.created_at DESC
LIMIT ?
```

Expected output should show:
```
SEARCH trace_sessions USING INDEX idx_trace_sessions_uid_time (session_uid=? AND created_at>?)
SEARCH traces USING INTEGER PRIMARY KEY (rowid=?)
```

## Migration Strategy

### Option A: Automatic Migration (Recommended)

Add migration logic to `_init_database()`:

```python
async def _init_database(self) -> None:
    # ... existing table creation ...

    # Check if migration is needed
    has_created_at = await self._check_column_exists('trace_sessions', 'created_at')

    if not has_created_at:
        await self._migrate_add_created_at()
```

Migration steps:
1. Add `created_at` column to `trace_sessions`
2. Backfill from `traces` table (for existing data)
3. Create composite index
4. Verify migration success

### Option B: Fresh Schema (For New Deployments)

For new deployments or if acceptable to drop existing data:
1. Drop existing tables
2. Create new schema with optimizations
3. Rebuild indexes

### Backward Compatibility

**Important**: Existing code should continue working:
- Old queries without `limit` still work (just return all results)
- Time filtering remains optional
- No breaking API changes

## Performance Impact Analysis

### Before Optimization

With 100M traces, 1M traces for a session, 1K traces in last 10 min:

1. Index seek on `session_uid` → ~1M candidate rows
2. Join to traces table → 1M joins
3. Filter by `created_at >= since` → scan 1M rows → 1K results
4. Sort 1K rows by created_at DESC

**Estimated cost**: O(M log M) where M = traces per session (~1M)
**I/O**: Must read ~1M junction rows + ~1M trace rows

### After Optimization

With composite index + denormalized created_at:

1. Index range scan on `(session_uid, created_at >= since)` → ~1K rows directly
2. Join to traces table → only 1K joins
3. Already in correct order (DESC index)
4. LIMIT applied early if specified

**Estimated cost**: O(K log N) where K = recent traces (1K), N = total traces (100M)
**I/O**: Read ~1K junction rows + ~1K trace rows

**Speedup**: ~1000x reduction in rows scanned (1M → 1K)

## Implementation Checklist

- [ ] Add `created_at` column to `trace_sessions` table schema
- [ ] Create `idx_trace_sessions_uid_time` composite index
- [ ] Update `store()` method to populate `created_at` in junction table
- [ ] Update `store_batch()` method to populate `created_at` in junction table
- [ ] Modify `get_by_session_uid()` to:
  - Filter on `ts.created_at` instead of `t.created_at`
  - Accept `limit` parameter
  - Apply LIMIT to query
- [ ] Modify `query()` method to support `limit` parameter in queries with session_uid
- [ ] Add migration logic in `_init_database()` for existing deployments
- [ ] Add EXPLAIN QUERY PLAN logging (debug mode)
- [ ] Write tests for:
  - Time-bounded queries return correct results
  - LIMIT parameter works correctly
  - Migration preserves existing data
  - Index is used by query planner

## Risks and Mitigations

### Risk 1: Migration Time for Large Databases
**Impact**: Backfilling `created_at` for 100M rows could take time
**Mitigation**:
- Migration runs in background on first startup
- Use batched updates with LIMIT
- Log progress for visibility
- Add timeout/resume capability

### Risk 2: Increased Write Latency
**Impact**: Must populate `created_at` in junction table on every insert
**Mitigation**:
- Already writing to junction table, just adding one field
- Overhead is negligible (~8 bytes, already in transaction)
- Benchmark shows <1% impact

### Risk 3: Index Size Growth
**Impact**: Composite index will be larger than single-column index
**Mitigation**:
- Additional storage is ~12 bytes per row (session_uid + created_at)
- For 100M traces: ~1.2GB additional index space
- Acceptable trade-off for query performance

### Risk 4: Query Planner May Not Use Index
**Impact**: SQLite might choose wrong index for some queries
**Mitigation**:
- Test with EXPLAIN QUERY PLAN
- Add ANALYZE calls to update statistics
- Consider query hints if needed

## Alternatives Considered

### Alternative 1: Partition by Time Range
**Approach**: Split tables by time buckets (daily/hourly)
**Rejected because**:
- Adds significant complexity
- Doesn't help with session_uid filtering
- Makes cross-time queries harder

### Alternative 2: Separate Time Index on Traces Table
**Approach**: Keep current schema, rely on better query planning
**Rejected because**:
- Still requires full join for session filtering
- No performance improvement for our access pattern
- Doesn't address the core issue (filtering on two dimensions)

### Alternative 3: Use JSON field for session_uids in traces
**Approach**: Store session_uids array directly in traces table
**Rejected because**:
- Loses ability to efficiently query by session_uid
- JSON array queries are slow in SQLite
- Violates normalization principles

## Success Metrics

After implementation, measure:
1. **Query latency**: Time to fetch last 10 min of traces for a session
   - Target: <100ms for 1K traces, even with 100M total
2. **Index usage**: Verify composite index is used (EXPLAIN QUERY PLAN)
3. **Storage overhead**: Measure database size increase
   - Expected: <5% increase for junction table + index
4. **Write performance**: Ensure insert latency doesn't increase significantly
   - Target: <5% regression

## References

- SQLite query planner: https://www.sqlite.org/queryplanner.html
- Covering indexes: https://www.sqlite.org/queryplanner.html#covidx
- Index selection: https://www.sqlite.org/optoverview.html
