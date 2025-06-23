> **Configuration (via environment variables)**
> - `API_PREFIX`: Base path for memory endpoints (e.g. `/api/v1`), default `/api/v1`.
> - `MEMORY_BACKEND`: Select backend (`dynamo`, `firestore`, `sqlite`, `json`), default `json`.
> - `AWS_REGION`: (for DynamoDB backend)
> - `DYNAMO_TABLE`: DynamoDB table name
> - `FIRESTORE_PROJECT`: Firestore project ID
# Memory Layer

The Memory Layer is the core, centralized state store for any Copilot framework built on ZSI principles. It ensures continuity, personalization, and auditability by managing user context, agent states, and interaction history across sessions and deployments.

## Key Responsibilities

1. **Context Management**  
   - **Load**: Retrieve `UserContext` by `user_id` and optional `date` or `namespace`.  
   - **Save**: Persist updates atomically, including version metadata.  
   - **Rollback**: Access immutable historical snapshots to recover or audit previous states.

2. **Synchronization & Consistency**  
   - Leverage event-driven pub/sub or distributed locks to synchronize updates across microservices.  
   - Implement optimistic locking or vector clocks to resolve write conflicts gracefully.

3. **Predictive Insights & Forecasting**  
   - Expose hooks for real-time analytics modules (e.g. trend detectors, proactive nudges).  
   - Support summarization APIs that compute aggregated metrics (e.g. weekly trends, goal progress).

4. **Security & Compliance**  
   - Enforce RBAC/ABAC policies at the store layer.  
   - Encrypt data at rest and in transit.  
   - Log all read/write operations with traceable request IDs.

## FastAPI Integration

### Endpoints

```yaml
GET    {API_PREFIX}/memory/{user_id}           # Load current context
POST   {API_PREFIX}/memory/{user_id}           # Merge and persist updates
GET    {API_PREFIX}/memory/{user_id}/history   # List versioned snapshots
POST   {API_PREFIX}/memory/{user_id}/rollback  # Rollback to a specific version
```

### Pydantic Models

```python
class UserContext(BaseModel):
    user_id: str
    version: str
    data: Dict[str, Any]
    updated_at: datetime

class MemoryUpdate(BaseModel):
    user_id: str
    delta: Dict[str, Any]
    tags: Optional[List[str]] = []
```

## Persistence Backends

- **DynamoDB**  
  - Table per environment, partitioned by `user_id`, sort key `version`.  
  - Supports conditional writes and ACID transactions.

- **Firestore / Supabase**  
  - Document-per-user with subcollections for versions.  
  - Real-time sync and offline persistence options.

- **Local JSON / SQLite (MVP)**  
  - File or local database for rapid prototyping.  
  - Useful fallback when cloud services are unavailable.

- **Environment Configuration**
  - Use `AWS_REGION`, `DYNAMO_TABLE`, or `FIRESTORE_PROJECT` to configure cloud backends.

## Implementation Best Practices

- Abstract store behind `MemoryStore` interface for pluggable backends, configured via environment variable `MEMORY_BACKEND`.  
- Keep update operations idempotent: replaying a delta should not duplicate state.  
- Batch writes and reads to optimize throughput and minimize latency.  
- Maintain comprehensive audit logs and expose them via `/api/v1/memory/logs`.  
- Expose feature toggles (via config flags) to enable/disable advanced capabilities (versioning, forecasting hooks) per deployment.

## Extension Points

- **Custom Hooks**: Plug in domain-specific processors (e.g., nutritional trend analyzers, trade signal predictors).  
- **Retention Policies**: Configure automatic cleanup of stale snapshots.  
- **Metrics & Monitoring**: Integrate with Prometheus/Grafana or CloudWatch for operational visibility.

## Environment Setup Example

```bash
export API_PREFIX="/api/v1"
export MEMORY_BACKEND="dynamo"
export AWS_REGION="us-west-2"
export DYNAMO_TABLE="zsi_memory_dev"
export FIRESTORE_PROJECT="my-firestore-project"
```