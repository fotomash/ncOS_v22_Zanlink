# Agent Routing Logic

## Dynamic Intent-Based Routing

Routing of incoming intents is fully driven by the intent payload and runtime configuration:

1. **Configuration**  
   - `MODULES_PATH` (env var): directory where all agents are discovered (default: `modules/`).  
   - `DEFAULT_AGENT` (env var): key of the fallback agent if none matches (default: `fallback`).

2. **Intent Extraction**  
   - Extract `business_type` (or legacy `agent`) from the intent payload.  
   - Validate the payload against the Pydantic `Intent` schema.

3. **Handler Resolution**  
   - Locate handler in the dynamic `agent_registry` loaded at application startup.  
   - If missing, fall back to `DEFAULT_AGENT` and log a warning.

4. **Context Management**  
   - Load user context from the configured memory store (JSON, DynamoDB, Firebase).  
   - Pass both intent and mutable `UserMemory` to the handler.

5. **Execution & Persistence**  
   - Execute `handler.handle_intent(intent, memory)`.  
   - Persist any updates returned by the handler back to the memory store.

6. **Response**  
   - Return the structured `AgentResult` (including messages, tags, triggers) to the caller.

## Implementation in FastAPI

```python
from fastapi import FastAPI, Header, HTTPException
import os
from core.orchestrator import agent_registry
from user_memory import load_user_memory, save_user_memory
from schema.models import Intent, AgentResult

app = FastAPI(
    title="Copilot Framework API",
    description="Dynamic, pluggable agents via ZSI pattern",
    version="1.0.0",
    openapi_tags=[{"name": "routing", "description": "Intent routing endpoints"}],
)

MODULES_PATH = os.getenv("MODULES_PATH", "modules/")
DEFAULT_AGENT = os.getenv("DEFAULT_AGENT", "fallback")

@app.post("/intent", tags=["routing"], response_model=AgentResult)
async def route_intent(
    intent: Intent,
    api_key: str = Header(..., description="API key for authentication")
):
    user_id = intent.user_id or "default_user"
    agent_key = intent.business_type or intent.agent or DEFAULT_AGENT
    handler = agent_registry.get(agent_key)
    if not handler:
        # fallback
        handler = agent_registry.get(DEFAULT_AGENT)
        if not handler:
            raise HTTPException(status_code=404, detail=f"No handler for {agent_key}")
    memory = load_user_memory(user_id, intent.session_id)
    result = await handler.handle_intent(intent, memory)
    save_user_memory(user_id, intent.session_id, memory)
    return result
```


## Extension Hooks

1. **Custom Pre-Processors**  
   - Insert middleware before loading memory (e.g., decrypt or audit logs).

2. **Post-Dispatch Notifications**  
   - Hook into result to trigger external events (e.g., webhooks, pub/sub).

3. **Observability**  
   - Integrate with OpenTelemetry or logging services by wrapping handlers.

4. **Security Enhancements**  
   - Replace `api_key` header with OAuth2/JWT authentication schemes.
