# Orchestrator Workflows

This document outlines typical orchestrator workflows for NCOS.

1. **Initialization**
   - Load configuration files and environment variables.
   - Instantiate agents defined in the configuration.
2. **Event Loop**
   - Receive market data or user requests.
   - Dispatch events to the relevant agents through the agent mesh.
3. **Execution**
   - Aggregate agent responses.
   - Execute trades or actions based on orchestrator logic.
4. **Persistence**
   - Store session data and vector embeddings.
5. **Shutdown**
   - Gracefully save state and terminate running tasks.

Refer to `docs/orchestrator/system_prompt.md` for the full system prompt specification.
