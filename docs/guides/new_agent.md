# Adding a New Agent

This guide explains how to create and register a new agent within NCOS.

1. **Create the agent class** in `src/ncos/agents/` or an appropriate package.
2. **Implement required methods** such as `analyze`, `act`, and any callbacks.
3. **Update configuration** files in `config/agents/` to include the new agent profile.
4. **Register the agent** with the orchestrator via the agent mesh or plugin system.
5. **Write tests** under `tests/` to validate agent behaviors.

See `docs/orchestrator/system_prompt.md` for orchestrator details.
