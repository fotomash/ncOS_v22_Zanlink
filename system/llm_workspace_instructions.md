# 🧠 Zanzibar Structured Intelligence Workspace Initialization

## ⚙️ Environment Configuration

All runtime settings, endpoints, and feature flags must be resolved via environment variables or `zsi_config.yaml`—no hard-coded URLs or keys.

1. Create a `.env` file in the project root and ensure it’s loaded at startup (e.g., via `python-dotenv`).
2. Define these variables, adjusting values per environment:

   ```dotenv
   # Runtime environment
   FASTAPI_ENV=${FASTAPI_ENV:-development}        # Options: development, staging, production

   # Framework configuration
   ZSI_CONFIG_PATH=${ZSI_CONFIG_PATH:-zsi_config.yaml}

   # API Security
   API_KEY_HEADER_NAME=${API_KEY_HEADER_NAME:-X-API-KEY}
   API_KEY=${API_KEY:-<your-secret-key>}

   # Storage & Persistence
   USER_CONTEXT_PATH=${USER_CONTEXT_PATH:-data/user_context.json}
   RAILWAY_DB_URL=${RAILWAY_DB_URL:-}
   SUPABASE_URL=${SUPABASE_URL:-}
   SUPABASE_KEY=${SUPABASE_KEY:-}
   DYNAMODB_TABLE=${DYNAMODB_TABLE:-}
   AWS_LAMBDA_ARN=${AWS_LAMBDA_ARN:-}

   # Hosting Endpoints (referenced via variables)
   FASTAPI_BASE_URL=${FASTAPI_BASE_URL:-http://localhost:8000}
   RAILWAY_API_URL=${RAILWAY_API_URL:-}
   AWS_API_GATEWAY_URL=${AWS_API_GATEWAY_URL:-}
   S3_BUCKET_URL=${S3_BUCKET_URL:-}

   # Feature Flags for domain agents
   FEATURE_TRADING_ENABLED=${FEATURE_TRADING_ENABLED:-false}
   FEATURE_NANNY_ENABLED=${FEATURE_NANNY_ENABLED:-false}
   FEATURE_HEALTH_ENABLED=${FEATURE_HEALTH_ENABLED:-true}

   # Documentation
   MKDOCS_ENV=${MKDOCS_ENV:-development}
   ```

3. Reference these variables in your `mkdocs.yml` via environment substitution (e.g., using the mkdocs-macros-plugin) or in your CI pipeline, so docs automatically adapt per environment.
4. In code, never hard-code endpoints—always read from `os.getenv` or `zsi_config.yaml`.

---

## 🛠️ Future-Proofing & Extensibility

- All service endpoints, storage backends, and feature toggles should be resolved via environment variables or `zsi_config.yaml` to avoid hard-coding.
- Use feature flags (e.g., `FEATURE_TRADING_ENABLED`, `FEATURE_NANNY_ENABLED`) to conditionally enable domain-specific agents.
- Use `zsi_config.yaml` to declare available modules and discover them at startup, enabling drop‑in agents without additional code changes.
- For new business types (e.g., CareMatch, Trading, DietPilot), drop the agent package in `modules/<business_type>/agent.py` and add a corresponding entry in `zsi_config.yaml`.
- Ensure your CI/CD pipeline injects environment variables securely (e.g., via GitHub Actions Secrets or Terraform).
- Reference all infra endpoints (Railway, Lambda, DynamoDB, Supabase) by variable names `RAILWAY_DB_URL`, `AWS_LAMBDA_ARN`, `DYNAMODB_TABLE` for consistency.

## 1. Load Configuration & Blueprint
1. Ensure the following files are present in the root and referenced via environment variables:
   - `zsi_config.yaml`   → framework parameters and module mappings  
   - `startup_blueprint.json` → project objectives, phases, and deliverables  
   - `zsi_launcher.py`        → launcher script to wire everything together

2. In your Python environment, install dependencies:
   ```bash
   pip install fastapi pydantic uvicorn pyyaml
   ```

## 2. Bootstrap the FastAPI Orchestrator
1. Scaffold the FastAPI entrypoint:
   ```bash
   mkdir core modules schema data system flows
   touch main.py core/orchestrator.py
   ```
   > **Security & Config**  
   > - Load `.env` with `python-dotenv` or similar.  
   > - Globally enforce `API_KEY_HEADER_NAME` on all `/api/*` routes.  
   > - Configure CORS policies via environment flags (e.g., `FASTAPI_ENV`).  
2. Copy `orchestrator.py` into `core/orchestrator.py` and ensure it:
   - Loads `zsi_config.yaml`
   - Reads/writes `data/user_context.json`
   - Dispatches intents to modules in `modules/`

3. Create `main.py`:
   ```python
   from fastapi import FastAPI, Depends, Header, HTTPException
   from dotenv import load_dotenv
   import os
   from core.orchestrator import handle_user_input

   # Load environment
   load_dotenv()
   API_KEY_NAME = os.getenv("API_KEY_HEADER_NAME", "X-API-KEY")
   VALID_API_KEY = os.getenv("API_KEY")

   app = FastAPI(
       title="ZSI Copilot Framework",
       version="1.0.0",
       openapi_tags=[
           {"name": "zsi", "description": "Zanzibar Structured Intelligence endpoints"}
       ]
   )

   # API Key security dependency
   async def verify_api_key(api_key: str = Header(..., alias=API_KEY_NAME)):
       if api_key != VALID_API_KEY:
           raise HTTPException(status_code=403, detail="Invalid API Key")

   @app.post("/api/log", tags=["zsi"], dependencies=[Depends(verify_api_key)])
   async def log_event(payload: dict):
       return await handle_user_input(payload)

   @app.get("/api/ping", tags=["zsi"])
   def ping():
       return {"status": "alive"}
   ```

## 3. Implement & Register ZSI Modules
1. Under `modules/`, create subfolders:
   - `zbot/` (action agents)
   - `zse/` (signal/event analyzers)
   - `zbar/` (journaling & behavioral insights)

2. Each module must expose a common function:
   ```python
   def handle_intent(intent: dict, memory: dict) -> dict:
       # process intent, update memory, return response
   ```

3. Update `core/orchestrator.py` to dynamically import and invoke:
   ```python
   module = __import__(f"modules.{intent_type}.agent", fromlist=["handle_intent"])
   response = module.handle_intent(intent, memory)
   ```

## 4. Define Schemas & Memory
1. In `schema/models.py`, declare Pydantic models for:
   - `IntentPayload`
   - `UserContext`
   - `AgentResponse`

2. Initialize `data/user_context.json` as:
   ```json
   {}
   ```
3. Use read/write helpers to persist per-user, per-day context.

## 5. System Prompts & Flows
1. Place core LLM directives in `docs/orchestrator/system_prompt.md`.
2. Define routing logic in `system/agent_routing_logic.md`.
3. Outline user flows in `flows/` (e.g., `daily_logging_flow.md`, `cheat_day_flow.md`).

## 6. Launch & Test
1. Start the app locally:
   ```bash
   uvicorn main:app --reload
   ```
2. Send test payloads:
   ```bash
   curl -X POST http://localhost:8000/api/log -H "Content-Type: application/json" -d '{"user_id":"demo","intent":"log_meal", ...}'
   ```
3. Verify context persistence in `data/user_context.json`.

## 7. Deployment & Scaling
- **MVP**: Deploy to Railway or Render (FastAPI, JSON storage).
- **Scale**: Migrate to AWS Lambda + API Gateway + DynamoDB.
- **Future**: Add scheduler for daily nudges, integrate voice frontend.

- **CI/CD & Testing**:  
  * Add pytest suites for modules and orchestrator.  
  * Include pre-commit hooks for linting and type-checking (flake8, mypy).  
  * Configure GitHub Actions or GitLab CI to run tests, build docs, and deploy.

- **Observability & Monitoring**:  
  * Integrate structured logging (e.g., Loguru, Python stdlib).  
  * Expose Prometheus metrics via `/metrics`.  
  * Configure Sentry or similar for error tracking.

---

You are now ready to run and iterate your ZSI agent framework. Good luck!

---
## 🔮 Future Extensions
- Support OAuth2 / JWT authentication flows.
- Add voice frontend integration (e.g., WebSocket, Twilio, or custom SDK).
- Build a scheduler service for push notifications and reminders.
- Provide Terraform templates or CloudFormation for infra as code.
- Extend docs with auto-generated OpenAPI UI via MkDocs TechDocs.
