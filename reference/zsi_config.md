<file name=1 path=zsi_config.md><!--
THIS CONFIGURATION IS FULLY PARAMETERIZED FOR ENVIRONMENTS, MODULES, AND HOSTING TARGETS.
Do not hard-code URLs or secrets here—use environment variables as placeholders.
-->

# ZSI Config

```yaml
version: 1.0.0
mission: Launch structural intelligence workspace for business incubation

# Core framework settings
environment:
  default: ${ZSI_ENV:-development}            # supported: development, staging, production
  available: 
    - development
    - staging
    - production
  variables:
    APP_NAME: ${APP_NAME:-copilot-framework}
    LOG_LEVEL: ${LOG_LEVEL:-info}
    FASTAPI_HOST: ${FASTAPI_HOST:-0.0.0.0}
    FASTAPI_PORT: ${FASTAPI_PORT:-8000}
    AWS_REGION: ${AWS_REGION:-us-east-1}
    DYNAMODB_TABLE: ${DYNAMODB_TABLE:-zsi-memory}
    RAILWAY_STATIC_URL: ${RAILWAY_STATIC_URL:-}
    OAUTH2_CLIENT_ID: ${OAUTH2_CLIENT_ID:-}
    OAUTH2_CLIENT_SECRET: ${OAUTH2_CLIENT_SECRET:-}

hosting:
  - name: fastapi
    description: Local dev server or containerized (Railway, Docker)
    entrypoint: uvicorn main:app --host ${FASTAPI_HOST} --port ${FASTAPI_PORT}
  - name: aws_lambda
    description: Serverless via API Gateway, Lambda function
    deployment: SAM/Serverless or Terraform module, region: ${AWS_REGION}
  - name: docker
    description: Containerized with Docker Compose or Kubernetes
    image: ${DOCKER_IMAGE:-${DOCKER_REGISTRY:-your-registry}/zsi-framework:${IMAGE_TAG:-latest}}
  - name: firebase_functions
    description: Google Cloud Functions with Firebase backend (optional)
  - name: supabase
    description: Postgres and Auth platform for staging/scale

security:
  api_key_header: ${API_KEY_HEADER:-X-API-KEY}
  cors_origins:
    development: ["*"]
    staging: ["https://staging.${APP_NAME}.com"]
    production: ["https://www.${APP_NAME}.com"]
  openapi:
    title: ${APP_NAME} API
    version: ${APP_VERSION:-0.1.0}
    docs_url: /docs
    redoc_url: /redoc
  authentication:
    oauth2:
      enabled: ${OAUTH2_ENABLED:-false}
      token_url: /auth/token
    jwt:
      secret: ${JWT_SECRET:-changeme}
      algorithm: HS256
      expires_in: 3600

modules:
  - name: scaffolding
    description: Base project structure and FastAPI entrypoint
  - name: agent_core
    description: BaseAgent class, registration, and dynamic discovery
  - name: orchestrator
    description: Intent routing, middleware hooks, memory persistence
  - name: persistence
    description: Pluggable storage adapters (JSON, DynamoDB, Supabase)
  - name: security
    description: API key enforcement, CORS, OpenAPI and auth layers
  - name: observability
    description: Logging, metrics, tracing (OpenTelemetry hooks)
  - name: extensions
    description: Future plugin points (OAuth2, JWT, third-party integrations)

future_expansion:
  - name: api_gateways
    description: Defines environment-specific API Gateway configs (FastAPI, AWS, GCP)
  - name: auth_strategies
    description: Placeholder for OAuth2 flows, JWT rotations, and custom auth hooks
  - name: monitoring
    description: Hook points for metrics, tracing, alert rules (OpenTelemetry, Prometheus)
```