# ncOS v22.0 - Complete Bundle Structure

## 📁 Directory Structure
```
ncOS_v22.0_Zanlink/
├── 📄 README.md
├── 📄 CHANGELOG.md
├── 📄 LICENSE
├── 📄 requirements.txt
├── 📄 requirements-dev.txt
├── 📄 Dockerfile
├── 📄 docker-compose.yml
├── 📄 .env.example
├── 📄 ncos_launcher.py
├── 📄 GPT_INSTRUCTIONS.md
│
├── 📁 agents/
│   ├── __init__.py
│   ├── master_orchestrator.py
│   ├── vector_memory_boot.py
│   ├── parquet_ingestor.py
│   ├── dimensional_fold.py
│   ├── market_conditioner.py
│   ├── signal_processor.py
│   ├── strategy_evaluator.py
│   ├── position_manager.py
│   ├── risk_analyzer.py
│   ├── metrics_aggregator.py
│   ├── smc_router.py
│   ├── maz2_executor.py
│   ├── tmc_executor.py
│   ├── liquidity_sniper_agent.py
│   └── entry_executor_smc_agent.py
│
├── 📁 config/
│   ├── ncos_config_zanlink.json
│   ├── agent_registry.yaml
│   ├── bootstrap_config.yaml
│   └── *_config.yaml (individual agent configs)
│
├── 📁 integrations/
│   ├── __init__.py
│   ├── ncos_zanlink_bridge.py
│   ├── ncos_llm_gateway.py
│   ├── ncos_data_package_manager.py
│   ├── ncos_prompt_templates.py
│   ├── ncos_integration_bridge.py
│   ├── ncos_chatgpt_actions.py
│   ├── ncos_chatgpt_schema_zanlink.yaml
│   └── offline_enrichment.py
│
├── 📁 core/
│   ├── __init__.py
│   ├── engine.py
│   ├── state_machine.py
│   ├── event_detector.py
│   ├── market_maker.py
│   └── entry_executor_smc.py
│
├── 📁 api/
│   ├── __init__.py
│   ├── app.py
│   ├── zbar_routes.py
│   ├── llm_assistant.py
│   └── unified_mt4_processor.py
│
├── 📁 processors/
│   ├── __init__.py
│   ├── tick_processor.py
│   ├── zbar_bridge.py
│   ├── zbar_parquet_bridge.py
│   ├── zbar_writer.py
│   ├── zbar_reader.py
│   └── menu_system.py
│
├── 📁 data/
│   ├── 📁 cache/
│   ├── 📁 zbar/
│   ├── 📁 journals/
│   ├── 📁 parquet/
│   └── 📁 models/
│
├── 📁 logs/
│   └── .gitkeep
│
├── 📁 tests/
│   ├── __init__.py
│   ├── test_integration.py
│   ├── test_llm_gateway.py
│   ├── test_patterns.py
│   └── test_zanlink_bridge.py
│
├── 📁 scripts/
│   ├── deploy.sh
│   ├── start_ncos.sh
│   ├── stop_ncos.sh
│   ├── status_ncos.sh
│   ├── integration_bootstrap.py
│   ├── add_structure.py
│   ├── test_llm_integration.py
│   └── quick_start_predictive.py
│
├── 📁 docs/
│   ├── NCOS_V22_DOCUMENTATION.md
│   ├── API_REFERENCE.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── MIGRATION_GUIDE.md
│   └── TROUBLESHOOTING.md
│
└── 📁 nginx/
    ├── nginx.conf
    └── ssl/
        ├── cert.pem
        └── key.pem
```

## 🔧 Configuration Files

### .env.example
```bash
# API Keys
ZANLINK_API_KEY=your-zanlink-api-key
OPENAI_API_KEY=your-openai-api-key

# Service Configuration
NCOS_ENV=production
NCOS_LOG_LEVEL=INFO
NCOS_CACHE_TTL=300
NCOS_MAX_RETRIES=3

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/ncos

# Redis (optional)
REDIS_URL=redis://localhost:6379

# MT4 Connection
MT4_SERVER=your-broker-server
MT4_LOGIN=your-account
MT4_PASSWORD=your-password
```

### nginx.conf
```nginx
events {
    worker_connections 1024;
}

http {
    upstream ncos_backend {
        server ncos-app:8000;
        server ncos-app:8004 backup;
    }

    server {
        listen 80;
        server_name zanlink.com;

        location /api/v1/ {
            proxy_pass http://ncos_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
```

## 🚀 Quick Start Commands

```bash
# 1. Clone and setup
git clone <repository>
cd ncOS_v22.0_Zanlink
cp .env.example .env
# Edit .env with your keys

# 2. Docker deployment
docker-compose up -d

# 3. Local development
pip install -r requirements.txt
python ncos_launcher.py

# 4. Test the integration
python scripts/test_llm_integration.py

# 5. Check status
docker-compose ps
curl https://zanlink.com/api/v1/health
```

## 📝 File Purposes

### Core Files
- `ncos_launcher.py` - Main application launcher
- `app.py` - Journal dashboard (Streamlit)
- `engine.py` - Vector processing engine
- `llm_assistant.py` - LLM integration service

### Integration Files
- `ncos_zanlink_bridge.py` - Zanlink API client
- `ncos_llm_gateway.py` - Unified LLM endpoint
- `ncos_data_package_manager.py` - Data preprocessing
- `ncos_prompt_templates.py` - Dynamic prompts
- `ncos_chatgpt_actions.py` - ChatGPT endpoints

### Configuration
- `ncos_config_zanlink.json` - Main configuration
- `ncos_chatgpt_schema_zanlink.yaml` - OpenAPI schema
- `GPT_INSTRUCTIONS.md` - ChatGPT setup guide

### Docker
- `Dockerfile` - Multi-stage build
- `docker-compose.yml` - Service orchestration
- `offline_enrichment.py` - Data enrichment engine

## 🎯 Key Features
1. Zanlink cloud integration
2. Pre-processed LLM responses
3. ChatGPT custom actions
4. Docker containerization
5. Offline data enrichment
6. Intelligent caching
7. Multi-agent orchestration
8. Real-time pattern detection

## 📞 Support
- Documentation: https://docs.zanlink.com/ncos
- API Status: https://status.zanlink.com
- Email: support@zanlink.com
