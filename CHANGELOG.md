# ncOS Changelog

## [v22.0] - 2024-01-15 - Zanlink Enhanced Edition

### Added
- **Zanlink Cloud Integration**
  - Global API endpoints at https://zanlink.com/api/v1
  - Built-in authentication and rate limiting
  - Automatic failover and retry logic

- **LLM Gateway System**
  - Unified endpoint for all LLM interactions
  - Pre-processed data packages
  - Context-aware response formatting

- **ChatGPT Custom Actions**
  - OpenAPI 3.0 schema for direct integration
  - Simplified endpoints optimized for ChatGPT
  - One-click setup with provided schema

- **Data Package Manager**
  - Intelligent caching with 5-minute TTL
  - Pre-computation of common queries
  - 80% reduction in processing time

- **Prompt Template Library**
  - Dynamic prompt generation
  - Context-aware templates
  - Consistent formatting across all responses

- **Docker Support**
  - Complete Dockerfile for easy deployment
  - Multi-stage build for optimized images
  - Environment-based configuration

- **Enhanced Pattern Recognition**
  - ML-enhanced Wyckoff detection
  - Improved SMC order block identification
  - Real-time pattern confidence scoring

### Changed
- **API Structure**
  - Consolidated from 50+ endpoints to 4 main endpoints
  - Standardized response format
  - Improved error handling

- **Performance**
  - 5x faster response times with caching
  - Reduced memory footprint by 40%
  - Optimized database queries

- **Configuration**
  - Centralized configuration in JSON format
  - Environment variable support
  - Hot-reload capability

### Fixed
- Memory leak in vector engine
- Pattern detection accuracy on low liquidity pairs
- WebSocket connection stability
- ZBAR journal file rotation
- MT4 processor timeout issues

### Deprecated
- Legacy API endpoints (will be removed in v23.0)
- XML configuration format
- Python 3.7 support

## [v21.7] - 2024-01-01

### Added
- ZBAR methodology integration
- Phoenix dashboard
- Voice command support
- Session replay functionality

### Changed
- Improved agent communication protocol
- Enhanced risk management algorithms
- Updated Wyckoff phase detection

### Fixed
- Agent synchronization issues
- Memory optimization in data ingestion
- Chart rendering performance

---

For migration guide from v21.x to v22.0, see MIGRATION_GUIDE.md
