# ncOS v22.0 - Zanlink Enhanced Edition
## Complete Documentation

### 📋 Table of Contents
1. [Overview](#overview)
2. [What's New](#whats-new)
3. [Installation](#installation)
4. [Zanlink Integration](#zanlink-integration)
5. [API Reference](#api-reference)
6. [ChatGPT Integration](#chatgpt-integration)
7. [Docker Deployment](#docker-deployment)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

## Overview

ncOS v22.0 is a sophisticated trading system that combines multiple AI agents, pattern recognition, and now features seamless LLM integration through Zanlink. This version introduces pre-processed data packages, unified API endpoints, and ChatGPT-optimized responses.

### Key Features
- **Unified LLM Gateway**: Single endpoint for all AI interactions
- **Zanlink Integration**: Cloud-hosted API for global access
- **Pre-processed Data**: 5x faster responses with intelligent caching
- **ChatGPT Actions**: Direct integration with custom GPTs
- **Docker Support**: Easy deployment with containerization
- **Enhanced Pattern Recognition**: Wyckoff + SMC with ML enhancements

## What's New in v22.0

### Major Enhancements
1. **Zanlink Cloud Integration**
   - Global API access at https://zanlink.com/api/v1
   - Built-in load balancing and redundancy
   - Automatic failover and retry logic

2. **LLM Optimization**
   - Pre-processed data packages
   - Context-aware prompt templates
   - Batch processing support
   - Response caching (5-minute TTL)

3. **Simplified API**
   - From 50+ endpoints to 4 main endpoints
   - Unified response format
   - ChatGPT-optimized JSON structure

4. **Enhanced Performance**
   - 5x faster response times
   - Reduced API calls by 80%
   - Intelligent data compression

## Installation

### Prerequisites
- Python 3.8+
- Docker (optional)
- Zanlink API key

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/ncos-v22
cd ncos-v22

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ZANLINK_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"

# Run the launcher
python ncos_launcher.py
```

### Docker Installation
```bash
# Build the image
docker build -t ncos-zanlink:v22 .

# Run the container
docker run -d \
  --name ncos-zanlink \
  -e ZANLINK_API_KEY="your-api-key" \
  -p 8000:8000 \
  -p 8004:8004 \
  ncos-zanlink:v22
```

## Zanlink Integration

### Authentication
All Zanlink API calls require authentication:
```python
headers = {
    "Authorization": "Bearer YOUR_ZANLINK_API_KEY"
}
```

### Main Endpoints

#### 1. Quick Status
```bash
GET https://zanlink.com/api/v1/quick/status?symbol=XAUUSD
```

Response:
```json
{
  "symbol": "XAUUSD",
  "price": 1948.75,
  "trend": "bullish",
  "one_line_summary": "XAUUSD bullish at 1948.75, watch 1950.50 resistance",
  "action": "Hold longs, add on dips to 1945"
}
```

#### 2. Market Analysis
```bash
POST https://zanlink.com/api/v1/analyze
{
  "type": "market",
  "symbol": "XAUUSD",
  "timeframe": "H1"
}
```

Response:
```json
{
  "summary": "XAUUSD bullish on H1, testing resistance at 1950.50",
  "insights": [
    "Strong buying pressure detected",
    "Key support holding at 1945.30",
    "Volume profile suggests accumulation"
  ],
  "data": {
    "trend": "bullish",
    "strength": 7.5,
    "next_targets": [1955, 1960]
  },
  "recommendations": [
    "Long entries on pullback to 1945-1947",
    "Stop loss below 1940",
    "Take profit at 1955 and 1960"
  ]
}
```

#### 3. Pattern Detection
```bash
GET https://zanlink.com/api/v1/patterns/detect?symbol=XAUUSD
```

Response:
```json
{
  "patterns": [
    {
      "name": "Wyckoff Spring",
      "confidence": 0.85,
      "location": 1945.30,
      "action": "long_entry"
    },
    {
      "name": "Bullish Order Block",
      "confidence": 0.78,
      "location": 1942.00,
      "action": "support_zone"
    }
  ],
  "summary": "3 patterns detected on XAUUSD",
  "bias": "bullish"
}
```

## ChatGPT Integration

### Setting Up Custom Actions

1. Go to your ChatGPT custom GPT configuration
2. Add new action with the OpenAPI schema from `ncos_chatgpt_schema_zanlink.yaml`
3. Set authentication header: `Authorization: Bearer YOUR_ZANLINK_API_KEY`

### Example Prompts
- "What's the current status of Gold?"
- "Analyze EURUSD on the 4-hour timeframe"
- "What patterns are forming on XAUUSD?"
- "Should I go long on Gold right now?"

### Response Format
ChatGPT will receive pre-formatted responses optimized for conversation:
```
**Summary**: [One-line market overview]

**Key Insights**:
• [Top 3 insights]

**Recommendations**:
1. [Actionable steps]

💡 *Suggested follow-up: [Next logical question]*
```

## Configuration

### Main Configuration File
`config/ncos_config_zanlink.json`:
```json
{
  "api": {
    "base_url": "https://zanlink.com/api/v1",
    "endpoints": {
      "journal": "https://zanlink.com/api/v1/journal",
      "market": "https://zanlink.com/api/v1/market",
      "patterns": "https://zanlink.com/api/v1/patterns"
    }
  },
  "cache": {
    "ttl_minutes": 5,
    "max_size": 100
  }
}
```

### Environment Variables
```bash
# Required
ZANLINK_API_KEY=your-api-key
OPENAI_API_KEY=your-openai-key

# Optional
NCOS_LOG_LEVEL=INFO
NCOS_CACHE_TTL=300
NCOS_MAX_RETRIES=3
```

## API Reference

### Python SDK
```python
from ncos_zanlink_bridge import ZanlinkIntegrationBridge

# Initialize
bridge = ZanlinkIntegrationBridge(api_key="your-key")

# Quick market check
status = await bridge.get_quick_status("XAUUSD")
print(status["one_line_summary"])

# Full analysis
analysis = await bridge.analyze_market(
    symbol="XAUUSD",
    timeframe="H1",
    analysis_type="market"
)

# Pattern detection
patterns = await bridge.detect_patterns("XAUUSD")
```

### REST API Examples
```python
import httpx

# Quick status
response = httpx.get(
    "https://zanlink.com/api/v1/quick/status",
    params={"symbol": "XAUUSD"},
    headers={"Authorization": "Bearer YOUR_KEY"}
)

# Analysis
response = httpx.post(
    "https://zanlink.com/api/v1/analyze",
    json={
        "type": "market",
        "symbol": "XAUUSD",
        "timeframe": "H1"
    },
    headers={"Authorization": "Bearer YOUR_KEY"}
)
```

## Troubleshooting

### Common Issues

1. **Authentication Error**
   - Check your ZANLINK_API_KEY
   - Ensure Bearer prefix in Authorization header

2. **Timeout Errors**
   - Default timeout is 30 seconds
   - For heavy analysis, use async endpoints

3. **Cache Issues**
   - Clear cache: `DELETE /api/v1/cache/clear`
   - Adjust TTL in configuration

4. **Pattern Detection Empty**
   - Ensure sufficient market data
   - Check symbol format (e.g., XAUUSD not XAU/USD)

### Debug Mode
Enable debug logging:
```bash
export NCOS_LOG_LEVEL=DEBUG
python ncos_launcher.py
```

### Health Check
```bash
curl https://zanlink.com/api/v1/health
```

## Support

- Documentation: https://docs.zanlink.com/ncos
- API Status: https://status.zanlink.com
- Support Email: support@zanlink.com
- Discord: https://discord.gg/zanlink

---

© 2024 ncOS - Zanlink Enhanced Edition
