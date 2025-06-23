"""
Enhanced Predictive Engine Schemas with comprehensive validation
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator, root_validator


class PredictionMode(str, Enum):
    STANDARD = "standard"
    ENHANCED = "enhanced"
    EXPERIMENTAL = "experimental"


class ModelType(str, Enum):
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    CNN = "cnn"
    ENSEMBLE = "ensemble"


class DataSourceConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether this data source is active")
    sampling_rate: Optional[str] = Field(default="1s", regex="^\d+[smh]$")
    anonymized: Optional[bool] = Field(default=True)
    sources: Optional[List[str]] = Field(default_factory=list)

    @validator('sampling_rate')
    def validate_sampling_rate(cls, v):
        if v:
            import re
            if not re.match(r'^\d+[smh]$', v):
                raise ValueError('Sampling rate must be in format: <number><s|m|h>')
        return v


class ModelParameters(BaseModel):
    layers: Optional[int] = Field(default=12, ge=1, le=100)
    hidden_size: Optional[int] = Field(default=768, ge=64, le=4096)
    attention_heads: Optional[int] = Field(default=12, ge=1, le=64)
    units: Optional[int] = Field(default=512, ge=32, le=2048)
    dropout: Optional[float] = Field(default=0.2, ge=0.0, le=0.9)

    @validator('attention_heads')
    def validate_attention_heads(cls, v, values):
        if 'hidden_size' in values and v:
            if values['hidden_size'] % v != 0:
                raise ValueError('hidden_size must be divisible by attention_heads')
        return v


class ModelConfig(BaseModel):
    type: ModelType
    parameters: ModelParameters


class PatternMatchingConfig(BaseModel):
    enabled: bool = Field(default=True)
    min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    max_patterns: int = Field(default=1000, ge=1, le=10000)


class EngineConfig(BaseModel):
    name: str = Field(default="Neural Predictive Engine")
    version: str = Field(regex="^\d+\.\d+$")
    mode: PredictionMode = Field(default=PredictionMode.ENHANCED)
    neural_depth: int = Field(default=5, ge=1, le=10)
    pattern_matching: PatternMatchingConfig


class PredictionParams(BaseModel):
    time_window: str = Field(default="24h", regex="^\d+[smhd]$")
    update_frequency: str = Field(default="5m", regex="^\d+[smh]$")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_predictions: int = Field(default=100, ge=1, le=1000)


class CachingConfig(BaseModel):
    enabled: bool = Field(default=True)
    ttl: int = Field(default=3600, ge=60, le=86400, description="Time to live in seconds")


class BatchProcessingConfig(BaseModel):
    enabled: bool = Field(default=True)
    batch_size: int = Field(default=32, ge=1, le=256)


class OptimizationConfig(BaseModel):
    caching: CachingConfig
    batch_processing: BatchProcessingConfig


class RotationConfig(BaseModel):
    enabled: bool = Field(default=True)
    max_size: str = Field(default="100MB", regex="^\d+[KMG]B$")
    max_files: int = Field(default=5, ge=1, le=20)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    destinations: List[str] = Field(default_factory=lambda: ["file", "monitoring"])
    rotation: RotationConfig


class JournalingConfig(BaseModel):
    enabled: bool = Field(default=True)
    path: str = Field(default="/var/log/ncOS/predictive")
    format: str = Field(default="structured_json", regex="^(plain|json|structured_json)$")
    retention_days: int = Field(default=30, ge=1, le=365)


class DataEnrichmentConfig(BaseModel):
    metadata_extraction: bool = Field(default=True)
    pattern_analysis: bool = Field(default=True)
    anomaly_detection: bool = Field(default=True)


class OutputConfig(BaseModel):
    format: str = Field(default="structured", regex="^(plain|structured|json)$")
    include_confidence: bool = Field(default=True)
    include_reasoning: bool = Field(default=True)


class PredictiveEngineConfig(BaseModel):
    """Complete configuration schema for the Predictive Engine"""
    engine: EngineConfig
    models: Dict[str, ModelConfig]
    prediction_params: PredictionParams
    data_sources: Dict[str, DataSourceConfig]
    output: OutputConfig
    optimization: OptimizationConfig
    logging: LoggingConfig
    journaling: JournalingConfig
    data_enrichment: DataEnrichmentConfig

    @root_validator
    def validate_models(cls, values):
        models = values.get('models', {})
        if not models:
            raise ValueError('At least one model must be configured')
        if 'primary' not in models:
            raise ValueError('A primary model must be configured')
        return values


# Additional validation schemas for runtime data

class PredictionRequest(BaseModel):
    """Schema for prediction requests"""
    input_data: Dict[str, Any]
    prediction_type: str
    priority: Optional[int] = Field(default=5, ge=1, le=10)
    timeout: Optional[int] = Field(default=30, ge=1, le=300)
    callback_url: Optional[str] = None


class PredictionResult(BaseModel):
    """Schema for prediction results"""
    request_id: str
    timestamp: datetime
    predictions: List[Dict[str, Any]]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @validator('predictions')
    def validate_predictions(cls, v):
        if not v:
            raise ValueError('At least one prediction must be provided')
        return v


class SystemMetrics(BaseModel):
    """Schema for system metrics validation"""
    cpu_usage: float = Field(ge=0.0, le=100.0)
    memory_usage: float = Field(ge=0.0, le=100.0)
    prediction_latency: float = Field(ge=0.0, description="Latency in milliseconds")
    queue_size: int = Field(ge=0)
    active_models: int = Field(ge=0)

    @root_validator
    def validate_resource_usage(cls, values):
        cpu = values.get('cpu_usage', 0)
        memory = values.get('memory_usage', 0)
        if cpu > 90 or memory > 90:
            values['high_resource_warning'] = True
        return values
