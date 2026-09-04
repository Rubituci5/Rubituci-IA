"""
Entity API Configuration

Centralized configuration using Pydantic Settings.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # APPLICATION
    # =========================================================================
    APP_NAME: str = "Entity"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # =========================================================================
    # DATABASE
    # =========================================================================
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "entity"
    POSTGRES_USER: str = "entity"
    POSTGRES_PASSWORD: str = "changeme_secure_password"
    DATABASE_URL: str = "postgresql+asyncpg://entity:changeme_secure_password@localhost:5432/entity"

    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600

    # =========================================================================
    # REDIS
    # =========================================================================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0
    REDIS_URL: str = "redis://localhost:6379/0"

    # =========================================================================
    # SECURITY / AUTH
    # =========================================================================
    SECRET_KEY: str = "generate-with-openssl-rand-hex-32"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    BCRYPT_ROUNDS: int = 12

    # =========================================================================
    # MODEL
    # =========================================================================
    MODEL_NAME: str = "entity-gen-000001"
    MODEL_PATH: str = "./snapshots/generation_000001/rubituci_conversational_v2/model_weights_rubituci_conversational_v2.pt"
    TOKENIZER_PATH: str = "./snapshots/generation_000001/tokenizer"
    VOCAB_SIZE: int = 8192
    D_MODEL: int = 256
    N_LAYERS: int = 6
    N_HEADS: int = 8
    D_FF: int = 1024
    MAX_SEQ_LEN: int = 512
    DROPOUT: float = 0.1

    # =========================================================================
    # INFERENCE
    # =========================================================================
    INFERENCE_DEVICE: str = "cpu"
    INFERENCE_BATCH_SIZE: int = 1
    INFERENCE_MAX_NEW_TOKENS: int = 256
    INFERENCE_TEMPERATURE: float = 0.8
    INFERENCE_TOP_K: int = 50
    INFERENCE_TOP_P: float = 0.9
    INFERENCE_REPETITION_PENALTY: float = 1.1

    # =========================================================================
    # TRAINING
    # =========================================================================
    TRAINING_DEVICE: str = "cpu"
    TRAINING_BATCH_SIZE: int = 4
    TRAINING_LEARNING_RATE: float = 3e-4
    TRAINING_WEIGHT_DECAY: float = 0.01
    TRAINING_WARMUP_STEPS: int = 100
    TRAINING_MAX_STEPS: int = 10000
    TRAINING_GRAD_ACCUM_STEPS: int = 4
    TRAINING_EVAL_EVERY: int = 500
    TRAINING_SAVE_EVERY: int = 1000
    TRAINING_SEED: int = 42
    SNAPSHOT_ROOT: str = "./snapshots"
    DATASET_ROOT: str = "./data"
    CHECKPOINT_ROOT: str = "./checkpoints"

    # Background jobs
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # =========================================================================
    # MEMORY
    # =========================================================================
    MEMORY_EMBEDDING_DIM: int = 384
    MEMORY_RETRIEVAL_TOP_K: int = 10
    MEMORY_IMPORTANCE_THRESHOLD: float = 0.5
    MEMORY_CONSOLIDATION_BATCH_SIZE: int = 100
    MEMORY_EPISODIC_TTL_DAYS: int = 365

    # =========================================================================
    # WEB SEARCH / RESEARCH
    # =========================================================================
    WEB_SEARCH_ENABLED: bool = True
    WEB_SEARCH_PROVIDER: str = "duckduckgo"
    WEB_SEARCH_MAX_RESULTS: int = 5
    WEB_SEARCH_TIMEOUT_SECONDS: int = 10
    WEB_SEARCH_USER_AGENT: str = "Entity/0.1.0 (+https://entity.example.com)"
    WEB_SEARCH_RATE_LIMIT_PER_MINUTE: int = 30

    # =========================================================================
    # REFLECTION
    # =========================================================================
    REFLECTION_ENABLED: bool = True
    REFLECTION_INTERVAL_SECONDS: int = 300
    REFLECTION_MAX_MEMORIES: int = 50
    REFLECTION_MIN_IMPORTANCE: float = 0.3

    # =========================================================================
    # CONSOLIDATION
    # =========================================================================
    CONSOLIDATION_ENABLED: bool = True
    CONSOLIDATION_SCHEDULE_CRON: str = "0 3 * * *"
    CONSOLIDATION_MIN_EXPERIENCES: int = 100
    CONSOLIDATION_MAX_EXPERIENCES: int = 10000
    SLEEP_MAX_RESEARCH_QUERIES: int = 3

    # =========================================================================
    # COMMUNITY / FEEDBACK
    # =========================================================================
    FEEDBACK_ENABLED: bool = True
    FEEDBACK_MIN_REPUTATION: float = 0.0
    FEEDBACK_WEIGHT_CORRECTION: float = 1.0
    FEEDBACK_WEIGHT_USEFUL: float = 0.5
    FEEDBACK_WEIGHT_ISSUE: float = 0.3

    # =========================================================================
    # OBSERVABILITY
    # =========================================================================
    OTEL_EXPORTER_PROMETHEUS_PORT: int = 9464
    PROMETHEUS_METRICS_PORT: int = 9090
    GRAFANA_PORT: int = 3001

    # =========================================================================
    # FRONTEND
    # =========================================================================
    NEXT_PUBLIC_API_URL: str = "http://localhost:8000"
    NEXT_PUBLIC_WS_URL: str = "ws://localhost:8000"
    NEXT_PUBLIC_APP_NAME: str = "Entity"
    FRONTEND_URL: str = "http://localhost:3000"

    # Google OAuth (server-side only; never expose GOOGLE_CLIENT_SECRET to web)
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/auth/google/callback"

    # =========================================================================
    # SAFETY / CONTAINMENT
    # =========================================================================
    SAFETY_KILL_SWITCH_ENABLED: bool = True
    SAFETY_RATE_LIMIT_REQUESTS: int = 100
    SAFETY_RATE_LIMIT_WINDOW: int = 60
    SAFETY_MAX_CONVERSATION_LENGTH: int = 100
    SAFETY_BLOCKED_DOMAINS: str = ""

    # =========================================================================
    # STORAGE (S3-compatible)
    # =========================================================================
    S3_ENDPOINT_URL: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_BUCKET: str = "entity-artifacts"
    S3_REGION: str = "us-east-1"

    # =========================================================================
    # CORS
    # =========================================================================
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:3001"])
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = Field(default_factory=lambda: ["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default_factory=lambda: ["*"])

    # =========================================================================
    # EXPERIMENTAL ECONOMIC CAPABILITIES
    # =========================================================================
    ECONOMIC_ENABLED: bool = False
    ECONOMIC_MAX_BUDGET: float = 0.0
    ECONOMIC_ALLOWED_CATEGORIES: List[str] = Field(default_factory=list)
    ECONOMIC_APPROVAL_REQUIRED_THRESHOLD: float = 100.0


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
