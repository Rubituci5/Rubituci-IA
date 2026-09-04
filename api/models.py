"""
Entity Database Models

SQLAlchemy models for the Entity project.
Organized by schema: auth, memory, evolution, community, safety.
"""

import enum
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean,
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
    Enum as SQLEnum, JSON, ARRAY, Float, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from sqlalchemy.orm import relationship, declarative_base, declared_attr
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class EntityUUID:
    """Mixin for UUID primary key."""
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )


class TimestampMixin:
    """Mixin for created_at/updated_at timestamps."""
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# =============================================================================
# ENUMS
# =============================================================================

class UserRole(str, enum.Enum):
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    RESEARCHER = "researcher"


class MessageRole(str, enum.Enum):
    USER = "user"
    ENTITY = "entity"
    SYSTEM = "system"


class FeedbackType(str, enum.Enum):
    USEFUL = "useful"
    INCORRECT = "incorrect"
    QUESTIONABLE = "questionable"
    GOOD_INTERPRETATION = "good_interpretation"
    FACTUAL_CORRECTION = "factual_correction"
    SAFETY_ISSUE = "safety_issue"
    OTHER = "other"


class MemorySource(str, enum.Enum):
    HUMAN_CLAIM = "human_claim"
    WEB_SOURCE = "web_source"
    DIRECT_OBSERVATION = "direct_observation"
    INTERNAL_INFERENCE = "internal_inference"
    PREVIOUS_BELIEF = "previous_belief"
    CONSOLIDATED_KNOWLEDGE = "consolidated_knowledge"


class ConfidenceLevel(str, enum.Enum):
    KNOWN = "known"
    LIKELY = "likely"
    UNCERTAIN = "uncertain"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class OperationalState(str, enum.Enum):
    INTERACTION = "interaction"
    REFLECTION = "reflection"
    RESEARCH = "research"
    CONSOLIDATION = "consolidation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    IDLE = "idle"


class GenerationStatus(str, enum.Enum):
    CREATED = "created"
    TRAINING = "training"
    EVALUATING = "evaluating"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class EventType(str, enum.Enum):
    GENERATION_CREATED = "generation_created"
    FIRST_INTERACTION = "first_interaction"
    FIRST_AUTONOMOUS_QUERY = "first_autonomous_query"
    CONSOLIDATION_CYCLE = "consolidation_cycle"
    GENERATION_PROMOTED = "generation_promoted"
    ARCHITECTURE_MODIFIED = "architecture_modified"
    SAFETY_EVENT = "safety_event"


# =============================================================================
# AUTH SCHEMA
# =============================================================================

class User(EntityUUID, TimestampMixin, Base):
    """User accounts for community interaction."""
    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_email", "email", unique=True),
        Index("ix_users_username", "username", unique=True),
        Index("ix_users_role", "role"),
        {"schema": "auth"},
    )

    email = Column(String(255), nullable=False, unique=True)
    username = Column(String(100), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    reputation_score = Column(Float, default=0.0, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    bio = Column(Text, nullable=True)
    preferences = Column(JSONB, default=dict, nullable=False)


class Session(EntityUUID, TimestampMixin, Base):
    """User sessions for authentication."""
    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_user_id", "user_id"),
        Index("ix_sessions_expires", "expires_at"),
        {"schema": "auth"},
    )

    user_id = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    is_revoked = Column(Boolean, default=False, nullable=False)

    user = relationship("User", backref="sessions")


class APIKey(EntityUUID, TimestampMixin, Base):
    """API keys for programmatic access."""
    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_key_hash", "key_hash", unique=True),
        Index("ix_api_keys_user_id", "user_id"),
        {"schema": "auth"},
    )

    user_id = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    key_prefix = Column(String(20), nullable=False)  # First 8 chars for identification
    scopes = Column(ARRAY(String), default=list, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    last_used = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    user = relationship("User", backref="api_keys")


# =============================================================================
# CONVERSATION SCHEMA
# =============================================================================

class Conversation(EntityUUID, TimestampMixin, Base):
    """Conversations between users and the entity."""
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_user_id", "user_id"),
        Index("ix_conversations_generation", "generation"),
        Index("ix_conversations_state", "state"),
        {"schema": "entity"},
    )

    user_id = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="SET NULL"), nullable=True)
    generation = Column(Integer, nullable=False, default=1)
    title = Column(String(500), nullable=True)
    state = Column(SQLEnum(OperationalState), default=OperationalState.INTERACTION, nullable=False)
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    metadata_ = Column("metadata", JSONB, default=dict, nullable=False)

    user = relationship("User", backref="conversations")
    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")


class Message(EntityUUID, TimestampMixin, Base):
    """Individual messages in a conversation."""
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        Index("ix_messages_role", "role"),
        Index("ix_messages_generation", "generation"),
        {"schema": "entity"},
    )

    conversation_id = Column(UUID(as_uuid=True), ForeignKey("entity.conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    generation = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=True)
    inference_time_ms = Column(Integer, nullable=True)
    memories_retrieved = Column(ARRAY(UUID(as_uuid=True)), default=list, nullable=False)
    web_sources_used = Column(ARRAY(UUID(as_uuid=True)), default=list, nullable=False)
    metadata_ = Column("metadata", JSONB, default=dict, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")
    feedback = relationship("Feedback", back_populates="message", cascade="all, delete-orphan")


# =============================================================================
# MEMORY SCHEMA
# =============================================================================

class EpisodicMemory(EntityUUID, TimestampMixin, Base):
    """Episodic memories - specific events and experiences."""
    __tablename__ = "episodic_memories"
    __table_args__ = (
        Index("ix_episodic_memories_source", "source"),
        Index("ix_episodic_memories_importance", "importance"),
        Index("ix_episodic_memories_generation", "generation"),
        Index("ix_episodic_memories_confidence", "confidence"),
        Index("ix_episodic_memories_timestamp", "timestamp"),
        {"schema": "memory"},
    )

    event_id = Column(String(100), nullable=False, unique=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    participants = Column(ARRAY(String), default=list, nullable=False)
    content = Column(Text, nullable=False)
    source = Column(SQLEnum(MemorySource), nullable=False)
    context = Column(JSONB, default=dict, nullable=False)
    importance = Column(Float, default=0.5, nullable=False)
    confidence = Column(Float, default=0.5, nullable=False)
    associations = Column(ARRAY(UUID(as_uuid=True)), default=list, nullable=False)
    generation = Column(Integer, nullable=False)
    embedding = Column(Vector(384), nullable=True)  # For semantic retrieval
    consolidated = Column(Boolean, default=False, nullable=False)
    consolidated_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    semantic_links = relationship("SemanticMemoryLink", back_populates="episodic_memory", cascade="all, delete-orphan")
    belief_evidence = relationship("BeliefEvidence", back_populates="episodic_memory", cascade="all, delete-orphan")


class SemanticMemory(EntityUUID, TimestampMixin, Base):
    """Semantic memories - consolidated knowledge patterns."""
    __tablename__ = "semantic_memories"
    __table_args__ = (
        Index("ix_semantic_memories_category", "category"),
        Index("ix_semantic_memories_confidence", "confidence"),
        Index("ix_semantic_memories_generation", "generation"),
        {"schema": "memory"},
    )

    concept = Column(String(500), nullable=False)
    category = Column(String(100), nullable=True)
    representation = Column(Text, nullable=False)  # Natural language description
    confidence = Column(Float, default=0.5, nullable=False)
    evidence_count = Column(Integer, default=0, nullable=False)
    generation = Column(Integer, nullable=False)
    last_reinforced = Column(DateTime(timezone=True), nullable=True)
    embedding = Column(Vector(384), nullable=True)

    # Relationships
    episodic_links = relationship("SemanticMemoryLink", back_populates="semantic_memory", cascade="all, delete-orphan")


class SemanticMemoryLink(EntityUUID, TimestampMixin, Base):
    """Links between semantic and episodic memories."""
    __tablename__ = "semantic_memory_links"
    __table_args__ = (
        Index("ix_semantic_links_semantic_id", "semantic_memory_id"),
        Index("ix_semantic_links_episodic_id", "episodic_memory_id"),
        UniqueConstraint("semantic_memory_id", "episodic_memory_id", name="uq_semantic_episodic_link"),
        {"schema": "memory"},
    )

    semantic_memory_id = Column(UUID(as_uuid=True), ForeignKey("memory.semantic_memories.id", ondelete="CASCADE"), nullable=False)
    episodic_memory_id = Column(UUID(as_uuid=True), ForeignKey("memory.episodic_memories.id", ondelete="CASCADE"), nullable=False)
    strength = Column(Float, default=1.0, nullable=False)
    link_type = Column(String(50), default="supports", nullable=False)

    semantic_memory = relationship("SemanticMemory", back_populates="episodic_links")
    episodic_memory = relationship("EpisodicMemory", back_populates="semantic_links")


# =============================================================================
# BELIEF SCHEMA
# =============================================================================

class Belief(EntityUUID, TimestampMixin, Base):
    """Beliefs held by the entity with confidence levels."""
    __tablename__ = "beliefs"
    __table_args__ = (
        Index("ix_beliefs_category", "category"),
        Index("ix_beliefs_confidence", "confidence"),
        Index("ix_beliefs_generation", "generation"),
        Index("ix_beliefs_status", "status"),
        {"schema": "memory"},
    )

    proposition = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)
    confidence = Column(Float, default=0.5, nullable=False)
    confidence_level = Column(SQLEnum(ConfidenceLevel), default=ConfidenceLevel.UNCERTAIN, nullable=False)
    status = Column(String(50), default="active", nullable=False)  # active, deprecated, contradicted
    generation = Column(Integer, nullable=False)
    formed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    last_updated = Column(DateTime(timezone=True), nullable=True)
    contradiction_count = Column(Integer, default=0, nullable=False)

    # Relationships
    evidence = relationship("BeliefEvidence", back_populates="belief", cascade="all, delete-orphan")


class BeliefEvidence(EntityUUID, TimestampMixin, Base):
    """Evidence supporting or contradicting a belief."""
    __tablename__ = "belief_evidence"
    __table_args__ = (
        Index("ix_belief_evidence_belief_id", "belief_id"),
        Index("ix_belief_evidence_episodic_id", "episodic_memory_id"),
        Index("ix_belief_evidence_source", "source"),
        {"schema": "memory"},
    )

    belief_id = Column(UUID(as_uuid=True), ForeignKey("memory.beliefs.id", ondelete="CASCADE"), nullable=False)
    episodic_memory_id = Column(UUID(as_uuid=True), ForeignKey("memory.episodic_memories.id", ondelete="CASCADE"), nullable=True)
    source = Column(SQLEnum(MemorySource), nullable=False)
    content = Column(Text, nullable=False)
    supports = Column(Boolean, nullable=False)  # True = supports, False = contradicts
    weight = Column(Float, default=1.0, nullable=False)
    metadata_ = Column("metadata", JSONB, default=dict, nullable=False)

    belief = relationship("Belief", back_populates="evidence")
    episodic_memory = relationship("EpisodicMemory", back_populates="belief_evidence")


# =============================================================================
# RESEARCH / WEB SEARCH SCHEMA
# =============================================================================

class WebQuery(EntityUUID, TimestampMixin, Base):
    """Web search queries initiated by the entity."""
    __tablename__ = "web_queries"
    __table_args__ = (
        Index("ix_web_queries_generation", "generation"),
        Index("ix_web_queries_trigger", "trigger_type"),
        Index("ix_web_queries_timestamp", "created_at"),
        {"schema": "memory"},
    )

    query = Column(Text, nullable=False)
    trigger_type = Column(String(50), nullable=False)  # autonomous, user_requested, reflection
    generation = Column(Integer, nullable=False)
    results_count = Column(Integer, default=0, nullable=False)
    search_time_ms = Column(Integer, nullable=True)
    metadata_ = Column("metadata", JSONB, default=dict, nullable=False)

    sources = relationship("WebSource", back_populates="query", cascade="all, delete-orphan")


class WebSource(EntityUUID, TimestampMixin, Base):
    """Web sources retrieved and their provenance."""
    __tablename__ = "web_sources"
    __table_args__ = (
        Index("ix_web_sources_query_id", "query_id"),
        Index("ix_web_sources_url", "url"),
        Index("ix_web_sources_domain", "domain"),
        Index("ix_web_sources_credibility", "credibility"),
        {"schema": "memory"},
    )

    query_id = Column(UUID(as_uuid=True), ForeignKey("memory.web_queries.id", ondelete="CASCADE"), nullable=False)
    url = Column(String(2048), nullable=False)
    domain = Column(String(255), nullable=True)
    title = Column(String(500), nullable=True)
    content = Column(Text, nullable=True)
    snippet = Column(Text, nullable=True)
    credibility = Column(Float, default=0.5, nullable=False)
    source_type = Column(String(50), nullable=True)  # news, wiki, blog, academic, etc.
    language = Column(String(10), default="en", nullable=False)
    fetched_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    content_hash = Column(String(64), nullable=True)  # For deduplication
    metadata_ = Column("metadata", JSONB, default=dict, nullable=False)

    query = relationship("WebQuery", back_populates="sources")
    citations = relationship("SourceCitation", back_populates="source", cascade="all, delete-orphan")


class SourceCitation(EntityUUID, TimestampMixin, Base):
    """Citations linking web sources to memories/beliefs."""
    __tablename__ = "source_citations"
    __table_args__ = (
        Index("ix_citations_source_id", "source_id"),
        Index("ix_citations_episodic_id", "episodic_memory_id"),
        Index("ix_citations_belief_id", "belief_id"),
        {"schema": "memory"},
    )

    source_id = Column(UUID(as_uuid=True), ForeignKey("memory.web_sources.id", ondelete="CASCADE"), nullable=False)
    episodic_memory_id = Column(UUID(as_uuid=True), ForeignKey("memory.episodic_memories.id", ondelete="CASCADE"), nullable=True)
    belief_id = Column(UUID(as_uuid=True), ForeignKey("memory.beliefs.id", ondelete="CASCADE"), nullable=True)
    context = Column(Text, nullable=True)
    relevance = Column(Float, default=1.0, nullable=False)

    source = relationship("WebSource", back_populates="citations")
    episodic_memory = relationship("EpisodicMemory", backref="citations")
    belief = relationship("Belief", backref="citations")


# =============================================================================
# REFLECTION SCHEMA
# =============================================================================

class Reflection(EntityUUID, TimestampMixin, Base):
    """Autonomous reflection cycles."""
    __tablename__ = "reflections"
    __table_args__ = (
        Index("ix_reflections_generation", "generation"),
        Index("ix_reflections_trigger", "trigger_type"),
        {"schema": "memory"},
    )

    trigger_type = Column(String(50), nullable=False)  # scheduled, contradiction, novelty, recurrence
    generation = Column(Integer, nullable=False)
    input_memory_ids = Column(ARRAY(UUID(as_uuid=True)), default=list, nullable=False)
    output = Column(Text, nullable=True)
    contradictions_found = Column(Integer, default=0, nullable=False)
    inferences_made = Column(Integer, default=0, nullable=False)
    new_memories_created = Column(Integer, default=0, nullable=False)
    duration_ms = Column(Integer, nullable=True)
    metadata_ = Column("metadata", JSONB, default=dict, nullable=False)


# =============================================================================
# CONSOLIDATION SCHEMA
# =============================================================================

class ConsolidationRun(EntityUUID, TimestampMixin, Base):
    """Consolidation (sleep) cycles."""
    __tablename__ = "consolidation_runs"
    __table_args__ = (
        Index("ix_consolidation_generation", "generation"),
        Index("ix_consolidation_status", "status"),
        {"schema": "evolution"},
    )

    generation = Column(Integer, nullable=False)
    status = Column(String(50), default="pending", nullable=False)  # pending, running, completed, failed
    experiences_processed = Column(Integer, default=0, nullable=False)
    semantic_memories_created = Column(Integer, default=0, nullable=False)
    semantic_memories_updated = Column(Integer, default=0, nullable=False)
    beliefs_formed = Column(Integer, default=0, nullable=False)
    beliefs_updated = Column(Integer, default=0, nullable=False)
    dataset_size = Column(Integer, default=0, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSONB, default=dict, nullable=False)


# =============================================================================
# TRAINING / EVOLUTION SCHEMA
# =============================================================================

class TrainingRun(EntityUUID, TimestampMixin, Base):
    """Model training runs."""
    __tablename__ = "training_runs"
    __table_args__ = (
        Index("ix_training_generation", "generation"),
        Index("ix_training_status", "status"),
        {"schema": "evolution"},
    )

    generation = Column(Integer, nullable=False)
    parent_generation = Column(Integer, nullable=True)
    status = Column(String(50), default="pending", nullable=False)
    config = Column(JSONB, default=dict, nullable=False)
    dataset_hash = Column(String(64), nullable=True)
    train_loss = Column(Float, nullable=True)
    eval_loss = Column(Float, nullable=True)
    perplexity = Column(Float, nullable=True)
    tokens_processed = Column(BigInteger, default=0, nullable=False)
    steps_completed = Column(Integer, default=0, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    checkpoint_path = Column(String(500), nullable=True)
    metadata_ = Column("metadata", JSONB, default=dict, nullable=False)


class ModelGeneration(EntityUUID, TimestampMixin, Base):
    """Model generations with full lineage tracking."""
    __tablename__ = "model_generations"
    __table_args__ = (
        Index("ix_generations_number", "generation_number", unique=True),
        Index("ix_generations_status", "status"),
        Index("ix_generations_parent", "parent_generation"),
        {"schema": "evolution"},
    )

    generation_number = Column(Integer, nullable=False, unique=True)
    parent_generation = Column(Integer, nullable=True)
    status = Column(SQLEnum(GenerationStatus), default=GenerationStatus.CREATED, nullable=False)
    model_path = Column(String(500), nullable=True)
    tokenizer_path = Column(String(500), nullable=True)
    config = Column(JSONB, default=dict, nullable=False)
    training_run_id = Column(UUID(as_uuid=True), ForeignKey("evolution.training_runs.id", ondelete="SET NULL"), nullable=True)
    eval_metrics = Column(JSONB, default=dict, nullable=False)
    architecture_changes = Column(JSONB, default=list, nullable=False)
    promoted_at = Column(DateTime(timezone=True), nullable=True)
    deprecated_at = Column(DateTime(timezone=True), nullable=True)
    metadata_ = Column("metadata", JSONB, default=dict, nullable=False)

    training_run = relationship("TrainingRun")


class EvolutionEvent(EntityUUID, TimestampMixin, Base):
    """Objective evolution timeline events."""
    __tablename__ = "evolution_events"
    __table_args__ = (
        Index("ix_evolution_events_type", "event_type"),
        Index("ix_evolution_events_generation", "generation"),
        Index("ix_evolution_events_timestamp", "created_at"),
        {"schema": "evolution"},
    )

    event_type = Column(SQLEnum(EventType), nullable=False)
    generation = Column(Integer, nullable=True)
    description = Column(Text, nullable=False)
    details = Column(JSONB, default=dict, nullable=False)
    source = Column(String(50), default="system", nullable=False)  # system, human, autonomous


# =============================================================================
# COMMUNITY / FEEDBACK SCHEMA
# =============================================================================

class Feedback(EntityUUID, TimestampMixin, Base):
    """Community feedback on entity responses."""
    __tablename__ = "feedback"
    __table_args__ = (
        Index("ix_feedback_message_id", "message_id"),
        Index("ix_feedback_user_id", "user_id"),
        Index("ix_feedback_type", "feedback_type"),
        Index("ix_feedback_generation", "generation"),
        {"schema": "community"},
    )

    message_id = Column(UUID(as_uuid=True), ForeignKey("entity.messages.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="CASCADE"), nullable=False)
    feedback_type = Column(SQLEnum(FeedbackType), nullable=False)
    content = Column(Text, nullable=True)  # Explanation/correction
    source_url = Column(String(2048), nullable=True)  # If providing a source
    weight = Column(Float, default=1.0, nullable=False)
    processed = Column(Boolean, default=False, nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    generation = Column(Integer, nullable=False)

    message = relationship("Message", back_populates="feedback")
    user = relationship("User", backref="feedback")


class Contribution(EntityUUID, TimestampMixin, Base):
    """Community contributions (datasets, corrections, experiments)."""
    __tablename__ = "contributions"
    __table_args__ = (
        Index("ix_contributions_user_id", "user_id"),
        Index("ix_contributions_status", "status"),
        Index("ix_contributions_type", "contribution_type"),
        {"schema": "community"},
    )

    user_id = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="CASCADE"), nullable=False)
    contribution_type = Column(String(50), nullable=False)  # dataset, correction, experiment, code
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    content = Column(JSONB, default=dict, nullable=False)  # Structured content
    status = Column(String(50), default="pending", nullable=False)  # pending, accepted, rejected
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="SET NULL"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    generation = Column(Integer, nullable=True)

    user = relationship("User", foreign_keys=[user_id], backref="contributions")
    reviewer = relationship("User", foreign_keys=[reviewed_by], backref="reviews")


# =============================================================================
# SAFETY / AUDIT SCHEMA
# =============================================================================

class SystemEvent(EntityUUID, TimestampMixin, Base):
    """System-level events for audit trail."""
    __tablename__ = "system_events"
    __table_args__ = (
        Index("ix_system_events_type", "event_type"),
        Index("ix_system_events_severity", "severity"),
        Index("ix_system_events_timestamp", "created_at"),
        {"schema": "safety"},
    )

    event_type = Column(String(100), nullable=False)
    severity = Column(String(20), default="info", nullable=False)  # debug, info, warning, error, critical
    component = Column(String(100), nullable=True)
    message = Column(Text, nullable=False)
    details = Column(JSONB, default=dict, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="SET NULL"), nullable=True)
    generation = Column(Integer, nullable=True)


class SafetyViolation(EntityUUID, TimestampMixin, Base):
    """Safety violations and containment events."""
    __tablename__ = "safety_violations"
    __table_args__ = (
        Index("ix_safety_violations_type", "violation_type"),
        Index("ix_safety_violations_severity", "severity"),
        Index("ix_safety_violations_user_id", "user_id"),
        Index("ix_safety_violations_resolved", "resolved"),
        {"schema": "safety"},
    )

    violation_type = Column(String(100), nullable=False)
    severity = Column(String(20), default="warning", nullable=False)  # warning, critical, emergency
    description = Column(Text, nullable=False)
    context = Column(JSONB, default=dict, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="SET NULL"), nullable=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("entity.conversations.id", ondelete="SET NULL"), nullable=True)
    action_taken = Column(String(50), nullable=True)  # blocked, quarantined, terminated, logged
    resolved = Column(Boolean, default=False, nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="SET NULL"), nullable=True)


class AuditLog(EntityUUID, TimestampMixin, Base):
    """Immutable audit log for critical operations."""
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
        Index("ix_audit_logs_timestamp", "created_at"),
        {"schema": "safety"},
    )

    action = Column(String(100), nullable=False)  # create, update, delete, promote, terminate
    resource_type = Column(String(50), nullable=False)  # model, config, user, memory
    resource_id = Column(String(100), nullable=True)
    actor_type = Column(String(20), nullable=False)  # human, system, entity
    actor_id = Column(UUID(as_uuid=True), nullable=True)
    changes = Column(JSONB, default=dict, nullable=False)
    ip_address = Column(String(45), nullable=True)
    signature = Column(String(255), nullable=True)  # Cryptographic signature


# =============================================================================
# REPORTS / CONSENTS
# =============================================================================

class Report(EntityUUID, TimestampMixin, Base):
    """User reports for moderation."""
    __tablename__ = "reports"
    __table_args__ = (
        Index("ix_reports_status", "status"),
        Index("ix_reports_reporter", "reporter_id"),
        Index("ix_reports_target", "target_type", "target_id"),
        {"schema": "community"},
    )

    reporter_id = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="CASCADE"), nullable=False)
    target_type = Column(String(50), nullable=False)  # message, user, contribution
    target_id = Column(UUID(as_uuid=True), nullable=False)
    reason = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="pending", nullable=False)  # pending, reviewed, dismissed, actioned
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="SET NULL"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    action_taken = Column(String(100), nullable=True)


class Consent(EntityUUID, TimestampMixin, Base):
    """Data processing consents."""
    __tablename__ = "consents"
    __table_args__ = (
        Index("ix_consents_user_id", "user_id"),
        Index("ix_consents_type", "consent_type"),
        UniqueConstraint("user_id", "consent_type", name="uq_user_consent"),
        {"schema": "community"},
    )

    user_id = Column(UUID(as_uuid=True), ForeignKey("auth.users.id", ondelete="CASCADE"), nullable=False)
    consent_type = Column(String(50), nullable=False)  # training_data, analytics, research, marketing
    granted = Column(Boolean, default=False, nullable=False)
    version = Column(String(20), default="1.0", nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)

    user = relationship("User", backref="consents")