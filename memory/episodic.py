"""
Episodic Memory Service

Stores and retrieves specific events and experiences.
Each memory is an episode with context, participants, and provenance.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from pgvector.sqlalchemy import Vector

from api.models import EpisodicMemory, MemorySource, User, Conversation, Message
from api.config import settings


@dataclass
class EpisodicMemoryData:
    """Data for creating an episodic memory."""
    event_id: str
    timestamp: datetime
    participants: List[str]
    content: str
    source: MemorySource
    context: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    confidence: float = 0.5
    associations: List[uuid.UUID] = field(default_factory=list)
    generation: int = 1
    embedding: Optional[List[float]] = None


class EpisodicMemoryService:
    """
    Service for managing episodic memories.

    Features:
    - Store experiences with full provenance
    - Retrieve by similarity (vector search)
    - Retrieve by importance, recency, relevance
    - Link to semantic memories and beliefs
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, memory_data: EpisodicMemoryData) -> EpisodicMemory:
        """Create a new episodic memory."""
        memory = EpisodicMemory(
            event_id=memory_data.event_id,
            timestamp=memory_data.timestamp,
            participants=memory_data.participants,
            content=memory_data.content,
            source=memory_data.source,
            context=memory_data.context,
            importance=memory_data.importance,
            confidence=memory_data.confidence,
            associations=memory_data.associations,
            generation=memory_data.generation,
            embedding=memory_data.embedding,
        )
        self.db.add(memory)
        await self.db.flush()
        return memory

    async def create_from_interaction(
        self,
        conversation: Conversation,
        message: Message,
        user: Optional[User] = None,
        importance: float = 0.5,
    ) -> EpisodicMemory:
        """Create episodic memory from a conversation interaction."""
        participants = ["user"]
        if user:
            participants.append(f"user:{user.username}")

        event_id = f"interaction_{conversation.id}_{message.id}"

        memory = await self.create(EpisodicMemoryData(
            event_id=event_id,
            timestamp=message.created_at,
            participants=participants,
            content=f"User said: {message.content}" if message.role.value == "user" else f"Entity responded: {message.content}",
            source=MemorySource.HUMAN_CLAIM if message.role.value == "user" else MemorySource.INTERNAL_INFERENCE,
            context={
                "conversation_id": str(conversation.id),
                "message_id": str(message.id),
                "generation": conversation.generation,
            },
            importance=importance,
            confidence=0.8 if message.role.value == "user" else 0.6,
            generation=conversation.generation,
        ))
        return memory

    async def create_from_web_research(
        self,
        query: str,
        source_content: str,
        source_url: str,
        generation: int,
        importance: float = 0.6,
    ) -> EpisodicMemory:
        """Create episodic memory from web research."""
        event_id = f"web_{uuid.uuid4().hex[:12]}"

        memory = await self.create(EpisodicMemoryData(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            participants=["web"],
            content=f"Web research for '{query}': {source_content[:500]}",
            source=MemorySource.WEB_SOURCE,
            context={
                "query": query,
                "url": source_url,
                "generation": generation,
            },
            importance=importance,
            confidence=0.5,  # Web content is uncertain by default
            generation=generation,
        ))
        return memory

    async def create_from_reflection(
        self,
        trigger_type: str,
        content: str,
        input_memory_ids: List[uuid.UUID],
        generation: int,
        importance: float = 0.7,
    ) -> EpisodicMemory:
        """Create episodic memory from autonomous reflection."""
        event_id = f"reflection_{uuid.uuid4().hex[:12]}"

        memory = await self.create(EpisodicMemoryData(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            participants=["self"],
            content=f"Reflection ({trigger_type}): {content}",
            source=MemorySource.INTERNAL_INFERENCE,
            context={
                "trigger_type": trigger_type,
                "input_memories": [str(mid) for mid in input_memory_ids],
                "generation": generation,
            },
            importance=importance,
            confidence=0.7,
            associations=input_memory_ids,
            generation=generation,
        ))
        return memory

    async def get_by_id(self, memory_id: uuid.UUID) -> Optional[EpisodicMemory]:
        """Get memory by ID."""
        result = await self.db.execute(
            select(EpisodicMemory).where(EpisodicMemory.id == memory_id)
        )
        return result.scalar_one_or_none()

    async def get_by_event_id(self, event_id: str) -> Optional[EpisodicMemory]:
        """Get memory by event ID."""
        result = await self.db.execute(
            select(EpisodicMemory).where(EpisodicMemory.event_id == event_id)
        )
        return result.scalar_one_or_none()

    async def retrieve_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_importance: float = 0.0,
        generation: Optional[int] = None,
        source: Optional[MemorySource] = None,
    ) -> List[EpisodicMemory]:
        """Retrieve memories similar to query embedding (vector search)."""
        stmt = select(EpisodicMemory).where(
            EpisodicMemory.embedding.is_not(None),
            EpisodicMemory.importance >= min_importance,
        )

        if generation:
            stmt = stmt.where(EpisodicMemory.generation == generation)
        if source:
            stmt = stmt.where(EpisodicMemory.source == source)

        stmt = stmt.order_by(EpisodicMemory.embedding.cosine_distance(query_embedding)).limit(top_k)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def retrieve_recent(
        self,
        limit: int = 50,
        min_importance: float = 0.0,
        generation: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> List[EpisodicMemory]:
        """Retrieve most recent memories."""
        stmt = select(EpisodicMemory).where(
            EpisodicMemory.importance >= min_importance,
        )

        if generation:
            stmt = stmt.where(EpisodicMemory.generation == generation)
        if since:
            stmt = stmt.where(EpisodicMemory.timestamp >= since)

        stmt = stmt.order_by(desc(EpisodicMemory.timestamp)).limit(limit)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def retrieve_important(
        self,
        limit: int = 50,
        min_importance: float = 0.7,
        generation: Optional[int] = None,
    ) -> List[EpisodicMemory]:
        """Retrieve most important memories."""
        stmt = select(EpisodicMemory).where(
            EpisodicMemory.importance >= min_importance,
        )

        if generation:
            stmt = stmt.where(EpisodicMemory.generation == generation)

        stmt = stmt.order_by(desc(EpisodicMemory.importance), desc(EpisodicMemory.timestamp)).limit(limit)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def retrieve_for_context(
        self,
        query: str,
        top_k: int = 10,
        generation: Optional[int] = None,
    ) -> List[EpisodicMemory]:
        """
        Retrieve memories relevant for context building.
        Uses a combination of recency, importance, and keyword matching.
        """
        # Simple keyword-based retrieval for now
        # In production, would use embedding similarity
        keywords = query.lower().split()

        stmt = select(EpisodicMemory).where(
            EpisodicMemory.importance >= 0.3,
        )

        if generation:
            stmt = stmt.where(EpisodicMemory.generation == generation)

        # Add keyword filters (simple text search)
        for kw in keywords[:5]:  # Limit to first 5 keywords
            stmt = stmt.where(EpisodicMemory.content.ilike(f"%{kw}%"))

        stmt = stmt.order_by(
            desc(EpisodicMemory.importance),
            desc(EpisodicMemory.timestamp),
        ).limit(top_k)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def retrieve_by_participant(
        self,
        participant: str,
        limit: int = 50,
        generation: Optional[int] = None,
    ) -> List[EpisodicMemory]:
        """Retrieve memories involving a specific participant."""
        stmt = select(EpisodicMemory).where(
            EpisodicMemory.participants.contains([participant]),
        )

        if generation:
            stmt = stmt.where(EpisodicMemory.generation == generation)

        stmt = stmt.order_by(desc(EpisodicMemory.timestamp)).limit(limit)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_importance(self, memory_id: uuid.UUID, importance: float) -> bool:
        """Update memory importance score."""
        memory = await self.get_by_id(memory_id)
        if memory:
            memory.importance = max(0.0, min(1.0, importance))
            memory.updated_at = datetime.now(timezone.utc)
            return True
        return False

    async def update_confidence(self, memory_id: uuid.UUID, confidence: float) -> bool:
        """Update memory confidence score."""
        memory = await self.get_by_id(memory_id)
        if memory:
            memory.confidence = max(0.0, min(1.0, confidence))
            memory.updated_at = datetime.now(timezone.utc)
            return True
        return False

    async def add_association(self, memory_id: uuid.UUID, associated_id: uuid.UUID) -> bool:
        """Add an association to another memory."""
        memory = await self.get_by_id(memory_id)
        if memory and associated_id not in memory.associations:
            memory.associations.append(associated_id)
            memory.updated_at = datetime.now(timezone.utc)
            return True
        return False

    async def mark_consolidated(self, memory_id: uuid.UUID) -> bool:
        """Mark memory as consolidated into semantic memory."""
        memory = await self.get_by_id(memory_id)
        if memory:
            memory.consolidated = True
            memory.consolidated_at = datetime.now(timezone.utc)
            return True
        return False

    async def get_stats(self, generation: Optional[int] = None) -> Dict[str, Any]:
        """Get memory statistics."""
        stmt = select(
            func.count(EpisodicMemory.id),
            func.avg(EpisodicMemory.importance),
            func.avg(EpisodicMemory.confidence),
            func.sum(func.case((EpisodicMemory.consolidated == True, 1), else_=0)),
        )

        if generation:
            stmt = stmt.where(EpisodicMemory.generation == generation)

        result = await self.db.execute(stmt)
        total, avg_importance, avg_confidence, consolidated = result.one()

        # By source
        stmt = select(EpisodicMemory.source, func.count(EpisodicMemory.id)).group_by(EpisodicMemory.source)
        if generation:
            stmt = stmt.where(EpisodicMemory.generation == generation)
        result = await self.db.execute(stmt)
        by_source = {row.source.value: row[1] for row in result.all()}

        return {
            "total_memories": total or 0,
            "avg_importance": float(avg_importance or 0),
            "avg_confidence": float(avg_confidence or 0),
            "consolidated_count": consolidated or 0,
            "by_source": by_source,
        }

    async def cleanup_old(self, ttl_days: int = 365, dry_run: bool = True) -> int:
        """Remove memories older than TTL (low importance only)."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)

        stmt = select(EpisodicMemory).where(
            EpisodicMemory.timestamp < cutoff,
            EpisodicMemory.importance < 0.3,
            EpisodicMemory.consolidated == False,
        )

        result = await self.db.execute(stmt)
        memories = result.scalars().all()

        if not dry_run:
            for mem in memories:
                await self.db.delete(mem)

        return len(memories)
