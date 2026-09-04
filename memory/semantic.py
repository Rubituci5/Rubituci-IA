"""
Semantic Memory Service

Consolidated knowledge patterns extracted from episodic memories.
Represents generalizations, concepts, and learned relationships.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from pgvector.sqlalchemy import Vector

from api.models import SemanticMemory, SemanticMemoryLink, EpisodicMemory
from api.config import settings


@dataclass
class SemanticMemoryData:
    """Data for creating a semantic memory."""
    concept: str
    category: Optional[str]
    representation: str
    confidence: float = 0.5
    generation: int = 1
    embedding: Optional[List[float]] = None
    source_episodic_ids: List[uuid.UUID] = field(default_factory=list)


class SemanticMemoryService:
    """
    Service for managing semantic memories.

    Features:
    - Store consolidated knowledge
    - Link to supporting episodic memories
    - Update confidence based on evidence
    - Retrieve by concept, category, similarity
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, memory_data: SemanticMemoryData) -> SemanticMemory:
        """Create a new semantic memory."""
        memory = SemanticMemory(
            concept=memory_data.concept,
            category=memory_data.category,
            representation=memory_data.representation,
            confidence=memory_data.confidence,
            generation=memory_data.generation,
            embedding=memory_data.embedding,
            evidence_count=len(memory_data.source_episodic_ids),
            last_reinforced=datetime.now(timezone.utc),
        )
        self.db.add(memory)
        await self.db.flush()

        # Create links to episodic memories
        for ep_id in memory_data.source_episodic_ids:
            link = SemanticMemoryLink(
                semantic_memory_id=memory.id,
                episodic_memory_id=ep_id,
                strength=1.0,
                link_type="supports",
            )
            self.db.add(link)

        return memory

    async def get_or_create(
        self,
        concept: str,
        category: Optional[str],
        representation: str,
        generation: int,
        source_episodic_ids: List[uuid.UUID],
    ) -> SemanticMemory:
        """Get existing semantic memory or create new one."""
        # Check for existing similar concept
        result = await self.db.execute(
            select(SemanticMemory).where(
                SemanticMemory.concept.ilike(concept),
                SemanticMemory.generation == generation,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing
            existing.representation = representation
            existing.confidence = min(1.0, existing.confidence + 0.1)
            existing.evidence_count += len(source_episodic_ids)
            existing.last_reinforced = datetime.now(timezone.utc)

            # Add new links
            for ep_id in source_episodic_ids:
                # Check if link exists
                result = await self.db.execute(
                    select(SemanticMemoryLink).where(
                        SemanticMemoryLink.semantic_memory_id == existing.id,
                        SemanticMemoryLink.episodic_memory_id == ep_id,
                    )
                )
                if not result.scalar_one_or_none():
                    link = SemanticMemoryLink(
                        semantic_memory_id=existing.id,
                        episodic_memory_id=ep_id,
                        strength=1.0,
                        link_type="supports",
                    )
                    self.db.add(link)

            return existing

        # Create new
        return await self.create(SemanticMemoryData(
            concept=concept,
            category=category,
            representation=representation,
            confidence=0.5,
            generation=generation,
            source_episodic_ids=source_episodic_ids,
        ))

    async def get_by_id(self, memory_id: uuid.UUID) -> Optional[SemanticMemory]:
        """Get semantic memory by ID."""
        result = await self.db.execute(
            select(SemanticMemory).where(SemanticMemory.id == memory_id)
        )
        return result.scalar_one_or_none()

    async def get_by_concept(self, concept: str, generation: Optional[int] = None) -> Optional[SemanticMemory]:
        """Get semantic memory by concept name."""
        stmt = select(SemanticMemory).where(SemanticMemory.concept.ilike(concept))
        if generation:
            stmt = stmt.where(SemanticMemory.generation == generation)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def retrieve_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_confidence: float = 0.0,
        generation: Optional[int] = None,
    ) -> List[SemanticMemory]:
        """Retrieve semantic memories by embedding similarity."""
        stmt = select(SemanticMemory).where(
            SemanticMemory.embedding.is_not(None),
            SemanticMemory.confidence >= min_confidence,
        )

        if generation:
            stmt = stmt.where(SemanticMemory.generation == generation)

        stmt = stmt.order_by(SemanticMemory.embedding.cosine_distance(query_embedding)).limit(top_k)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def retrieve_by_category(
        self,
        category: str,
        limit: int = 50,
        generation: Optional[int] = None,
    ) -> List[SemanticMemory]:
        """Retrieve semantic memories by category."""
        stmt = select(SemanticMemory).where(SemanticMemory.category == category)
        if generation:
            stmt = stmt.where(SemanticMemory.generation == generation)
        stmt = stmt.order_by(desc(SemanticMemory.confidence), desc(SemanticMemory.last_reinforced)).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def retrieve_high_confidence(
        self,
        limit: int = 50,
        min_confidence: float = 0.8,
        generation: Optional[int] = None,
    ) -> List[SemanticMemory]:
        """Retrieve high-confidence semantic memories."""
        stmt = select(SemanticMemory).where(SemanticMemory.confidence >= min_confidence)
        if generation:
            stmt = stmt.where(SemanticMemory.generation == generation)
        stmt = stmt.order_by(desc(SemanticMemory.confidence), desc(SemanticMemory.evidence_count)).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_supporting_episodes(
        self,
        semantic_memory_id: uuid.UUID,
        limit: int = 20,
    ) -> List[EpisodicMemory]:
        """Get episodic memories supporting a semantic memory."""
        result = await self.db.execute(
            select(EpisodicMemory)
            .join(SemanticMemoryLink, SemanticMemoryLink.episodic_memory_id == EpisodicMemory.id)
            .where(SemanticMemoryLink.semantic_memory_id == semantic_memory_id)
            .order_by(desc(SemanticMemoryLink.strength))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_confidence(
        self,
        memory_id: uuid.UUID,
        delta: float,
        reason: str = "",
    ) -> bool:
        """Update confidence based on new evidence."""
        memory = await self.get_by_id(memory_id)
        if memory:
            memory.confidence = max(0.0, min(1.0, memory.confidence + delta))
            memory.last_reinforced = datetime.now(timezone.utc)
            # Could log the reason in metadata
            return True
        return False

    async def reinforce(self, memory_id: uuid.UUID, new_episodic_ids: List[uuid.UUID]) -> bool:
        """Reinforce semantic memory with new episodic evidence."""
        memory = await self.get_by_id(memory_id)
        if not memory:
            return False

        added = 0
        for ep_id in new_episodic_ids:
            result = await self.db.execute(
                select(SemanticMemoryLink).where(
                    SemanticMemoryLink.semantic_memory_id == memory_id,
                    SemanticMemoryLink.episodic_memory_id == ep_id,
                )
            )
            if not result.scalar_one_or_none():
                link = SemanticMemoryLink(
                    semantic_memory_id=memory_id,
                    episodic_memory_id=ep_id,
                    strength=1.0,
                    link_type="supports",
                )
                self.db.add(link)
                added += 1

        if added > 0:
            memory.evidence_count += added
            memory.confidence = min(1.0, memory.confidence + 0.05 * added)
            memory.last_reinforced = datetime.now(timezone.utc)

        return added > 0

    async def contradict(self, memory_id: uuid.UUID, contradicting_episodic_ids: List[uuid.UUID]) -> bool:
        """Record contradicting evidence."""
        memory = await self.get_by_id(memory_id)
        if not memory:
            return False

        for ep_id in contradicting_episodic_ids:
            link = SemanticMemoryLink(
                semantic_memory_id=memory_id,
                episodic_memory_id=ep_id,
                strength=1.0,
                link_type="contradicts",
            )
            self.db.add(link)

        # Reduce confidence
        memory.confidence = max(0.0, memory.confidence - 0.1 * len(contradicting_episodic_ids))
        return True

    async def get_stats(self, generation: Optional[int] = None) -> Dict[str, Any]:
        """Get semantic memory statistics."""
        stmt = select(
            func.count(SemanticMemory.id),
            func.avg(SemanticMemory.confidence),
            func.avg(SemanticMemory.evidence_count),
            func.count(SemanticMemory.category.distinct()),
        )
        if generation:
            stmt = stmt.where(SemanticMemory.generation == generation)

        result = await self.db.execute(stmt)
        total, avg_confidence, avg_evidence, categories = result.one()

        # By category
        stmt = select(SemanticMemory.category, func.count(SemanticMemory.id)).group_by(SemanticMemory.category)
        if generation:
            stmt = stmt.where(SemanticMemory.generation == generation)
        result = await self.db.execute(stmt)
        by_category = {row.category or "uncategorized": row[1] for row in result.all()}

        return {
            "total_concepts": total or 0,
            "avg_confidence": float(avg_confidence or 0),
            "avg_evidence_per_concept": float(avg_evidence or 0),
            "categories": categories or 0,
            "by_category": by_category,
        }

    async def find_related(
        self,
        concept: str,
        limit: int = 10,
        generation: Optional[int] = None,
    ) -> List[SemanticMemory]:
        """Find semantically related concepts (by shared episodic evidence)."""
        # Get the semantic memory
        target = await self.get_by_concept(concept, generation)
        if not target:
            return []

        # Find other semantic memories sharing episodic evidence
        result = await self.db.execute(
            select(SemanticMemory, func.count(SemanticMemoryLink.episodic_memory_id).label("shared_count"))
            .join(SemanticMemoryLink, SemanticMemoryLink.semantic_memory_id == SemanticMemory.id)
            .where(
                SemanticMemoryLink.episodic_memory_id.in_(
                    select(SemanticMemoryLink.episodic_memory_id).where(
                        SemanticMemoryLink.semantic_memory_id == target.id
                    )
                ),
                SemanticMemory.id != target.id,
            )
            .group_by(SemanticMemory.id)
            .order_by(desc("shared_count"))
            .limit(limit)
        )
        return [row.SemanticMemory for row in result.all()]
