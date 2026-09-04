"""
Belief System

Manages the entity's beliefs with confidence levels, evidence tracking,
and contradiction detection. This is the entity's "model of the world".
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import selectinload

from api.models import Belief, BeliefEvidence, EpisodicMemory, MemorySource, ConfidenceLevel
from memory.episodic import EpisodicMemoryService
from api.config import settings


class BeliefStatus(str, Enum):
    """Status of a belief."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    CONTRADICTED = "contradicted"
    PENDING = "pending"


@dataclass
class BeliefData:
    """Data for creating a belief."""
    proposition: str
    category: Optional[str]
    confidence: float = 0.5
    generation: int = 1
    source_evidence: List[Dict[str, Any]] = field(default_factory=list)


class BeliefSystem:
    """
    Manages the entity's belief network.

    Features:
    - Track beliefs with confidence levels
    - Link beliefs to evidence (episodic memories, web sources)
    - Detect contradictions
    - Update confidence based on evidence
    - Query beliefs by category, confidence, status
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.episodic_service = EpisodicMemoryService(db)

    async def create_belief(self, belief_data: BeliefData) -> Belief:
        """Create a new belief."""
        # Determine confidence level
        confidence_level = self._confidence_to_level(belief_data.confidence)

        belief = Belief(
            proposition=belief_data.proposition,
            category=belief_data.category,
            confidence=belief_data.confidence,
            confidence_level=confidence_level,
            status=BeliefStatus.ACTIVE.value,
            generation=belief_data.generation,
            formed_at=datetime.now(timezone.utc),
        )
        self.db.add(belief)
        await self.db.flush()

        # Add evidence
        for ev in belief_data.source_evidence:
            evidence = BeliefEvidence(
                belief_id=belief.id,
                episodic_memory_id=ev.get("episodic_memory_id"),
                source=ev.get("source", MemorySource.INTERNAL_INFERENCE),
                content=ev.get("content", ""),
                supports=ev.get("supports", True),
                weight=ev.get("weight", 1.0),
            )
            self.db.add(evidence)

        return belief

    async def get_or_create_belief(
        self,
        proposition: str,
        category: Optional[str],
        initial_confidence: float,
        generation: int,
        evidence: List[Dict[str, Any]],
    ) -> Belief:
        """Get existing belief or create new one."""
        # Check for similar existing belief
        result = await self.db.execute(
            select(Belief).where(
                Belief.proposition.ilike(proposition),
                Belief.generation == generation,
                Belief.status != BeliefStatus.DEPRECATED.value,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update with new evidence
            await self.add_evidence(existing.id, evidence)
            return existing

        return await self.create_belief(BeliefData(
            proposition=proposition,
            category=category,
            confidence=initial_confidence,
            generation=generation,
            source_evidence=evidence,
        ))

    async def add_evidence(
        self,
        belief_id: uuid.UUID,
        evidence_list: List[Dict[str, Any]],
    ) -> bool:
        """Add evidence to a belief."""
        result = await self.db.execute(
            select(Belief).where(Belief.id == belief_id)
        )
        belief = result.scalar_one_or_none()
        if not belief:
            return False

        supporting = 0
        contradicting = 0

        for ev in evidence_list:
            evidence = BeliefEvidence(
                belief_id=belief_id,
                episodic_memory_id=ev.get("episodic_memory_id"),
                source=ev.get("source", MemorySource.INTERNAL_INFERENCE),
                content=ev.get("content", ""),
                supports=ev.get("supports", True),
                weight=ev.get("weight", 1.0),
            )
            self.db.add(evidence)

            if ev.get("supports", True):
                supporting += 1
            else:
                contradicting += 1

        # Update confidence based on evidence balance
        if supporting + contradicting > 0:
            net_support = (supporting - contradicting) / (supporting + contradicting)
            # Adjust confidence toward net support
            target_confidence = 0.5 + net_support * 0.5
            belief.confidence = belief.confidence * 0.7 + target_confidence * 0.3
            belief.confidence = max(0.0, min(1.0, belief.confidence))
            belief.confidence_level = self._confidence_to_level(belief.confidence)
            belief.last_updated = datetime.now(timezone.utc)

            if contradicting > 0:
                belief.contradiction_count += contradicting
                if belief.confidence < 0.3:
                    belief.status = BeliefStatus.CONTRADICTED.value

        return True

    async def get_belief(self, belief_id: uuid.UUID) -> Optional[Belief]:
        """Get belief by ID with evidence."""
        result = await self.db.execute(
            select(Belief)
            .options(selectinload(Belief.evidence))
            .where(Belief.id == belief_id)
        )
        return result.scalar_one_or_none()

    async def get_beliefs(
        self,
        category: Optional[str] = None,
        status: Optional[BeliefStatus] = None,
        min_confidence: float = 0.0,
        generation: Optional[int] = None,
        limit: int = 100,
    ) -> List[Belief]:
        """Query beliefs with filters."""
        stmt = select(Belief).where(
            Belief.confidence >= min_confidence,
        )

        if category:
            stmt = stmt.where(Belief.category == category)
        if status:
            stmt = stmt.where(Belief.status == status.value)
        if generation:
            stmt = stmt.where(Belief.generation == generation)

        stmt = stmt.order_by(desc(Belief.confidence), desc(Belief.last_updated)).limit(limit)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def find_contradictions(
        self,
        generation: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find beliefs that contradict each other."""
        # This is a simplified version - in practice would use NLP to detect contradictions
        # For now, find beliefs with high contradiction_count
        stmt = select(Belief).where(
            Belief.contradiction_count > 0,
            Belief.status != BeliefStatus.DEPRECATED.value,
        )
        if generation:
            stmt = stmt.where(Belief.generation == generation)

        result = await self.db.execute(stmt)
        beliefs = result.scalars().all()

        contradictions = []
        for belief in beliefs:
            # Get contradicting evidence
            result = await self.db.execute(
                select(BeliefEvidence).where(
                    BeliefEvidence.belief_id == belief.id,
                    BeliefEvidence.supports == False,
                )
            )
            contradicting_evidence = result.scalars().all()

            contradictions.append({
                "belief_id": str(belief.id),
                "proposition": belief.proposition,
                "confidence": belief.confidence,
                "contradiction_count": belief.contradiction_count,
                "contradicting_evidence": [
                    {"content": e.content, "source": e.source.value}
                    for e in contradicting_evidence
                ],
            })

        return contradictions

    async def resolve_contradiction(
        self,
        belief_id: uuid.UUID,
        resolution: str,  # "keep", "deprecate", "split"
        reason: str,
    ) -> bool:
        """Resolve a contradiction."""
        belief = await self.get_belief(belief_id)
        if not belief:
            return False

        if resolution == "deprecate":
            belief.status = BeliefStatus.DEPRECATED.value
        elif resolution == "split":
            # Would create a new belief with modified proposition
            # For now just mark as needing review
            belief.status = BeliefStatus.PENDING.value
        # "keep" means no change

        belief.last_updated = datetime.now(timezone.utc)
        return True

    async def update_confidence(
        self,
        belief_id: uuid.UUID,
        new_confidence: float,
        reason: str = "",
    ) -> bool:
        """Manually update belief confidence."""
        belief = await self.get_belief(belief_id)
        if not belief:
            return False

        belief.confidence = max(0.0, min(1.0, new_confidence))
        belief.confidence_level = self._confidence_to_level(belief.confidence)
        belief.last_updated = datetime.now(timezone.utc)

        if belief.confidence < 0.3:
            belief.status = BeliefStatus.CONTRADICTED.value
        elif belief.status == BeliefStatus.CONTRADICTED.value and belief.confidence > 0.5:
            belief.status = BeliefStatus.ACTIVE.value

        return True

    async def get_belief_network(
        self,
        concept: str,
        depth: int = 2,
        generation: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get network of related beliefs."""
        # Find seed belief
        seed = await self.db.execute(
            select(Belief).where(
                Belief.proposition.ilike(f"%{concept}%"),
                Belief.status != BeliefStatus.DEPRECATED.value,
            )
        )
        seed_belief = seed.scalar_one_or_none()

        if not seed_belief:
            return {"nodes": [], "edges": []}

        # Get related beliefs (same category or shared evidence)
        related = await self.db.execute(
            select(Belief).where(
                Belief.category == seed_belief.category,
                Belief.id != seed_belief.id,
                Belief.status != BeliefStatus.DEPRECATED.value,
            )
        )
        related_beliefs = related.scalars().all()

        nodes = [{
            "id": str(seed_belief.id),
            "proposition": seed_belief.proposition,
            "confidence": seed_belief.confidence,
            "status": seed_belief.status,
            "is_seed": True,
        }]
        for b in related_beliefs[:20]:
            nodes.append({
                "id": str(b.id),
                "proposition": b.proposition,
                "confidence": b.confidence,
                "status": b.status,
                "is_seed": False,
            })

        # Edges based on shared evidence
        edges = []
        # Would need more complex query for actual shared evidence
        # Simplified: connect seed to all related
        for node in nodes[1:]:
            edges.append({
                "source": str(seed_belief.id),
                "target": node["id"],
                "type": "category",
            })

        return {"nodes": nodes, "edges": edges}

    async def get_stats(self, generation: Optional[int] = None) -> Dict[str, Any]:
        """Get belief system statistics."""
        stmt = select(
            func.count(Belief.id),
            func.avg(Belief.confidence),
            func.sum(func.case((Belief.status == BeliefStatus.ACTIVE.value, 1), else_=0)),
            func.sum(func.case((Belief.status == BeliefStatus.CONTRADICTED.value, 1), else_=0)),
            func.sum(func.case((Belief.status == BeliefStatus.DEPRECATED.value, 1), else_=0)),
        )
        if generation:
            stmt = stmt.where(Belief.generation == generation)

        result = await self.db.execute(stmt)
        total, avg_conf, active, contradicted, deprecated = result.one()

        # By category
        stmt = select(Belief.category, func.count(Belief.id)).group_by(Belief.category)
        if generation:
            stmt = stmt.where(Belief.generation == generation)
        result = await self.db.execute(stmt)
        by_category = {row.category or "uncategorized": row[1] for row in result.all()}

        # By confidence level
        stmt = select(Belief.confidence_level, func.count(Belief.id)).group_by(Belief.confidence_level)
        if generation:
            stmt = stmt.where(Belief.generation == generation)
        result = await self.db.execute(stmt)
        by_level = {row.confidence_level.value: row[1] for row in result.all()}

        return {
            "total_beliefs": total or 0,
            "avg_confidence": float(avg_conf or 0),
            "active": active or 0,
            "contradicted": contradicted or 0,
            "deprecated": deprecated or 0,
            "by_category": by_category,
            "by_confidence_level": by_level,
        }

    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to level."""
        if confidence >= 0.9:
            return ConfidenceLevel.KNOWN
        elif confidence >= 0.7:
            return ConfidenceLevel.LIKELY
        elif confidence >= 0.4:
            return ConfidenceLevel.UNCERTAIN
        elif confidence >= 0.2:
            return ConfidenceLevel.CONTRADICTED
        else:
            return ConfidenceLevel.UNKNOWN

    async def form_belief_from_experience(
        self,
        experience: EpisodicMemory,
        generation: int,
    ) -> Optional[Belief]:
        """
        Attempt to form a belief from an episodic memory.

        This is a placeholder for the actual belief formation logic
        which would use NLP to extract propositions from experiences.
        """
        # Simple heuristic: if memory is important and from direct observation
        if experience.importance > 0.7 and experience.source == MemorySource.DIRECT_OBSERVATION:
            # Extract a simple proposition (in reality would use LLM)
            proposition = f"Observed: {experience.content[:200]}"

            return await self.get_or_create_belief(
                proposition=proposition,
                category="observation",
                initial_confidence=experience.confidence,
                generation=generation,
                evidence=[{
                    "episodic_memory_id": experience.id,
                    "source": experience.source,
                    "content": experience.content,
                    "supports": True,
                    "weight": experience.importance,
                }]
            )
        return None