"""
Reflection Scheduler

Celery beat tasks for autonomous reflection cycles.
Selects memories, detects contradictions/novelty, infers new insights,
and creates new memories from reflection.
"""

import uuid
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from celery import Celery
from celery.schedules import crontab
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import selectinload

from api.config import settings
from api.models import (
    EpisodicMemory, SemanticMemory, Belief, Reflection,
    MemorySource, BeliefState, Generation
)
from memory.episodic import EpisodicMemoryService
from memory.semantic import SemanticMemoryService
from memory.belief import BeliefSystem
from memory.retrieval import MemoryRetriever, RetrievalStrategy, RetrievalResult
from research.web_search import WebSearchService
from research.provenance import ProvenanceTracker, ProvenanceType
from brain.inference import InferenceEngine


celery_app = Celery(
    "reflection",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)


@dataclass
class ReflectionCandidate:
    """A memory or belief candidate for reflection."""
    id: uuid.UUID
    type: str  # "episodic", "semantic", "belief"
    content: str
    importance: float
    recency: float
    connections: List[uuid.UUID]
    metadata: Dict[str, Any]


@dataclass
class ReflectionResult:
    """Result of a reflection cycle."""
    reflection_id: uuid.UUID
    trigger_type: str
    input_memories: List[uuid.UUID]
    output: str
    new_insights: List[str]
    contradictions_found: List[Dict[str, Any]]
    novelty_detected: List[Dict[str, Any]]
    beliefs_updated: List[uuid.UUID]
    memories_created: List[uuid.UUID]
    research_triggered: List[str]
    duration_seconds: float
    generation: int


class ReflectionTrigger(str, Enum):
    """Types of reflection triggers."""
    SCHEDULED = "scheduled"
    CONTRADICTION = "contradiction"
    NOVELTY = "novelty"
    HIGH_IMPORTANCE = "high_importance"
    RESEARCH_COMPLETE = "research_complete"
    HUMAN_FEEDBACK = "human_feedback"
    GENERATION_TRANSITION = "generation_transition"


class ReflectionScheduler:
    """
    Manages autonomous reflection cycles.

    The reflection process:
    1. Select candidate memories/beliefs based on importance, recency, connections
    2. Detect contradictions between beliefs and memories
    3. Detect novelty (new patterns, unexpected connections)
    4. Infer new insights through reasoning
    5. Create new memories from reflection
    6. Update beliefs based on new insights
    7. Trigger research for gaps
    """

    def __init__(self, db: AsyncSession, inference_engine: InferenceEngine):
        self.db = db
        self.inference = inference_engine
        self.episodic_service = EpisodicMemoryService(db)
        self.semantic_service = SemanticMemoryService(db)
        self.belief_system = BeliefSystem(db)
        self.retriever = MemoryRetriever(db)
        self.web_search = WebSearchService(db)
        self.provenance = ProvenanceTracker(db)
        self._current_generation = None

    async def _get_current_generation(self) -> int:
        """Get current active generation."""
        if self._current_generation is None:
            stmt = select(Generation).where(Generation.is_active == True).order_by(Generation.number.desc())
            result = await self.db.execute(stmt)
            gen = result.scalar_one_or_none()
            self._current_generation = gen.number if gen else 1
        return self._current_generation

    async def select_reflection_candidates(
        self,
        trigger: ReflectionTrigger,
        context: Dict[str, Any],
        max_candidates: int = 20,
    ) -> List[ReflectionCandidate]:
        """
        Select memories and beliefs for reflection based on trigger and context.
        """
        candidates = []
        generation = await self._get_current_generation()

        # Strategy varies by trigger
        if trigger == ReflectionTrigger.SCHEDULED:
            # Mix of recent important memories, active beliefs, and semantic concepts
            candidates.extend(await self._select_recent_important(max_candidates // 3))
            candidates.extend(await self._select_active_beliefs(max_candidates // 3))
            candidates.extend(await self._select_connected_concepts(max_candidates // 3))

        elif trigger == ReflectionTrigger.CONTRADICTION:
            # Focus on contradictory beliefs and related memories
            candidates.extend(await self._select_contradiction_context(context, max_candidates))

        elif trigger == ReflectionTrigger.NOVELTY:
            # Focus on new/unusual patterns
            candidates.extend(await self._select_novel_patterns(context, max_candidates))

        elif trigger == ReflectionTrigger.HIGH_IMPORTANCE:
            # Focus on high-importance items
            candidates.extend(await self._select_high_importance(max_candidates))

        elif trigger == ReflectionTrigger.RESEARCH_COMPLETE:
            # Focus on research results and related knowledge
            candidates.extend(await self._select_research_context(context, max_candidates))

        # Deduplicate by ID
        seen = set()
        unique = []
        for c in candidates:
            if c.id not in seen:
                seen.add(c.id)
                unique.append(c)

        return unique[:max_candidates]

    async def _select_recent_important(self, limit: int) -> List[ReflectionCandidate]:
        """Select recent high-importance episodic memories."""
        results = await self.retriever.retrieve(
            query="",
            strategies=[RetrievalStrategy.RECENCY, RetrievalStrategy.IMPORTANCE],
            limit=limit,
            generation=await self._get_current_generation(),
        )

        candidates = []
        for r in results:
            if r.source_type == "episodic" and r.importance_score > 0.5:
                candidates.append(ReflectionCandidate(
                    id=r.memory_id,
                    type="episodic",
                    content=r.content[:500],
                    importance=r.importance_score,
                    recency=r.recency_score,
                    connections=r.associations,
                    metadata={"source": "recent_important"},
                ))
        return candidates

    async def _select_active_beliefs(self, limit: int) -> List[ReflectionCandidate]:
        """Select active beliefs with medium confidence (ripe for update)."""
        stmt = select(Belief).where(
            and_(
                Belief.state == BeliefState.ACTIVE,
                Belief.confidence > 0.3,
                Belief.confidence < 0.9,
                Belief.generation == await self._get_current_generation(),
            )
        ).order_by(desc(Belief.last_updated)).limit(limit)

        result = await self.db.execute(stmt)
        beliefs = result.scalars().all()

        candidates = []
        for b in beliefs:
            candidates.append(ReflectionCandidate(
                id=b.id,
                type="belief",
                content=b.proposition,
                importance=b.confidence,
                recency=1.0 if (datetime.now(timezone.utc) - b.last_updated).days < 7 else 0.5,
                connections=[e for e in b.evidence_episodic] + [e for e in b.evidence_semantic],
                metadata={"source": "active_belief", "confidence": b.confidence},
            ))
        return candidates

    async def _select_connected_concepts(self, limit: int) -> List[ReflectionCandidate]:
        """Select highly-connected semantic concepts."""
        stmt = select(SemanticMemory).where(
            SemanticMemory.generation == await self._get_current_generation()
        ).order_by(desc(SemanticMemory.connection_count)).limit(limit)

        result = await self.db.execute(stmt)
        concepts = result.scalars().all()

        candidates = []
        for c in concepts:
            candidates.append(ReflectionCandidate(
                id=c.id,
                type="semantic",
                content=f"{c.concept}: {c.description}",
                importance=min(c.connection_count / 10.0, 1.0),
                recency=0.8,
                connections=c.related_concepts or [],
                metadata={"source": "connected_concept", "connection_count": c.connection_count},
            ))
        return candidates

    async def _select_contradiction_context(
        self,
        context: Dict[str, Any],
        limit: int,
    ) -> List[ReflectionCandidate]:
        """Select memories/beliefs related to a contradiction."""
        belief_id = context.get("belief_id")
        if not belief_id:
            return []

        # Get the contradictory belief
        stmt = select(Belief).where(Belief.id == belief_id)
        result = await self.db.execute(stmt)
        belief = result.scalar_one_or_none()
        if not belief:
            return []

        candidates = [ReflectionCandidate(
            id=belief.id,
            type="belief",
            content=belief.proposition,
            importance=1.0,
            recency=1.0,
            connections=[],
            metadata={"source": "contradiction_focus", "confidence": belief.confidence},
        )]

        # Get evidence for and against
        for e_id in belief.evidence_episodic + belief.evidence_semantic:
            # Would fetch the memory
            pass

        # Get related beliefs
        related = await self.belief_system.find_related_beliefs(
            belief.proposition, threshold=0.6, limit=limit - 1
        )
        for b, sim in related:
            candidates.append(ReflectionCandidate(
                id=b.id,
                type="belief",
                content=b.proposition,
                importance=sim,
                recency=0.8,
                connections=[],
                metadata={"source": "contradiction_related", "similarity": sim},
            ))

        return candidates[:limit]

    async def _select_novel_patterns(
        self,
        context: Dict[str, Any],
        limit: int,
    ) -> List[ReflectionCandidate]:
        """Select memories showing novel patterns."""
        # Look for memories with low similarity to existing beliefs
        recent = await self.retriever.retrieve(
            query="",
            strategies=[RetrievalStrategy.RECENCY],
            limit=limit * 2,
            generation=await self._get_current_generation(),
        )

        candidates = []
        for r in recent:
            # Check if this memory contradicts or extends beliefs
            related_beliefs = await self.belief_system.find_related_beliefs(
                r.content, threshold=0.5, limit=5
            )

            novelty_score = 1.0 - max([s for _, s in related_beliefs], default=0.0)
            if novelty_score > 0.5:  # Relatively novel
                candidates.append(ReflectionCandidate(
                    id=r.memory_id,
                    type=r.source_type,
                    content=r.content[:500],
                    importance=novelty_score,
                    recency=r.recency_score,
                    connections=r.associations,
                    metadata={"source": "novelty", "novelty_score": novelty_score},
                ))

        # Sort by novelty
        candidates.sort(key=lambda c: c.metadata.get("novelty_score", 0), reverse=True)
        return candidates[:limit]

    async def _select_high_importance(self, limit: int) -> List[ReflectionCandidate]:
        """Select highest importance memories across all types."""
        # Top episodic
        episodic = await self.retriever.retrieve(
            query="",
            strategies=[RetrievalStrategy.IMPORTANCE],
            limit=limit,
            generation=await self._get_current_generation(),
        )

        candidates = []
        for r in episodic:
            candidates.append(ReflectionCandidate(
                id=r.memory_id,
                type=r.source_type,
                content=r.content[:500],
                importance=r.importance_score,
                recency=r.recency_score,
                connections=r.associations,
                metadata={"source": "high_importance"},
            ))
        return candidates

    async def _select_research_context(
        self,
        context: Dict[str, Any],
        limit: int,
    ) -> List[ReflectionCandidate]:
        """Select context around completed research."""
        query_id = context.get("query_id")
        if not query_id:
            return []

        # Get memories created from this research
        stmt = select(EpisodicMemory).where(
            EpisodicMemory.context.contains({"query_id": str(query_id)})
        ).limit(limit)

        result = await self.db.execute(stmt)
        memories = result.scalars().all()

        candidates = []
        for m in memories:
            candidates.append(ReflectionCandidate(
                id=m.id,
                type="episodic",
                content=m.content[:500],
                importance=m.importance_score,
                recency=1.0,
                connections=m.associations,
                metadata={"source": "research_context"},
            ))
        return candidates

    async def detect_contradictions(
        self,
        candidates: List[ReflectionCandidate],
    ) -> List[Dict[str, Any]]:
        """Detect contradictions among candidates."""
        contradictions = []

        # Check beliefs against each other
        beliefs = [c for c in candidates if c.type == "belief"]
        for i, b1 in enumerate(beliefs):
            for b2 in beliefs[i+1:]:
                # Simple contradiction detection: check if propositions are mutually exclusive
                # In practice, would use model to detect semantic contradiction
                contradiction = await self._check_contradiction(b1.content, b2.content)
                if contradiction:
                    contradictions.append({
                        "type": "belief_belief",
                        "items": [str(b1.id), str(b2.id)],
                        "description": contradiction,
                        "severity": "high",
                    })

        # Check memories against beliefs
        memories = [c for c in candidates if c.type in ("episodic", "semantic")]
        for m in memories:
            for b in beliefs:
                contradiction = await self._check_contradiction(m.content, b.content)
                if contradiction:
                    contradictions.append({
                        "type": "memory_belief",
                        "items": [str(m.id), str(b.id)],
                        "description": contradiction,
                        "severity": "medium",
                    })

        return contradictions

    async def _check_contradiction(self, text1: str, text2: str) -> Optional[str]:
        """Check if two texts contradict each other using the model."""
        prompt = f"""Do these two statements contradict each other? Answer only "yes: <reason>" or "no".

Statement 1: {text1}
Statement 2: {text2}"""

        try:
            response = await self.inference.generate(
                prompt,
                max_tokens=100,
                temperature=0.1,
            )
            if response.strip().lower().startswith("yes:"):
                return response[4:].strip()
        except Exception:
            pass
        return None

    async def detect_novelty(
        self,
        candidates: List[ReflectionCandidate],
    ) -> List[Dict[str, Any]]:
        """Detect novel patterns or unexpected connections."""
        novelty = []

        # Look for memories that don't fit existing semantic concepts
        for c in candidates:
            if c.type == "episodic":
                # Check against semantic memory
                related = await self.semantic_service.find_related_concepts(
                    c.content, threshold=0.6, limit=3
                )

                if not related:
                    # No related concepts - potentially novel
                    novelty.append({
                        "type": "isolated_memory",
                        "item": str(c.id),
                        "description": "Memory has no strong connection to existing concepts",
                        "novelty_score": 0.9,
                    })
                elif len(related) == 1 and related[0][1] < 0.7:
                    # Weak connection to single concept
                    novelty.append({
                        "type": "weakly_connected",
                        "item": str(c.id),
                        "related_concept": str(related[0][0].id),
                        "similarity": related[0][1],
                        "novelty_score": 1.0 - related[0][1],
                    })

        # Look for unexpected connections between candidates
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i+1:]:
                if c1.id in c2.connections or c2.id in c1.connections:
                    continue  # Already known connection

                # Check for semantic similarity without explicit connection
                similarity = await self._compute_similarity(c1.content, c2.content)
                if similarity > 0.7:
                    novelty.append({
                        "type": "latent_connection",
                        "items": [str(c1.id), str(c2.id)],
                        "similarity": similarity,
                        "description": f"Strong semantic similarity ({similarity:.2f}) without explicit link",
                        "novelty_score": similarity * 0.8,
                    })

        return novelty

    async def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        # Would use embeddings in practice
        # For now, simple heuristic
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    async def infer_insights(
        self,
        candidates: List[ReflectionCandidate],
        contradictions: List[Dict[str, Any]],
        novelty: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str]]:
        """
        Generate insights from reflection.

        Returns:
            - new_insights: General insights from reflection
            - research_questions: Questions that need external research
        """
        # Build context from candidates
        context_parts = []
        for c in candidates[:10]:  # Limit context
            context_parts.append(f"[{c.type.upper()}] {c.content}")

        context = "\n".join(context_parts)

        contradictions_text = "\n".join([
            f"- {c['description']} (between {', '.join(c['items'])})"
            for c in contradictions[:5]
        ]) or "None detected."

        novelty_text = "\n".join([
            f"- {n['description']} (score: {n.get('novelty_score', 0):.2f})"
            for n in novelty[:5]
        ]) or "None detected."

        prompt = f"""You are an autonomous AI entity reflecting on your memories and beliefs.

CURRENT KNOWLEDGE:
{context}

CONTRADICTIONS DETECTED:
{contradictions_text}

NOVELTY DETECTED:
{novelty_text}

TASK: Reflect on this information and produce:
1. KEY INSIGHTS (2-5): New understandings, patterns, or syntheses
2. RESEARCH QUESTIONS (0-3): Specific questions needing external verification

Format:
INSIGHTS:
- Insight 1
- Insight 2
...

QUESTIONS:
- Question 1
- Question 2
...

If no insights or questions, write "None" for that section."""

        try:
            response = await self.inference.generate(
                prompt,
                max_tokens=500,
                temperature=0.7,
            )

            insights = []
            questions = []
            section = None

            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("INSIGHTS:"):
                    section = "insights"
                elif line.startswith("QUESTIONS:"):
                    section = "questions"
                elif line.startswith("- ") and section == "insights":
                    insights.append(line[2:])
                elif line.startswith("- ") and section == "questions":
                    questions.append(line[2:])

            return insights, questions

        except Exception as e:
            return [f"Reflection error: {str(e)}"], []

    async def create_reflection_memories(
        self,
        reflection_id: uuid.UUID,
        insights: List[str],
        candidates: List[ReflectionCandidate],
        generation: int,
    ) -> List[uuid.UUID]:
        """Create new episodic memories from reflection insights."""
        memory_ids = []

        for insight in insights:
            # Determine source memories this insight derives from
            derived_from = [c.id for c in candidates[:5]]  # Top 5 candidates

            memory = await self.episodic_service.create_memory(
                content=f"Reflection insight: {insight}",
                source=MemorySource.REFLECTION,
                importance=0.8,
                tags=["reflection", "insight", "autonomous"],
                associations=derived_from,
                context={
                    "reflection_id": str(reflection_id),
                    "generation": generation,
                    "insight_type": "reflection",
                },
            )
            memory_ids.append(memory.id)

            # Record provenance
            await self.provenance.record_memory_created(
                memory_id=memory.id,
                content=insight,
                source=MemorySource.REFLECTION,
                context={"reflection_id": str(reflection_id), "generation": generation},
                derived_from=derived_from,
                generation=generation,
            )

        return memory_ids

    async def update_beliefs_from_reflection(
        self,
        insights: List[str],
        contradictions: List[Dict[str, Any]],
        generation: int,
    ) -> List[uuid.UUID]:
        """Update or create beliefs based on reflection."""
        updated = []

        # Handle contradictions - may reduce confidence
        for c in contradictions:
            if c["type"] == "belief_belief":
                for belief_id_str in c["items"]:
                    belief_id = uuid.UUID(belief_id_str)
                    # Reduce confidence slightly due to contradiction
                    await self.belief_system.update_confidence(
                        belief_id=belief_id,
                        delta=-0.1,
                        reason=f"Contradiction detected: {c['description']}",
                    )
                    updated.append(belief_id)

        # Create new beliefs from strong insights
        for insight in insights:
            # Check if similar belief exists
            related = await self.belief_system.find_related_beliefs(insight, threshold=0.8, limit=1)
            if related:
                # Strengthen existing belief
                belief, sim = related[0]
                await self.belief_system.update_confidence(
                    belief_id=belief.id,
                    delta=0.05,
                    reason=f"Supported by reflection insight: {insight[:100]}",
                )
                updated.append(belief.id)
            else:
                # Create new tentative belief
                belief = await self.belief_system.create_belief(
                    proposition=insight,
                    initial_confidence=0.5,
                    evidence_episodic=[],  # Would link to reflection memories
                    evidence_semantic=[],
                    generation=generation,
                )
                updated.append(belief.id)

        return updated

    async def trigger_research_for_gaps(
        self,
        questions: List[str],
        context: Dict[str, Any],
        generation: int,
    ) -> List[str]:
        """Trigger autonomous research for unanswered questions."""
        triggered = []

        for question in questions[:3]:  # Limit concurrent research
            # Check if recently researched
            # Would query for recent WebQuery with similar query

            # Trigger research
            query_id = await self.web_search.autonomous_research(
                seed_query=question,
                generation=generation,
                max_depth=2,
                max_pages=5,
            )
            triggered.append(str(query_id))

        return triggered

    async def run_reflection_cycle(
        self,
        trigger: ReflectionTrigger,
        context: Dict[str, Any] = None,
    ) -> ReflectionResult:
        """Run a complete reflection cycle."""
        start_time = datetime.now(timezone.utc)
        context = context or {}
        generation = await self._get_current_generation()

        # 1. Select candidates
        candidates = await self.select_reflection_candidates(trigger, context)

        # 2. Detect contradictions
        contradictions = await self.detect_contradictions(candidates)

        # 3. Detect novelty
        novelty = await self.detect_novelty(candidates)

        # 4. Infer insights
        insights, questions = await self.infer_insights(candidates, contradictions, novelty)

        # 5. Create reflection record
        reflection = Reflection(
            id=uuid.uuid4(),
            generation=generation,
            trigger_type=trigger.value,
            trigger_context=context,
            input_memory_ids=[c.id for c in candidates],
            output_summary="\n".join(insights) if insights else "No insights generated",
            contradictions_found=contradictions,
            novelty_detected=novelty,
            insights_generated=insights,
            research_questions=questions,
            duration_seconds=0,  # Will update at end
        )
        self.db.add(reflection)
        await self.db.flush()

        # 6. Create memories from insights
        memory_ids = await self.create_reflection_memories(
            reflection.id, insights, candidates, generation
        )

        # 7. Update beliefs
        belief_ids = await self.update_beliefs_from_reflection(
            insights, contradictions, generation
        )

        # 8. Trigger research for gaps
        research_ids = await self.trigger_research_for_gaps(questions, context, generation)

        # 9. Record provenance
        await self.provenance.record_reflection(
            reflection_id=reflection.id,
            trigger_type=trigger.value,
            input_memories=[c.id for c in candidates],
            output=reflection.output_summary,
            generation=generation,
        )

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        reflection.duration_seconds = duration
        await self.db.commit()

        return ReflectionResult(
            reflection_id=reflection.id,
            trigger_type=trigger.value,
            input_memories=[c.id for c in candidates],
            output=reflection.output_summary,
            new_insights=insights,
            contradictions_found=contradictions,
            novelty_detected=novelty,
            beliefs_updated=belief_ids,
            memories_created=memory_ids,
            research_triggered=research_ids,
            duration_seconds=duration,
            generation=generation,
        )


# Celery Tasks

@celery_app.task(bind=True, max_retries=3)
def scheduled_reflection(self):
    """Periodic scheduled reflection task."""
    # This runs in a sync Celery worker, so we need to handle async
    import asyncio
    from api.database import get_async_session

    async def _run():
        async for db in get_async_session():
            # Need inference engine - would be initialized from checkpoint
            # For now, create a placeholder
            inference = InferenceEngine(
                model=None,  # Would load from current generation
                tokenizer=None,
                device="cpu",
            )

            scheduler = ReflectionScheduler(db, inference)
            result = await scheduler.run_reflection_cycle(
                ReflectionTrigger.SCHEDULED,
                {"scheduled": True, "timestamp": datetime.now(timezone.utc).isoformat()},
            )
            return {
                "reflection_id": str(result.reflection_id),
                "insights": len(result.new_insights),
                "contradictions": len(result.contradictions_found),
                "novelty": len(result.novelty_detected),
                "memories_created": len(result.memories_created),
                "beliefs_updated": len(result.beliefs_updated),
                "research_triggered": len(result.research_triggered),
                "duration": result.duration_seconds,
            }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, max_retries=3)
def contradiction_reflection(self, belief_id: str):
    """Triggered reflection when contradiction detected."""
    import asyncio
    from api.database import get_async_session

    async def _run():
        async for db in get_async_session():
            inference = InferenceEngine(model=None, tokenizer=None, device="cpu")
            scheduler = ReflectionScheduler(db, inference)
            result = await scheduler.run_reflection_cycle(
                ReflectionTrigger.CONTRADICTION,
                {"belief_id": belief_id},
            )
            return {"reflection_id": str(result.reflection_id)}

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(bind=True, max_retries=3)
def research_completion_reflection(self, query_id: str):
    """Triggered reflection when research completes."""
    import asyncio
    from api.database import get_async_session

    async def _run():
        async for db in get_async_session():
            inference = InferenceEngine(model=None, tokenizer=None, device="cpu")
            scheduler = ReflectionScheduler(db, inference)
            result = await scheduler.run_reflection_cycle(
                ReflectionTrigger.RESEARCH_COMPLETE,
                {"query_id": query_id},
            )
            return {"reflection_id": str(result.reflection_id)}

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)


# Celery Beat Schedule
celery_app.conf.beat_schedule = {
    # Run reflection every 30 minutes
    "scheduled-reflection-every-30-minutes": {
        "task": "reflection.scheduler.scheduled_reflection",
        "schedule": 1800.0,  # 30 minutes in seconds
    },
    # Daily deep reflection at 3 AM
    "daily-deep-reflection": {
        "task": "reflection.scheduler.scheduled_reflection",
        "schedule": crontab(hour=3, minute=0),
        "kwargs": {"deep": True},
    },
    # Weekly comprehensive reflection on Sundays
    "weekly-comprehensive-reflection": {
        "task": "reflection.scheduler.scheduled_reflection",
        "schedule": crontab(hour=4, minute=0, day_of_week=0),
        "kwargs": {"comprehensive": True},
    },
}

celery_app.conf.timezone = "UTC"