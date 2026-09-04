"""
Consolidation Pipeline (Computational Sleep)

The entity's "sleep" cycle:
1. Select experiences from waking period
2. Detect patterns, contradictions, abstractions
3. Form semantic concepts from episodic memories
4. Update belief confidences
5. Generate training data from consolidated knowledge
6. Trigger training if enough new data
7. Evaluate and promote new generation

This runs as a background Celery task.
"""

import uuid
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, not_
from sqlalchemy.orm import selectinload

from api.config import settings
from api.models import (
    EpisodicMemory, SemanticMemory, Belief, ConsolidationRun,
    MemorySource, BeliefState, Generation
)
from memory.episodic import EpisodicMemoryService
from memory.semantic import SemanticMemoryService
from memory.belief import BeliefSystem
from memory.retrieval import MemoryRetriever, RetrievalStrategy
from research.provenance import ProvenanceTracker, ProvenanceType
from brain.inference import InferenceEngine


celery_app = Celery(
    "consolidation",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation run."""
    generation: int
    max_memories: int = 1000
    min_importance: float = 0.3
    lookback_hours: int = 24
    min_concept_frequency: int = 3
    min_belief_evidence: int = 2
    contradiction_threshold: float = 0.7
    novelty_threshold: float = 0.6
    max_new_concepts: int = 50
    max_training_examples: int = 10000


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""
    run_id: uuid.UUID
    generation: int
    started_at: datetime
    completed_at: Optional[datetime]
    input_memories: int
    memories_consolidated: int
    new_concepts: int
    updated_concepts: int
    beliefs_updated: int
    contradictions_resolved: int
    training_examples_generated: int
    dataset_path: Optional[str]
    duration_seconds: float
    status: str
    error: Optional[str] = None


class ConsolidationStage(str, Enum):
    """Stages of consolidation."""
    SELECTION = "selection"
    PATTERN_DETECTION = "pattern_detection"
    CONCEPT_FORMATION = "concept_formation"
    BELIEF_UPDATE = "belief_update"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    TRAINING_DATA_GENERATION = "training_data_generation"
    COMPLETE = "complete"


class ConsolidationPipeline:
    """
    Executes the computational sleep cycle.

    Processes waking experiences into consolidated knowledge.
    """

    def __init__(self, db: AsyncSession, inference_engine: InferenceEngine):
        self.db = db
        self.inference = inference_engine
        self.episodic_service = EpisodicMemoryService(db)
        self.semantic_service = SemanticMemoryService(db)
        self.belief_system = BeliefSystem(db)
        self.retriever = MemoryRetriever(db)
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

    async def run_consolidation(
        self,
        config: Optional[ConsolidationConfig] = None,
    ) -> ConsolidationResult:
        """
        Run a complete consolidation cycle.

        This is the entity's "sleep" - processing experiences into knowledge.
        """
        config = config or ConsolidationConfig(generation=await self._get_current_generation())
        run_id = uuid.uuid4()
        started_at = datetime.now(timezone.utc)

        # Create consolidation run record
        run = ConsolidationRun(
            id=run_id,
            generation=config.generation,
            status=ConsolidationStage.SELECTION.value,
            started_at=started_at,
            config={
                "max_memories": config.max_memories,
                "min_importance": config.min_importance,
                "lookback_hours": config.lookback_hours,
            },
        )
        self.db.add(run)
        await self.db.commit()

        try:
            # Stage 1: Select memories for consolidation
            await self._update_run_status(run_id, ConsolidationStage.SELECTION)
            memories = await self._select_memories(config)

            # Stage 2: Detect patterns
            await self._update_run_status(run_id, ConsolidationStage.PATTERN_DETECTION)
            patterns = await self._detect_patterns(memories, config)

            # Stage 3: Form/update semantic concepts
            await self._update_run_status(run_id, ConsolidationStage.CONCEPT_FORMATION)
            new_concepts, updated_concepts = await self._form_concepts(patterns, memories, config)

            # Stage 4: Update beliefs
            await self._update_run_status(run_id, ConsolidationStage.BELIEF_UPDATE)
            beliefs_updated = await self._update_beliefs(memories, new_concepts, config)

            # Stage 5: Resolve contradictions
            await self._update_run_status(run_id, ConsolidationStage.CONTRADICTION_RESOLUTION)
            contradictions_resolved = await self._resolve_contradictions(config)

            # Stage 6: Generate training data
            await self._update_run_status(run_id, ConsolidationStage.TRAINING_DATA_GENERATION)
            training_examples, dataset_path = await self._generate_training_data(
                memories, new_concepts, config
            )

            completed_at = datetime.now(timezone.utc)
            duration = (completed_at - started_at).total_seconds()

            # Update run record
            run.status = ConsolidationStage.COMPLETE.value
            run.completed_at = completed_at
            run.duration_seconds = duration
            run.input_memories = len(memories)
            run.memories_consolidated = len(memories)
            run.new_concepts = new_concepts
            run.updated_concepts = updated_concepts
            run.beliefs_updated = beliefs_updated
            run.contradictions_resolved = contradictions_resolved
            run.training_examples_generated = training_examples
            run.dataset_path = dataset_path
            await self.db.commit()

            # Record provenance
            memory_ids = [m.id for m in memories]
            await self.provenance.record_consolidation(
                consolidation_run_id=run_id,
                input_memories=memory_ids,
                output_concepts=[],  # Would track concept IDs
                generation=config.generation,
            )

            return ConsolidationResult(
                run_id=run_id,
                generation=config.generation,
                started_at=started_at,
                completed_at=completed_at,
                input_memories=len(memories),
                memories_consolidated=len(memories),
                new_concepts=new_concepts,
                updated_concepts=updated_concepts,
                beliefs_updated=beliefs_updated,
                contradictions_resolved=contradictions_resolved,
                training_examples_generated=training_examples,
                dataset_path=dataset_path,
                duration_seconds=duration,
                status="success",
            )

        except Exception as e:
            completed_at = datetime.now(timezone.utc)
            duration = (completed_at - started_at).total_seconds()

            run.status = "failed"
            run.completed_at = completed_at
            run.duration_seconds = duration
            run.error = str(e)
            await self.db.commit()

            return ConsolidationResult(
                run_id=run_id,
                generation=config.generation,
                started_at=started_at,
                completed_at=completed_at,
                input_memories=0,
                memories_consolidated=0,
                new_concepts=0,
                updated_concepts=0,
                beliefs_updated=0,
                contradictions_resolved=0,
                training_examples_generated=0,
                dataset_path=None,
                duration_seconds=duration,
                status="failed",
                error=str(e),
            )

    async def _select_memories(self, config: ConsolidationConfig) -> List[EpisodicMemory]:
        """Select memories for consolidation."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=config.lookback_hours)

        stmt = select(EpisodicMemory).where(
            and_(
                EpisodicMemory.generation == config.generation,
                EpisodicMemory.created_at >= cutoff,
                EpisodicMemory.importance_score >= config.min_importance,
                EpisodicMemory.consolidated == False,
            )
        ).order_by(desc(EpisodicMemory.importance_score)).limit(config.max_memories)

        result = await self.db.execute(stmt)
        memories = list(result.scalars().all())

        # Also include some recent lower-importance memories for context
        if len(memories) < config.max_memories // 2:
            stmt = select(EpisodicMemory).where(
                and_(
                    EpisodicMemory.generation == config.generation,
                    EpisodicMemory.created_at >= cutoff,
                    EpisodicMemory.consolidated == False,
                )
            ).order_by(desc(EpisodicMemory.created_at)).limit(config.max_memories - len(memories))

            result = await self.db.execute(stmt)
            additional = list(result.scalars().all())
            memories.extend(additional)

        return memories

    async def _detect_patterns(
        self,
        memories: List[EpisodicMemory],
        config: ConsolidationConfig,
    ) -> Dict[str, Any]:
        """Detect patterns in memories: recurring themes, entities, relations."""
        patterns = {
            "themes": defaultdict(int),
            "entities": defaultdict(int),
            "relations": defaultdict(int),
            "temporal": defaultdict(int),
            "source_distribution": defaultdict(int),
        }

        # Analyze each memory
        for memory in memories:
            # Source distribution
            patterns["source_distribution"][memory.source.value] += 1

            # Temporal patterns (hour of day)
            hour = memory.created_at.hour
            patterns["temporal"][hour] += 1

            # Extract themes/entities using model
            analysis = await self._analyze_memory_content(memory.content)
            for theme in analysis.get("themes", []):
                patterns["themes"][theme] += 1
            for entity in analysis.get("entities", []):
                patterns["entities"][entity] += 1
            for relation in analysis.get("relations", []):
                patterns["relations"][relation] += 1

        # Filter by frequency
        for key in ["themes", "entities", "relations"]:
            patterns[key] = {k: v for k, v in patterns[key].items() if v >= config.min_concept_frequency}

        return patterns

    async def _analyze_memory_content(self, content: str) -> Dict[str, List[str]]:
        """Use model to extract themes, entities, relations from memory."""
        prompt = f"""Extract key themes, entities, and relations from this text. Return as JSON.

Text: {content[:2000]}

Return format:
{{
  "themes": ["theme1", "theme2"],
  "entities": ["entity1", "entity2"],
  "relations": ["relation1", "relation2"]
}}"""

        try:
            response = await self.inference.generate(prompt, max_tokens=300, temperature=0.3)
            import json
            return json.loads(response.strip())
        except Exception:
            return {"themes": [], "entities": [], "relations": []}

    async def _form_concepts(
        self,
        patterns: Dict[str, Any],
        memories: List[EpisodicMemory],
        config: ConsolidationConfig,
    ) -> Tuple[int, int]:
        """Form new semantic concepts from detected patterns."""
        new_concepts = 0
        updated_concepts = 0

        # Process themes as potential concepts
        for theme, frequency in sorted(patterns["themes"].items(), key=lambda x: -x[1])[:config.max_new_concepts]:
            # Find memories related to this theme
            related_memories = [
                m for m in memories
                if theme.lower() in m.content.lower()
            ]

            if len(related_memories) < config.min_concept_frequency:
                continue

            # Check if concept already exists
            existing = await self.semantic_service.find_concept(theme)
            if existing:
                # Update existing concept
                await self._update_concept(existing, related_memories, theme, frequency)
                updated_concepts += 1
            else:
                # Create new concept
                await self._create_concept(theme, related_memories, frequency, config.generation)
                new_concepts += 1

        # Process entities
        for entity, frequency in sorted(patterns["entities"].items(), key=lambda x: -x[1])[:config.max_new_concepts // 2]:
            if frequency < config.min_concept_frequency:
                continue

            related_memories = [
                m for m in memories
                if entity.lower() in m.content.lower()
            ]

            existing = await self.semantic_service.find_concept(entity)
            if existing:
                await self._update_concept(existing, related_memories, entity, frequency)
                updated_concepts += 1
            else:
                await self._create_concept(entity, related_memories, frequency, config.generation)
                new_concepts += 1

        return new_concepts, updated_concepts

    async def _create_concept(
        self,
        concept: str,
        memories: List[EpisodicMemory],
        frequency: int,
        generation: int,
    ):
        """Create a new semantic concept."""
        # Generate description using model
        memory_texts = "\n---\n".join([m.content[:500] for m in memories[:10]])

        prompt = f"""Create a concise definition/description for the concept "{concept}" based on these memories.

Memories:
{memory_texts}

Provide a clear, factual description (2-3 sentences)."""

        try:
            description = await self.inference.generate(prompt, max_tokens=200, temperature=0.4)
        except Exception:
            description = f"Concept derived from {frequency} memories."

        # Find related concepts
        related = await self.semantic_service.find_related_concepts(concept, threshold=0.5, limit=10)
        related_ids = [str(r[0].id) for r in related]

        # Calculate confidence based on frequency and memory quality
        avg_importance = sum(m.importance_score for m in memories) / len(memories)
        confidence = min(0.5 + (frequency / 20.0) + (avg_importance * 0.3), 0.95)

        await self.semantic_service.create_concept(
            concept=concept,
            description=description.strip(),
            source_memories=[m.id for m in memories],
            confidence=confidence,
            generation=generation,
            related_concepts=related_ids,
        )

    async def _update_concept(
        self,
        concept: SemanticMemory,
        memories: List[EpisodicMemory],
        theme: str,
        frequency: int,
    ):
        """Update an existing concept with new evidence."""
        # Add new source memories
        new_memory_ids = [m.id for m in memories if m.id not in (concept.source_memories or [])]
        if new_memory_ids:
            concept.source_memories = (concept.source_memories or []) + new_memory_ids

        # Update description if significant new info
        if frequency > len(concept.source_memories or []) * 0.5:
            memory_texts = "\n---\n".join([m.content[:500] for m in memories[:5]])
            prompt = f"""Update the description for concept "{theme}" with new information.

Current description: {concept.description}

New memories:
{memory_texts}

Provide updated description (2-3 sentences)."""

            try:
                new_description = await self.inference.generate(prompt, max_tokens=200, temperature=0.4)
                concept.description = new_description.strip()
            except Exception:
                pass

        # Update confidence
        concept.access_count += 1
        concept.last_accessed = datetime.now(timezone.utc)
        concept.connection_count = len(concept.related_concepts or [])

        await self.db.commit()

    async def _update_beliefs(
        self,
        memories: List[EpisodicMemory],
        new_concepts: int,
        config: ConsolidationConfig,
    ) -> int:
        """Update beliefs based on consolidated memories."""
        updated = 0

        # Group memories by potential belief topics
        topics = defaultdict(list)
        for memory in memories:
            # Use tags and source to group
            for tag in memory.tags or []:
                topics[tag].append(memory)
            topics[memory.source.value].append(memory)

        # For each topic with enough memories, update or create beliefs
        for topic, topic_memories in topics.items():
            if len(topic_memories) < config.min_belief_evidence:
                continue

            # Extract propositions from memories
            propositions = await self._extract_propositions(topic_memories)

            for prop in propositions:
                # Find or create belief
                belief = await self.belief_system.get_or_create_belief(
                    proposition=prop,
                    category=topic,
                    initial_confidence=0.5,
                    generation=config.generation,
                    evidence=[{
                        "episodic_memory_id": m.id,
                        "source": m.source,
                        "content": m.content[:500],
                        "supports": True,
                        "weight": m.importance_score,
                    } for m in topic_memories[:5]]
                )
                updated += 1

        return updated

    async def _extract_propositions(self, memories: List[EpisodicMemory]) -> List[str]:
        """Extract belief propositions from memories."""
        memory_texts = "\n---\n".join([m.content[:300] for m in memories[:10]])

        prompt = f"""Extract 1-3 clear, factual propositions from these memories.
Each proposition should be a standalone statement that could be true or false.

Memories:
{memory_texts}

Return as JSON array of strings."""

        try:
            response = await self.inference.generate(prompt, max_tokens=300, temperature=0.4)
            import json
            return json.loads(response.strip())
        except Exception:
            return []

    async def _resolve_contradictions(self, config: ConsolidationConfig) -> int:
        """Resolve contradictions detected during consolidation."""
        resolved = 0

        # Find beliefs with contradictions
        contradictions = await self.belief_system.find_contradictions(config.generation)

        for c in contradictions:
            belief_id = uuid.UUID(c["belief_id"])
            # Auto-resolve low-confidence contradictions
            belief = await self.belief_system.get_belief(belief_id)
            if belief and belief.confidence < config.contradiction_threshold:
                # Deprecate low-confidence contradicted beliefs
                await self.belief_system.resolve_contradiction(
                    belief_id=belief_id,
                    resolution="deprecate",
                    reason=f"Auto-resolved during consolidation: confidence {belief.confidence:.2f} below threshold",
                )
                resolved += 1

        return resolved

    async def _generate_training_data(
        self,
        memories: List[EpisodicMemory],
        new_concepts: int,
        config: ConsolidationConfig,
    ) -> Tuple[int, Optional[str]]:
        """Generate training dataset from consolidated knowledge."""
        if new_concepts == 0 and len(memories) < 10:
            return 0, None

        examples = []

        # 1. Memory completion tasks
        for memory in memories[:config.max_training_examples // 4]:
            prompt = memory.content[:200]
            completion = memory.content[200:500]
            if len(completion) > 50:
                examples.append({
                    "type": "memory_completion",
                    "prompt": prompt,
                    "completion": completion,
                    "source_memory": str(memory.id),
                })

        # 2. Concept definition tasks
        stmt = select(SemanticMemory).where(
            SemanticMemory.generation == config.generation
        ).limit(config.max_training_examples // 4)
        result = await self.db.execute(stmt)
        concepts = result.scalars().all()

        for concept in concepts:
            examples.append({
                "type": "concept_definition",
                "prompt": f"Define: {concept.concept}",
                "completion": concept.description,
                "source_concept": str(concept.id),
            })

        # 3. Belief reasoning tasks
        stmt = select(Belief).where(
            and_(
                Belief.generation == config.generation,
                Belief.state == BeliefState.ACTIVE,
            )
        ).limit(config.max_training_examples // 4)
        result = await self.db.execute(stmt)
        beliefs = result.scalars().all()

        for belief in beliefs:
            examples.append({
                "type": "belief_reasoning",
                "prompt": f"Assess: {belief.proposition}",
                "completion": f"Confidence: {belief.confidence:.2f}. {belief.proposition}",
                "source_belief": str(belief.id),
            })

        # 4. Reflection synthesis tasks
        stmt = select(EpisodicMemory).where(
            and_(
                EpisodicMemory.generation == config.generation,
                EpisodicMemory.source == MemorySource.REFLECTION,
            )
        ).limit(config.max_training_examples // 4)
        result = await self.db.execute(stmt)
        reflections = result.scalars().all()

        for reflection in reflections:
            examples.append({
                "type": "reflection_synthesis",
                "prompt": "Reflect on: " + reflection.content[:200],
                "completion": reflection.content,
                "source_memory": str(reflection.id),
            })

        # Save dataset
        if examples:
            dataset_dir = Path(settings.DATASET_ROOT) / f"generation_{config.generation:06d}" / "consolidation"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            dataset_path = dataset_dir / f"consolidation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
            import json
            with open(dataset_path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            return len(examples), str(dataset_path)

        return 0, None

    async def _update_run_status(self, run_id: uuid.UUID, stage: ConsolidationStage):
        """Update consolidation run status."""
        stmt = select(ConsolidationRun).where(ConsolidationRun.id == run_id)
        result = await self.db.execute(stmt)
        run = result.scalar_one_or_none()
        if run:
            run.status = stage.value
            await self.db.commit()


# Celery Tasks

@celery_app.task(bind=True, max_retries=2)
def run_consolidation(self, generation: Optional[int] = None):
    """Run consolidation pipeline as Celery task."""
    import asyncio
    from api.database import get_async_session

    async def _run():
        async for db in get_async_session():
            inference = InferenceEngine(model=None, tokenizer=None, device="cpu")
            pipeline = ConsolidationPipeline(db, inference)

            config = ConsolidationConfig(generation=generation or 1)
            result = await pipeline.run_consolidation(config)

            return {
                "run_id": str(result.run_id),
                "status": result.status,
                "memories": result.input_memories,
                "new_concepts": result.new_concepts,
                "beliefs_updated": result.beliefs_updated,
                "training_examples": result.training_examples_generated,
                "duration": result.duration_seconds,
            }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=300)


@celery_app.task(bind=True)
def run_deep_consolidation(self, generation: int):
    """Run deep consolidation with extended lookback."""
    import asyncio
    from api.database import get_async_session

    async def _run():
        async for db in get_async_session():
            inference = InferenceEngine(model=None, tokenizer=None, device="cpu")
            pipeline = ConsolidationPipeline(db, inference)

            config = ConsolidationConfig(
                generation=generation,
                max_memories=5000,
                min_importance=0.1,
                lookback_hours=168,  # 1 week
                min_concept_frequency=5,
                max_new_concepts=100,
                max_training_examples=50000,
            )
            result = await pipeline.run_consolidation(config)

            return {
                "run_id": str(result.run_id),
                "status": result.status,
                "memories": result.input_memories,
                "new_concepts": result.new_concepts,
                "training_examples": result.training_examples_generated,
            }

    return asyncio.run(_run())


# Beat Schedule for Consolidation
celery_app.conf.beat_schedule.update({
    # Nightly consolidation at 2 AM
    "nightly-consolidation": {
        "task": "consolidation.pipeline.run_consolidation",
        "schedule": crontab(hour=2, minute=0),
    },
    # Weekly deep consolidation on Sundays at 1 AM
    "weekly-deep-consolidation": {
        "task": "consolidation.pipeline.run_deep_consolidation",
        "schedule": crontab(hour=1, minute=0, day_of_week=0),
        "args": [1],  # generation
    },
})

celery_app.conf.timezone = "UTC"