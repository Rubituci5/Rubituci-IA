"""
Memory Retrieval System

Unified retrieval interface for episodic and semantic memories.
Supports vector search, keyword search, and hybrid retrieval.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
import math

from api.models import EpisodicMemory, SemanticMemory, MemorySource
from memory.episodic import EpisodicMemoryService
from memory.semantic import SemanticMemoryService
from api.config import settings


class RetrievalStrategy(str, Enum):
    """Memory retrieval strategies."""
    VECTOR = "vector"
    KEYWORD = "keyword"
    RECENT = "recent"
    IMPORTANT = "important"
    HYBRID = "hybrid"
    ASSOCIATIVE = "associative"


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    memory_id: uuid.UUID
    memory_type: str  # "episodic" or "semantic"
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[MemorySource] = None
    timestamp: Optional[datetime] = None
    importance: float = 0.0
    confidence: float = 0.0


def _cosine_similarity(left, right) -> float:
    if left is None or right is None:
        return 0.0
    a, b = list(left), list(right)
    denominator = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 10
    min_score: float = 0.0
    generation: Optional[int] = None
    include_episodic: bool = True
    include_semantic: bool = True
    source_filter: Optional[List[MemorySource]] = None
    time_range: Optional[tuple[datetime, datetime]] = None


class MemoryRetriever:
    """
    Unified memory retrieval service.

    Combines episodic and semantic memory retrieval with multiple strategies.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.episodic_service = EpisodicMemoryService(db)
        self.semantic_service = SemanticMemoryService(db)

    async def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant memories for a query.

        Args:
            query: Text query
            query_embedding: Optional embedding vector for semantic search
            config: Retrieval configuration

        Returns:
            List of retrieval results ranked by relevance
        """
        config = config or RetrievalConfig()
        results = []

        # Determine which strategies to use
        strategies = [config.strategy] if config.strategy != RetrievalStrategy.HYBRID else [
            RetrievalStrategy.VECTOR,
            RetrievalStrategy.KEYWORD,
            RetrievalStrategy.RECENT,
            RetrievalStrategy.IMPORTANT,
        ]

        for strategy in strategies:
            if strategy == RetrievalStrategy.VECTOR and query_embedding:
                results.extend(await self._vector_search(query_embedding, config))
            elif strategy == RetrievalStrategy.KEYWORD:
                results.extend(await self._keyword_search(query, config))
            elif strategy == RetrievalStrategy.RECENT:
                results.extend(await self._recent_search(config))
            elif strategy == RetrievalStrategy.IMPORTANT:
                results.extend(await self._important_search(config))
            elif strategy == RetrievalStrategy.ASSOCIATIVE:
                results.extend(await self._associative_search(query, config))

        # Deduplicate and rank
        results = self._deduplicate_and_rank(results, config.top_k)

        return results

    async def _vector_search(
        self,
        query_embedding: List[float],
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        """Vector similarity search."""
        results = []

        if config.include_episodic:
            episodic = await self.episodic_service.retrieve_similar(
                query_embedding,
                top_k=config.top_k,
                generation=config.generation,
            )
            for mem in episodic:
                if config.source_filter and mem.source not in config.source_filter:
                    continue
                results.append(RetrievalResult(
                    memory_id=mem.id,
                    memory_type="episodic",
                    content=mem.content,
                    score=_cosine_similarity(mem.embedding, query_embedding) if mem.embedding else 0.5,
                    metadata={
                        "event_id": mem.event_id,
                        "participants": mem.participants,
                        "context": mem.context,
                    },
                    source=mem.source,
                    timestamp=mem.timestamp,
                    importance=mem.importance,
                    confidence=mem.confidence,
                ))

        if config.include_semantic:
            semantic = await self.semantic_service.retrieve_similar(
                query_embedding,
                top_k=config.top_k,
                generation=config.generation,
            )
            for mem in semantic:
                results.append(RetrievalResult(
                    memory_id=mem.id,
                    memory_type="semantic",
                    content=mem.representation,
                    score=_cosine_similarity(mem.embedding, query_embedding) if mem.embedding else 0.5,
                    metadata={
                        "concept": mem.concept,
                        "category": mem.category,
                        "evidence_count": mem.evidence_count,
                    },
                    timestamp=mem.last_reinforced,
                    importance=mem.confidence,
                    confidence=mem.confidence,
                ))

        return results

    async def _keyword_search(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        """Keyword-based search."""
        results = []
        keywords = query.lower().split()

        if config.include_episodic:
            episodic = await self.episodic_service.retrieve_for_context(
                query,
                top_k=config.top_k,
                generation=config.generation,
            )
            for mem in episodic:
                if config.source_filter and mem.source not in config.source_filter:
                    continue
                # Simple keyword scoring
                score = sum(1 for kw in keywords if kw in mem.content.lower()) / max(len(keywords), 1)
                if score >= config.min_score:
                    results.append(RetrievalResult(
                        memory_id=mem.id,
                        memory_type="episodic",
                        content=mem.content,
                        score=score * mem.importance,
                        metadata={
                            "event_id": mem.event_id,
                            "participants": mem.participants,
                            "context": mem.context,
                        },
                        source=mem.source,
                        timestamp=mem.timestamp,
                        importance=mem.importance,
                        confidence=mem.confidence,
                    ))

        return results

    async def _recent_search(
        self,
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        """Retrieve recent memories."""
        results = []

        if config.include_episodic:
            episodic = await self.episodic_service.retrieve_recent(
                limit=config.top_k,
                min_importance=config.min_score,
                generation=config.generation,
            )
            for mem in episodic:
                if config.source_filter and mem.source not in config.source_filter:
                    continue
                if config.time_range:
                    if not (config.time_range[0] <= mem.timestamp <= config.time_range[1]):
                        continue
                results.append(RetrievalResult(
                    memory_id=mem.id,
                    memory_type="episodic",
                    content=mem.content,
                    score=mem.importance * 0.5,  # Recency weighted by importance
                    metadata={
                        "event_id": mem.event_id,
                        "participants": mem.participants,
                    },
                    source=mem.source,
                    timestamp=mem.timestamp,
                    importance=mem.importance,
                    confidence=mem.confidence,
                ))

        return results

    async def _important_search(
        self,
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        """Retrieve important memories."""
        results = []

        if config.include_episodic:
            episodic = await self.episodic_service.retrieve_important(
                limit=config.top_k,
                min_importance=max(0.5, config.min_score),
                generation=config.generation,
            )
            for mem in episodic:
                if config.source_filter and mem.source not in config.source_filter:
                    continue
                results.append(RetrievalResult(
                    memory_id=mem.id,
                    memory_type="episodic",
                    content=mem.content,
                    score=mem.importance,
                    metadata={
                        "event_id": mem.event_id,
                        "participants": mem.participants,
                    },
                    source=mem.source,
                    timestamp=mem.timestamp,
                    importance=mem.importance,
                    confidence=mem.confidence,
                ))

        if config.include_semantic:
            semantic = await self.semantic_service.retrieve_high_confidence(
                limit=config.top_k,
                min_confidence=max(0.5, config.min_score),
                generation=config.generation,
            )
            for mem in semantic:
                results.append(RetrievalResult(
                    memory_id=mem.id,
                    memory_type="semantic",
                    content=mem.representation,
                    score=mem.confidence,
                    metadata={
                        "concept": mem.concept,
                        "category": mem.category,
                    },
                    timestamp=mem.last_reinforced,
                    importance=mem.confidence,
                    confidence=mem.confidence,
                ))

        return results

    async def _associative_search(
        self,
        query: str,
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        """Associative retrieval - find memories linked to query concepts."""
        results = []

        # Find semantic memories matching query concepts
        keywords = query.lower().split()
        for kw in keywords[:3]:
            semantic = await self.semantic_service.find_related(kw, limit=5, generation=config.generation)
            for mem in semantic:
                # Get supporting episodes
                episodes = await self.semantic_service.get_supporting_episodes(mem.id, limit=3)
                for ep in episodes:
                    if config.source_filter and ep.source not in config.source_filter:
                        continue
                    results.append(RetrievalResult(
                        memory_id=ep.id,
                        memory_type="episodic",
                        content=ep.content,
                        score=mem.confidence * 0.8,
                        metadata={
                            "event_id": ep.event_id,
                            "associated_concept": mem.concept,
                        },
                        source=ep.source,
                        timestamp=ep.timestamp,
                        importance=ep.importance,
                        confidence=ep.confidence,
                    ))

        return results

    def _deduplicate_and_rank(
        self,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Deduplicate by memory_id and rank by score."""
        seen = set()
        unique = []
        for r in results:
            if r.memory_id not in seen:
                seen.add(r.memory_id)
                unique.append(r)

        # Sort by score descending
        unique.sort(key=lambda x: x.score, reverse=True)

        return unique[:top_k]

    async def retrieve_for_generation(
        self,
        prompt: str,
        query_embedding: Optional[List[float]] = None,
        max_memories: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve memories optimized for generation context.

        Prioritizes:
        1. High-importance episodic memories
        2. High-confidence semantic memories
        3. Recent relevant interactions
        """
        config = RetrievalConfig(
            strategy=RetrievalStrategy.HYBRID,
            top_k=max_memories * 2,
            min_score=0.3,
        )

        results = await self.retrieve(prompt, query_embedding, config)

        # Balance episodic vs semantic
        episodic_results = [r for r in results if r.memory_type == "episodic"]
        semantic_results = [r for r in results if r.memory_type == "semantic"]

        # Interleave: 60% episodic, 40% semantic
        balanced = []
        ep_idx = sem_idx = 0
        target_ep = int(max_memories * 0.6)
        target_sem = max_memories - target_ep

        while len(balanced) < max_memories and (ep_idx < len(episodic_results) or sem_idx < len(semantic_results)):
            if ep_idx < target_ep and ep_idx < len(episodic_results):
                balanced.append(episodic_results[ep_idx])
                ep_idx += 1
            elif sem_idx < target_sem and sem_idx < len(semantic_results):
                balanced.append(semantic_results[sem_idx])
                sem_idx += 1
            else:
                break

        return balanced

    async def get_context_summary(
        self,
        results: List[RetrievalResult],
        max_tokens: int = 1000,
    ) -> str:
        """Format retrieval results as context for generation."""
        if not results:
            return ""

        parts = ["[Relevant Memories]"]
        token_count = 0

        for r in results:
            prefix = f"[{r.memory_type.capitalize()}"
            if r.source:
                prefix += f": {r.source.value}"
            prefix += f" | Importance: {r.importance:.2f} | Confidence: {r.confidence:.2f}] "

            content = f"{prefix}{r.content}"
            # Rough token estimate
            est_tokens = len(content) // 4

            if token_count + est_tokens > max_tokens:
                break

            parts.append(content)
            token_count += est_tokens

        return "\n".join(parts)
