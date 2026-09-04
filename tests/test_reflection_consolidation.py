"""
Tests for Reflection and Consolidation Modules
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from reflection.scheduler import ReflectionScheduler, ReflectionCycle
from consolidation.pipeline import ConsolidationPipeline, ConsolidationResult


class TestReflectionScheduler:
    @pytest.fixture
    def mock_services(self):
        memory = AsyncMock()
        belief = AsyncMock()
        web_search = AsyncMock()
        return memory, belief, web_search

    @pytest.fixture
    def scheduler(self, mock_services):
        memory, belief, web_search = mock_services
        return ReflectionScheduler(memory, belief, web_search)

    def test_reflection_cycle_creation(self):
        cycle = ReflectionCycle(
            cycle_id="cycle-123",
            trigger="scheduled",
            selected_memories=["mem-1", "mem-2"],
            contradictions_found=["contr-1"],
            novel_patterns=["pattern-1"],
            inferences=["inference-1"],
            new_memories_created=["mem-3"],
            beliefs_updated=["belief-1"],
        )
        assert cycle.cycle_id == "cycle-123"
        assert cycle.trigger == "scheduled"
        assert len(cycle.selected_memories) == 2

    @pytest.mark.asyncio
    async def test_select_memories_for_reflection(self, scheduler, mock_services):
        memory, belief, web_search = mock_services

        # Mock recent memories
        mock_memories = [
            MagicMock(id=f"mem-{i}", content=f"Memory {i}", importance=0.5 + i * 0.1)
            for i in range(5)
        ]
        memory.get_recent_memories = AsyncMock(return_value=mock_memories)

        # Mock important memories
        important_memories = [
            MagicMock(id=f"imp-{i}", content=f"Important {i}", importance=0.9)
            for i in range(2)
        ]
        memory.get_important_memories = AsyncMock(return_value=important_memories)

        selected = await scheduler._select_memories_for_reflection()
        assert len(selected) > 0
        # Should include both recent and important
        assert len(selected) <= 10  # max_memories_per_cycle default

    @pytest.mark.asyncio
    async def test_detect_contradictions(self, scheduler, mock_services):
        memory, belief, web_search = mock_services

        # Mock beliefs with potential contradictions
        belief.get_all_propositions = AsyncMock(return_value=[
            MagicMock(id="p1", statement="AI is beneficial", confidence=0.8),
            MagicMock(id="p2", statement="AI poses risks", confidence=0.7),
        ])

        beliefs = await belief.get_all_propositions()
        contradictions = scheduler._detect_contradictions(beliefs)
        # Should detect some contradiction between beneficial and risks
        assert isinstance(contradictions, list)

    @pytest.mark.asyncio
    async def test_run_reflection_cycle(self, scheduler, mock_services):
        memory, belief, web_search = mock_services

        # Mock all the internal methods
        scheduler._select_memories_for_reflection = AsyncMock(return_value=["mem-1", "mem-2"])
        scheduler._detect_contradictions = AsyncMock(return_value=["contr-1"])
        scheduler._detect_novel_patterns = AsyncMock(return_value=["pattern-1"])
        scheduler._generate_inferences = AsyncMock(return_value=["inference-1"])
        scheduler._create_reflection_memories = AsyncMock(return_value=["mem-3"])
        scheduler._update_beliefs = AsyncMock(return_value=["belief-1"])

        cycle = await scheduler.run_reflection_cycle(trigger="test")
        assert isinstance(cycle, ReflectionCycle)
        assert cycle.trigger == "test"
        assert cycle.selected_memories == ["mem-1", "mem-2"]
        assert cycle.contradictions_found == ["contr-1"]


class TestConsolidationPipeline:
    @pytest.fixture
    def mock_services(self):
        episodic = AsyncMock()
        semantic = AsyncMock()
        belief = AsyncMock()
        return episodic, semantic, belief

    @pytest.fixture
    def pipeline(self, mock_services):
        episodic, semantic, belief = mock_services
        return ConsolidationPipeline(episodic, semantic, belief)

    def test_consolidation_result_creation(self):
        result = ConsolidationResult(
            cycle_id="cons-123",
            memories_processed=100,
            concepts_extracted=10,
            concepts_new=5,
            concepts_updated=3,
            beliefs_updated=2,
            contradictions_resolved=1,
            training_examples_generated=50,
            duration_seconds=30.5,
        )
        assert result.memories_processed == 100
        assert result.concepts_new == 5

    @pytest.mark.asyncio
    async def test_select_memories_for_consolidation(self, pipeline, mock_services):
        episodic, semantic, belief = mock_services

        # Mock unconsolidated memories
        memories = [
            MagicMock(id=f"mem-{i}", content=f"Content {i}", importance=0.6, created_at=datetime.utcnow())
            for i in range(20)
        ]
        episodic.get_unconsolidated_memories = AsyncMock(return_value=memories)

        selected = await pipeline._select_memories_for_consolidation(limit=10)
        assert len(selected) == 10
        # Should prioritize by importance
        assert selected[0].importance >= selected[-1].importance

    @pytest.mark.asyncio
    async def test_detect_patterns(self, pipeline, mock_services):
        episodic, semantic, belief = mock_services

        memories = [
            MagicMock(id=f"mem-{i}", content=f"AI and machine learning {i}")
            for i in range(10)
        ]

        patterns = await pipeline._detect_patterns(memories)
        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_form_concepts(self, pipeline, mock_services):
        episodic, semantic, belief = mock_services

        patterns = [
            {"theme": "AI", "keywords": ["AI", "machine learning", "neural"], "supporting_memories": ["m1", "m2"]},
            {"theme": "Safety", "keywords": ["safety", "alignment", "risk"], "supporting_memories": ["m3"]},
        ]

        # Mock semantic service
        semantic.get_concept_by_name = AsyncMock(return_value=None)
        semantic.create_concept = AsyncMock(return_value=MagicMock(id="concept-1"))

        concepts = await pipeline._form_concepts(patterns)
        assert len(concepts) == 2
        assert semantic.create_concept.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_training_data(self, pipeline, mock_services):
        episodic, semantic, belief = mock_services

        concepts = [
            MagicMock(id="c1", name="AI", description="Artificial Intelligence", confidence=0.9),
            MagicMock(id="c2", name="Safety", description="AI Safety", confidence=0.8),
        ]

        training_data = await pipeline._generate_training_data(concepts)
        assert isinstance(training_data, list)
        # Should generate QA pairs for each concept
        assert len(training_data) >= 2

    @pytest.mark.asyncio
    async def test_run_consolidation_cycle(self, pipeline, mock_services):
        episodic, semantic, belief = mock_services

        # Mock all internal methods
        pipeline._select_memories_for_consolidation = AsyncMock(return_value=[MagicMock(id="m1")] * 10)
        pipeline._detect_patterns = AsyncMock(return_value=[{"theme": "test"}])
        pipeline._form_concepts = AsyncMock(return_value=[MagicMock(id="c1")] * 2)
        pipeline._update_beliefs = AsyncMock(return_value=1)
        pipeline._resolve_contradictions = AsyncMock(return_value=1)
        pipeline._generate_training_data = AsyncMock(return_value=["ex1", "ex2"] * 25)
        pipeline._save_training_data = AsyncMock()

        result = await pipeline.run_consolidation_cycle()
        assert isinstance(result, ConsolidationResult)
        assert result.memories_processed == 10
        assert result.concepts_extracted == 2
        assert result.training_examples_generated == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])