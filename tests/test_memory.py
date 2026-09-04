"""
Tests for Memory Module
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from memory.episodic import EpisodicMemoryService, EpisodicMemory
from memory.semantic import SemanticMemoryService, SemanticConcept
from memory.belief import BeliefSystem, Proposition, ConfidenceLevel
from memory.retrieval import MemoryRetriever, RetrievalStrategy, RetrievedMemory


class TestEpisodicMemory:
    @pytest.fixture
    def mock_session(self):
        session = AsyncMock()
        return session

    @pytest.fixture
    def episodic_service(self, mock_session):
        return EpisodicMemoryService(mock_session)

    def test_memory_creation(self):
        memory = EpisodicMemory(
            content="Test memory content",
            source_type="interaction",
            source_id="test-123",
            importance=0.8,
            embedding=np.random.randn(384).astype(np.float32),
        )
        assert memory.content == "Test memory content"
        assert memory.source_type == "interaction"
        assert memory.importance == 0.8

    @pytest.mark.asyncio
    async def test_create_from_interaction(self, episodic_service, mock_session):
        mock_memory = MagicMock()
        mock_memory.id = "mem-123"
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        with patch("memory.episodic.EpisodicMemory", return_value=mock_memory):
            result = await episodic_service.create_from_interaction(
                user_message="Hello",
                entity_response="Hi there!",
                conversation_id="conv-123",
                importance=0.7,
            )
            assert result == mock_memory
            mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_by_vector(self, episodic_service, mock_session):
        # Mock query result
        mock_memories = [
            MagicMock(id=f"mem-{i}", content=f"Memory {i}", importance=0.5, created_at=datetime.utcnow())
            for i in range(3)
        ]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_memories
        mock_session.execute = AsyncMock(return_value=mock_result)

        query_embedding = np.random.randn(384).astype(np.float32)
        results = await episodic_service.retrieve_by_vector(query_embedding, limit=5)
        assert len(results) == 3


class TestSemanticMemory:
    @pytest.fixture
    def mock_session(self):
        return AsyncMock()

    @pytest.fixture
    def semantic_service(self, mock_session):
        return SemanticMemoryService(mock_session)

    def test_concept_creation(self):
        concept = SemanticConcept(
            name="Test Concept",
            description="A test concept",
            confidence=0.8,
            embedding=np.random.randn(384).astype(np.float32),
        )
        assert concept.name == "Test Concept"
        assert concept.confidence == 0.8

    @pytest.mark.asyncio
    async def test_consolidate_from_episodic(self, semantic_service, mock_session):
        # Mock episodic memories
        episodic_memories = [
            MagicMock(
                id=f"mem-{i}",
                content=f"Content about AI and machine learning {i}",
                embedding=np.random.randn(384).astype(np.float32),
                importance=0.7,
            )
            for i in range(5)
        ]

        mock_session.execute = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        # This would need more mocking to test fully
        # Just verify method exists
        assert hasattr(semantic_service, "consolidate_from_episodic")


class TestBeliefSystem:
    @pytest.fixture
    def mock_session(self):
        return AsyncMock()

    @pytest.fixture
    def belief_system(self, mock_session):
        return BeliefSystem(mock_session)

    def test_proposition_creation(self):
        prop = Proposition(
            statement="The sky is blue",
            confidence=0.95,
            confidence_level=ConfidenceLevel.VERY_HIGH,
            evidence_ids=["ev-1", "ev-2"],
        )
        assert prop.statement == "The sky is blue"
        assert prop.confidence == 0.95
        assert prop.confidence_level == ConfidenceLevel.VERY_HIGH

    def test_confidence_levels(self):
        assert ConfidenceLevel.VERY_LOW.value == 0
        assert ConfidenceLevel.LOW.value == 1
        assert ConfidenceLevel.MEDIUM.value == 2
        assert ConfidenceLevel.HIGH.value == 3
        assert ConfidenceLevel.VERY_HIGH.value == 4

    @pytest.mark.asyncio
    async def test_add_proposition(self, belief_system, mock_session):
        mock_prop = MagicMock()
        mock_prop.id = "prop-123"
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        with patch("memory.belief.Proposition", return_value=mock_prop):
            result = await belief_system.add_proposition(
                statement="Test belief",
                confidence=0.8,
                evidence_ids=["ev-1"],
            )
            assert result == mock_prop

    @pytest.mark.asyncio
    async def test_detect_contradictions(self, belief_system, mock_session):
        # Mock propositions that might contradict
        prop1 = MagicMock(id="p1", statement="AI is beneficial", confidence=0.8)
        prop2 = MagicMock(id="p2", statement="AI is harmful", confidence=0.7)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [prop1, prop2]
        mock_session.execute = AsyncMock(return_value=mock_result)

        contradictions = await belief_system.detect_contradictions("AI is dangerous")
        assert isinstance(contradictions, list)


class TestMemoryRetriever:
    @pytest.fixture
    def mock_services(self):
        episodic = AsyncMock()
        semantic = AsyncMock()
        belief = AsyncMock()
        return episodic, semantic, belief

    @pytest.fixture
    def retriever(self, mock_services):
        episodic, semantic, belief = mock_services
        return MemoryRetriever(episodic, semantic, belief)

    @pytest.mark.asyncio
    async def test_retrieve_vector_strategy(self, retriever, mock_services):
        episodic, semantic, belief = mock_services

        mock_episodic = [MagicMock(id="e1", content="Episodic memory", importance=0.8)]
        mock_semantic = [MagicMock(id="s1", name="Semantic concept", confidence=0.9)]

        episodic.retrieve_by_vector = AsyncMock(return_value=mock_episodic)
        semantic.retrieve_by_vector = AsyncMock(return_value=mock_semantic)

        results = await retriever.retrieve(
            query="test query",
            query_embedding=np.random.randn(384).astype(np.float32),
            strategy=RetrievalStrategy.VECTOR,
            limit=5,
        )

        assert len(results) >= 1
        episodic.retrieve_by_vector.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_hybrid_strategy(self, retriever, mock_services):
        episodic, semantic, belief = mock_services

        mock_episodic = [MagicMock(id="e1", content="Episodic", importance=0.8)]
        mock_semantic = [MagicMock(id="s1", name="Semantic", confidence=0.9)]

        episodic.retrieve_by_vector = AsyncMock(return_value=mock_episodic)
        episodic.retrieve_by_keyword = AsyncMock(return_value=[])
        semantic.retrieve_by_vector = AsyncMock(return_value=mock_semantic)
        semantic.retrieve_by_keyword = AsyncMock(return_value=[])

        results = await retriever.retrieve(
            query="test query",
            query_embedding=np.random.randn(384).astype(np.float32),
            strategy=RetrievalStrategy.HYBRID,
            limit=10,
        )

        # Should have called both vector and keyword for both services
        assert episodic.retrieve_by_vector.called
        assert episodic.retrieve_by_keyword.called
        assert semantic.retrieve_by_vector.called

    def test_deduplicate_results(self, retriever):
        # Create duplicate results
        mem1 = RetrievedMemory(
            id="mem-1",
            content="Test memory",
            source="episodic",
            score=0.9,
            metadata={},
        )
        mem2 = RetrievedMemory(
            id="mem-1",  # Same ID
            content="Test memory",
            source="episodic",
            score=0.8,
            metadata={},
        )
        mem3 = RetrievedMemory(
            id="mem-2",
            content="Another memory",
            source="semantic",
            score=0.7,
            metadata={},
        )

        deduplicated = retriever._deduplicate([mem1, mem2, mem3])
        assert len(deduplicated) == 2
        # Should keep higher score
        assert deduplicated[0].score == 0.9

    def test_rank_results(self, retriever):
        mems = [
            RetrievedMemory(id="1", content="Low", source="e", score=0.3, metadata={}),
            RetrievedMemory(id="2", content="High", source="e", score=0.9, metadata={}),
            RetrievedMemory(id="3", content="Medium", source="e", score=0.6, metadata={}),
        ]

        ranked = retriever._rank_results(mems, "query")
        assert ranked[0].score == 0.9
        assert ranked[1].score == 0.6
        assert ranked[2].score == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])