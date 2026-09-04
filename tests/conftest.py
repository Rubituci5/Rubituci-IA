"""
Pytest Configuration and Fixtures
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    session.delete = AsyncMock()
    session.scalar = AsyncMock()
    session.scalars = AsyncMock()
    return session


@pytest.fixture
def mock_embedding():
    """Mock embedding vector."""
    import numpy as np
    return np.random.randn(384).astype(np.float32)


@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    return {
        "content": "This is a test memory about AI and machine learning.",
        "source_type": "interaction",
        "source_id": "test-source-123",
        "importance": 0.75,
        "metadata": {"topic": "AI", "language": "en"},
    }


@pytest.fixture
def sample_belief_data():
    """Sample belief data for testing."""
    return {
        "statement": "Machine learning models can generalize from training data.",
        "confidence": 0.85,
        "evidence_ids": ["ev-1", "ev-2"],
        "domain": "AI",
    }


@pytest.fixture
def sample_research_query():
    """Sample research query."""
    return {
        "query": "What are the latest developments in AI alignment?",
        "depth": 2,
        "breadth": 3,
    }


# Async test utilities
class AsyncMockIterator:
    """Helper to make async iterators for testing."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


def async_mock_return(value):
    """Create an async mock that returns a value."""
    mock = AsyncMock()
    mock.return_value = value
    return mock


def async_mock_side_effect(*values):
    """Create an async mock with side effects."""
    mock = AsyncMock()
    mock.side_effect = values
    return mock


# Test markers
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_db: Tests requiring database")