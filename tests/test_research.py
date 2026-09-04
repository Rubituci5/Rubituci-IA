"""
Tests for Research Module
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

from research.web_search import WebSearchService, WebSearchProvider, SearchResult, WebQuery
from research.browser import BrowserService, BrowserSession, AutonomousBrowser
from research.provenance import ProvenanceTracker, SourceType, SourceCitation


class TestWebSearch:
    @pytest.fixture
    def mock_session(self):
        return AsyncMock()

    @pytest.fixture
    def web_search(self, mock_session):
        with patch("research.web_search.aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__.return_value = mock_session
            service = WebSearchService()
            service.session = mock_session
            return service

    def test_search_result_creation(self):
        result = SearchResult(
            title="Test Result",
            url="https://example.com",
            snippet="This is a test snippet",
            source="duckduckgo",
            rank=1,
        )
        assert result.title == "Test Result"
        assert result.url == "https://example.com"
        assert result.rank == 1

    def test_web_query_creation(self):
        query = WebQuery(query="test query", provider=WebSearchProvider.DUCKDUCKGO, max_results=10)
        assert query.query == "test query"
        assert query.provider == WebSearchProvider.DUCKDUCKGO
        assert query.max_results == 10

    @pytest.mark.asyncio
    async def test_search_duckduckgo(self, web_search, mock_session):
        # Mock HTML response
        mock_response = MagicMock()
        mock_response.text = AsyncMock(return_value="""
        <html>
            <body>
                <div class="result">
                    <a class="result__url" href="https://example.com">example.com</a>
                    <a class="result__snippet" href="https://example.com">Test snippet</a>
                </div>
            </body>
        </html>
        """)
        mock_response.status = 200
        mock_session.get.return_value.__aenter__.return_value = mock_response

        results = await web_search.search("test query", max_results=5)
        assert isinstance(results, list)
        mock_session.get.assert_called()

    @pytest.mark.asyncio
    async def test_autonomous_research(self, web_search):
        # Mock search and follow-up
        web_search.search = AsyncMock(return_value=[
            SearchResult(title="Result 1", url="https://ex1.com", snippet="Snippet 1", source="ddg", rank=1),
            SearchResult(title="Result 2", url="https://ex2.com", snippet="Snippet 2", source="ddg", rank=2),
        ])
        web_search.fetch_page = AsyncMock(return_value="Full page content")

        report = await web_search.autonomous_research("AI safety", depth=2, breadth=2)
        assert "topic" in report
        assert "findings" in report
        assert "sources" in report


class TestBrowser:
    @pytest.fixture
    def browser_service(self):
        return BrowserService()

    def test_session_creation(self, browser_service):
        session = browser_service.create_session("session-1")
        assert session.session_id == "session-1"
        assert session.current_url is None
        assert len(session.history) == 0

    def test_navigate(self, browser_service):
        session = browser_service.create_session("session-1")
        # Mock the fetch
        browser_service._fetch_page = AsyncMock(return_value=("Page content", "https://example.com"))

        import asyncio
        result = asyncio.run(session.navigate("https://example.com"))
        assert result == "Page content"
        assert session.current_url == "https://example.com"
        assert len(session.history) == 1

    def test_back_forward(self, browser_service):
        session = browser_service.create_session("session-1")
        session.history = [
            {"url": "https://a.com", "title": "A"},
            {"url": "https://b.com", "title": "B"},
            {"url": "https://c.com", "title": "C"},
        ]
        session.current_index = 2
        session.current_url = "https://c.com"

        # Go back
        session.go_back()
        assert session.current_url == "https://b.com"
        assert session.current_index == 1

        # Go forward
        session.go_forward()
        assert session.current_url == "https://c.com"
        assert session.current_index == 2

    def test_extract_links(self, browser_service):
        html = """
        <html>
            <body>
                <a href="https://example.com">Link 1</a>
                <a href="/relative">Relative</a>
                <a href="javascript:void(0)">JS</a>
            </body>
        </html>
        """
        links = browser_service.extract_links(html, "https://base.com")
        assert len(links) == 2  # javascript filtered out
        assert "https://example.com" in links
        assert "https://base.com/relative" in links

    def test_sanitize_content(self, browser_service):
        html = """
        <html>
            <head><script>alert('xss')</script></head>
            <body>
                <div class="content">Main content</div>
                <nav>Navigation</nav>
                <footer>Footer</footer>
            </body>
        </html>
        """
        sanitized = browser_service.sanitize_content(html)
        assert "alert" not in sanitized
        assert "Main content" in sanitized
        assert "Navigation" not in sanitized  # nav removed
        assert "Footer" not in sanitized  # footer removed


class TestAutonomousBrowser:
    @pytest.fixture
    def autonomous_browser(self):
        browser = BrowserService()
        return AutonomousBrowser(browser)

    @pytest.mark.asyncio
    async def test_research_cycle(self, autonomous_browser):
        # Mock browser methods
        autonomous_browser.browser._fetch_page = AsyncMock(return_value=("Content about AI", "https://ex.com"))
        autonomous_browser.browser.extract_links = MagicMock(return_value=["https://ex.com/link1"])
        autonomous_browser.web_search = AsyncMock()
        autonomous_browser.web_search.search = AsyncMock(return_value=[])
        autonomous_browser.web_search.fetch_page = AsyncMock(return_value="Content")

        findings = await autonomous_browser.research_cycle("AI", max_pages=3)
        assert isinstance(findings, list)


class TestProvenanceTracker:
    @pytest.fixture
    def provenance(self):
        return ProvenanceTracker()

    def test_create_web_source(self, provenance):
        source = provenance.create_web_source(
            url="https://example.com",
            title="Test Page",
            content="Page content",
            query="test query",
        )
        assert source.source_type == SourceType.WEB
        assert source.url == "https://example.com"
        assert source.title == "Test Page"
        assert source.content_hash is not None

    def test_create_citation(self, provenance):
        source = provenance.create_web_source(
            url="https://example.com",
            title="Test",
            content="Content",
        )
        citation = provenance.create_citation(source, "Relevant excerpt", confidence=0.9)
        assert citation.source_id == source.source_id
        assert citation.excerpt == "Relevant excerpt"
        assert citation.confidence == 0.9

    def test_lineage_tracking(self, provenance):
        # Create sources
        source1 = provenance.create_web_source(url="https://a.com", title="A", content="A content")
        source2 = provenance.create_web_source(url="https://b.com", title="B", content="B content")

        # Create derived memory with both sources
        memory_id = provenance.track_lineage(
            derived_id="mem-123",
            source_ids=[source1.source_id, source2.source_id],
            derivation_type="consolidation",
        )
        assert memory_id == "mem-123"

        # Get lineage
        lineage = provenance.get_lineage("mem-123")
        assert len(lineage) == 2
        assert source1.source_id in lineage
        assert source2.source_id in lineage

    def test_verify_integrity(self, provenance):
        source = provenance.create_web_source(
            url="https://example.com",
            title="Test",
            content="Original content",
        )
        # Verify with same content
        assert provenance.verify_integrity(source.source_id, "Original content") is True
        # Verify with different content
        assert provenance.verify_integrity(source.source_id, "Modified content") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])