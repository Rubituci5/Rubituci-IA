"""
Web Search Service

Autonomous web search capabilities for the entity.
Supports multiple search providers and tracks full provenance.
"""

from __future__ import annotations

import uuid
import asyncio
import hashlib
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from enum import Enum

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from api.models import WebQuery, WebSource, MemorySource
from api.config import settings
from api.security import ContainmentPolicy


class SearchProvider(str, Enum):
    """Supported search providers."""
    DUCKDUCKGO = "duckduckgo"
    BING = "bing"
    GOOGLE = "google"
    SEARXNG = "searxng"


@dataclass
class SearchResult:
    """A single search result."""
    url: str
    title: str
    snippet: str
    domain: str
    rank: int
    source_type: Optional[str] = None
    language: str = "en"


@dataclass
class SearchConfig:
    """Configuration for web search."""
    provider: SearchProvider = SearchProvider.DUCKDUCKGO
    max_results: int = 5
    timeout_seconds: int = 10
    user_agent: str = "Entity/0.1.0 (+https://entity.example.com)"
    rate_limit_per_minute: int = 30
    safe_search: bool = True
    region: str = "wt-wt"  # Worldwide


def normalize_search_result_url(url: str) -> str:
    """Unwrap provider redirect links and return the original public result URL."""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            decoded = unquote(target)
            if urlparse(decoded).scheme in {"http", "https"}:
                return decoded
    return url


class WebSearchService:
    """
    Autonomous web search service.

    Features:
    - Multiple search provider support
    - Rate limiting
    - Provenance tracking
    - Content extraction
    - Autonomous query generation
    """

    def __init__(self, db: AsyncSession, config: Optional[SearchConfig] = None):
        self.db = db
        self.config = config or SearchConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._request_times: List[float] = []

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_seconds),
            headers={"User-Agent": self.config.user_agent},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                headers={"User-Agent": self.config.user_agent},
                follow_redirects=True,
            )
        return self._client

    async def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        now = asyncio.get_event_loop().time()
        window_start = now - 60  # 1 minute window

        # Clean old requests
        self._request_times = [t for t in self._request_times if t > window_start]

        if len(self._request_times) >= self.config.rate_limit_per_minute:
            # Wait until we can make a request
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self._request_times.append(now)

    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        generation: int = 1,
        trigger_type: str = "autonomous",
    ) -> List[SearchResult]:
        """
        Perform a web search.

        Args:
            query: Search query
            max_results: Maximum results (overrides config)
            generation: Current model generation
            trigger_type: What triggered this search (autonomous, user_requested, reflection)

        Returns:
            List of search results
        """
        await self._rate_limit()

        max_results = max_results or self.config.max_results

        # Record the query
        web_query = WebQuery(
            query=query,
            trigger_type=trigger_type,
            generation=generation,
        )
        self.db.add(web_query)
        await self.db.flush()

        # Perform search based on provider
        if self.config.provider == SearchProvider.DUCKDUCKGO:
            results = await self._search_duckduckgo(query, max_results)
        elif self.config.provider == SearchProvider.BING:
            results = await self._search_bing(query, max_results)
        elif self.config.provider == SearchProvider.GOOGLE:
            results = await self._search_google(query, max_results)
        elif self.config.provider == SearchProvider.SEARXNG:
            results = await self._search_searxng(query, max_results)
        else:
            results = []

        # Save results as web sources
        for i, result in enumerate(results):
            source = WebSource(
                query_id=web_query.id,
                url=result.url,
                domain=result.domain,
                title=result.title,
                snippet=result.snippet,
                source_type=result.source_type,
                language=result.language,
                credibility=0.5,  # Default, updated after content fetch
            )
            self.db.add(source)

        web_query.results_count = len(results)
        await self.db.flush()

        return results

    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo HTML."""
        client = await self._get_client()

        # DuckDuckGo HTML endpoint
        params = {
            "q": query,
            "kl": self.config.region,
            "safe": "on" if self.config.safe_search else "off",
        }

        try:
            response = await client.get("https://html.duckduckgo.com/html/", params=params)
            response.raise_for_status()
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        # Parse results
        for i, result_div in enumerate(soup.select(".result__snippet")):
            if i >= max_results:
                break

            # Get parent result container
            container = result_div.find_parent("div", class_="result")
            if not container:
                continue

            # Extract title and URL
            title_elem = container.select_one(".result__title a")
            if not title_elem:
                continue

            title = title_elem.get_text(strip=True)
            url = normalize_search_result_url(title_elem.get("href", ""))

            # Extract snippet
            snippet = result_div.get_text(strip=True)

            # Parse domain
            domain = urlparse(url).netloc if url else ""

            results.append(SearchResult(
                url=url,
                title=title,
                snippet=snippet,
                domain=domain,
                rank=i + 1,
                source_type="web",
            ))

        return results

    async def _search_bing(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Bing (requires API key)."""
        # Placeholder - would need Bing API key
        return []

    async def _search_google(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Google (requires API key)."""
        # Placeholder - would need Google Custom Search API
        return []

    async def _search_searxng(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using SearXNG instance."""
        # Placeholder - would need SearXNG instance URL
        return []

    async def fetch_page(
        self,
        url: str,
        query_id: Optional[uuid.UUID] = None,
    ) -> Optional["PageContent"]:
        """
        Fetch and extract content from a web page.

        Args:
            url: URL to fetch
            query_id: Optional associated query ID

        Returns:
            PageContent with extracted text and metadata
        """
        # Check containment policy - entity can READ web but not EXECUTE
        if ContainmentPolicy.is_forbidden("fetch_web_page"):
            return None

        await self._rate_limit()
        client = await self._get_client()

        try:
            response = await client.get(url)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return None

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts, styles, ads
        for elem in soup(["script", "style", "nav", "footer", "aside", "iframe", "noscript"]):
            elem.decompose()

        # Extract main content
        # Try common content selectors
        main_content = (
            soup.select_one("main") or
            soup.select_one("article") or
            soup.select_one('[role="main"]') or
            soup.select_one(".content") or
            soup.select_one("#content") or
            soup.body
        )

        if not main_content:
            return None

        text = main_content.get_text(separator="\n", strip=True)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Extract links for further navigation
        links = []
        for a in main_content.find_all("a", href=True):
            href = a["href"]
            absolute_url = urljoin(url, href)
            if urlparse(absolute_url).scheme in ("http", "https"):
                links.append({
                    "url": absolute_url,
                    "text": a.get_text(strip=True)[:200],
                })

        # Compute content hash for deduplication
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        return PageContent(
            url=url,
            title=soup.title.string.strip() if soup.title else "",
            content=text[:50000],  # Limit size
            links=links[:50],  # Limit links
            content_hash=content_hash,
            fetched_at=datetime.now(timezone.utc),
            metadata={
                "content_length": len(text),
                "link_count": len(links),
            },
        )

    async def autonomous_research(
        self,
        seed_query: str,
        max_depth: int = 2,
        max_pages: int = 10,
        generation: int = 1,
    ) -> List[PageContent]:
        """
        Perform autonomous research starting from a seed query.

        The entity chooses what to explore based on information gaps.
        """
        visited = set()
        to_visit = [(seed_query, 0)]
        results = []

        while to_visit and len(results) < max_pages:
            query, depth = to_visit.pop(0)

            if depth > max_depth:
                continue

            # Search
            search_results = await self.search(query, max_results=5, generation=generation, trigger_type="autonomous")

            for sr in search_results:
                if sr.url in visited:
                    continue
                visited.add(sr.url)

                # Fetch page
                page = await self.fetch_page(sr.url)
                if page and page.content:
                    results.append(page)

                    # Extract potential follow-up queries from content
                    if depth < max_depth:
                        followup_queries = self._extract_followup_queries(page.content, query)
                        for fq in followup_queries[:2]:
                            to_visit.append((fq, depth + 1))

        return results

    def _extract_followup_queries(self, content: str, original_query: str) -> List[str]:
        """Extract potential follow-up queries from page content."""
        # Simple heuristic: find question-like phrases or capitalized concepts
        queries = []

        # Find questions
        questions = re.findall(r'[^.?]*\?', content)
        for q in questions[:3]:
            q = q.strip()
            if len(q) > 10 and len(q) < 200:
                queries.append(q)

        # Find capitalized phrases (potential concepts)
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        for c in concepts[:5]:
            if len(c) > 3 and c.lower() not in original_query.lower():
                queries.append(f"{c} {original_query}")

        return list(dict.fromkeys(queries))[:5]  # Deduplicate

    async def get_search_history(
        self,
        generation: Optional[int] = None,
        limit: int = 50,
    ) -> List[WebQuery]:
        """Get search history."""
        stmt = select(WebQuery).order_by(WebQuery.created_at.desc()).limit(limit)
        if generation:
            stmt = stmt.where(WebQuery.generation == generation)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_sources_for_query(self, query_id: uuid.UUID) -> List[WebSource]:
        """Get all sources for a query."""
        result = await self.db.execute(
            select(WebSource).where(WebSource.query_id == query_id)
        )
        return list(result.scalars().all())


@dataclass
class PageContent:
    """Extracted page content."""
    url: str
    title: str
    content: str
    links: List[Dict[str, str]]
    content_hash: str
    fetched_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
