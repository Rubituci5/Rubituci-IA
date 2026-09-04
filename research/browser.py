"""
Browser Service

Web navigation capabilities for the entity.
Allows following links, navigating between pages, and maintaining session state.
"""

import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from collections import deque

import httpx
from bs4 import BeautifulSoup

from research.web_search import PageContent, WebSearchService
from api.config import settings
from api.security import ContainmentPolicy


@dataclass
class NavigationStep:
    """A single navigation step."""
    url: str
    title: str
    timestamp: datetime
    query_id: Optional[uuid.UUID] = None
    referrer: Optional[str] = None
    action: str = "click"  # click, search, back, forward


class BrowserSession:
    """Browser session with history and state."""

    def __init__(self, session_id: uuid.UUID, generation: int):
        self.session_id = session_id
        self.generation = generation
        self.history: List[NavigationStep] = []
        self.current_index = -1
        self.visited_urls: Set[str] = set()
        self.cookies: Dict[str, str] = {}
        self.created_at = datetime.now(timezone.utc)

    def navigate(self, url: str, title: str, query_id: Optional[uuid.UUID] = None, referrer: Optional[str] = None, action: str = "click"):
        """Record a navigation."""
        # Trim forward history if we're not at the end
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]

        step = NavigationStep(
            url=url,
            title=title,
            timestamp=datetime.now(timezone.utc),
            query_id=query_id,
            referrer=referrer,
            action=action,
        )
        self.history.append(step)
        self.current_index = len(self.history) - 1
        self.visited_urls.add(url)

    def go_back(self) -> Optional[NavigationStep]:
        """Go back in history."""
        if self.current_index > 0:
            self.current_index -= 1
            return self.history[self.current_index]
        return None

    def go_forward(self) -> Optional[NavigationStep]:
        """Go forward in history."""
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            return self.history[self.current_index]
        return None

    def get_current(self) -> Optional[NavigationStep]:
        """Get current page."""
        if 0 <= self.current_index < len(self.history):
            return self.history[self.current_index]
        return None


class BrowserService:
    """
    Web browser for the entity.

    Capabilities:
    - Search the web
    - Open pages
    - Follow links
    - Navigate history (back/forward)
    - Maintain session state
    - Extract content
    - Track provenance

    Security:
    - No JavaScript execution
    - No file downloads
    - No form submission with credentials
    - Rate limited
    - Content sanitized
    """

    def __init__(self, db_session, generation: int = 1):
        self.db = db_session
        self.generation = generation
        self.session = BrowserSession(uuid.uuid4(), generation)
        self.search_service = WebSearchService(db_session)
        self._client: Optional[httpx.AsyncClient] = None
        self._request_count = 0

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.WEB_SEARCH_TIMEOUT_SECONDS),
            headers={"User-Agent": settings.WEB_SEARCH_USER_AGENT},
            follow_redirects=True,
            cookies=self.session.cookies,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            # Save cookies back to session
            self.session.cookies = dict(self._client.cookies)
            await self._client.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.WEB_SEARCH_TIMEOUT_SECONDS),
                headers={"User-Agent": settings.WEB_SEARCH_USER_AGENT},
                follow_redirects=True,
                cookies=self.session.cookies,
            )
        return self._client

    async def search(self, query: str, max_results: int = 5) -> List[PageContent]:
        """Search and return results as PageContent objects."""
        self._enforce_rate_limit()

        search_results = await self.search_service.search(
            query=query,
            max_results=max_results,
            generation=self.generation,
            trigger_type="browser_search",
        )

        # Fetch content for each result
        pages = []
        for sr in search_results:
            page = await self.open(sr.url, query_id=sr.query_id if hasattr(sr, 'query_id') else None)
            if page:
                pages.append(page)

        return pages

    async def open(self, url: str, query_id: Optional[uuid.UUID] = None) -> Optional[PageContent]:
        """Open a URL and extract content."""
        # Security checks
        if not self._is_allowed_url(url):
            return None

        if url in self.session.visited_urls:
            # Already visited, just navigate
            self.session.navigate(url, "", query_id, action="revisit")
            # Return cached or re-fetch
            pass

        self._enforce_rate_limit()
        client = await self._get_client()

        try:
            response = await client.get(url)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to open {url}: {e}")
            return None

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            return None

        # Parse
        soup = BeautifulSoup(response.text, "html.parser")

        # Sanitize - remove dangerous elements
        for elem in soup(["script", "style", "iframe", "object", "embed", "form", "input", "button", "noscript"]):
            elem.decompose()

        # Remove event handlers
        for elem in soup.find_all(True):
            for attr in list(elem.attrs.keys()):
                if attr.startswith("on"):
                    del elem.attrs[attr]

        # Extract content
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
        text = self._clean_text(text)

        # Extract links
        links = []
        for a in main_content.find_all("a", href=True):
            href = a["href"]
            absolute_url = urljoin(url, href)
            parsed = urlparse(absolute_url)
            if parsed.scheme in ("http", "https"):
                links.append({
                    "url": absolute_url,
                    "text": a.get_text(strip=True)[:200],
                    "title": a.get("title", "")[:200],
                })

        # Record navigation
        title = soup.title.string.strip() if soup.title else ""
        self.session.navigate(url, title, query_id, referrer=self._get_referrer(), action="click")

        # Update cookies
        self.session.cookies = dict(client.cookies)

        return PageContent(
            url=url,
            title=title,
            content=text[:50000],
            links=links[:100],
            content_hash="",
            fetched_at=datetime.now(timezone.utc),
        )

    async def follow_link(self, link_index: int, query_id: Optional[uuid.UUID] = None) -> Optional[PageContent]:
        """Follow a link from the current page by index."""
        current = self.session.get_current()
        if not current:
            return None

        # We'd need to store links with the page - simplified for now
        # In practice, would fetch current page again or cache links
        return None

    async def go_back(self) -> Optional[PageContent]:
        """Navigate back."""
        step = self.session.go_back()
        if step:
            return await self.open(step.url, step.query_id)
        return None

    async def go_forward(self) -> Optional[PageContent]:
        """Navigate forward."""
        step = self.session.go_forward()
        if step:
            return await self.open(step.url, step.query_id)
        return None

    def get_history(self) -> List[NavigationStep]:
        """Get navigation history."""
        return self.session.history

    def get_visited_domains(self) -> Set[str]:
        """Get unique domains visited."""
        domains = set()
        for url in self.session.visited_urls:
            try:
                domains.add(urlparse(url).netloc)
            except Exception:
                pass
        return domains

    def _is_allowed_url(self, url: str) -> bool:
        """Check if URL is allowed."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False

            # Block local/private addresses
            hostname = parsed.hostname or ""
            if hostname in ("localhost", "127.0.0.1", "0.0.0.0") or hostname.endswith(".local"):
                return False

            # Block private IP ranges
            import ipaddress
            try:
                ip = ipaddress.ip_address(hostname)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    return False
            except ValueError:
                pass  # Not an IP address

            # Check blocked domains
            blocked = settings.SAFETY_BLOCKED_DOMAINS.split(",") if settings.SAFETY_BLOCKED_DOMAINS else []
            if any(hostname.endswith(b.strip()) for b in blocked if b.strip()):
                return False

            return True
        except Exception:
            return False

    def _enforce_rate_limit(self):
        """Enforce rate limiting."""
        self._request_count += 1
        if self._request_count > settings.WEB_SEARCH_RATE_LIMIT_PER_MINUTE:
            raise Exception("Rate limit exceeded")

    def _get_referrer(self) -> Optional[str]:
        """Get referrer URL."""
        if self.session.current_index > 0:
            return self.session.history[self.session.current_index - 1].url
        return None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        import re
        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\t+", " ", text)
        return text.strip()


class AutonomousBrowser:
    """
    High-level autonomous browser for research cycles.

    The entity can:
    - Start research from a question
    - Choose what to read
    - Follow interesting links
    - Compare sources
    - Return to previous pages
    - Build knowledge over time
    """

    def __init__(self, db_session, generation: int = 1):
        self.db = db_session
        self.generation = generation
        self.browser: Optional[BrowserService] = None

    async def __aenter__(self):
        self.browser = BrowserService(self.db, self.generation)
        await self.browser.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await self.browser.__aexit__(exc_type, exc_val, exc_tb)

    async def research_topic(
        self,
        initial_query: str,
        max_pages: int = 10,
        max_depth: int = 3,
    ) -> List[PageContent]:
        """
        Autonomous research on a topic.

        The browser decides what to explore based on:
        - Information gaps
        - Contradictions found
        - Novel concepts
        - Recurring themes
        """
        results = []
        to_explore = deque([(initial_query, 0, "initial")])  # (query, depth, reason)
        explored_queries = set()

        while to_explore and len(results) < max_pages:
            query, depth, reason = to_explore.popleft()

            if query in explored_queries or depth > max_depth:
                continue

            explored_queries.add(query)

            # Search
            pages = await self.browser.search(query, max_results=3)

            for page in pages:
                if len(results) >= max_pages:
                    break

                results.append(page)

                # Analyze page for follow-up
                if depth < max_depth:
                    followups = self._analyze_for_followups(page, query)
                    for fu_query, fu_reason in followups:
                        to_explore.append((fu_query, depth + 1, fu_reason))

        return results

    def _analyze_for_followups(self, page: PageContent, original_query: str) -> List[tuple[str, str]]:
        """Analyze page content for follow-up research directions."""
        followups = []
        content = page.content.lower()

        # Detect contradictions (simple heuristic)
        contradiction_indicators = ["however", "but", "contrary", "disagree", "debate", "controversial", "disputed"]
        if any(ind in content for ind in contradiction_indicators):
            followups.append((f"contradiction {original_query}", "contradiction_detected"))

        # Detect uncertainty
        uncertainty_indicators = ["unclear", "unknown", "debated", "hypothesis", "speculation", "may be", "could be"]
        if any(ind in content for ind in uncertainty_indicators):
            followups.append((f"uncertainty {original_query}", "uncertainty_detected"))

        # Extract capitalized concepts as potential topics
        import re
        concepts = re.findall(r'\b[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})*\b', page.content)
        for concept in concepts[:3]:
            if concept.lower() not in original_query.lower():
                followups.append((f"{concept} {original_query}", "new_concept"))

        # Extract questions
        questions = re.findall(r'[^.?]*\?', page.content)
        for q in questions[:2]:
            q = q.strip()
            if len(q) > 15:
                followups.append((q, "question_found"))

        return followups[:5]

    async def compare_sources(self, urls: List[str]) -> Dict[str, Any]:
        """Fetch and compare multiple sources on the same topic."""
        pages = []
        for url in urls:
            page = await self.browser.open(url)
            if page:
                pages.append(page)

        # Simple comparison: find common and unique terms
        all_words = []
        for page in pages:
            words = set(page.content.lower().split())
            all_words.append(words)

        if not all_words:
            return {"common": [], "unique": {}}

        common = set.intersection(*all_words) if len(all_words) > 1 else all_words[0]
        unique = {}
        for i, words in enumerate(all_words):
            unique[pages[i].url] = words - common

        return {
            "pages_compared": len(pages),
            "common_terms": list(common)[:50],
            "unique_terms": {url: list(terms)[:20] for url, terms in unique.items()},
            "sources": [{"url": p.url, "title": p.title} for p in pages],
        }