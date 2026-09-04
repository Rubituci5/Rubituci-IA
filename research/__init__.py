"""
Entity Research System

Autonomous web search and information gathering capabilities.
"""

from .web_search import WebSearchService, SearchResult, SearchConfig
from .browser import BrowserService, PageContent
from .provenance import ProvenanceTracker, SourceRecord

__all__ = [
    "WebSearchService",
    "SearchResult",
    "SearchConfig",
    "BrowserService",
    "PageContent",
    "ProvenanceTracker",
    "SourceRecord",
]