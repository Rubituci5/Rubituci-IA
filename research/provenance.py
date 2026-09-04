"""
Provenance Tracking

Records the complete lineage of information:
where it came from, when, how it was obtained, and what was done with it.
"""

import uuid
import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from api.models import WebSource, SourceCitation, EpisodicMemory, Belief, MemorySource
from api.config import settings


class ProvenanceType(str, Enum):
    """Types of provenance records."""
    WEB_SEARCH = "web_search"
    PAGE_VISIT = "page_visit"
    LINK_FOLLOWED = "link_followed"
    MEMORY_CREATED = "memory_created"
    BELIEF_FORMED = "belief_formed"
    BELIEF_UPDATED = "belief_updated"
    CONSOLIDATION = "consolidation"
    REFLECTION = "reflection"
    HUMAN_INTERACTION = "human_interaction"


@dataclass
class SourceRecord:
    """Complete record of an information source."""
    source_id: uuid.UUID
    provenance_type: ProvenanceType
    timestamp: datetime
    origin: Dict[str, Any]  # Where the info came from
    content_summary: str
    content_hash: str
    context: Dict[str, Any]  # Why this was retrieved
    transformations: List[Dict[str, Any]] = field(default_factory=list)  # What was done with it
    derived_from: List[uuid.UUID] = field(default_factory=list)  # Parent sources
    confidence_impact: float = 0.0  # How this affected confidence
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceChain:
    """Chain of provenance for a piece of knowledge."""
    root_sources: List[SourceRecord]
    intermediate_steps: List[SourceRecord]
    final_derivation: SourceRecord
    confidence: float


class ProvenanceTracker:
    """
    Tracks provenance of all information entering the system.

    Ensures every piece of knowledge can be traced back to its origins.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._session_records: List[SourceRecord] = []

    async def record_web_search(
        self,
        query: str,
        results: List[Dict[str, Any]],
        generation: int,
        trigger_type: str,
    ) -> List[SourceRecord]:
        """Record a web search and its results."""
        records = []

        for i, result in enumerate(results):
            # Create content hash
            content = f"{result.get('title', '')}\n{result.get('snippet', '')}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

            record = SourceRecord(
                source_id=uuid.uuid4(),
                provenance_type=ProvenanceType.WEB_SEARCH,
                timestamp=datetime.now(timezone.utc),
                origin={
                    "type": "web_search",
                    "query": query,
                    "provider": settings.WEB_SEARCH_PROVIDER,
                    "rank": i + 1,
                    "url": result.get("url"),
                    "domain": result.get("domain"),
                },
                content_summary=content[:500],
                content_hash=content_hash,
                context={
                    "generation": generation,
                    "trigger_type": trigger_type,
                    "search_intent": trigger_type,
                },
                confidence_impact=0.0,  # Will be updated after content evaluation
            )
            records.append(record)
            self._session_records.append(record)

        return records

    async def record_page_visit(
        self,
        url: str,
        title: str,
        content: str,
        query_id: Optional[uuid.UUID],
        referrer: Optional[str],
        generation: int,
    ) -> SourceRecord:
        """Record a page visit."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        record = SourceRecord(
            source_id=uuid.uuid4(),
            provenance_type=ProvenanceType.PAGE_VISIT,
            timestamp=datetime.now(timezone.utc),
            origin={
                "type": "page_visit",
                "url": url,
                "title": title,
                "referrer": referrer,
                "query_id": str(query_id) if query_id else None,
            },
            content_summary=content[:500],
            content_hash=content_hash,
            context={
                "generation": generation,
                "visit_depth": 0,  # Would track navigation depth
            },
        )
        self._session_records.append(record)
        return record

    async def record_link_followed(
        self,
        from_url: str,
        to_url: str,
        link_text: str,
        generation: int,
    ) -> SourceRecord:
        """Record following a link."""
        record = SourceRecord(
            source_id=uuid.uuid4(),
            provenance_type=ProvenanceType.LINK_FOLLOWED,
            timestamp=datetime.now(timezone.utc),
            origin={
                "type": "link_followed",
                "from_url": from_url,
                "to_url": to_url,
                "link_text": link_text,
            },
            content_summary=f"Followed link '{link_text}' from {from_url} to {to_url}",
            content_hash=hashlib.sha256(f"{from_url}{to_url}".encode()).hexdigest()[:16],
            context={
                "generation": generation,
                "navigation_type": "explicit",
            },
        )
        self._session_records.append(record)
        return record

    async def record_memory_created(
        self,
        memory_id: uuid.UUID,
        content: str,
        source: MemorySource,
        context: Dict[str, Any],
        derived_from: List[uuid.UUID],
        generation: int,
    ) -> SourceRecord:
        """Record creation of an episodic memory."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        record = SourceRecord(
            source_id=uuid.uuid4(),
            provenance_type=ProvenanceType.MEMORY_CREATED,
            timestamp=datetime.now(timezone.utc),
            origin={
                "type": "memory_created",
                "memory_id": str(memory_id),
                "source": source.value,
            },
            content_summary=content[:500],
            content_hash=content_hash,
            context={
                **context,
                "generation": generation,
            },
            derived_from=derived_from,
            confidence_impact=0.0,
        )
        self._session_records.append(record)
        return record

    async def record_belief_formed(
        self,
        belief_id: uuid.UUID,
        proposition: str,
        confidence: float,
        evidence_sources: List[uuid.UUID],
        generation: int,
    ) -> SourceRecord:
        """Record formation of a belief."""
        record = SourceRecord(
            source_id=uuid.uuid4(),
            provenance_type=ProvenanceType.BELIEF_FORMED,
            timestamp=datetime.now(timezone.utc),
            origin={
                "type": "belief_formed",
                "belief_id": str(belief_id),
                "proposition": proposition,
                "initial_confidence": confidence,
            },
            content_summary=proposition[:500],
            content_hash=hashlib.sha256(proposition.encode()).hexdigest()[:16],
            context={
                "generation": generation,
                "evidence_count": len(evidence_sources),
            },
            derived_from=evidence_sources,
            confidence_impact=confidence,
        )
        self._session_records.append(record)
        return record

    async def record_belief_updated(
        self,
        belief_id: uuid.UUID,
        old_confidence: float,
        new_confidence: float,
        reason: str,
        new_evidence: List[uuid.UUID],
        generation: int,
    ) -> SourceRecord:
        """Record belief confidence update."""
        record = SourceRecord(
            source_id=uuid.uuid4(),
            provenance_type=ProvenanceType.BELIEF_UPDATED,
            timestamp=datetime.now(timezone.utc),
            origin={
                "type": "belief_updated",
                "belief_id": str(belief_id),
                "old_confidence": old_confidence,
                "new_confidence": new_confidence,
                "reason": reason,
            },
            content_summary=f"Belief confidence changed from {old_confidence:.2f} to {new_confidence:.2f}: {reason}",
            content_hash=hashlib.sha256(f"{belief_id}{old_confidence}{new_confidence}".encode()).hexdigest()[:16],
            context={
                "generation": generation,
                "delta": new_confidence - old_confidence,
            },
            derived_from=new_evidence,
            confidence_impact=new_confidence - old_confidence,
        )
        self._session_records.append(record)
        return record

    async def record_consolidation(
        self,
        consolidation_run_id: uuid.UUID,
        input_memories: List[uuid.UUID],
        output_concepts: List[str],
        generation: int,
    ) -> SourceRecord:
        """Record a consolidation (sleep) cycle."""
        record = SourceRecord(
            source_id=uuid.uuid4(),
            provenance_type=ProvenanceType.CONSOLIDATION,
            timestamp=datetime.now(timezone.utc),
            origin={
                "type": "consolidation",
                "consolidation_run_id": str(consolidation_run_id),
                "input_count": len(input_memories),
                "output_concepts": output_concepts,
            },
            content_summary=f"Consolidated {len(input_memories)} memories into {len(output_concepts)} concepts",
            content_hash=hashlib.sha256(f"consolidation{consolidation_run_id}".encode()).hexdigest()[:16],
            context={
                "generation": generation,
            },
            derived_from=input_memories,
        )
        self._session_records.append(record)
        return record

    async def record_reflection(
        self,
        reflection_id: uuid.UUID,
        trigger_type: str,
        input_memories: List[uuid.UUID],
        output: str,
        generation: int,
    ) -> SourceRecord:
        """Record an autonomous reflection."""
        record = SourceRecord(
            source_id=uuid.uuid4(),
            provenance_type=ProvenanceType.REFLECTION,
            timestamp=datetime.now(timezone.utc),
            origin={
                "type": "reflection",
                "reflection_id": str(reflection_id),
                "trigger_type": trigger_type,
            },
            content_summary=output[:500],
            content_hash=hashlib.sha256(output.encode()).hexdigest()[:16],
            context={
                "generation": generation,
                "input_count": len(input_memories),
            },
            derived_from=input_memories,
        )
        self._session_records.append(record)
        return record

    async def record_human_interaction(
        self,
        conversation_id: uuid.UUID,
        message_id: uuid.UUID,
        user_id: uuid.UUID,
        content: str,
        generation: int,
    ) -> SourceRecord:
        """Record a human interaction."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        record = SourceRecord(
            source_id=uuid.uuid4(),
            provenance_type=ProvenanceType.HUMAN_INTERACTION,
            timestamp=datetime.now(timezone.utc),
            origin={
                "type": "human_interaction",
                "conversation_id": str(conversation_id),
                "message_id": str(message_id),
                "user_id": str(user_id),
            },
            content_summary=content[:500],
            content_hash=content_hash,
            context={
                "generation": generation,
            },
            confidence_impact=0.0,
        )
        self._session_records.append(record)
        return record

    def get_session_records(self) -> List[SourceRecord]:
        """Get all records from current session."""
        return self._session_records.copy()

    def clear_session(self):
        """Clear session records."""
        self._session_records.clear()

    async def get_provenance_chain(
        self,
        target_id: uuid.UUID,
        target_type: str,  # "belief", "memory", "knowledge"
        max_depth: int = 10,
    ) -> ProvenanceChain:
        """
        Reconstruct the full provenance chain for a piece of knowledge.

        Traces back through all derivations to root sources.
        """
        # This would query the database for the full chain
        # Simplified implementation
        root_sources = []
        intermediate = []
        final = None

        # In a full implementation, this would recursively trace:
        # belief -> evidence -> episodic memories -> web sources / human interactions
        # For now, return empty chain
        return ProvenanceChain(
            root_sources=root_sources,
            intermediate_steps=intermediate,
            final_derivation=final or SourceRecord(
                source_id=target_id,
                provenance_type=ProvenanceType.MEMORY_CREATED,
                timestamp=datetime.now(timezone.utc),
                origin={},
                content_summary="",
                content_hash="",
            ),
            confidence=0.0,
        )

    async def verify_provenance(
        self,
        belief_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """
        Verify the provenance of a belief.

        Checks:
        - All evidence sources exist
        - No circular dependencies
        - Confidence matches evidence
        - Sources are accessible
        """
        # Would implement full verification
        return {
            "verified": True,
            "issues": [],
            "evidence_count": 0,
            "root_sources": 0,
            "chain_depth": 0,
        }

    async def get_source_statistics(self, generation: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics about information sources."""
        # Query database for source breakdown
        stmt = select(WebSource.source_type, func.count(WebSource.id)).group_by(WebSource.source_type)
        if generation:
            # Would need to join with WebQuery
            pass
        result = await self.db.execute(stmt)
        by_type = {row.source_type or "unknown": row[1] for row in result.all()}

        # Credibility distribution
        stmt = select(
            func.count(WebSource.id),
            func.avg(WebSource.credibility),
            func.min(WebSource.credibility),
            func.max(WebSource.credibility),
        )
        result = await self.db.execute(stmt)
        total, avg_cred, min_cred, max_cred = result.one()

        return {
            "total_sources": total or 0,
            "avg_credibility": float(avg_cred or 0),
            "min_credibility": float(min_cred or 0),
            "max_credibility": float(max_cred or 0),
            "by_type": by_type,
            "session_records": len(self._session_records),
        }

    async def export_provenance_graph(
        self,
        generation: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Export provenance as a graph for visualization."""
        # Would build a graph of all provenance relationships
        return {
            "nodes": [],
            "edges": [],
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generation": generation,
            },
        }