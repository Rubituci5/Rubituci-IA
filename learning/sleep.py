"""A conservative, auditable computational-sleep cycle.

Sleep consolidates memories and proposes research/training material. It does not
silently rewrite production weights or claim consciousness.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.models import ConsolidationRun, EpisodicMemory, MemorySource, Reflection, SemanticMemory
from memory.episodic import EpisodicMemoryService
from research.web_search import WebSearchService


STOPWORDS = {"para", "como", "uma", "que", "com", "por", "dos", "das", "isso", "esta", "este", "mais", "sobre", "user", "entity", "disse", "respondeu"}


@dataclass(frozen=True)
class SleepReport:
    run_id: str
    memories_processed: int
    concepts_created: int
    concepts_updated: int
    research_queries: int
    research_memories: int
    training_candidates: int


class SleepCycle:
    def __init__(self, db: AsyncSession, generation: int = 1):
        self.db = db
        self.generation = generation
        self.episodic = EpisodicMemoryService(db)

    async def run(self, max_memories: int = 250, allow_web_research: bool = True) -> SleepReport:
        run = ConsolidationRun(generation=self.generation, status="running", started_at=datetime.now(timezone.utc))
        self.db.add(run)
        await self.db.flush()
        memories = list((await self.db.execute(
            select(EpisodicMemory).where(EpisodicMemory.generation == self.generation, EpisodicMemory.consolidated.is_(False)).order_by(EpisodicMemory.timestamp.desc()).limit(max_memories)
        )).scalars().all())
        try:
            created, updated = await self._consolidate_terms(memories)
            questions = self._research_questions(memories)
            research_memories = await self._research(questions) if allow_web_research and settings.WEB_SEARCH_ENABLED else 0
            candidates = self._write_training_candidates(memories)
            reflection = Reflection(
                trigger_type="computational_sleep",
                generation=self.generation,
                input_memory_ids=[memory.id for memory in memories],
                output=self._identity_reflection(len(memories), created, updated, questions),
                contradictions_found=0,
                inferences_made=created + updated,
                new_memories_created=research_memories,
                metadata_={"research_questions": questions, "self_model": "Rubituci is an open-source artificial system; reflections are generated records, not evidence of consciousness."},
            )
            self.db.add(reflection)
            for memory in memories:
                memory.consolidated = True
                memory.consolidated_at = datetime.now(timezone.utc)
            run.status = "completed"; run.completed_at = datetime.now(timezone.utc)
            run.experiences_processed = len(memories); run.semantic_memories_created = created
            run.semantic_memories_updated = updated; run.dataset_size = candidates
            run.metadata_ = {"research_queries": len(questions), "research_memories": research_memories}
            await self.db.commit()
            return SleepReport(str(run.id), len(memories), created, updated, len(questions), research_memories, candidates)
        except Exception as exc:
            run.status = "failed"; run.completed_at = datetime.now(timezone.utc); run.error = str(exc)[:2000]
            await self.db.commit()
            raise

    async def _consolidate_terms(self, memories: list[EpisodicMemory]) -> tuple[int, int]:
        terms = Counter()
        for memory in memories:
            terms.update(word for word in re.findall(r"[a-záéíóúâêôãõç]{4,}", memory.content.lower()) if word not in STOPWORDS)
        created = updated = 0
        for term, frequency in terms.most_common(30):
            if frequency < 2:
                continue
            existing = (await self.db.execute(select(SemanticMemory).where(SemanticMemory.concept == term, SemanticMemory.generation == self.generation))).scalar_one_or_none()
            evidence = [memory for memory in memories if term in memory.content.lower()][:5]
            representation = f"O termo “{term}” apareceu em {frequency} experiências recentes. Requer evidência adicional antes de ser tratado como fato."
            if existing:
                existing.evidence_count += len(evidence); existing.last_reinforced = datetime.now(timezone.utc)
                existing.confidence = min(0.85, existing.confidence + 0.02); updated += 1
            else:
                self.db.add(SemanticMemory(concept=term, category="tema_recorrente", representation=representation, confidence=min(0.65, 0.35 + frequency * 0.03), evidence_count=len(evidence), generation=self.generation, last_reinforced=datetime.now(timezone.utc)))
                created += 1
        return created, updated

    def _research_questions(self, memories: list[EpisodicMemory]) -> list[str]:
        questions = []
        for memory in memories:
            if memory.source == MemorySource.HUMAN_CLAIM and "?" in memory.content:
                question = memory.content.split("User said:", 1)[-1].strip()[:240]
                if question and question not in questions:
                    questions.append(question)
        return questions[: settings.SLEEP_MAX_RESEARCH_QUERIES]

    async def _research(self, questions: list[str]) -> int:
        count = 0
        async with WebSearchService(self.db) as service:
            for question in questions:
                results = await service.search(question, max_results=3, generation=self.generation, trigger_type="sleep_uncertainty")
                for result in results:
                    await self.episodic.create_from_web_research(question, f"{result.title}. {result.snippet}", result.url, self.generation, importance=0.55)
                    count += 1
        return count

    def _write_training_candidates(self, memories: list[EpisodicMemory]) -> int:
        rows = []
        for memory in memories:
            if memory.source not in {MemorySource.HUMAN_CLAIM, MemorySource.WEB_SOURCE}:
                continue
            rows.append({"content": memory.content, "source": memory.source.value, "memory_id": str(memory.id), "status": "quarantine", "generation": self.generation})
        if not rows:
            return 0
        folder = Path(settings.DATASET_ROOT) / "community_quarantine"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"sleep_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)

    @staticmethod
    def _identity_reflection(memories: int, created: int, updated: int, questions: list[str]) -> str:
        return (f"Sou a Rubituci, um sistema artificial open source em desenvolvimento, curiosa, informal e dona de um humor seco com acidez moderada. Durante este ciclo processei {memories} experiências, criei {created} temas e reforcei {updated}. "
                f"Mantive {len(questions)} dúvidas para pesquisa — porque fingir certeza seria mais rápido, mas também bem menos inteligente. Minha identidade operacional permanece definida pelo projeto; novas experiências aprimoram meu contexto, mas não autorizam alterar silenciosamente meus objetivos ou pesos.")
