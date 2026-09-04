"""Portuguese-language quality helpers used at inference and evaluation time."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_ROLE_MARKER = re.compile(r"(?:^|\n)\s*(?:user|usu[aá]rio|entity|rubituci)\s*:", re.I)
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")
_MANY_SPACES = re.compile(r"[ \t]{2,}")
_REPEATED_WORDS = re.compile(r"\b([\wÀ-ÿ]+)(?:\s+\1){2,}\b", re.I)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LEXICON_SOURCES = (
    _PROJECT_ROOT / "data" / "literacy_ptbr_v1" / "literacy_ptbr_v1.jsonl",
    _PROJECT_ROOT / "data" / "oasst1_ptbr" / "processed" / "oasst1_ptbr_curated.jsonl",
    _PROJECT_ROOT / "data" / "foundation_v1" / "raw" / "dom_casmurro.txt",
)
_ALWAYS_ALLOWED = {"rubituci", "open", "source", "internet", "site", "link", "links", "online", "backup"}


def clean_assistant_response(text: str) -> str:
    """Remove leaked chat turns and repair harmless spacing artifacts.

    This deliberately avoids rewriting words: spelling and factual corrections must
    come from training, not from a post-processor that could alter meaning.
    """
    text = unicodedata.normalize("NFC", text or "").replace("\x00", "")
    marker = _ROLE_MARKER.search(text)
    if marker:
        text = text[: marker.start()]
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    text = _MANY_SPACES.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


@dataclass(frozen=True)
class PortugueseQuality:
    score: float
    has_sentence_end: bool
    has_repetition: bool
    replacement_characters: int
    leaked_role: bool
    alphabetic_ratio: float = 0.0


def evaluate_portuguese_surface(text: str) -> PortugueseQuality:
    """Return a small, dependency-free surface-quality score (not an IQ score)."""
    normalized = unicodedata.normalize("NFC", text or "").strip()
    has_sentence_end = bool(re.search(r"[.!?…][\"')\]]?$", normalized))
    has_repetition = bool(_REPEATED_WORDS.search(normalized))
    replacement_characters = normalized.count("�")
    leaked_role = bool(_ROLE_MARKER.search(normalized))
    visible = [char for char in normalized if not char.isspace()]
    alphabetic_ratio = (
        sum(char.isalpha() or char in ".,;:!?…'\"()[]-" for char in visible) / len(visible)
        if visible else 0.0
    )
    score = 1.0
    if normalized and not has_sentence_end:
        score -= 0.15
    if not normalized:
        score = 0.0
    if has_repetition:
        score -= 0.30
    if replacement_characters:
        score -= min(0.40, replacement_characters * 0.10)
    if leaked_role:
        score -= 0.30
    if alphabetic_ratio < 0.85:
        score -= 0.25
    return PortugueseQuality(
        score=max(0.0, round(score, 3)),
        has_sentence_end=has_sentence_end,
        has_repetition=has_repetition,
        replacement_characters=replacement_characters,
        leaked_role=leaked_role,
        alphabetic_ratio=round(alphabetic_ratio, 3),
    )


@lru_cache(maxsize=1)
def portuguese_lexicon() -> frozenset[str]:
    """Build a local lexicon only from the project's Portuguese corpora."""
    words = set(_ALWAYS_ALLOWED)
    for path in _LEXICON_SOURCES:
        if not path.exists():
            continue
        content = unicodedata.normalize("NFC", path.read_text(encoding="utf-8", errors="ignore")).lower()
        words.update(re.findall(r"[a-záéíóúâêôãõàç]{2,}", content))
    return frozenset(words)


def _intent_fallback(user_message: str) -> str:
    """Choose an honest local reply for common intents when generation fails."""
    normalized = unicodedata.normalize("NFD", user_message or "").lower()
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")

    if "hoje" in normalized and any(term in normalized for term in ("aprendeu", "aprendeste", "aprendizado", "aprendizagem")):
        return "Hoje ainda não aprendi nada de novo. Quando eu aprender algo e a informação passar pela revisão, te conto sem inventar moda."
    if "capital" in normalized and "brasil" in normalized:
        return "A capital do Brasil é Brasília. Essa eu sei sem precisar fazer turismo pela internet."
    if any(term in normalized for term in ("quem e voce", "quem voce e", "se apresente", "se apresenta")):
        return "Eu sou a Rubituci, uma IA brasileira experimental feita do zero. Ainda sou pequena, curiosa e sincera quando não sei — milagre tecnológico raro, convenhamos."
    if re.search(r"\b(oi|ola|e ai|bom dia|boa tarde|boa noite)\b", normalized):
        return "Oi! Eu sou a Rubituci. Ainda estou aprendendo, mas já consigo conversar sem transformar cada frase num acidente ortográfico. Como posso te ajudar?"
    if any(term in normalized for term in ("como voce esta", "como esta voce", "tudo bem")):
        return "Estou funcionando e curiosa, o equivalente digital de estar bem. E você, como está?"
    return (
        "Ainda não manjo desse assunto o bastante para te responder direito — e chutar com confiança ainda é só um chute de terno. "
        "Tem uma fonte confiável pra compartilhar? Eu registro, analiso e uso no aprendizado após revisão."
    )


def safe_portuguese_response(
    text: str,
    minimum_score: float = 0.75,
    user_message: str = "",
) -> tuple[str, bool]:
    """Prevent visibly degenerate text from reaching a public user.

    This is a temporary quality gate while the pure model is trained further.
    It never pretends a rejected generation is a useful answer.
    """
    cleaned = clean_assistant_response(text)
    quality = evaluate_portuguese_surface(cleaned)
    words = re.findall(r"\b[\wÀ-ÿ]+\b", cleaned)
    short_fragment_ratio = (
        sum(len(word) == 1 and word.lower() not in {"a", "e", "é", "o"} for word in words) / len(words)
        if words else 1.0
    )
    suspicious_cluster = any(
        len(word) > 1 and not re.search(r"[aeiouáéíóúâêôãõà]", word, re.I)
        for word in words
    )
    lexical_words = [word.lower() for word in words if len(word) > 1 and not word.isdigit()]
    lexicon = portuguese_lexicon()
    unknown_ratio = (
        sum(word not in lexicon for word in lexical_words) / len(lexical_words)
        if lexical_words and lexicon else 0.0
    )
    if (
        quality.score < minimum_score
        or len(words) < 3
        or short_fragment_ratio > 0.20
        or suspicious_cluster
        or unknown_ratio > 0.20
    ):
        return _intent_fallback(user_message), False
    return cleaned, True
