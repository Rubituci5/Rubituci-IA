"""Incremental lexical observations collected from user writing.

This module does not mutate model weights during a conversation. It builds a
reviewable vocabulary/ngram ledger that can feed the next training cycle.
"""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path

from brain.language import portuguese_lexicon


_LOCK = threading.Lock()
_WORD = re.compile(r"[a-záéíóúâêôãõàç]{2,}", re.I)


def _ledger_path() -> Path:
    configured = os.getenv("LEXICAL_LEDGER_PATH")
    return Path(configured) if configured else Path("data/lexical_observations.json")


def observe_writing(text: str) -> None:
    """Record word frequency and character construction patterns safely."""
    normalized = unicodedata.normalize("NFC", text).lower()
    words = _WORD.findall(normalized)[:500]
    if not words:
        return
    known_lexicon = portuguese_lexicon()
    word_counts = Counter(words)
    ngrams = Counter(
        word[index:index + size]
        for word in words
        for size in (2, 3, 4)
        for index in range(len(word) - size + 1)
    )
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        try:
            ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            ledger = {}
        ledger.setdefault("known_words", {})
        ledger.setdefault("candidate_words", {})
        ledger.setdefault("character_ngrams", {})
        ledger["observations"] = int(ledger.get("observations", 0)) + 1
        for word, count in word_counts.items():
            bucket = "known_words" if word in known_lexicon else "candidate_words"
            ledger[bucket][word] = int(ledger[bucket].get(word, 0)) + count
        for ngram, count in ngrams.most_common(1500):
            ledger["character_ngrams"][ngram] = int(ledger["character_ngrams"].get(ngram, 0)) + count
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(ledger, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

