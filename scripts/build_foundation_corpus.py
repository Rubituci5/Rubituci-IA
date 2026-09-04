#!/usr/bin/env python3

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "foundation_v1" / "raw"
OUT_DIR = ROOT / "data" / "foundation_v1" / "processed"
MANIFEST_DIR = ROOT / "data" / "foundation_v1" / "manifests"

OUTPUT_FILE = OUT_DIR / "foundation_v1.jsonl"
MANIFEST_FILE = MANIFEST_DIR / "foundation_v1_manifest.json"


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def chunk_text(text: str, max_chars: int = 1600):
    import re

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current = ""

    def add_piece(piece: str):
        nonlocal current

        piece = piece.strip()
        if not piece:
            return

        candidate = piece if not current else current + "\n" + piece

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            add_piece(paragraph)
            continue

        sentences = re.split(
            r'(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9\"“—-])',
            paragraph,
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) <= max_chars:
                add_piece(sentence)
                continue

            words = sentence.split()
            piece = ""

            for word in words:
                candidate = word if not piece else piece + " " + word

                if len(candidate) <= max_chars:
                    piece = candidate
                else:
                    if piece:
                        add_piece(piece)
                    piece = word

            if piece:
                add_piece(piece)

    if current:
        chunks.append(current)

    return chunks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.txt"))

    if not files:
        print("Nenhum arquivo .txt encontrado em:")
        print(RAW_DIR)
        return 1

    records = []
    source_stats = []

    for path in files:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(raw)

        if not text:
            continue

        chunks = chunk_text(text)

        for chunk in chunks:
            records.append({
                "text": chunk,
                "source": path.name,
                "corpus": "foundation_v1",
            })

        source_stats.append({
            "file": path.name,
            "characters": len(text),
            "chunks": len(chunks),
        })

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    sha256 = hashlib.sha256(OUTPUT_FILE.read_bytes()).hexdigest()

    manifest = {
        "name": "Entity Foundation Corpus v1",
        "records": len(records),
        "sources": source_stats,
        "output": str(OUTPUT_FILE),
        "sha256": sha256,
    }

    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print("ENTITY FOUNDATION CORPUS V1")
    print("=" * 60)
    print("Fontes:", len(source_stats))
    print("Registros:", len(records))
    print("Dataset:", OUTPUT_FILE)
    print("Manifest:", MANIFEST_FILE)
    print("SHA256:", sha256)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
