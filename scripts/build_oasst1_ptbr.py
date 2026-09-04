#!/usr/bin/env python3

import json
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent.parent

OUT = (
    ROOT
    / "data"
    / "oasst1_ptbr"
    / "processed"
    / "oasst1_ptbr.jsonl"
)


def main():
    print("=" * 60)
    print("DOWNLOADING OASST1")
    print("=" * 60)

    ds = load_dataset("OpenAssistant/oasst1")

    rows = []

    for split_name in ds.keys():
        split = ds[split_name]

        for row in split:
            lang = str(row.get("lang", "")).lower()

            if lang not in {"pt-br", "pt"}:
                continue

            text = (row.get("text") or "").strip()
            role = row.get("role")

            if not text:
                continue

            if role not in {"prompter", "assistant"}:
                continue

            rows.append({
                "message_id": row.get("message_id"),
                "parent_id": row.get("parent_id"),
                "role": role,
                "text": text,
                "lang": lang,
            })

    by_id = {
        row["message_id"]: row
        for row in rows
        if row["message_id"]
    }

    examples = []

    for row in rows:
        if row["role"] != "assistant":
            continue

        parent_id = row.get("parent_id")

        if not parent_id:
            continue

        parent = by_id.get(parent_id)

        if not parent:
            continue

        if parent["role"] != "prompter":
            continue

        conversation = (
            f"User: {parent['text']}\n"
            f"Entity: {row['text']}"
        )

        examples.append({
            "text": conversation,
            "source": "OpenAssistant/oasst1",
            "language": row["lang"],
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", encoding="utf-8") as f:
        for item in examples:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 60)
    print("OASST1 PT-BR READY")
    print("=" * 60)
    print("Portuguese messages:", len(rows))
    print("Conversation pairs:", len(examples))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
