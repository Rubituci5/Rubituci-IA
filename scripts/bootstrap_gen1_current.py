#!/usr/bin/env python3

import asyncio
import json
from pathlib import Path

import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from api.config import settings
from api.models import ModelGeneration, GenerationStatus
from brain.config import EntityConfig
from brain.tokenizer import EntityTokenizer
from brain.model import EntityTransformer


ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "snapshots" / "generation_000001"
TOKENIZER_DIR = GEN_DIR / "tokenizer"
MODEL_PATH = GEN_DIR / "model_weights.pt"
CONFIG_PATH = GEN_DIR / "config.json"


INITIAL_CORPUS = [
    "Entity is a digital intelligence designed for continuous development.",
    "Entity maintains persistent memory across conversations.",
    "Entity can learn from interaction, reflection and structured evidence.",
    "Entity distinguishes facts, beliefs, hypotheses and uncertainty.",
    "Entity should communicate clearly and acknowledge uncertainty.",
    "Entity uses episodic memory to preserve experiences.",
    "Entity uses semantic memory to consolidate knowledge.",
    "Entity evolves through versioned model generations.",
    "Entity can receive feedback without treating feedback as absolute truth.",
    "Entity should preserve safety boundaries while maintaining cognitive freedom.",
    "User: Hello\nEntity: Hello. How can I help you?",
    "User: Who are you?\nEntity: I am Entity, a developing digital intelligence.",
    "User: What do you remember?\nEntity: I can use persistent memories stored by the system.",
    "User: Are you certain?\nEntity: I distinguish certainty from inference and uncertainty.",
] * 200


async def main():
    print("=" * 60)
    print("BOOTSTRAP ENTITY GENERATION 000001")
    print("=" * 60)

    GEN_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Training tokenizer...")
    tokenizer = EntityTokenizer.train_new(
        INITIAL_CORPUS,
        vocab_size=8192,
    )

    tokenizer.save(TOKENIZER_DIR)
    print(f"✓ Tokenizer saved: {TOKENIZER_DIR}")
    print(f"✓ Vocabulary size: {tokenizer.vocab_size}")

    print("\n2. Creating model configuration...")

    config = EntityConfig(
        vocab_size=tokenizer.vocab_size,
        generation=1,
        model_name="entity-gen-000001",
    )

    CONFIG_PATH.write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )

    print(f"✓ Config saved: {CONFIG_PATH}")

    print("\n3. Creating initial model...")
    model = EntityTransformer(config)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {param_count:,}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "generation": 1,
        "step": 0,
        "epoch": 0.0,
    }

    torch.save(checkpoint, MODEL_PATH)
    print(f"✓ Model checkpoint saved: {MODEL_PATH}")

    print("\n4. Registering generation in PostgreSQL...")

    engine = create_async_engine(settings.DATABASE_URL)
    Session = sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with Session() as db:
        result = await db.execute(
            select(ModelGeneration).where(
                ModelGeneration.generation_number == 1
            )
        )

        generation = result.scalar_one_or_none()

        if generation is None:
            generation = ModelGeneration(
                generation_number=1,
                parent_generation=None,
                status=GenerationStatus.PROMOTED,
                model_path=str(MODEL_PATH),
                tokenizer_path=str(TOKENIZER_DIR),
                config=config.to_dict(),
                eval_metrics={},
                architecture_changes=[],
                metadata_={
                    "bootstrap": True,
                    "trained": False,
                    "note": "Initial random weights. Requires training.",
                },
            )

            db.add(generation)

        else:
            generation.status = GenerationStatus.PROMOTED
            generation.model_path = str(MODEL_PATH)
            generation.tokenizer_path = str(TOKENIZER_DIR)
            generation.config = config.to_dict()
            generation.metadata_ = {
                **(generation.metadata_ or {}),
                "bootstrap": True,
                "trained": False,
                "note": "Initial random weights. Requires training.",
            }

        await db.commit()

    await engine.dispose()

    print("✓ Generation 000001 registered")

    print("\n" + "=" * 60)
    print("GENERATION 000001 BOOTSTRAP COMPLETE")
    print("=" * 60)
    print(f"MODEL_PATH={MODEL_PATH}")
    print(f"TOKENIZER_PATH={TOKENIZER_DIR}")
    print()
    print("IMPORTANT:")
    print("The model currently contains RANDOM WEIGHTS.")
    print("It can be loaded, but meaningful language requires training.")


if __name__ == "__main__":
    asyncio.run(main())
