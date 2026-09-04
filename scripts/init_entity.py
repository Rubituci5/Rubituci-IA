#!/usr/bin/env python3
"""
Entity Initialization Script

This script MUST be run before any public interaction or autonomous navigation.
It creates the immutable Generation 000001 snapshot containing:
- Complete source code (git commit)
- Model configuration
- Tokenizer
- Initial model weights (checkpoint)
- Dataset manifest
- Initial metrics
- Cryptographic hashes for verification

The snapshot is immutable and will never be overwritten by subsequent generations.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from api.config import settings
from api.database import Base
from api.models import Generation
from brain.config import EntityConfig
from brain.tokenizer import BPETokenizer
from brain.model import EntityTransformer
from evolution.snapshot import SnapshotManager, ensure_generation_000001_snapshot


async def create_database_tables(engine):
    """Create all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Database tables created")


async def create_initial_generation(db: AsyncSession) -> Generation:
    """Create Generation 000001 record if it doesn't exist."""
    from sqlalchemy import select

    stmt = select(Generation).where(Generation.number == 1)
    result = await db.execute(stmt)
    gen = result.scalar_one_or_none()

    if gen:
        print(f"✓ Generation 000001 already exists (ID: {gen.id})")
        return gen

    config = EntityConfig()
    gen = Generation(
        number=1,
        parent_generation=None,
        config_snapshot=config.__dict__,
        metrics={},
        status="initialized",
        is_active=True,
    )
    db.add(gen)
    await db.commit()
    await db.refresh(gen)
    print(f"✓ Created Generation 000001 (ID: {gen.id})")
    return gen


async def prepare_initial_tokenizer(db: AsyncSession, generation: int) -> BPETokenizer:
    """Train or load initial tokenizer."""
    from evolution.snapshot import SnapshotManager

    snapshot_manager = SnapshotManager(db)
    manifest = await snapshot_manager.get_snapshot_manifest(1)

    if manifest:
        # Load from snapshot
        tokenizer_path = snapshot_manager._get_snapshot_dir(1) / manifest.tokenizer_path
        if tokenizer_path.exists():
            tokenizer = BPETokenizer.load(tokenizer_path)
            print(f"✓ Loaded tokenizer from snapshot (vocab: {tokenizer.vocab_size})")
            return tokenizer

    # Train new tokenizer from initial corpus
    print("Training initial tokenizer...")
    tokenizer = BPETokenizer(vocab_size=32000)

    # Initial training corpus - would be replaced with real data
    initial_corpus = [
        "The entity is a community digital entity for continuous development.",
        "It has persistent memory, autonomous learning, and collaborative evolution.",
        "The entity uses a custom transformer architecture with RoPE and RMSNorm.",
        "Knowledge is stored in episodic memory, semantic memory, and belief systems.",
        "Autonomous web research allows the entity to gather new information.",
        "Reflection cycles process memories and update beliefs.",
        "Consolidation during sleep forms semantic concepts from experiences.",
        "Generations are versioned with full lineage tracking.",
        "Community feedback provides evidence but not ground truth.",
        "Containment policy: cognitive freedom does not equal operational authority.",
    ] * 100  # Repeat for minimal training

    tokenizer.train(initial_corpus)
    print(f"✓ Trained tokenizer (vocab: {tokenizer.vocab_size})")
    return tokenizer


async def create_initial_model(config: EntityConfig, tokenizer: BPETokenizer) -> EntityTransformer:
    """Create initial model with random weights."""
    model = EntityTransformer(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Created initial model ({param_count:,} parameters)")
    return model


async def save_initial_checkpoint(model: EntityTransformer, tokenizer: BPETokenizer, output_dir: Path):
    """Save initial model checkpoint."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.config.__dict__,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "generation": 1,
        "step": 0,
        "epoch": 0.0,
    }

    checkpoint_path = output_dir / "model_weights.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Saved initial checkpoint to {checkpoint_path}")
    return checkpoint_path


async def create_dataset_manifest(data_dir: Path) -> dict:
    """Create initial dataset manifest."""
    manifest = {
        "version": "1.0",
        "generation": 1,
        "created_at": "2026-09-03T00:00:00Z",
        "datasets": [
            {
                "name": "initial_corpus",
                "type": "text",
                "size_bytes": 0,
                "num_examples": 0,
                "description": "Initial training corpus for Generation 000001",
                "hash": "pending",
            }
        ],
        "total_examples": 0,
        "total_tokens": 0,
    }

    manifest_path = data_dir / "dataset_manifest.json"
    import json
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"✓ Created dataset manifest at {manifest_path}")
    return manifest


async def create_initial_metrics() -> dict:
    """Create initial metrics (placeholder for untrained model)."""
    return {
        "eval_loss": None,
        "perplexity": None,
        "train_loss": None,
        "steps": 0,
        "epoch": 0.0,
        "tokens_trained": 0,
        "note": "Generation 000001 - initial random weights, not yet trained",
    }


def get_code_paths() -> list[Path]:
    """Get paths to archive for code snapshot."""
    root = Path(__file__).parent.parent
    return [
        root / "brain",
        root / "api",
        root / "memory",
        root / "research",
        root / "reflection",
        root / "consolidation",
        root / "evolution",
        root / "training",
        root / "pyproject.toml",
        root / "docker-compose.yml",
    ]


async def main():
    parser = argparse.ArgumentParser(description="Initialize Entity and create Generation 000001 snapshot")
    parser.add_argument("--force", action="store_true", help="Force recreate even if snapshot exists")
    parser.add_argument("--skip-db", action="store_true", help="Skip database initialization")
    args = parser.parse_args()

    print("=" * 60)
    print("ENTITY INITIALIZATION - GENERATION 000001 SNAPSHOT")
    print("=" * 60)
    print()

    # Database setup
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    if not args.skip_db:
        await create_database_tables(engine)

    async with async_session() as db:
        # Create Generation 000001 record
        generation = await create_initial_generation(db)

        # Prepare components
        config = EntityConfig()
        tokenizer = await prepare_initial_tokenizer(db, 1)
        model = await create_initial_model(config, tokenizer)

        # Paths
        snapshot_root = Path(settings.SNAPSHOT_ROOT)
        gen_dir = snapshot_root / "generation_000001"
        gen_dir.mkdir(parents=True, exist_ok=True)

        weights_path = await save_initial_checkpoint(model, tokenizer, gen_dir)
        dataset_manifest = await create_dataset_manifest(gen_dir)
        initial_metrics = await create_initial_metrics()
        code_paths = get_code_paths()
        config_path = gen_dir / "config.json"
        import json
        config_path.write_text(json.dumps(config.__dict__, indent=2))

        # Check if snapshot already exists
        snapshot_manager = SnapshotManager(db)
        existing = await snapshot_manager.get_snapshot_manifest(1)

        if existing and not args.force:
            print()
            print("⚠ Generation 000001 snapshot already exists!")
            print(f"  Snapshot ID: {existing.snapshot_id}")
            print(f"  Created: {existing.created_at}")
            print(f"  Status: {existing.status}")
            print()
            print("Use --force to recreate (NOT RECOMMENDED - snapshots are immutable)")
            return 0

        if existing and args.force:
            print()
            print("⚠ FORCE MODE: Overwriting existing snapshot!")
            print("  THIS BREAKS IMMUTABILITY GUARANTEES")
            print()

        # Create the immutable snapshot
        print()
        print("Creating immutable Generation 000001 snapshot...")
        print("-" * 60)

        try:
            manifest = await ensure_generation_000001_snapshot(
                db=db,
                config=config,
                tokenizer=tokenizer,
                model_weights_path=weights_path,
                dataset_manifest=dataset_manifest,
                initial_metrics=initial_metrics,
                code_paths=code_paths,
                config_path=config_path,
            )

            print()
            print("=" * 60)
            print("✓ GENERATION 000001 SNAPSHOT CREATED SUCCESSFULLY")
            print("=" * 60)
            print()
            print(f"Snapshot ID: {manifest.snapshot_id}")
            print(f"Generation: {manifest.generation}")
            print(f"Created: {manifest.created_at}")
            print(f"Status: {manifest.status}")
            print(f"Total Size: {manifest.total_size_bytes:,} bytes")
            print()
            print("Component Hashes:")
            print(f"  Code:           {manifest.code_hash[:16]}...")
            print(f"  Config:         {manifest.config_hash[:16]}...")
            print(f"  Tokenizer:      {manifest.tokenizer_hash[:16]}...")
            print(f"  Model Weights:  {manifest.model_weights_hash[:16]}...")
            print(f"  Dataset:        {manifest.dataset_manifest_hash[:16]}...")
            print(f"  Metrics:        {manifest.metrics_hash[:16]}...")
            print()
            print(f"Manifest Hash:  {manifest.manifest_hash[:16]}...")
            print(f"Signature:      {manifest.signature[:16]}...")
            print()
            print("Git State:")
            print(f"  Commit: {manifest.git_commit or 'N/A'}")
            print(f"  Branch: {manifest.git_branch or 'N/A'}")
            print(f"  Dirty:  {manifest.git_dirty}")
            print()
            print("⚠ THIS SNAPSHOT IS IMMUTABLE AND WILL NEVER BE OVERWRITTEN")
            print("   All future generations will have their own separate snapshots.")
            print()

            return 0

        except Exception as e:
            print(f"\n✗ SNAPSHOT CREATION FAILED: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))