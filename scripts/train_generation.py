#!/usr/bin/env python3
"""
Training Script for Entity Generations

Runs training loop for a specific generation using consolidated dataset.
Creates checkpoints, evaluates, and determines promotion.
"""

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from api.config import settings
from api.database import Base
from api.models import Generation, TrainingRun
from brain.config import EntityConfig
from brain.tokenizer import BPETokenizer
from training.loop import TrainingLoop, TrainingConfig


async def main():
    parser = argparse.ArgumentParser(description="Train an entity generation")
    parser.add_argument("generation", type=int, help="Generation number to train")
    parser.add_argument("--dataset", type=str, help="Path to training dataset")
    parser.add_argument("--max-steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"TRAINING GENERATION {args.generation:06d}")
    print("=" * 60)

    # Database setup
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Verify generation exists
        stmt = select(Generation).where(Generation.number == args.generation)
        result = await db.execute(stmt)
        gen = result.scalar_one_or_none()

        if not gen:
            print(f"✗ Generation {args.generation} not found")
            return 1

        print(f"✓ Found Generation {args.generation} (ID: {gen.id})")
        print(f"  Status: {gen.status}")
        print(f"  Active: {gen.is_active}")

        # Load tokenizer from snapshot
        from evolution.snapshot import SnapshotManager
        snapshot_manager = SnapshotManager(db)
        manifest = await snapshot_manager.get_snapshot_manifest(args.generation)

        if not manifest:
            print(f"✗ No snapshot found for generation {args.generation}")
            return 1

        tokenizer_path = snapshot_manager._get_snapshot_dir(args.generation) / manifest.tokenizer_path
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"✓ Loaded tokenizer (vocab: {tokenizer.vocab_size})")

        # Dataset path
        dataset_path = args.dataset
        if not dataset_path:
            # Default to consolidation dataset
            dataset_path = Path(settings.DATASET_ROOT) / f"generation_{args.generation:06d}" / "consolidation"
            dataset_files = list(dataset_path.glob("*.jsonl"))
            if not dataset_files:
                print(f"✗ No dataset found at {dataset_path}")
                return 1
            # Use most recent
            dataset_path = max(dataset_files, key=lambda f: f.stat().st_mtime)

        dataset_path = Path(dataset_path)
        print(f"✓ Using dataset: {dataset_path}")

        # Load entity config from snapshot
        config = EntityConfig(**gen.config_snapshot)

        # Training config
        training_config = TrainingConfig(
            generation=args.generation,
            dataset_path=str(dataset_path),
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            device=args.device,
        )

        # Initialize training loop
        loop = TrainingLoop(db, training_config, config, tokenizer)

        # Resume if requested
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            await loop.load_checkpoint(Path(args.resume))

        # Run training
        print()
        print("Starting training...")
        print("-" * 60)

        result = await loop.run_training()

        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Run ID: {result.run_id}")
        print(f"Status: {result.status}")
        print(f"Steps: {result.total_steps}")
        print(f"Epochs: {result.total_epochs:.2f}")
        print(f"Final Train Loss: {result.final_train_loss:.4f}")
        print(f"Final Eval Loss: {result.final_eval_loss:.4f}" if result.final_eval_loss else "Final Eval Loss: N/A")
        print(f"Final Perplexity: {result.final_perplexity:.2f}" if result.final_perplexity else "Final Perplexity: N/A")
        print(f"Best Eval Loss: {result.best_eval_loss:.4f}" if result.best_eval_loss else "Best Eval Loss: N/A")
        print(f"Best Step: {result.best_step}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Promoted: {result.promoted}")
        if result.promoted:
            print(f"Promotion Reason: {result.promotion_reason}")

        return 0 if result.status == "success" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))