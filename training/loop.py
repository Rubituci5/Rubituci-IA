"""
Training Loop with Checkpoints

Manages the training process for each generation:
- Dataset preparation from consolidated data
- Training with checkpointing
- Evaluation metrics
- Generation promotion criteria
"""

import uuid
import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from api.config import settings
from api.models import Generation, GenerationSnapshot, TrainingRun, TrainingCheckpoint
from brain.config import EntityConfig
from brain.model import EntityTransformer
from brain.tokenizer import BPETokenizer
from brain.inference import InferenceEngine
from evolution.snapshot import SnapshotManager, ensure_generation_000001_snapshot


celery_app = Celery(
    "training",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)


class TrainingStage(str, Enum):
    """Training stages."""
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    CHECKPOINTING = "checkpointing"
    PROMOTING = "promoting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    """Training configuration."""
    generation: int
    dataset_path: str
    max_steps: int = 100000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation: int = 1
    eval_interval: int = 500
    save_interval: int = 1000
    max_checkpoints: int = 5
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    mixed_precision: bool = True
    compile_model: bool = False
    device: str = "auto"


@dataclass
class TrainingMetrics:
    """Training metrics at a checkpoint."""
    step: int
    epoch: float
    train_loss: float
    eval_loss: Optional[float]
    perplexity: Optional[float]
    learning_rate: float
    grad_norm: float
    tokens_per_second: float
    memory_allocated_gb: float
    timestamp: str


@dataclass
class TrainingResult:
    """Final training result."""
    run_id: uuid.UUID
    generation: int
    status: str
    total_steps: int
    total_epochs: float
    final_train_loss: float
    final_eval_loss: Optional[float]
    final_perplexity: Optional[float]
    best_eval_loss: Optional[float]
    best_step: int
    checkpoints: List[str]
    final_checkpoint: str
    duration_seconds: float
    promoted: bool
    promotion_reason: str = ""


class TextDataset(Dataset):
    """Simple text dataset for training."""

    def __init__(self, file_path: Path, tokenizer: BPETokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        # Load JSONL dataset
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                # Handle different formats
                if "prompt" in data and "completion" in data:
                    text = data["prompt"] + "\n" + data["completion"]
                elif "text" in data:
                    text = data["text"]
                elif "content" in data:
                    text = data["content"]
                else:
                    text = str(data)

                tokens = tokenizer.encode(text)
                if len(tokens) > 10:  # Minimum length
                    self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # Truncate or pad
        if len(tokens) > self.max_seq_len:
            start = torch.randint(0, len(tokens) - self.max_seq_len, (1,)).item()
            tokens = tokens[start:start + self.max_seq_len]
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch, pad_token_id: int, max_seq_len: int):
    """Collate batch with padding."""
    max_len = min(max(len(x) for x in batch), max_seq_len)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq[:max_len]
    return padded


class TrainingLoop:
    """
    Manages training for a generation.

    Features:
    - Checkpointing with rotation
    - Mixed precision training
    - Evaluation during training
    - Early stopping
    - Generation promotion evaluation
    """

    def __init__(
        self,
        db: AsyncSession,
        config: TrainingConfig,
        entity_config: EntityConfig,
        tokenizer: BPETokenizer,
    ):
        self.db = db
        self.config = config
        self.entity_config = entity_config
        self.tokenizer = tokenizer

        # Device setup
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Model
        self.model = EntityTransformer(entity_config).to(self.device)
        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.01,
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and self.device.type == "cuda" else None

        # State
        self.step = 0
        self.epoch = 0.0
        self.best_eval_loss = float("inf")
        self.best_step = 0
        self.patience_counter = 0
        self.checkpoints: List[Path] = []
        self.metrics_history: List[TrainingMetrics] = []

        # Dataset
        self.dataset_path = Path(config.dataset_path)
        self.train_loader = None
        self.eval_loader = None

        # Output dirs
        self.run_id = uuid.uuid4()
        self.output_dir = Path(settings.CHECKPOINT_ROOT) / f"generation_{config.generation:06d}" / f"run_{self.run_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def prepare_data(self, eval_split: float = 0.05) -> Tuple[DataLoader, DataLoader]:
        """Prepare train and eval dataloaders."""
        dataset = TextDataset(
            self.dataset_path,
            self.tokenizer,
            self.entity_config.max_seq_len,
        )

        # Split
        eval_size = int(len(dataset) * eval_split)
        train_size = len(dataset) - eval_size
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

        # Dataloaders
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, pad_id, self.entity_config.max_seq_len),
            num_workers=4,
            pin_memory=True,
        )

        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, pad_id, self.entity_config.max_seq_len),
            num_workers=2,
            pin_memory=True,
        )

        return self.train_loader, self.eval_loader

    async def train_step(self, batch: torch.Tensor) -> Tuple[float, float]:
        """Single training step."""
        self.model.train()
        batch = batch.to(self.device)

        # Forward
        if self.scaler:
            with torch.cuda.amp.autocast():
                logits, loss = self.model(batch, targets=batch)
                loss = loss / self.config.gradient_accumulation

            self.scaler.scale(loss).backward()
        else:
            logits, loss = self.model(batch, targets=batch)
            loss = loss / self.config.gradient_accumulation
            loss.backward()

        # Gradient clipping and step
        if (self.step + 1) % self.config.gradient_accumulation == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()
        else:
            grad_norm = 0.0

        return loss.item() * self.config.gradient_accumulation, grad_norm.item()

    async def evaluate(self) -> Tuple[float, float]:
        """Run evaluation."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating", leave=False):
                batch = batch.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        logits, loss = self.model(batch, targets=batch)
                else:
                    logits, loss = self.model(batch, targets=batch)

                # Loss is averaged over batch, multiply by tokens
                seq_len = batch.size(1)
                total_loss += loss.item() * seq_len * batch.size(0)
                total_tokens += seq_len * batch.size(0)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 20 else float("inf")

        return avg_loss, perplexity

    def save_checkpoint(self, metrics: TrainingMetrics, is_best: bool = False) -> Path:
        """Save training checkpoint."""
        checkpoint_name = f"checkpoint_step_{self.step:08d}.pt"
        if is_best:
            checkpoint_name = "checkpoint_best.pt"

        checkpoint_path = self.output_dir / checkpoint_name

        checkpoint = {
            "run_id": str(self.run_id),
            "generation": self.config.generation,
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "entity_config": self.entity_config.__dict__,
            "training_config": self.config.__dict__,
            "metrics": metrics.__dict__,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
        }

        torch.save(checkpoint, checkpoint_path)

        # Track checkpoints for rotation
        self.checkpoints.append(checkpoint_path)
        if len(self.checkpoints) > self.config.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if oldest.name != "checkpoint_best.pt" and oldest.exists():
                oldest.unlink()

        return checkpoint_path

    async def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load training checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if self.scaler and checkpoint.get("scaler_state_dict"):
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            self.step = checkpoint["step"]
            self.epoch = checkpoint["epoch"]
            self.best_eval_loss = checkpoint["metrics"].get("eval_loss", float("inf"))
            self.best_step = checkpoint.get("best_step", self.step)

            return True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False

    async def run_training(self) -> TrainingResult:
        """Run the complete training loop."""
        start_time = datetime.now(timezone.utc)

        # Record training run in DB
        run = TrainingRun(
            id=self.run_id,
            generation=self.config.generation,
            status=TrainingStage.PREPARING.value,
            config=self.config.__dict__,
            started_at=start_time,
        )
        self.db.add(run)
        await self.db.commit()

        try:
            # Prepare data
            await self.prepare_data()

            run.status = TrainingStage.TRAINING.value
            await self.db.commit()

            # Training loop
            tokens_processed = 0
            step_start_time = datetime.now(timezone.utc)

            for epoch in range(int(self.config.max_steps / len(self.train_loader)) + 1):
                self.epoch = epoch + self.step / len(self.train_loader)

                for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False):
                    # Training step
                    loss, grad_norm = await self.train_step(batch)

                    tokens_processed += batch.numel()
                    self.step += 1

                    # Logging
                    if self.step % 10 == 0:
                        step_duration = (datetime.now(timezone.utc) - step_start_time).total_seconds()
                        tokens_per_sec = tokens_processed / step_duration if step_duration > 0 else 0
                        mem_gb = torch.cuda.memory_allocated() / 1e9 if self.device.type == "cuda" else 0

                        metrics = TrainingMetrics(
                            step=self.step,
                            epoch=self.epoch,
                            train_loss=loss,
                            eval_loss=None,
                            perplexity=None,
                            learning_rate=self.scheduler.get_last_lr()[0],
                            grad_norm=grad_norm,
                            tokens_per_second=tokens_per_sec,
                            memory_allocated_gb=mem_gb,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                        self.metrics_history.append(metrics)

                    # Evaluation
                    if self.step % self.config.eval_interval == 0:
                        run.status = TrainingStage.EVALUATING.value
                        await self.db.commit()

                        eval_loss, perplexity = await self.evaluate()

                        # Update metrics
                        if self.metrics_history:
                            self.metrics_history[-1].eval_loss = eval_loss
                            self.metrics_history[-1].perplexity = perplexity

                        # Check for improvement
                        is_best = eval_loss < self.best_eval_loss - self.config.early_stopping_min_delta
                        if is_best:
                            self.best_eval_loss = eval_loss
                            self.best_step = self.step
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1

                        run.status = TrainingStage.TRAINING.value
                        await self.db.commit()

                        print(f"Step {self.step}: train_loss={loss:.4f}, eval_loss={eval_loss:.4f}, ppl={perplexity:.2f}")

                        # Early stopping
                        if self.patience_counter >= self.config.early_stopping_patience:
                            print(f"Early stopping at step {self.step}")
                            break

                    # Checkpointing
                    if self.step % self.config.save_interval == 0:
                        run.status = TrainingStage.CHECKPOINTING.value
                        await self.db.commit()

                        metrics = TrainingMetrics(
                            step=self.step,
                            epoch=self.epoch,
                            train_loss=loss,
                            eval_loss=self.metrics_history[-1].eval_loss if self.metrics_history else None,
                            perplexity=self.metrics_history[-1].perplexity if self.metrics_history else None,
                            learning_rate=self.scheduler.get_last_lr()[0],
                            grad_norm=grad_norm,
                            tokens_per_second=tokens_per_sec,
                            memory_allocated_gb=mem_gb,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )

                        checkpoint_path = self.save_checkpoint(metrics, is_best=is_best)

                        # Record checkpoint in DB
                        cp = TrainingCheckpoint(
                            id=uuid.uuid4(),
                            run_id=self.run_id,
                            step=self.step,
                            path=str(checkpoint_path),
                            metrics=metrics.__dict__,
                            is_best=is_best,
                        )
                        self.db.add(cp)
                        await self.db.commit()

                    # Max steps check
                    if self.step >= self.config.max_steps:
                        break

                if self.step >= self.config.max_steps:
                    break
                if self.patience_counter >= self.config.early_stopping_patience:
                    break

            # Final evaluation
            final_eval_loss, final_perplexity = await self.evaluate()

            # Save final checkpoint
            final_metrics = TrainingMetrics(
                step=self.step,
                epoch=self.epoch,
                train_loss=loss,
                eval_loss=final_eval_loss,
                perplexity=final_perplexity,
                learning_rate=self.scheduler.get_last_lr()[0],
                grad_norm=grad_norm,
                tokens_per_second=tokens_processed / (datetime.now(timezone.utc) - step_start_time).total_seconds(),
                memory_allocated_gb=torch.cuda.memory_allocated() / 1e9 if self.device.type == "cuda" else 0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            final_checkpoint = self.save_checkpoint(final_metrics)

            # Determine promotion
            promoted, reason = await self._evaluate_promotion(final_eval_loss, final_perplexity)

            completed_at = datetime.now(timezone.utc)
            duration = (completed_at - start_time).total_seconds()

            # Update run record
            run.status = TrainingStage.COMPLETE.value if not promoted else TrainingStage.PROMOTING.value
            run.completed_at = completed_at
            run.duration_seconds = duration
            run.total_steps = self.step
            run.total_epochs = self.epoch
            run.final_train_loss = loss
            run.final_eval_loss = final_eval_loss
            run.final_perplexity = final_perplexity
            run.best_eval_loss = self.best_eval_loss
            run.best_step = self.best_step
            run.checkpoints = [str(c) for c in self.checkpoints]
            run.final_checkpoint = str(final_checkpoint)
            run.promoted = promoted
            run.promotion_reason = reason
            await self.db.commit()

            # If promoted, create new generation
            if promoted:
                await self._promote_generation(final_checkpoint, final_metrics)

            return TrainingResult(
                run_id=self.run_id,
                generation=self.config.generation,
                status="success",
                total_steps=self.step,
                total_epochs=self.epoch,
                final_train_loss=loss,
                final_eval_loss=final_eval_loss,
                final_perplexity=final_perplexity,
                best_eval_loss=self.best_eval_loss,
                best_step=self.best_step,
                checkpoints=[str(c) for c in self.checkpoints],
                final_checkpoint=str(final_checkpoint),
                duration_seconds=duration,
                promoted=promoted,
                promotion_reason=reason,
            )

        except Exception as e:
            completed_at = datetime.now(timezone.utc)
            duration = (completed_at - start_time).total_seconds()

            run.status = TrainingStage.FAILED.value
            run.completed_at = completed_at
            run.duration_seconds = duration
            run.error = str(e)
            await self.db.commit()

            return TrainingResult(
                run_id=self.run_id,
                generation=self.config.generation,
                status="failed",
                total_steps=self.step,
                total_epochs=self.epoch,
                final_train_loss=0,
                final_eval_loss=None,
                final_perplexity=None,
                best_eval_loss=None,
                best_step=0,
                checkpoints=[],
                final_checkpoint="",
                duration_seconds=duration,
                promoted=False,
                promotion_reason="",
            )

    async def _evaluate_promotion(self, eval_loss: float, perplexity: float) -> Tuple[bool, str]:
        """Evaluate if generation should be promoted."""
        # Get previous generation metrics for comparison
        stmt = select(Generation).where(Generation.number == self.config.generation - 1)
        result = await self.db.execute(stmt)
        prev_gen = result.scalar_one_or_none()

        if not prev_gen:
            # First generation after 000001 - promote if reasonable
            return eval_loss < 5.0, "First trainable generation"

        prev_metrics = prev_gen.metrics or {}
        prev_eval_loss = prev_metrics.get("eval_loss", float("inf"))
        prev_perplexity = prev_metrics.get("perplexity", float("inf"))

        # Promotion criteria
        improvement_threshold = 0.95  # 5% improvement minimum
        min_perplexity = 20.0  # Must achieve reasonable perplexity

        loss_improved = eval_loss < prev_eval_loss * improvement_threshold
        ppl_reasonable = perplexity < min_perplexity
        ppl_improved = perplexity < prev_perplexity * improvement_threshold

        if loss_improved and ppl_reasonable and ppl_improved:
            return True, f"Improved: loss {prev_eval_loss:.4f} -> {eval_loss:.4f}, ppl {prev_perplexity:.2f} -> {perplexity:.2f}"

        return False, f"Not promoted: loss {prev_eval_loss:.4f} -> {eval_loss:.4f}, ppl {prev_perplexity:.2f} -> {perplexity:.2f}"

    async def _promote_generation(
        self,
        checkpoint_path: Path,
        metrics: TrainingMetrics,
    ):
        """Promote to next generation."""
        run = TrainingRun(
            id=uuid.uuid4(),
            generation=self.config.generation + 1,
            status="promoted",
            config={},
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_seconds=0,
            total_steps=self.step,
            final_eval_loss=metrics.eval_loss,
            final_perplexity=metrics.perplexity,
            promoted=True,
            promotion_reason=f"Promoted from generation {self.config.generation}",
        )
        self.db.add(run)

        # Create new generation record
        new_gen = Generation(
            number=self.config.generation + 1,
            parent_generation=self.config.generation,
            training_run_id=self.run_id,
            config_snapshot=self.entity_config.__dict__,
            metrics={
                "eval_loss": metrics.eval_loss,
                "perplexity": metrics.perplexity,
                "train_loss": metrics.train_loss,
                "steps": self.step,
            },
            status="training",
            is_active=False,  # Will be activated after snapshot
        )
        self.db.add(new_gen)
        await self.db.commit()

        # The snapshot creation would be triggered separately


# Celery Tasks

@celery_app.task(bind=True, max_retries=2)
def train_generation(self, generation: int, dataset_path: str, config_override: Optional[Dict] = None):
    """Train a generation as Celery task."""
    import asyncio
    from api.database import get_async_session

    async def _run():
        async for db in get_async_session():
            # Load entity config
            entity_config = EntityConfig()  # Would load from snapshot

            # Load tokenizer
            snapshot_manager = SnapshotManager(db)
            manifest = await snapshot_manager.get_snapshot_manifest(generation)
            if not manifest:
                raise ValueError(f"No snapshot for generation {generation}")

            tokenizer_path = snapshot_manager._get_snapshot_dir(generation) / manifest.tokenizer_path
            tokenizer = BPETokenizer.load(tokenizer_path)

            # Training config
            training_config = TrainingConfig(
                generation=generation,
                dataset_path=dataset_path,
                **(config_override or {}),
            )

            loop = TrainingLoop(db, training_config, entity_config, tokenizer)
            result = await loop.run_training()

            return {
                "run_id": str(result.run_id),
                "status": result.status,
                "steps": result.total_steps,
                "eval_loss": result.final_eval_loss,
                "perplexity": result.final_perplexity,
                "promoted": result.promoted,
                "reason": result.promotion_reason,
            }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=600)


@celery_app.task(bind=True)
def resume_training(self, run_id: str, checkpoint_path: str):
    """Resume training from checkpoint."""
    import asyncio
    from api.database import get_async_session

    async def _run():
        async for db in get_async_session():
            # Load run
            stmt = select(TrainingRun).where(TrainingRun.id == uuid.UUID(run_id))
            result = await db.execute(stmt)
            run = result.scalar_one_or_none()
            if not run:
                raise ValueError(f"Run {run_id} not found")

            # Setup same as train_generation but load checkpoint
            entity_config = EntityConfig()
            snapshot_manager = SnapshotManager(db)
            manifest = await snapshot_manager.get_snapshot_manifest(run.generation)
            tokenizer = BPETokenizer.load(snapshot_manager._get_snapshot_dir(run.generation) / manifest.tokenizer_path)

            training_config = TrainingConfig(
                generation=run.generation,
                dataset_path=run.config.get("dataset_path", ""),
            )

            loop = TrainingLoop(db, training_config, entity_config, tokenizer)
            await loop.load_checkpoint(Path(checkpoint_path))
            result = await loop.run_training()

            return {
                "run_id": str(result.run_id),
                "status": result.status,
                "steps": result.total_steps,
            }

    return asyncio.run(_run())


# Beat schedule - training triggered by consolidation, not on fixed schedule
# But we can have a periodic check for pending training
celery_app.conf.beat_schedule.update({
    "check-pending-training-every-hour": {
        "task": "training.loop.check_pending_training",
        "schedule": 3600.0,
    },
})

celery_app.conf.timezone = "UTC"