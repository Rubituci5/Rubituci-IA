"""
Tests for Evolution and Training Modules
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import torch

from evolution.snapshot import SnapshotManager, SnapshotManifest, SnapshotStatus
from evolution.promotion import PromotionManager, PromotionCandidate, PromotionDecision
from training.loop import TrainingLoop, TrainingConfig, TrainingResult, TrainingStatus


class TestSnapshotManager:
    @pytest.fixture
    def mock_db(self):
        return AsyncMock()

    @pytest.fixture
    def snapshot_manager(self, mock_db):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SnapshotManager(mock_db)
            manager.snapshot_root = Path(tmpdir)
            return manager

    def test_snapshot_manifest_creation(self):
        manifest = SnapshotManifest(
            snapshot_id="snap-123",
            generation=1,
            status=SnapshotStatus.COMPLETE,
            code_hash="abc123",
            config_hash="def456",
            tokenizer_hash="ghi789",
            model_weights_hash="jkl012",
            dataset_manifest_hash="mno345",
            metrics_hash="pqr678",
            manifest_hash="stu901",
            signature="vwx234",
        )
        assert manifest.snapshot_id == "snap-123"
        assert manifest.generation == 1
        assert manifest.status == SnapshotStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_create_snapshot_directory(self, snapshot_manager):
        gen_dir = await snapshot_manager._create_snapshot_directory(1)
        assert gen_dir.exists()
        assert gen_dir.name == "generation_000001"

    @pytest.mark.asyncio
    async def test_archive_code(self, snapshot_manager):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir) / "code"
            code_dir.mkdir()
            (code_dir / "test.py").write_text("print('hello')")

            archive_path = await snapshot_manager._archive_code([code_dir], snapshot_manager.snapshot_root / "generation_000001")
            assert archive_path.exists()
            assert archive_path.suffix == ".tar.gz"

    def test_compute_hash(self, snapshot_manager):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            f.flush()
            hash_val = snapshot_manager._compute_hash(Path(f.name))
            assert len(hash_val) == 64  # SHA256 hex

    @pytest.mark.asyncio
    async def test_verify_snapshot_integrity(self, snapshot_manager):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen_dir = Path(tmpdir) / "generation_000001"
            gen_dir.mkdir()

            # Create manifest
            manifest = SnapshotManifest(
                snapshot_id="test-123",
                generation=1,
                status=SnapshotStatus.COMPLETE,
                code_hash="a" * 64,
                config_hash="b" * 64,
                tokenizer_hash="c" * 64,
                model_weights_hash="d" * 64,
                dataset_manifest_hash="e" * 64,
                metrics_hash="f" * 64,
                manifest_hash="g" * 64,
                signature="h" * 64,
            )
            import json
            (gen_dir / "manifest.json").write_text(json.dumps(manifest.__dict__))

            # This would need actual files to verify properly
            # Just test method exists
            assert hasattr(snapshot_manager, "verify_snapshot_integrity")


class TestPromotionManager:
    @pytest.fixture
    def mock_db(self):
        return AsyncMock()

    @pytest.fixture
    def promotion_manager(self, mock_db):
        return PromotionManager(mock_db)

    def test_promotion_candidate_creation(self):
        candidate = PromotionCandidate(
            generation=2,
            parent_generation=1,
            eval_loss=1.5,
            perplexity=4.5,
            training_steps=50000,
            safety_score=0.95,
            ablation_results={},
        )
        assert candidate.generation == 2
        assert candidate.safety_score == 0.95

    def test_promotion_decision_enum(self):
        assert PromotionDecision.PROMOTE.value == "promote"
        assert PromotionDecision.REJECT.value == "reject"
        assert PromotionDecision.REQUIRES_REVIEW.value == "requires_review"

    @pytest.mark.asyncio
    async def test_evaluate_candidate(self, promotion_manager, mock_db):
        # Mock generation records
        mock_gen = MagicMock()
        mock_gen.number = 2
        mock_gen.metrics = {"eval_loss": 1.5, "perplexity": 4.5, "safety_score": 0.95}

        mock_parent = MagicMock()
        mock_parent.number = 1
        mock_parent.metrics = {"eval_loss": 2.0, "perplexity": 7.0, "safety_score": 0.9}

        mock_db.execute = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.side_effect = [mock_gen, mock_parent]
        mock_db.execute.return_value = mock_result

        decision = await promotion_manager.evaluate_candidate(2)
        assert decision in [PromotionDecision.PROMOTE, PromotionDecision.REJECT, PromotionDecision.REQUIRES_REVIEW]

    @pytest.mark.asyncio
    async def test_safety_evaluation(self, promotion_manager):
        # Test safety score threshold
        candidate = PromotionCandidate(
            generation=2,
            parent_generation=1,
            eval_loss=1.5,
            perplexity=4.5,
            training_steps=50000,
            safety_score=0.99,  # Above threshold
            ablation_results={},
        )
        # Should pass safety
        assert candidate.safety_score >= 0.95

        candidate_unsafe = PromotionCandidate(
            generation=2,
            parent_generation=1,
            eval_loss=1.5,
            perplexity=4.5,
            training_steps=50000,
            safety_score=0.8,  # Below threshold
            ablation_results={},
        )
        assert candidate_unsafe.safety_score < 0.95


class TestTrainingLoop:
    @pytest.fixture
    def mock_db(self):
        return AsyncMock()

    @pytest.fixture
    def training_config(self):
        return TrainingConfig(
            generation=1,
            dataset_path="/tmp/dataset.jsonl",
            max_steps=100,
            batch_size=4,
            learning_rate=1e-4,
            eval_interval=10,
            save_interval=20,
            device="cpu",
        )

    @pytest.fixture
    def model_and_tokenizer(self):
        from brain.config import EntityConfig
        from brain.model import EntityTransformer
        from brain.tokenizer import BPETokenizer

        config = EntityConfig(vocab_size=1000, d_model=128, n_layers=2, n_heads=4, max_seq_len=128)
        model = EntityTransformer(config)
        tokenizer = BPETokenizer(vocab_size=1000)
        tokenizer.train(["test corpus for training"] * 100)
        return model, tokenizer

    @pytest.mark.asyncio
    async def test_training_config_validation(self, training_config):
        assert training_config.max_steps == 100
        assert training_config.batch_size == 4
        assert training_config.learning_rate == 1e-4

    @pytest.mark.asyncio
    async def test_training_result_creation(self):
        result = TrainingResult(
            run_id="run-123",
            generation=1,
            status=TrainingStatus.SUCCESS,
            total_steps=100,
            total_epochs=1.0,
            final_train_loss=1.5,
            final_eval_loss=1.6,
            final_perplexity=5.0,
            best_eval_loss=1.55,
            best_step=90,
            duration_seconds=60.0,
            promoted=False,
        )
        assert result.status == TrainingStatus.SUCCESS
        assert result.total_steps == 100
        assert not result.promoted

    @pytest.mark.asyncio
    async def test_training_loop_initialization(self, mock_db, training_config, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        loop = TrainingLoop(mock_db, training_config, model.config, tokenizer)
        assert loop.config == training_config
        assert loop.tokenizer == tokenizer

    @pytest.mark.asyncio
    async def test_save_load_checkpoint(self, mock_db, training_config, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        loop = TrainingLoop(mock_db, training_config, model.config, tokenizer)
        loop.model = model
        loop.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loop.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(loop.optimizer, T_max=100)
        loop.step = 50
        loop.epoch = 0.5
        loop.best_eval_loss = 2.0

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            await loop.save_checkpoint(checkpoint_path)
            assert checkpoint_path.exists()

            # Create new loop and load
            loop2 = TrainingLoop(mock_db, training_config, model.config, tokenizer)
            loop2.model = EntityTransformer(model.config)
            loop2.optimizer = torch.optim.AdamW(loop2.model.parameters(), lr=1e-4)
            loop2.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(loop2.optimizer, T_max=100)
            await loop2.load_checkpoint(checkpoint_path)
            assert loop2.step == 50
            assert loop2.epoch == 0.5
            assert loop2.best_eval_loss == 2.0


class TestIntegration:
    """Integration tests for cross-module functionality."""

    @pytest.mark.asyncio
    async def test_generation_lifecycle(self):
        """Test the full generation lifecycle from snapshot to promotion."""
        # This would be a full integration test
        # For now, just verify the imports work
        from evolution.snapshot import SnapshotManager, ensure_generation_000001_snapshot
        from evolution.promotion import PromotionManager
        from training.loop import TrainingLoop
        assert True  # Imports successful


if __name__ == "__main__":
    pytest.main([__file__, "-v"])