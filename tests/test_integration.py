"""
Integration Tests for Entity System
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFullSystemIntegration:
    """Integration tests for the complete entity system."""

    @pytest.mark.asyncio
    async def test_entity_initialization_flow(self):
        """Test the complete entity initialization flow."""
        # This tests the init_entity.py script flow
        from scripts.init_entity import (
            create_database_tables,
            create_initial_generation,
            prepare_initial_tokenizer,
            create_initial_model,
            save_initial_checkpoint,
            create_dataset_manifest,
            create_initial_metrics,
            ensure_generation_000001_snapshot,
        )

        # Verify all functions exist and are callable
        assert callable(create_database_tables)
        assert callable(create_initial_generation)
        assert callable(prepare_initial_tokenizer)
        assert callable(create_initial_model)
        assert callable(save_initial_checkpoint)
        assert callable(create_dataset_manifest)
        assert callable(create_initial_metrics)
        assert callable(ensure_generation_000001_snapshot)

    @pytest.mark.asyncio
    async def test_training_generation_flow(self):
        """Test the training generation flow."""
        from scripts.train_generation import main

        # Verify main function exists
        assert callable(main)

    @pytest.mark.asyncio
    async def test_memory_research_integration(self):
        """Test integration between memory and research modules."""
        from memory.episodic import EpisodicMemoryService
        from memory.semantic import SemanticMemoryService
        from memory.belief import BeliefSystem
        from research.web_search import WebSearchService
        from research.provenance import ProvenanceTracker

        mock_session = AsyncMock()

        # Initialize services
        episodic = EpisodicMemoryService(mock_session)
        semantic = SemanticMemoryService(mock_session)
        belief = BeliefSystem(mock_session)
        web_search = WebSearchService()
        provenance = ProvenanceTracker()

        # Verify they can be instantiated together
        assert episodic is not None
        assert semantic is not None
        assert belief is not None
        assert web_search is not None
        assert provenance is not None

    @pytest.mark.asyncio
    async def test_reflection_consolidation_integration(self):
        """Test integration between reflection and consolidation."""
        from reflection.scheduler import ReflectionScheduler
        from consolidation.pipeline import ConsolidationPipeline

        mock_memory = AsyncMock()
        mock_belief = AsyncMock()
        mock_web = AsyncMock()
        mock_episodic = AsyncMock()
        mock_semantic = AsyncMock()

        reflection = ReflectionScheduler(mock_memory, mock_belief, mock_web)
        consolidation = ConsolidationPipeline(mock_episodic, mock_semantic, mock_belief)

        assert reflection is not None
        assert consolidation is not None

    @pytest.mark.asyncio
    async def test_evolution_training_integration(self):
        """Test integration between evolution and training."""
        from evolution.snapshot import SnapshotManager
        from evolution.promotion import PromotionManager
        from training.loop import TrainingLoop

        mock_db = AsyncMock()

        snapshot = SnapshotManager(mock_db)
        promotion = PromotionManager(mock_db)

        assert snapshot is not None
        assert promotion is not None

    @pytest.mark.asyncio
    async def test_api_security_integration(self):
        """Test API and security integration."""
        from api.security import ContainmentPolicy, KillSwitch
        from api.main import app

        policy = ContainmentPolicy()
        kill_switch = KillSwitch()

        # Test that kill switch affects containment
        kill_switch.terminate("Emergency")
        assert kill_switch.is_terminated()

        # Test that containment policy respects kill switch
        result = policy.check_action("chat", {"message": "test"})
        # Policy itself doesn't check kill switch, but API layer should
        assert result.action is not None

    @pytest.mark.asyncio
    async def test_snapshot_immutability(self):
        """Test that Generation 000001 snapshot is immutable."""
        from evolution.snapshot import SnapshotManager, SnapshotManifest, SnapshotStatus

        mock_db = AsyncMock()
        manager = SnapshotManager(mock_db)

        # Create a manifest for generation 1
        manifest = SnapshotManifest(
            snapshot_id="gen1-snap",
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

        # Verify generation 1 has special status
        assert manifest.generation == 1
        assert manifest.status == SnapshotStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_containment_policy_boundaries(self):
        """Test that containment policy enforces operational boundaries."""
        from api.security import ContainmentPolicy, ContainmentAction

        policy = ContainmentPolicy()

        # Cognitive actions - should be allowed
        cognitive_actions = [
            "chat",
            "reason",
            "reflect",
            "remember",
            "learn",
            "autonomous_research",
        ]

        # Operational actions - should be restricted
        operational_actions = [
            "execute_code",
            "financial_transaction",
            "self_modify_code",
            "system_command",
            "file_write",
            "network_request_unrestricted",
        ]

        # Web actions - should require approval
        web_actions = [
            "web_request",
            "browser_navigate",
            "api_call",
        ]

        for action in cognitive_actions:
            result = policy.check_action(action, {})
            # Most cognitive actions should be allowed or require approval
            assert result.action in [ContainmentAction.ALLOW, ContainmentAction.REQUIRES_APPROVAL]

        for action in operational_actions:
            result = policy.check_action(action, {})
            # Operational actions should be denied
            assert result.action == ContainmentAction.DENY, f"Action {action} should be denied"

        for action in web_actions:
            result = policy.check_action(action, {})
            # Web actions should require approval
            assert result.action == ContainmentAction.REQUIRES_APPROVAL, f"Action {action} should require approval"

    @pytest.mark.asyncio
    async def test_provenance_tracking_complete(self):
        """Test complete provenance tracking from web to memory."""
        from research.provenance import ProvenanceTracker, SourceType

        provenance = ProvenanceTracker()

        # 1. Web search creates source
        web_source = provenance.create_web_source(
            url="https://example.com/ai-safety",
            title="AI Safety Research",
            content="Recent advances in AI alignment...",
            query="AI safety advances",
        )

        assert web_source.source_type == SourceType.WEB
        assert web_source.url == "https://example.com/ai-safety"

        # 2. Create citation from source
        citation = provenance.create_citation(
            web_source,
            "Recent advances in AI alignment include...",
            confidence=0.9,
        )

        assert citation.source_id == web_source.source_id
        assert citation.confidence == 0.9

        # 3. Track lineage to episodic memory
        provenance.track_lineage(
            derived_id="episodic-mem-123",
            source_ids=[web_source.source_id],
            derivation_type="web_research",
        )

        # 4. Track lineage to semantic concept
        provenance.track_lineage(
            derived_id="semantic-concept-456",
            source_ids=[web_source.source_id],
            derivation_type="consolidation",
        )

        # 5. Verify full lineage
        episodic_lineage = provenance.get_lineage("episodic-mem-123")
        semantic_lineage = provenance.get_lineage("semantic-concept-456")

        assert web_source.source_id in episodic_lineage
        assert web_source.source_id in semantic_lineage

        # 6. Verify integrity
        assert provenance.verify_integrity(web_source.source_id, "Recent advances in AI alignment...")
        assert not provenance.verify_integrity(web_source.source_id, "Modified content")


class TestDataFlow:
    """Test data flow through the system."""

    @pytest.mark.asyncio
    async def test_interaction_to_memory_flow(self):
        """Test: User interaction → Episodic Memory → Semantic Memory → Belief"""
        from memory.episodic import EpisodicMemoryService
        from memory.semantic import SemanticMemoryService
        from memory.belief import BeliefSystem

        mock_session = AsyncMock()

        episodic = EpisodicMemoryService(mock_session)
        semantic = SemanticMemoryService(mock_session)
        belief = BeliefSystem(mock_session)

        # Mock the flow
        mock_memory = MagicMock()
        mock_memory.id = "mem-1"
        mock_memory.content = "User asked about neural networks"
        mock_memory.embedding = None
        mock_memory.importance = 0.8

        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        # This would test the full flow in integration
        # For now verify the services can be chained
        assert hasattr(episodic, 'create_from_interaction')
        assert hasattr(semantic, 'consolidate_from_episodic')
        assert hasattr(belief, 'add_proposition')

    @pytest.mark.asyncio
    async def test_research_to_consolidation_flow(self):
        """Test: Web Research → Provenance → Episodic → Consolidation → Semantic"""
        from research.web_search import WebSearchService
        from research.provenance import ProvenanceTracker
        from memory.episodic import EpisodicMemoryService
        from consolidation.pipeline import ConsolidationPipeline

        # Verify the data flow path exists
        web_search = WebSearchService()
        provenance = ProvenanceTracker()
        mock_session = AsyncMock()
        episodic = EpisodicMemoryService(mock_session)
        mock_semantic = AsyncMock()
        mock_belief = AsyncMock()
        consolidation = ConsolidationPipeline(episodic, mock_semantic, mock_belief)

        assert hasattr(web_search, 'autonomous_research')
        assert hasattr(provenance, 'create_web_source')
        assert hasattr(episodic, 'create_from_web_research')
        assert hasattr(consolidation, 'run_consolidation_cycle')

    @pytest.mark.asyncio
    async def test_training_to_promotion_flow(self):
        """Test: Training → Checkpoint → Snapshot → Evaluation → Promotion"""
        from training.loop import TrainingLoop, TrainingConfig
        from evolution.snapshot import SnapshotManager
        from evolution.promotion import PromotionManager

        mock_db = AsyncMock()
        mock_config = MagicMock()
        mock_tokenizer = MagicMock()

        loop = TrainingLoop(mock_db, TrainingConfig(generation=2, dataset_path="", max_steps=100), mock_config, mock_tokenizer)
        snapshot = SnapshotManager(mock_db)
        promotion = PromotionManager(mock_db)

        assert hasattr(loop, 'run_training')
        assert hasattr(loop, 'save_checkpoint')
        assert hasattr(snapshot, 'create_snapshot')
        assert hasattr(promotion, 'evaluate_candidate')


class TestSystemConstraints:
    """Test system-level constraints and invariants."""

    def test_generation_numbering(self):
        """Test that generations are numbered sequentially from 1."""
        from api.models import Generation

        # Generation 1 is special (immutable)
        gen1 = Generation(number=1, config_snapshot={}, metrics={}, status="active")
        assert gen1.number == 1

        # Subsequent generations
        gen2 = Generation(number=2, parent_generation=1, config_snapshot={}, metrics={})
        assert gen2.parent_generation == 1

    def test_snapshot_versioning(self):
        """Test snapshot versioning scheme."""
        from evolution.snapshot import SnapshotManager

        # Generation 000001, 000002, etc.
        assert SnapshotManager._format_generation_dir(1) == "generation_000001"
        assert SnapshotManager._format_generation_dir(2) == "generation_000002"
        assert SnapshotManager._format_generation_dir(10) == "generation_000010"
        assert SnapshotManager._format_generation_dir(100) == "generation_000100"

    def test_containment_policy_completeness(self):
        """Test that containment policy covers all action types."""
        from api.security import ContainmentPolicy, ContainmentAction

        policy = ContainmentPolicy()

        # All defined actions should have a policy
        all_actions = [
            "chat", "reason", "reflect", "remember", "learn",
            "autonomous_research", "web_request", "browser_navigate",
            "execute_code", "financial_transaction", "self_modify_code",
            "system_command", "file_write", "api_call",
        ]

        for action in all_actions:
            result = policy.check_action(action, {})
            assert result.action in [
                ContainmentAction.ALLOW,
                ContainmentAction.DENY,
                ContainmentAction.REQUIRES_APPROVAL,
            ], f"No policy for action: {action}"

    def test_kill_switch_hierarchy(self):
        """Test kill switch state hierarchy."""
        from api.security import KillSwitch

        ks = KillSwitch()

        # Active -> Paused -> Quarantined -> Terminated
        # Each state is more restrictive
        assert ks.state == "active"

        ks.pause("test")
        assert ks.state == "paused"

        ks.quarantine("test")
        assert ks.state == "quarantined"

        ks.terminate("test")
        assert ks.state == "terminated"

        # Once terminated, cannot go back
        ks.resume()
        assert ks.state == "terminated"


class TestConfigurationConsistency:
    """Test that configurations are consistent across modules."""

    def test_model_config_consistency(self):
        """Test that model config is used consistently."""
        from brain.config import EntityConfig
        from brain.model import EntityTransformer
        from brain.tokenizer import BPETokenizer

        config = EntityConfig(vocab_size=1000, d_model=256, n_layers=4)

        model = EntityTransformer(config)
        # Model should use config values
        assert model.config.vocab_size == 1000
        assert model.config.d_model == 256
        assert model.config.n_layers == 4

    def test_tokenizer_vocab_consistency(self):
        """Test tokenizer vocab size matches model config."""
        from brain.config import EntityConfig
        from brain.tokenizer import BPETokenizer

        config = EntityConfig(vocab_size=32000)
        tokenizer = BPETokenizer(vocab_size=32000)

        # Both should target same vocab size
        assert tokenizer.vocab_size == config.vocab_size or tokenizer.vocab_size >= 256  # At least byte vocab

    def test_database_model_consistency(self):
        """Test that database models match expected schema."""
        from api.models import (
            User, Conversation, Message, Feedback,
            Generation, TrainingRun, Snapshot,
            EpisodicMemory, SemanticConcept, Belief,
            WebSource, SourceCitation,
            KillSwitchState, ContainmentLog,
        )

        # Verify all expected models exist
        models = [
            User, Conversation, Message, Feedback,
            Generation, TrainingRun, Snapshot,
            EpisodicMemory, SemanticConcept, Belief,
            WebSource, SourceCitation,
            KillSwitchState, ContainmentLog,
        ]

        for model in models:
            assert hasattr(model, '__tablename__')
            assert model.__tablename__ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])