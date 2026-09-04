"""
Immutable Generation Snapshot System

Creates and manages cryptographically verified, immutable snapshots
of each generation. Generation 000001 snapshot is created before
any public interaction or autonomous navigation.

Key properties:
- Immutable: Never overwritten by subsequent generations
- Complete: Code, config, tokenizer, weights, dataset, metrics, hashes
- Verifiable: Cryptographic signatures and hash chains
- Auditable: Full provenance of what constitutes each generation
"""

import hashlib
import json
import shutil
import tarfile
import uuid
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, BinaryIO
from dataclasses import dataclass, field, asdict
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.models import Generation, GenerationSnapshot
from api.config import settings
from brain.config import EntityConfig
from brain.tokenizer import BPETokenizer


class SnapshotStatus(str, Enum):
    """Status of a snapshot."""
    CREATING = "creating"
    COMPLETE = "complete"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"
    ARCHIVED = "archived"


@dataclass
class SnapshotManifest:
    """Complete manifest of a generation snapshot."""
    snapshot_id: str
    generation: int
    created_at: str
    status: str

    # Component hashes
    code_hash: str
    config_hash: str
    tokenizer_hash: str
    model_weights_hash: str
    dataset_manifest_hash: str
    metrics_hash: str

    # Component paths (relative to snapshot root)
    code_path: str
    config_path: str
    tokenizer_path: str
    model_weights_path: str
    dataset_manifest_path: str
    metrics_path: str

    # Metadata
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False
    python_version: str = ""
    dependencies_hash: str = ""
    architecture_hash: str = ""

    # Verification
    manifest_hash: str = ""
    signature: str = ""

    # Size info
    total_size_bytes: int = 0
    component_sizes: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotManifest":
        return cls(**data)

    def compute_manifest_hash(self) -> str:
        """Compute hash of manifest (excluding manifest_hash and signature)."""
        data = {k: v for k, v in self.to_dict().items() if k not in ("manifest_hash", "signature")}
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()


class SnapshotManager:
    """
    Manages immutable generation snapshots.

    Snapshots are stored in: {SNAPSHOT_ROOT}/generation_{gen:06d}/
    Each snapshot is a self-contained directory with manifest and components.
    """

    def __init__(self, db: AsyncSession, snapshot_root: Optional[Path] = None):
        self.db = db
        self.snapshot_root = snapshot_root or Path(settings.SNAPSHOT_ROOT)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)

    def _get_snapshot_dir(self, generation: int) -> Path:
        """Get snapshot directory for a generation."""
        return self.snapshot_root / f"generation_{generation:06d}"

    def _get_manifest_path(self, generation: int) -> Path:
        """Get manifest file path."""
        return self._get_snapshot_dir(generation) / "manifest.json"

    async def create_generation_000001_snapshot(
        self,
        config: EntityConfig,
        tokenizer: BPETokenizer,
        model_weights_path: Path,
        dataset_manifest: Dict[str, Any],
        initial_metrics: Dict[str, Any],
        code_paths: List[Path],
        config_path: Path,
    ) -> SnapshotManifest:
        """
        Create the immutable Generation 000001 snapshot.

        This MUST be called before any public interaction or autonomous navigation.
        The snapshot is cryptographically sealed and never modified.
        """
        generation = 1
        snapshot_dir = self._get_snapshot_dir(generation)

        if snapshot_dir.exists():
            raise RuntimeError(
                f"Generation 000001 snapshot already exists at {snapshot_dir}. "
                "This snapshot is immutable and cannot be recreated."
            )

        snapshot_dir.mkdir(parents=True)

        snapshot_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        # Track component hashes and sizes
        component_hashes = {}
        component_sizes = {}

        # 1. Archive code
        code_archive = snapshot_dir / "code.tar.gz"
        code_hash = await self._archive_code(code_paths, code_archive)
        component_hashes["code"] = code_hash
        component_sizes["code"] = code_archive.stat().st_size

        # 2. Copy config
        config_dest = snapshot_dir / "config.json"
        shutil.copy2(config_path, config_dest)
        config_hash = self._hash_file(config_dest)
        component_hashes["config"] = config_hash
        component_sizes["config"] = config_dest.stat().st_size

        # 3. Save tokenizer
        tokenizer_dest = snapshot_dir / "tokenizer"
        tokenizer.save(tokenizer_dest)
        tokenizer_hash = self._hash_directory(tokenizer_dest)
        component_hashes["tokenizer"] = tokenizer_hash
        component_sizes["tokenizer"] = self._dir_size(tokenizer_dest)

        # 4. Copy model weights
        weights_dest = snapshot_dir / "model_weights.pt"
        shutil.copy2(model_weights_path, weights_dest)
        weights_hash = self._hash_file(weights_dest)
        component_hashes["model_weights"] = weights_hash
        component_sizes["model_weights"] = weights_dest.stat().st_size

        # 5. Save dataset manifest
        dataset_dest = snapshot_dir / "dataset_manifest.json"
        dataset_dest.write_text(json.dumps(dataset_manifest, sort_keys=True, indent=2))
        dataset_hash = self._hash_file(dataset_dest)
        component_hashes["dataset_manifest"] = dataset_hash
        component_sizes["dataset_manifest"] = dataset_dest.stat().st_size

        # 6. Save initial metrics
        metrics_dest = snapshot_dir / "initial_metrics.json"
        metrics_dest.write_text(json.dumps(initial_metrics, sort_keys=True, indent=2))
        metrics_hash = self._hash_file(metrics_dest)
        component_hashes["metrics"] = metrics_hash
        component_sizes["metrics"] = metrics_dest.stat().st_size

        # 7. Capture git state
        git_commit, git_branch, git_dirty = self._get_git_state()

        # 8. Capture environment
        python_version = self._get_python_version()
        dependencies_hash = self._get_dependencies_hash()
        architecture_hash = self._compute_architecture_hash(config)

        # Create manifest
        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            generation=generation,
            created_at=created_at,
            status=SnapshotStatus.CREATING.value,
            code_hash=component_hashes["code"],
            config_hash=component_hashes["config"],
            tokenizer_hash=component_hashes["tokenizer"],
            model_weights_hash=component_hashes["model_weights"],
            dataset_manifest_hash=component_hashes["dataset_manifest"],
            metrics_hash=component_hashes["metrics"],
            code_path="code.tar.gz",
            config_path="config.json",
            tokenizer_path="tokenizer",
            model_weights_path="model_weights.pt",
            dataset_manifest_path="dataset_manifest.json",
            metrics_path="initial_metrics.json",
            git_commit=git_commit,
            git_branch=git_branch,
            git_dirty=git_dirty,
            python_version=python_version,
            dependencies_hash=dependencies_hash,
            architecture_hash=architecture_hash,
            total_size_bytes=sum(component_sizes.values()),
            component_sizes=component_sizes,
        )

        # Compute manifest hash
        manifest.manifest_hash = manifest.compute_manifest_hash()

        # Sign manifest (in production, would use HSM or key management)
        manifest.signature = self._sign_manifest(manifest)

        # Save manifest
        manifest_path = self._get_manifest_path(generation)
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        # Update status
        manifest.status = SnapshotStatus.COMPLETE.value
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        # Verify immediately
        verified = await self.verify_snapshot(generation)
        if not verified:
            manifest.status = SnapshotStatus.CORRUPTED.value
            manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
            raise RuntimeError("Generation 000001 snapshot verification failed after creation")

        manifest.status = SnapshotStatus.VERIFIED.value
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        # Record in database
        await self._record_snapshot_in_db(manifest)

        return manifest

    async def create_generation_snapshot(
        self,
        generation: int,
        config: EntityConfig,
        tokenizer: BPETokenizer,
        model_weights_path: Path,
        dataset_manifest: Dict[str, Any],
        metrics: Dict[str, Any],
        code_paths: List[Path],
        config_path: Path,
        parent_snapshot_id: Optional[str] = None,
    ) -> SnapshotManifest:
        """
        Create a snapshot for a new generation (000002+).

        Unlike Generation 000001, subsequent generations can be created
        but each gets its own immutable snapshot directory.
        """
        snapshot_dir = self._get_snapshot_dir(generation)

        if snapshot_dir.exists():
            raise RuntimeError(
                f"Generation {generation:06d} snapshot already exists. "
                "Snapshots are immutable."
            )

        snapshot_dir.mkdir(parents=True)

        snapshot_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        component_hashes = {}
        component_sizes = {}

        # Archive code (may be same as parent or modified)
        code_archive = snapshot_dir / "code.tar.gz"
        code_hash = await self._archive_code(code_paths, code_archive)
        component_hashes["code"] = code_hash
        component_sizes["code"] = code_archive.stat().st_size

        # Config
        config_dest = snapshot_dir / "config.json"
        shutil.copy2(config_path, config_dest)
        config_hash = self._hash_file(config_dest)
        component_hashes["config"] = config_hash
        component_sizes["config"] = config_dest.stat().st_size

        # Tokenizer (may have evolved)
        tokenizer_dest = snapshot_dir / "tokenizer"
        tokenizer.save(tokenizer_dest)
        tokenizer_hash = self._hash_directory(tokenizer_dest)
        component_hashes["tokenizer"] = tokenizer_hash
        component_sizes["tokenizer"] = self._dir_size(tokenizer_dest)

        # Model weights
        weights_dest = snapshot_dir / "model_weights.pt"
        shutil.copy2(model_weights_path, weights_dest)
        weights_hash = self._hash_file(weights_dest)
        component_hashes["model_weights"] = weights_hash
        component_sizes["model_weights"] = weights_dest.stat().st_size

        # Dataset manifest
        dataset_dest = snapshot_dir / "dataset_manifest.json"
        dataset_dest.write_text(json.dumps(dataset_manifest, sort_keys=True, indent=2))
        dataset_hash = self._hash_file(dataset_dest)
        component_hashes["dataset_manifest"] = dataset_hash
        component_sizes["dataset_manifest"] = dataset_dest.stat().st_size

        # Metrics
        metrics_dest = snapshot_dir / "metrics.json"
        metrics_dest.write_text(json.dumps(metrics, sort_keys=True, indent=2))
        metrics_hash = self._hash_file(metrics_dest)
        component_hashes["metrics"] = metrics_hash
        component_sizes["metrics"] = metrics_dest.stat().st_size

        # Git state
        git_commit, git_branch, git_dirty = self._get_git_state()

        # Environment
        python_version = self._get_python_version()
        dependencies_hash = self._get_dependencies_hash()
        architecture_hash = self._compute_architecture_hash(config)

        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            generation=generation,
            created_at=created_at,
            status=SnapshotStatus.CREATING.value,
            code_hash=component_hashes["code"],
            config_hash=component_hashes["config"],
            tokenizer_hash=component_hashes["tokenizer"],
            model_weights_hash=component_hashes["model_weights"],
            dataset_manifest_hash=component_hashes["dataset_manifest"],
            metrics_hash=component_hashes["metrics"],
            code_path="code.tar.gz",
            config_path="config.json",
            tokenizer_path="tokenizer",
            model_weights_path="model_weights.pt",
            dataset_manifest_path="dataset_manifest.json",
            metrics_path="metrics.json",
            git_commit=git_commit,
            git_branch=git_branch,
            git_dirty=git_dirty,
            python_version=python_version,
            dependencies_hash=dependencies_hash,
            architecture_hash=architecture_hash,
            total_size_bytes=sum(component_sizes.values()),
            component_sizes=component_sizes,
        )

        manifest.manifest_hash = manifest.compute_manifest_hash()
        manifest.signature = self._sign_manifest(manifest)

        manifest_path = self._get_manifest_path(generation)
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        manifest.status = SnapshotStatus.COMPLETE.value
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        verified = await self.verify_snapshot(generation)
        if not verified:
            manifest.status = SnapshotStatus.CORRUPTED.value
            manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))
            raise RuntimeError(f"Generation {generation:06d} snapshot verification failed")

        manifest.status = SnapshotStatus.VERIFIED.value
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        await self._record_snapshot_in_db(manifest, parent_snapshot_id)

        return manifest

    async def verify_snapshot(self, generation: int) -> bool:
        """Verify a snapshot's integrity."""
        snapshot_dir = self._get_snapshot_dir(generation)
        manifest_path = self._get_manifest_path(generation)

        if not manifest_path.exists():
            return False

        try:
            manifest_data = json.loads(manifest_path.read_text())
            manifest = SnapshotManifest.from_dict(manifest_data)
        except Exception:
            return False

        # Verify manifest hash
        if manifest.manifest_hash != manifest.compute_manifest_hash():
            return False

        # Verify each component
        checks = [
            ("code", manifest.code_path, manifest.code_hash),
            ("config", manifest.config_path, manifest.config_hash),
            ("tokenizer", manifest.tokenizer_path, manifest.tokenizer_hash),
            ("model_weights", manifest.model_weights_path, manifest.model_weights_hash),
            ("dataset_manifest", manifest.dataset_manifest_path, manifest.dataset_manifest_hash),
            ("metrics", manifest.metrics_path, manifest.metrics_hash),
        ]

        for name, rel_path, expected_hash in checks:
            abs_path = snapshot_dir / rel_path
            if not abs_path.exists():
                return False

            if abs_path.is_dir():
                actual_hash = self._hash_directory(abs_path)
            else:
                actual_hash = self._hash_file(abs_path)

            if actual_hash != expected_hash:
                return False

        # Verify signature
        if not self._verify_signature(manifest):
            return False

        return True

    async def get_snapshot_manifest(self, generation: int) -> Optional[SnapshotManifest]:
        """Get snapshot manifest if it exists."""
        manifest_path = self._get_manifest_path(generation)
        if not manifest_path.exists():
            return None

        try:
            data = json.loads(manifest_path.read_text())
            return SnapshotManifest.from_dict(data)
        except Exception:
            return None

    async def list_snapshots(self) -> List[SnapshotManifest]:
        """List all available snapshots."""
        snapshots = []
        for gen_dir in sorted(self.snapshot_root.glob("generation_*")):
            try:
                gen_num = int(gen_dir.name.split("_")[1])
                manifest = await self.get_snapshot_manifest(gen_num)
                if manifest:
                    snapshots.append(manifest)
            except (IndexError, ValueError):
                continue
        return snapshots

    async def export_snapshot(self, generation: int, output_path: Path) -> bool:
        """Export a snapshot as a verified archive."""
        if not await self.verify_snapshot(generation):
            return False

        snapshot_dir = self._get_snapshot_dir(generation)

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(snapshot_dir, arcname=f"generation_{generation:06d}")

        return True

    async def _record_snapshot_in_db(
        self,
        manifest: SnapshotManifest,
        parent_snapshot_id: Optional[str] = None,
    ):
        """Record snapshot in database."""
        snapshot = GenerationSnapshot(
            snapshot_id=uuid.UUID(manifest.snapshot_id),
            generation=manifest.generation,
            manifest=manifest.to_dict(),
            status=manifest.status,
            created_at=datetime.fromisoformat(manifest.created_at.replace("Z", "+00:00")),
            verified_at=datetime.now(timezone.utc) if manifest.status == SnapshotStatus.VERIFIED.value else None,
            parent_snapshot_id=uuid.UUID(parent_snapshot_id) if parent_snapshot_id else None,
        )
        self.db.add(snapshot)
        await self.db.commit()

    def _hash_file(self, path: Path) -> str:
        """Compute SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _hash_directory(self, path: Path) -> str:
        """Compute deterministic hash of a directory."""
        hasher = hashlib.sha256()

        # Get all files in deterministic order
        files = sorted(path.rglob("*"))
        for f in files:
            if f.is_file():
                # Hash relative path
                rel_path = f.relative_to(path)
                hasher.update(str(rel_path).encode())
                hasher.update(b"\0")
                # Hash content
                with open(f, "rb") as fp:
                    for chunk in iter(lambda: fp.read(8192), b""):
                        hasher.update(chunk)
                hasher.update(b"\0")

        return hasher.hexdigest()

    def _dir_size(self, path: Path) -> int:
        """Get total size of directory."""
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    async def _archive_code(self, code_paths: List[Path], output: Path) -> str:
        """Create deterministic tar.gz of code."""
        with tarfile.open(output, "w:gz", format=tarfile.GNU_FORMAT) as tar:
            for code_path in sorted(code_paths):
                if code_path.exists():
                    if code_path.is_file():
                        tar.add(code_path, arcname=code_path.name)
                    else:
                        for f in sorted(code_path.rglob("*")):
                            if f.is_file():
                                arcname = f.relative_to(code_path.parent)
                                tar.add(f, arcname=str(arcname))

        # Ensure deterministic archive (sort members, fixed timestamps)
        # Recreate with fixed mtime
        import tarfile
        with tarfile.open(output, "r:gz") as tar:
            members = tar.getmembers()
            for m in members:
                m.mtime = 0  # Fixed timestamp
                m.uid = 0
                m.gid = 0
                m.uname = ""
                m.gname = ""

        with tarfile.open(output, "w:gz", format=tarfile.GNU_FORMAT) as tar:
            for m in sorted(members, key=lambda x: x.name):
                f = tar.extractfile(m)
                if f:
                    tar.addfile(m, f)

        return self._hash_file(output)

    def _get_git_state(self) -> tuple[Optional[str], Optional[str], bool]:
        """Get current git commit, branch, and dirty status."""
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path.cwd(),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            commit = None

        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=Path.cwd(),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            branch = None

        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=Path.cwd(),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            dirty = bool(status)
        except Exception:
            dirty = False

        return commit, branch, dirty

    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_dependencies_hash(self) -> str:
        """Get hash of pinned dependencies."""
        # Would read from pyproject.toml or requirements.txt
        try:
            pyproject = Path("pyproject.toml")
            if pyproject.exists():
                return self._hash_file(pyproject)
        except Exception:
            pass
        return ""

    def _compute_architecture_hash(self, config: EntityConfig) -> str:
        """Compute hash of model architecture configuration."""
        arch_data = {
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "d_ff": config.d_ff,
            "max_seq_len": config.max_seq_len,
            "rope_theta": config.rope_theta,
            "rms_norm_eps": config.rms_norm_eps,
            "tie_embeddings": config.tie_embeddings,
            "use_bias": config.use_bias,
        }
        canonical = json.dumps(arch_data, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    def _sign_manifest(self, manifest: SnapshotManifest) -> str:
        """Sign manifest (placeholder - would use proper signing in production)."""
        # In production: use HSM, KMS, or proper key management
        # For now: HMAC with a secret (not secure, just for structure)
        secret = settings.SNAPSHOT_SIGNING_SECRET.encode() if settings.SNAPSHOT_SIGNING_SECRET else b"dev-secret"
        data = manifest.manifest_hash.encode()
        import hmac
        return hmac.new(secret, data, hashlib.sha256).hexdigest()

    def _verify_signature(self, manifest: SnapshotManifest) -> bool:
        """Verify manifest signature."""
        expected = self._sign_manifest(manifest)
        return hmac.compare_digest(manifest.signature, expected)


# Convenience function for Generation 000001
async def ensure_generation_000001_snapshot(
    db: AsyncSession,
    config: EntityConfig,
    tokenizer: BPETokenizer,
    model_weights_path: Path,
    dataset_manifest: Dict[str, Any],
    initial_metrics: Dict[str, Any],
    code_paths: List[Path],
    config_path: Path,
) -> SnapshotManifest:
    """
    Ensure Generation 000001 snapshot exists.

    Call this BEFORE any public interaction or autonomous navigation.
    Raises if snapshot doesn't exist and cannot be created.
    """
    manager = SnapshotManager(db)

    # Check if already exists
    existing = await manager.get_snapshot_manifest(1)
    if existing:
        # Verify it's still valid
        if await manager.verify_snapshot(1):
            return existing
        else:
            raise RuntimeError("Generation 000001 snapshot exists but verification failed!")

    # Create it
    return await manager.create_generation_000001_snapshot(
        config=config,
        tokenizer=tokenizer,
        model_weights_path=model_weights_path,
        dataset_manifest=dataset_manifest,
        initial_metrics=initial_metrics,
        code_paths=code_paths,
        config_path=config_path,
    )


# Import hmac for signature verification
import hmac