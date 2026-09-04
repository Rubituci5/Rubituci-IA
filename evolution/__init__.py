"""
Entity Evolution System

Manages generation versioning, snapshots, and the evolution pipeline.
"""

from .snapshot import (
    SnapshotManager,
    SnapshotManifest,
    SnapshotStatus,
    ensure_generation_000001_snapshot,
)

__all__ = [
    "SnapshotManager",
    "SnapshotManifest",
    "SnapshotStatus",
    "ensure_generation_000001_snapshot",
]