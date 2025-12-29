"""
Checkpoint handler for PSTM.

Provides save/resume functionality for long-running migrations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pstm.config.models import MigrationConfig
from pstm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointState:
    """State stored in a checkpoint."""

    # Identity
    config_hash: str
    created_at: str
    updated_at: str

    # Progress
    total_tiles: int
    completed_tiles: list[int] = field(default_factory=list)

    # Metrics
    total_traces_processed: int = 0
    total_compute_time_s: float = 0.0

    # Warnings/errors accumulated
    warnings: list[str] = field(default_factory=list)

    @property
    def n_completed(self) -> int:
        """Number of completed tiles."""
        return len(self.completed_tiles)

    @property
    def progress_fraction(self) -> float:
        """Completion progress as fraction."""
        if self.total_tiles == 0:
            return 0.0
        return self.n_completed / self.total_tiles

    def is_tile_completed(self, tile_id: int) -> bool:
        """Check if a tile has been completed."""
        return tile_id in self.completed_tiles

    def mark_tile_completed(self, tile_id: int) -> None:
        """Mark a tile as completed."""
        if tile_id not in self.completed_tiles:
            self.completed_tiles.append(tile_id)
            self.updated_at = datetime.now().isoformat()


class CheckpointHandler:
    """
    Handles checkpoint save/load for migration resume.

    Checkpoint contents:
    - state.json: Progress state and metadata
    - completed.npy: Boolean mask of completed tiles (for fast loading)
    - metrics.jsonl: Per-tile metrics log
    """

    STATE_FILE = "state.json"
    COMPLETED_FILE = "completed.npy"
    METRICS_FILE = "metrics.jsonl"

    def __init__(
        self,
        checkpoint_dir: Path | str,
        config: MigrationConfig,
        total_tiles: int,
    ):
        """
        Initialize checkpoint handler.

        Args:
            checkpoint_dir: Directory for checkpoint files
            config: Migration configuration (for hash verification)
            total_tiles: Total number of tiles
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.total_tiles = total_tiles

        # Compute config hash for verification
        self.config_hash = self._compute_config_hash(config)

        # Current state
        self._state: CheckpointState | None = None

    def _compute_config_hash(self, config: MigrationConfig) -> str:
        """Compute hash of configuration for verification."""
        # Serialize config to JSON for hashing
        # Exclude execution settings that don't affect output
        config_dict = config.model_dump(mode='json')

        # Remove execution-specific settings
        if "execution" in config_dict:
            exec_cfg = config_dict["execution"]
            exec_cfg.pop("checkpoint", None)
            exec_cfg.pop("verbose", None)
            exec_cfg.pop("log_file", None)

        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    @property
    def state(self) -> CheckpointState:
        """Get current state (load if needed)."""
        if self._state is None:
            self._state = self._create_new_state()
        return self._state

    def _create_new_state(self) -> CheckpointState:
        """Create a new checkpoint state."""
        now = datetime.now().isoformat()
        return CheckpointState(
            config_hash=self.config_hash,
            created_at=now,
            updated_at=now,
            total_tiles=self.total_tiles,
        )

    def exists(self) -> bool:
        """Check if a checkpoint exists."""
        state_file = self.checkpoint_dir / self.STATE_FILE
        return state_file.exists()

    def load(self) -> CheckpointState | None:
        """
        Load checkpoint if it exists and is valid.

        Returns:
            CheckpointState if valid checkpoint exists, None otherwise
        """
        if not self.exists():
            logger.debug("No checkpoint found")
            return None

        state_file = self.checkpoint_dir / self.STATE_FILE

        try:
            with open(state_file, "r") as f:
                data = json.load(f)

            state = CheckpointState(**data)

            # Verify config hash
            if state.config_hash != self.config_hash:
                logger.warning(
                    "Checkpoint config hash mismatch - configuration has changed. "
                    "Starting fresh."
                )
                return None

            # Verify total tiles
            if state.total_tiles != self.total_tiles:
                logger.warning(
                    f"Checkpoint tile count mismatch: {state.total_tiles} != {self.total_tiles}. "
                    "Starting fresh."
                )
                return None

            # Try to load completed mask for faster lookup
            completed_file = self.checkpoint_dir / self.COMPLETED_FILE
            if completed_file.exists():
                completed_mask = np.load(completed_file)
                state.completed_tiles = list(np.where(completed_mask)[0])

            self._state = state
            logger.info(
                f"Loaded checkpoint: {state.n_completed}/{state.total_tiles} tiles completed "
                f"({state.progress_fraction * 100:.1f}%)"
            )
            return state

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            return None

    def save(self) -> None:
        """Save current state to checkpoint."""
        if self._state is None:
            return

        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        self._state.updated_at = datetime.now().isoformat()

        # Save state JSON (convert numpy types to native Python for JSON serialization)
        state_file = self.checkpoint_dir / self.STATE_FILE
        state_dict = asdict(self._state)
        # Convert completed_tiles list to native Python ints
        state_dict["completed_tiles"] = [int(t) for t in state_dict["completed_tiles"]]
        with open(state_file, "w") as f:
            json.dump(state_dict, f, indent=2)

        # Save completed mask for fast loading
        completed_file = self.checkpoint_dir / self.COMPLETED_FILE
        completed_mask = np.zeros(self.total_tiles, dtype=bool)
        completed_mask[self._state.completed_tiles] = True
        np.save(completed_file, completed_mask)

        logger.debug(f"Saved checkpoint: {self._state.n_completed} tiles completed")

    def mark_completed(
        self,
        tile_id: int,
        n_traces: int = 0,
        compute_time: float = 0.0,
    ) -> None:
        """
        Mark a tile as completed.

        Args:
            tile_id: Tile ID
            n_traces: Number of traces processed
            compute_time: Compute time in seconds
        """
        self.state.mark_tile_completed(tile_id)
        self.state.total_traces_processed += n_traces
        self.state.total_compute_time_s += compute_time

        # Log metrics
        self._log_metrics(tile_id, n_traces, compute_time)

    def _log_metrics(
        self,
        tile_id: int,
        n_traces: int,
        compute_time: float,
    ) -> None:
        """Append tile metrics to log file."""
        metrics_file = self.checkpoint_dir / self.METRICS_FILE
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "tile_id": tile_id,
            "n_traces": n_traces,
            "compute_time_s": compute_time,
            "timestamp": datetime.now().isoformat(),
        }

        with open(metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def add_warning(self, message: str) -> None:
        """Add a warning to the checkpoint state."""
        self.state.warnings.append(message)

    def get_remaining_tiles(self) -> list[int]:
        """Get list of tile IDs that haven't been completed."""
        completed = set(self.state.completed_tiles)
        return [i for i in range(self.total_tiles) if i not in completed]

    def cleanup(self) -> None:
        """Remove checkpoint files after successful completion."""
        if not self.checkpoint_dir.exists():
            return

        try:
            for file in [self.STATE_FILE, self.COMPLETED_FILE, self.METRICS_FILE]:
                path = self.checkpoint_dir / file
                if path.exists():
                    path.unlink()

            # Remove directory if empty
            if not any(self.checkpoint_dir.iterdir()):
                self.checkpoint_dir.rmdir()

            logger.info("Checkpoint cleaned up")

        except Exception as e:
            logger.warning(f"Failed to clean up checkpoint: {e}")

    def get_summary(self) -> dict[str, Any]:
        """Get checkpoint summary."""
        return {
            "exists": self.exists(),
            "config_hash": self.config_hash,
            "total_tiles": self.total_tiles,
            "completed_tiles": self.state.n_completed if self._state else 0,
            "progress_percent": f"{self.state.progress_fraction * 100:.1f}%" if self._state else "0%",
            "checkpoint_dir": str(self.checkpoint_dir),
        }


def should_checkpoint(
    tile_id: int,
    interval: int,
    force: bool = False,
) -> bool:
    """
    Determine if we should save a checkpoint.

    Args:
        tile_id: Current tile ID (0-indexed)
        interval: Checkpoint interval (save every N tiles)
        force: Force checkpoint regardless of interval

    Returns:
        True if checkpoint should be saved
    """
    if force:
        return True

    # Checkpoint every `interval` tiles
    return (tile_id + 1) % interval == 0
