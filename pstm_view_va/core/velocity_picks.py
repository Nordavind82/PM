"""Velocity pick data structures and management."""

import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Callable
import numpy as np


@dataclass
class VelocityPick:
    """Single velocity pick point."""
    time_ms: float          # Time in ms
    velocity: float         # Velocity in m/s
    confidence: float = 1.0 # Confidence 0-1 (for visualization)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'VelocityPick':
        return cls(
            time_ms=float(d['time_ms']),
            velocity=float(d['velocity']),
            confidence=float(d.get('confidence', 1.0))
        )


@dataclass
class VelocityPickSet:
    """Collection of velocity picks for one location."""
    il: int                              # Inline position
    xl: int                              # Crossline position
    picks: List[VelocityPick] = field(default_factory=list)
    modified: bool = False               # Has unsaved changes
    created: str = ""                    # ISO timestamp
    last_modified: str = ""              # ISO timestamp

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        self.last_modified = datetime.now().isoformat()

    def add_pick(self, time_ms: float, velocity: float, confidence: float = 1.0) -> VelocityPick:
        """Add a new pick, maintaining sort order by time."""
        pick = VelocityPick(time_ms, velocity, confidence)
        self.picks.append(pick)
        self.picks.sort(key=lambda p: p.time_ms)
        self.modified = True
        self.last_modified = datetime.now().isoformat()
        return pick

    def remove_pick(self, index: int) -> Optional[VelocityPick]:
        """Remove pick at index."""
        if 0 <= index < len(self.picks):
            pick = self.picks.pop(index)
            self.modified = True
            self.last_modified = datetime.now().isoformat()
            return pick
        return None

    def remove_pick_near(self, time_ms: float, velocity: float,
                         time_tol: float = 50.0, vel_tol: float = 100.0) -> Optional[VelocityPick]:
        """Remove pick nearest to given time/velocity within tolerance."""
        idx = self.find_pick_near(time_ms, velocity, time_tol, vel_tol)
        if idx is not None:
            return self.remove_pick(idx)
        return None

    def move_pick(self, index: int, new_time_ms: float, new_velocity: float):
        """Move pick to new position."""
        if 0 <= index < len(self.picks):
            self.picks[index].time_ms = new_time_ms
            self.picks[index].velocity = new_velocity
            self.picks.sort(key=lambda p: p.time_ms)
            self.modified = True
            self.last_modified = datetime.now().isoformat()

    def find_pick_near(self, time_ms: float, velocity: float,
                       time_tol: float = 50.0, vel_tol: float = 100.0) -> Optional[int]:
        """Find index of pick nearest to given position within tolerance."""
        best_idx = None
        best_dist = float('inf')

        for i, pick in enumerate(self.picks):
            dt = abs(pick.time_ms - time_ms) / time_tol
            dv = abs(pick.velocity - velocity) / vel_tol
            dist = np.sqrt(dt**2 + dv**2)

            if dist < best_dist and dist < 1.0:  # Within tolerance ellipse
                best_dist = dist
                best_idx = i

        return best_idx

    def get_velocity_function(self, t_coords: np.ndarray) -> np.ndarray:
        """Interpolate picks to get velocity function at given times."""
        if len(self.picks) == 0:
            return None

        if len(self.picks) == 1:
            # Single pick - constant velocity
            return np.full(len(t_coords), self.picks[0].velocity)

        # Extract times and velocities
        times = np.array([p.time_ms for p in self.picks])
        vels = np.array([p.velocity for p in self.picks])

        # Linear interpolation with extrapolation
        return np.interp(t_coords, times, vels)

    def to_dict(self) -> dict:
        return {
            'il': self.il,
            'xl': self.xl,
            'picks': [p.to_dict() for p in self.picks],
            'created': self.created,
            'last_modified': self.last_modified,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'VelocityPickSet':
        picks = [VelocityPick.from_dict(p) for p in d.get('picks', [])]
        return cls(
            il=int(d['il']),
            xl=int(d['xl']),
            picks=picks,
            created=d.get('created', ''),
            last_modified=d.get('last_modified', ''),
            modified=False
        )


class UndoRedoStack:
    """Manages undo/redo for velocity picks."""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.undo_stack: List[VelocityPickSet] = []
        self.redo_stack: List[VelocityPickSet] = []

    def push(self, state: VelocityPickSet):
        """Save current state for undo."""
        # Deep copy the state
        state_copy = VelocityPickSet.from_dict(state.to_dict())
        self.undo_stack.append(state_copy)

        # Clear redo stack on new action
        self.redo_stack.clear()

        # Limit stack size
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)

    def undo(self, current: VelocityPickSet) -> Optional[VelocityPickSet]:
        """Undo last action, return previous state."""
        if not self.undo_stack:
            return None

        # Save current to redo stack
        self.redo_stack.append(VelocityPickSet.from_dict(current.to_dict()))

        # Pop and return previous state
        return self.undo_stack.pop()

    def redo(self, current: VelocityPickSet) -> Optional[VelocityPickSet]:
        """Redo last undone action."""
        if not self.redo_stack:
            return None

        # Save current to undo stack
        self.undo_stack.append(VelocityPickSet.from_dict(current.to_dict()))

        # Pop and return redo state
        return self.redo_stack.pop()

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()


class VelocityPickManager:
    """Manages velocity picks with auto-save and persistence."""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else None
        self.picks_dir: Optional[Path] = None
        self.current_picks: Optional[VelocityPickSet] = None
        self.undo_redo = UndoRedoStack()
        self.on_change_callbacks: List[Callable] = []

        if self.base_dir:
            self.setup_directory(self.base_dir)

    def setup_directory(self, base_dir: str):
        """Setup picks directory."""
        self.base_dir = Path(base_dir)
        self.picks_dir = self.base_dir / 'velocity_picks'
        self.picks_dir.mkdir(parents=True, exist_ok=True)

        # Create backup directory
        backup_dir = self.picks_dir / 'backups'
        backup_dir.mkdir(exist_ok=True)

    def get_pick_file_path(self, il: int, xl: int) -> Path:
        """Get file path for picks at given location."""
        if not self.picks_dir:
            return None
        return self.picks_dir / f'picks_IL{il:04d}_XL{xl:04d}.json'

    def load_picks(self, il: int, xl: int) -> VelocityPickSet:
        """Load picks for location, create empty if not exists."""
        file_path = self.get_pick_file_path(il, xl)

        if file_path and file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.current_picks = VelocityPickSet.from_dict(data)
            except Exception as e:
                print(f"Error loading picks: {e}")
                self.current_picks = VelocityPickSet(il=il, xl=xl)
        else:
            self.current_picks = VelocityPickSet(il=il, xl=xl)

        self.undo_redo.clear()
        return self.current_picks

    def save_picks(self, picks: VelocityPickSet = None) -> bool:
        """Save picks to file."""
        picks = picks or self.current_picks
        if not picks or not self.picks_dir:
            return False

        file_path = self.get_pick_file_path(picks.il, picks.xl)
        if not file_path:
            return False

        try:
            # Create backup if file exists
            if file_path.exists():
                backup_dir = self.picks_dir / 'backups'
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = backup_dir / f'picks_IL{picks.il:04d}_XL{picks.xl:04d}_{timestamp}.json'
                shutil.copy(file_path, backup_path)

                # Keep only last 10 backups per location
                self._cleanup_backups(picks.il, picks.xl, keep=10)

            # Save current picks
            with open(file_path, 'w') as f:
                json.dump(picks.to_dict(), f, indent=2)

            picks.modified = False
            return True

        except Exception as e:
            print(f"Error saving picks: {e}")
            return False

    def _cleanup_backups(self, il: int, xl: int, keep: int = 10):
        """Remove old backup files, keeping only the most recent ones."""
        if not self.picks_dir:
            return

        backup_dir = self.picks_dir / 'backups'
        pattern = f'picks_IL{il:04d}_XL{xl:04d}_*.json'

        backups = sorted(backup_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

        # Remove old backups
        for backup in backups[:-keep]:
            try:
                backup.unlink()
            except Exception:
                pass

    def add_pick(self, time_ms: float, velocity: float, confidence: float = 1.0) -> VelocityPick:
        """Add a pick and auto-save."""
        if not self.current_picks:
            return None

        # Save state for undo
        self.undo_redo.push(self.current_picks)

        # Add pick
        pick = self.current_picks.add_pick(time_ms, velocity, confidence)

        # Auto-save
        self.save_picks()

        # Notify listeners
        self._notify_change()

        return pick

    def remove_pick(self, index: int) -> Optional[VelocityPick]:
        """Remove pick and auto-save."""
        if not self.current_picks:
            return None

        # Save state for undo
        self.undo_redo.push(self.current_picks)

        # Remove pick
        pick = self.current_picks.remove_pick(index)

        if pick:
            # Auto-save
            self.save_picks()

            # Notify listeners
            self._notify_change()

        return pick

    def move_pick(self, index: int, new_time_ms: float, new_velocity: float):
        """Move pick and auto-save."""
        if not self.current_picks:
            return

        # Save state for undo (only on drag start, not every move)
        # This should be called separately before drag starts

        # Move pick
        self.current_picks.move_pick(index, new_time_ms, new_velocity)

        # Auto-save
        self.save_picks()

        # Notify listeners
        self._notify_change()

    def save_undo_state(self):
        """Explicitly save state for undo (call before drag operations)."""
        if self.current_picks:
            self.undo_redo.push(self.current_picks)

    def undo(self) -> bool:
        """Undo last action."""
        if not self.current_picks:
            return False

        prev_state = self.undo_redo.undo(self.current_picks)
        if prev_state:
            self.current_picks = prev_state
            self.save_picks()
            self._notify_change()
            return True
        return False

    def redo(self) -> bool:
        """Redo last undone action."""
        if not self.current_picks:
            return False

        next_state = self.undo_redo.redo(self.current_picks)
        if next_state:
            self.current_picks = next_state
            self.save_picks()
            self._notify_change()
            return True
        return False

    def can_undo(self) -> bool:
        return self.undo_redo.can_undo()

    def can_redo(self) -> bool:
        return self.undo_redo.can_redo()

    def add_change_callback(self, callback: Callable):
        """Add callback to be notified on changes."""
        self.on_change_callbacks.append(callback)

    def remove_change_callback(self, callback: Callable):
        """Remove change callback."""
        if callback in self.on_change_callbacks:
            self.on_change_callbacks.remove(callback)

    def _notify_change(self):
        """Notify all listeners of change."""
        for callback in self.on_change_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in change callback: {e}")

    def get_velocity_function(self, t_coords: np.ndarray) -> Optional[np.ndarray]:
        """Get interpolated velocity function from current picks."""
        if not self.current_picks:
            return None
        return self.current_picks.get_velocity_function(t_coords)

    def get_all_pick_locations(self) -> List[Tuple[int, int]]:
        """Get list of all IL/XL locations that have picks."""
        if not self.picks_dir:
            return []

        locations = []
        for file in self.picks_dir.glob('picks_IL*_XL*.json'):
            try:
                # Parse IL and XL from filename
                name = file.stem
                parts = name.split('_')
                il = int(parts[1][2:])  # Remove 'IL' prefix
                xl = int(parts[2][2:])  # Remove 'XL' prefix
                locations.append((il, xl))
            except Exception:
                pass

        return sorted(locations)
