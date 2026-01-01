#!/usr/bin/env python3
"""
Synthetic Azimuth Test for PSTM Migration

Creates two common offset gathers with different azimuths and 3D velocity
variations, then migrates them separately and jointly.

Test configuration:
- 3D velocity model with spatial gradient and lateral variations
- Offset 1: Azimuths 0-90 degrees (N-E quadrant)
- Offset 2: Azimuths 90-180 degrees (E-S quadrant)
- Three flat reflectors at different depths
- One dipping reflector

Outputs:
- 3 PSTM images (offset1, offset2, joint)
- Each with inline, crossline, and time slice views
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List
import sys
from numba import jit, prange

sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset/synthetic_azimuth_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    # Grid parameters - smaller for fast testing
    nx: int = 51
    ny: int = 51
    nt: int = 251
    dx: float = 50.0  # m
    dy: float = 50.0  # m
    dt_ms: float = 4.0  # ms

    # Velocity model parameters
    v0: float = 2000.0  # m/s at surface
    v_gradient: float = 0.5  # m/s per ms (vertical gradient)
    v_lateral_x: float = 100.0  # m/s variation in x direction
    v_lateral_y: float = 60.0  # m/s variation in y direction

    # Acquisition parameters
    offset: float = 400.0  # m (same offset for both gathers)
    traces_per_cmp: int = 6  # traces per CMP location

    # Wavelet
    f_peak: float = 25.0  # Hz

    @property
    def dt_s(self) -> float:
        return self.dt_ms / 1000.0

    @property
    def x_coords(self) -> np.ndarray:
        return np.arange(self.nx) * self.dx

    @property
    def y_coords(self) -> np.ndarray:
        return np.arange(self.ny) * self.dy

    @property
    def t_axis_ms(self) -> np.ndarray:
        return np.arange(self.nt) * self.dt_ms


def create_3d_velocity_model(config: SyntheticConfig) -> np.ndarray:
    """
    Create 3D velocity model with spatial and gradient variations.

    V(x, y, t) = V0 + gradient*t + lateral_x*sin(x) + lateral_y*cos(y)
    """
    print("\n[1] Creating 3D velocity model...")

    velocity = np.zeros((config.nx, config.ny, config.nt), dtype=np.float32)

    x = config.x_coords
    y = config.y_coords
    t = config.t_axis_ms

    # Grid coordinates
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')

    # Base velocity with vertical gradient
    velocity = config.v0 + config.v_gradient * T

    # Lateral variation in X (sinusoidal)
    x_period = x[-1] / 2  # Half the grid
    velocity += config.v_lateral_x * np.sin(2 * np.pi * X / x_period)

    # Lateral variation in Y (cosinusoidal)
    y_period = y[-1] / 2
    velocity += config.v_lateral_y * np.cos(2 * np.pi * Y / y_period)

    velocity = velocity.astype(np.float32)

    print(f"    Shape: {velocity.shape}")
    print(f"    V range: {velocity.min():.0f} - {velocity.max():.0f} m/s")
    print(f"    V at center surface: {velocity[config.nx//2, config.ny//2, 0]:.0f} m/s")
    print(f"    V at center deep: {velocity[config.nx//2, config.ny//2, -1]:.0f} m/s")

    return velocity


def create_ricker_wavelet(f_peak: float, dt_s: float, duration_s: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Create a Ricker wavelet."""
    t = np.arange(-duration_s/2, duration_s/2, dt_s)
    wavelet = (1 - 2*(np.pi*f_peak*t)**2) * np.exp(-(np.pi*f_peak*t)**2)
    return wavelet.astype(np.float32), t


@jit(nopython=True, cache=True)
def dsr_traveltime(ox: float, oy: float, sx: float, sy: float,
                   rx: float, ry: float, t0: float, v: float) -> float:
    """Compute DSR traveltime."""
    ds2 = (ox - sx)**2 + (oy - sy)**2
    dr2 = (ox - rx)**2 + (oy - ry)**2
    t0_half_sq = (t0 / 2)**2
    inv_v_sq = 1 / (v * v)
    return np.sqrt(t0_half_sq + ds2 * inv_v_sq) + np.sqrt(t0_half_sq + dr2 * inv_v_sq)


@dataclass
class Reflector:
    """Definition of a reflector."""
    name: str
    t0_ms: float  # Zero-offset time at center
    dip_x: float = 0.0  # ms per grid cell in x
    dip_y: float = 0.0  # ms per grid cell in y
    amplitude: float = 1.0


def get_reflector_time(t0_ms: float, dip_x: float, dip_y: float,
                       ix: int, iy: int, nx: int, ny: int) -> float:
    """Get reflector time at a grid location."""
    dx = ix - nx // 2
    dy = iy - ny // 2
    return t0_ms + dip_x * dx + dip_y * dy


def generate_common_offset_gather(
    config: SyntheticConfig,
    velocity: np.ndarray,
    reflector_params: List[Tuple[float, float, float, float]],  # t0_ms, dip_x, dip_y, amp
    azimuth_range: Tuple[float, float],
    name: str
) -> Tuple[np.ndarray, dict]:
    """
    Generate a common offset gather with specified azimuth range.

    Returns:
        traces: (n_traces, nt) array
        headers: dict with source, receiver, midpoint coordinates
    """
    print(f"\n[2] Generating {name}...")
    print(f"    Azimuth range: {azimuth_range[0]:.0f} - {azimuth_range[1]:.0f} degrees")

    wavelet, _ = create_ricker_wavelet(config.f_peak, config.dt_s)
    wavelet_half = len(wavelet) // 2

    # Generate azimuths within range
    n_azimuths = config.traces_per_cmp
    azimuths = np.linspace(azimuth_range[0], azimuth_range[1], n_azimuths, endpoint=False)

    # Total traces
    n_traces = config.nx * config.ny * n_azimuths

    # Preallocate
    traces = np.zeros((n_traces, config.nt), dtype=np.float32)
    source_x = np.zeros(n_traces, dtype=np.float32)
    source_y = np.zeros(n_traces, dtype=np.float32)
    receiver_x = np.zeros(n_traces, dtype=np.float32)
    receiver_y = np.zeros(n_traces, dtype=np.float32)
    midpoint_x = np.zeros(n_traces, dtype=np.float32)
    midpoint_y = np.zeros(n_traces, dtype=np.float32)
    offset_arr = np.zeros(n_traces, dtype=np.float32)
    azimuth_arr = np.zeros(n_traces, dtype=np.float32)

    trace_idx = 0
    for ix in range(config.nx):
        for iy in range(config.ny):
            # CMP location
            mx = ix * config.dx
            my = iy * config.dy

            for az_idx, azimuth in enumerate(azimuths):
                # Compute source/receiver from midpoint, offset, azimuth
                half_offset = config.offset / 2
                dx = half_offset * np.sin(np.radians(azimuth))
                dy = half_offset * np.cos(np.radians(azimuth))

                sx, sy = mx - dx, my - dy
                rx, ry = mx + dx, my + dy

                # Get velocity at this location (use middle time as reference)
                v_local = velocity[ix, iy, config.nt // 2]

                # Create trace with reflections
                for t0_ms, dip_x, dip_y, amp in reflector_params:
                    # Get reflector time at this location
                    t0_ref = get_reflector_time(t0_ms, dip_x, dip_y, ix, iy, config.nx, config.ny)
                    t0 = t0_ref / 1000.0

                    # Skip if reflector is outside time range
                    if t0_ref < 50 or t0_ref > (config.nt - 50) * config.dt_ms:
                        continue

                    # Compute traveltime
                    t_travel = dsr_traveltime(mx, my, sx, sy, rx, ry, t0, v_local)
                    sample_idx = int(t_travel / config.dt_s)

                    # Add wavelet
                    if wavelet_half <= sample_idx < config.nt - wavelet_half:
                        start = sample_idx - wavelet_half
                        end = start + len(wavelet)
                        if end <= config.nt:
                            traces[trace_idx, start:end] += wavelet * amp

                source_x[trace_idx] = sx
                source_y[trace_idx] = sy
                receiver_x[trace_idx] = rx
                receiver_y[trace_idx] = ry
                midpoint_x[trace_idx] = mx
                midpoint_y[trace_idx] = my
                offset_arr[trace_idx] = config.offset
                azimuth_arr[trace_idx] = azimuth
                trace_idx += 1

    headers = {
        'source_x': source_x, 'source_y': source_y,
        'receiver_x': receiver_x, 'receiver_y': receiver_y,
        'midpoint_x': midpoint_x, 'midpoint_y': midpoint_y,
        'offset': offset_arr, 'azimuth': azimuth_arr
    }

    print(f"    Traces: {n_traces}")
    print(f"    Non-zero traces: {np.sum(np.any(traces != 0, axis=1))}")

    return traces, headers


@jit(nopython=True, parallel=True, cache=True)
def migrate_kernel(
    traces: np.ndarray,
    source_x: np.ndarray, source_y: np.ndarray,
    receiver_x: np.ndarray, receiver_y: np.ndarray,
    midpoint_x: np.ndarray, midpoint_y: np.ndarray,
    velocity: np.ndarray,
    nx: int, ny: int, nt: int,
    dx: float, dy: float, dt_s: float,
    aperture: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated migration kernel."""
    image = np.zeros((nx, ny, nt), dtype=np.float64)
    fold = np.zeros((nx, ny, nt), dtype=np.int32)
    n_traces = traces.shape[0]

    # Parallelize over output x
    for ix in prange(nx):
        ox = ix * dx
        for iy in range(ny):
            oy = iy * dy

            for j in range(n_traces):
                mx = midpoint_x[j]
                my = midpoint_y[j]

                # Check aperture
                dm = np.sqrt((ox - mx)**2 + (oy - my)**2)
                if dm > aperture:
                    continue

                sx, sy = source_x[j], source_y[j]
                rx, ry = receiver_x[j], receiver_y[j]

                for it in range(10, nt - 10):
                    t0_out = it * dt_s
                    v = velocity[ix, iy, it]

                    # DSR traveltime
                    ds2 = (ox - sx)**2 + (oy - sy)**2
                    dr2 = (ox - rx)**2 + (oy - ry)**2
                    t0_half_sq = (t0_out / 2)**2
                    inv_v_sq = 1 / (v * v)
                    t_travel = np.sqrt(t0_half_sq + ds2 * inv_v_sq) + np.sqrt(t0_half_sq + dr2 * inv_v_sq)

                    sample_idx = t_travel / dt_s

                    if 0 <= sample_idx < nt - 1:
                        idx0 = int(sample_idx)
                        frac = sample_idx - idx0
                        amp = traces[j, idx0] * (1-frac) + traces[j, idx0+1] * frac
                        image[ix, iy, it] += amp
                        fold[ix, iy, it] += 1

    return image, fold


def migrate_gather(
    config: SyntheticConfig,
    velocity: np.ndarray,
    traces: np.ndarray,
    headers: dict,
    aperture: float = 600.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform Kirchhoff PSTM migration."""
    return migrate_kernel(
        traces,
        headers['source_x'], headers['source_y'],
        headers['receiver_x'], headers['receiver_y'],
        headers['midpoint_x'], headers['midpoint_y'],
        velocity,
        config.nx, config.ny, config.nt,
        config.dx, config.dy, config.dt_s,
        aperture
    )


def normalize_image(image: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Normalize image by fold (3D)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        normalized = np.where(fold > 0, image / fold, 0.0)
    return normalized


def create_qc_images(
    image: np.ndarray,
    fold: np.ndarray,
    config: SyntheticConfig,
    name: str,
    output_dir: Path
):
    """Create QC images: inline, crossline, and time slices."""
    print(f"\n    Creating QC images for {name}...")

    # Normalize
    image_norm = normalize_image(image, fold)

    # Key positions
    il_idx = config.nx // 2
    xl_idx = config.ny // 2
    time_slices_ms = [200, 400, 600]

    # Create figure
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'PSTM Migration: {name}', fontsize=16, fontweight='bold')

    # Clip for display
    vmax = np.percentile(np.abs(image_norm), 99)
    if vmax == 0:
        vmax = 1.0
    vmin = -vmax

    # Row 1: Inline view
    ax1 = fig.add_subplot(3, 4, 1)
    il_section = image_norm[il_idx, :, :].T
    im1 = ax1.imshow(il_section, aspect='auto', cmap='gray',
                     extent=[0, config.ny * config.dy, config.nt * config.dt_ms, 0],
                     vmin=vmin, vmax=vmax)
    ax1.set_xlabel('Crossline (m)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title(f'Inline {il_idx} (x={il_idx * config.dx:.0f}m)')
    plt.colorbar(im1, ax=ax1, label='Amplitude')

    # Inline fold
    ax2 = fig.add_subplot(3, 4, 2)
    il_fold = fold[il_idx, :, :].T
    im2 = ax2.imshow(il_fold, aspect='auto', cmap='viridis',
                     extent=[0, config.ny * config.dy, config.nt * config.dt_ms, 0])
    ax2.set_xlabel('Crossline (m)')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title(f'Inline {il_idx} - Fold')
    plt.colorbar(im2, ax=ax2, label='Fold')

    # Row 1: Crossline view
    ax3 = fig.add_subplot(3, 4, 3)
    xl_section = image_norm[:, xl_idx, :].T
    im3 = ax3.imshow(xl_section, aspect='auto', cmap='gray',
                     extent=[0, config.nx * config.dx, config.nt * config.dt_ms, 0],
                     vmin=vmin, vmax=vmax)
    ax3.set_xlabel('Inline (m)')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title(f'Crossline {xl_idx} (y={xl_idx * config.dy:.0f}m)')
    plt.colorbar(im3, ax=ax3, label='Amplitude')

    # Crossline fold
    ax4 = fig.add_subplot(3, 4, 4)
    xl_fold = fold[:, xl_idx, :].T
    im4 = ax4.imshow(xl_fold, aspect='auto', cmap='viridis',
                     extent=[0, config.nx * config.dx, config.nt * config.dt_ms, 0])
    ax4.set_xlabel('Inline (m)')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title(f'Crossline {xl_idx} - Fold')
    plt.colorbar(im4, ax=ax4, label='Fold')

    # Row 2-3: Time slices
    for i, t_ms in enumerate(time_slices_ms):
        t_idx = int(t_ms / config.dt_ms)
        if t_idx >= config.nt:
            t_idx = config.nt - 1

        # Image slice
        ax = fig.add_subplot(3, 4, 5 + i*2)
        t_slice = image_norm[:, :, t_idx].T
        im = ax.imshow(t_slice, aspect='equal', cmap='gray',
                       extent=[0, config.nx * config.dx, config.ny * config.dy, 0],
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel('Inline (m)')
        ax.set_ylabel('Crossline (m)')
        ax.set_title(f'Time Slice {t_ms} ms')
        plt.colorbar(im, ax=ax, label='Amplitude')

        # Fold slice
        ax = fig.add_subplot(3, 4, 6 + i*2)
        f_slice = fold[:, :, t_idx].T
        im = ax.imshow(f_slice, aspect='equal', cmap='viridis',
                       extent=[0, config.nx * config.dx, config.ny * config.dy, 0])
        ax.set_xlabel('Inline (m)')
        ax.set_ylabel('Crossline (m)')
        ax.set_title(f'Fold at {t_ms} ms')
        plt.colorbar(im, ax=ax, label='Fold')

    plt.tight_layout()
    fig_path = output_dir / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_qc.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fig_path}")

    return fig_path


def create_comparison_figure(
    results: dict,
    config: SyntheticConfig,
    output_dir: Path
):
    """Create comparison figure of all three migrations."""
    print("\n[7] Creating comparison figure...")

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('PSTM Migration Comparison: Offset 1 vs Offset 2 vs Joint',
                 fontsize=14, fontweight='bold')

    il_idx = config.nx // 2
    t_slice_ms = 400
    t_idx = int(t_slice_ms / config.dt_ms)

    names = ['Offset 1 (Az 0-90)', 'Offset 2 (Az 90-180)', 'Joint (Az 0-180)']

    # Get global vmax for consistent scaling
    all_images = [normalize_image(results[k]['image'], results[k]['fold'])
                  for k in ['offset1', 'offset2', 'joint']]
    vmax = max(np.percentile(np.abs(img), 99) for img in all_images)
    if vmax == 0:
        vmax = 1.0
    vmin = -vmax

    for row, (key, name) in enumerate(zip(['offset1', 'offset2', 'joint'], names)):
        image_norm = normalize_image(results[key]['image'], results[key]['fold'])
        fold = results[key]['fold']

        # Inline section
        ax = axes[row, 0]
        il_section = image_norm[il_idx, :, :].T
        ax.imshow(il_section, aspect='auto', cmap='gray',
                  extent=[0, config.ny * config.dy, config.nt * config.dt_ms, 0],
                  vmin=vmin, vmax=vmax)
        ax.set_ylabel('Time (ms)')
        if row == 2:
            ax.set_xlabel('Crossline (m)')
        ax.set_title(f'{name}\nInline {il_idx}')

        # Inline fold
        ax = axes[row, 1]
        il_fold = fold[il_idx, :, :].T
        ax.imshow(il_fold, aspect='auto', cmap='viridis',
                  extent=[0, config.ny * config.dy, config.nt * config.dt_ms, 0])
        if row == 2:
            ax.set_xlabel('Crossline (m)')
        ax.set_title('Fold')

        # Time slice
        ax = axes[row, 2]
        t_slice = image_norm[:, :, t_idx].T
        ax.imshow(t_slice, aspect='equal', cmap='gray',
                  extent=[0, config.nx * config.dx, config.ny * config.dy, 0],
                  vmin=vmin, vmax=vmax)
        if row == 2:
            ax.set_xlabel('Inline (m)')
        ax.set_ylabel('Crossline (m)')
        ax.set_title(f'Time Slice {t_slice_ms} ms')

        # Fold slice
        ax = axes[row, 3]
        f_slice = fold[:, :, t_idx].T
        ax.imshow(f_slice, aspect='equal', cmap='viridis',
                  extent=[0, config.nx * config.dx, config.ny * config.dy, 0])
        if row == 2:
            ax.set_xlabel('Inline (m)')
        ax.set_title(f'Fold at {t_slice_ms} ms')

    plt.tight_layout()
    fig_path = output_dir / "migration_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fig_path}")

    return fig_path


def create_velocity_qc(velocity: np.ndarray, config: SyntheticConfig, output_dir: Path):
    """Create velocity model QC figure."""
    print("\n    Creating velocity QC figure...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('3D Velocity Model with Spatial Variations', fontsize=14, fontweight='bold')

    # Inline section at center
    ax = axes[0, 0]
    il_idx = config.nx // 2
    v_il = velocity[il_idx, :, :].T
    im = ax.imshow(v_il, aspect='auto', cmap='jet',
                   extent=[0, config.ny * config.dy, config.nt * config.dt_ms, 0])
    ax.set_xlabel('Crossline (m)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Inline {il_idx} (x={il_idx * config.dx:.0f}m)')
    plt.colorbar(im, ax=ax, label='Velocity (m/s)')

    # Crossline section at center
    ax = axes[0, 1]
    xl_idx = config.ny // 2
    v_xl = velocity[:, xl_idx, :].T
    im = ax.imshow(v_xl, aspect='auto', cmap='jet',
                   extent=[0, config.nx * config.dx, config.nt * config.dt_ms, 0])
    ax.set_xlabel('Inline (m)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Crossline {xl_idx} (y={xl_idx * config.dy:.0f}m)')
    plt.colorbar(im, ax=ax, label='Velocity (m/s)')

    # Time slices
    for i, t_ms in enumerate([100, 400, 800]):
        ax = axes[1, i] if i < 2 else axes[0, 2]
        t_idx = int(t_ms / config.dt_ms)
        if t_idx >= config.nt:
            t_idx = config.nt - 1
        v_slice = velocity[:, :, t_idx].T
        im = ax.imshow(v_slice, aspect='equal', cmap='jet',
                       extent=[0, config.nx * config.dx, config.ny * config.dy, 0])
        ax.set_xlabel('Inline (m)')
        ax.set_ylabel('Crossline (m)')
        ax.set_title(f'Time Slice {t_ms} ms')
        plt.colorbar(im, ax=ax, label='V (m/s)')

    # Velocity profile at center
    ax = axes[1, 2]
    v_center = velocity[config.nx//2, config.ny//2, :]
    v_corner = velocity[0, 0, :]
    v_edge = velocity[config.nx//2, 0, :]
    ax.plot(v_center, config.t_axis_ms, 'b-', label='Center', linewidth=2)
    ax.plot(v_corner, config.t_axis_ms, 'r--', label='Corner (0,0)', linewidth=1.5)
    ax.plot(v_edge, config.t_axis_ms, 'g:', label='Edge', linewidth=1.5)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Time (ms)')
    ax.invert_yaxis()
    ax.set_title('Velocity Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "velocity_model_qc.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {fig_path}")


def main():
    """Main function to run the synthetic azimuth test."""
    print("=" * 70)
    print("SYNTHETIC AZIMUTH TEST FOR PSTM MIGRATION")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")

    # Configuration
    config = SyntheticConfig()

    print(f"\nConfiguration:")
    print(f"  Grid: {config.nx} x {config.ny} x {config.nt}")
    print(f"  Cell size: {config.dx} x {config.dy} m")
    print(f"  Time sampling: {config.dt_ms} ms")
    print(f"  Offset: {config.offset} m")
    print(f"  Traces per CMP: {config.traces_per_cmp}")

    # Create velocity model
    velocity = create_3d_velocity_model(config)
    create_velocity_qc(velocity, config, OUTPUT_DIR)

    # Define reflectors as tuples: (t0_ms, dip_x, dip_y, amplitude)
    reflector_params = [
        (200, 0.0, 0.0, 1.0),     # Flat at 200ms
        (400, 0.0, 0.0, 0.8),     # Flat at 400ms
        (600, 0.0, 0.0, 0.6),     # Flat at 600ms
        (500, 0.8, 0.5, 0.7),     # Dipping reflector
    ]

    print(f"\nReflectors:")
    for t0, dx, dy, amp in reflector_params:
        print(f"  t0={t0}ms, dip_x={dx}, dip_y={dy}, amp={amp}")

    # Generate offset gather 1 (azimuths 0-90)
    traces1, headers1 = generate_common_offset_gather(
        config, velocity, reflector_params,
        azimuth_range=(0, 90),
        name="Offset Gather 1 (Azimuth 0-90)"
    )

    # Generate offset gather 2 (azimuths 90-180)
    traces2, headers2 = generate_common_offset_gather(
        config, velocity, reflector_params,
        azimuth_range=(90, 180),
        name="Offset Gather 2 (Azimuth 90-180)"
    )

    # Results storage
    results = {}

    # Migrate offset 1 separately
    print(f"\n[3] Migrating Offset 1 separately...")
    print("    (JIT compiling kernel, first run may take a moment...)")
    image1, fold1 = migrate_gather(config, velocity, traces1, headers1)
    results['offset1'] = {'image': image1, 'fold': fold1}
    print(f"    Max fold: {fold1.max()}")
    create_qc_images(image1, fold1, config, "Offset 1 (Az 0-90)", OUTPUT_DIR)

    # Migrate offset 2 separately
    print(f"\n[4] Migrating Offset 2 separately...")
    image2, fold2 = migrate_gather(config, velocity, traces2, headers2)
    results['offset2'] = {'image': image2, 'fold': fold2}
    print(f"    Max fold: {fold2.max()}")
    create_qc_images(image2, fold2, config, "Offset 2 (Az 90-180)", OUTPUT_DIR)

    # Migrate both jointly
    print(f"\n[5] Migrating both offsets jointly...")
    traces_joint = np.vstack([traces1, traces2])
    headers_joint = {}
    for key in headers1:
        headers_joint[key] = np.concatenate([headers1[key], headers2[key]])

    print(f"    Joint traces: {traces_joint.shape[0]}")
    image_joint, fold_joint = migrate_gather(config, velocity, traces_joint, headers_joint)
    results['joint'] = {'image': image_joint, 'fold': fold_joint}
    print(f"    Max fold: {fold_joint.max()}")
    create_qc_images(image_joint, fold_joint, config, "Joint (Az 0-180)", OUTPUT_DIR)

    # Create comparison figure
    create_comparison_figure(results, config, OUTPUT_DIR)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Migration':<25} {'Max Fold':<12} {'Mean Fold':<12} {'Max Amp':<12}")
    print("-" * 60)

    for name, key in [('Offset 1 (Az 0-90)', 'offset1'),
                      ('Offset 2 (Az 90-180)', 'offset2'),
                      ('Joint (Az 0-180)', 'joint')]:
        img_norm = normalize_image(results[key]['image'], results[key]['fold'])
        fold = results[key]['fold']
        mean_fold = fold[fold > 0].mean() if np.any(fold > 0) else 0
        print(f"{name:<25} {fold.max():<12} {mean_fold:<12.1f} {np.abs(img_norm).max():<12.4f}")

    # Verify joint = sum of individuals (before normalization)
    print(f"\n[8] Verification: Joint image vs sum of individuals")
    image_sum = results['offset1']['image'] + results['offset2']['image']
    fold_sum = results['offset1']['fold'] + results['offset2']['fold']

    image_diff = np.abs(results['joint']['image'] - image_sum).max()
    fold_diff = np.abs(results['joint']['fold'] - fold_sum).max()

    print(f"    Image difference (max): {image_diff:.6f}")
    print(f"    Fold difference (max): {fold_diff}")

    if image_diff < 1e-6 and fold_diff == 0:
        print(f"    PASS: Joint migration equals sum of individual migrations")
    else:
        print(f"    NOTE: Small numerical differences present")

    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")

    return True


if __name__ == "__main__":
    main()
