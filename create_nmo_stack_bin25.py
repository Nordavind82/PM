#!/usr/bin/env python3
"""
Create NMO Stack for Offset Bin 25.

Uses the same velocities as PSTM migration for comparison.
NMO correction: t_nmo = sqrt(t0^2 + (offset/v)^2)
"""

import numpy as np
import polars as pl
import zarr
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
INPUT_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_20m")
VELOCITY_PATH = INPUT_DIR / "velocity_pstm.zarr"
OUTPUT_DIR = INPUT_DIR / "nmo_stack_bin25"
BIN_NAME = "offset_bin_25"

# Output grid (matching PSTM)
N_IL = 511
N_XL = 427
N_T = 1001
DT_MS = 2.0
T_MAX_MS = 2000.0


def load_velocity():
    """Load velocity model."""
    print("Loading velocity model...")
    v = zarr.open_array(str(VELOCITY_PATH), mode='r')
    velocity = np.array(v[:])
    attrs = dict(v.attrs)

    x_axis = np.array(attrs['x_axis'])  # IL numbers
    y_axis = np.array(attrs['y_axis'])  # XL numbers
    t_axis = np.array(attrs['t_axis_ms'])

    print(f"  Shape: {velocity.shape}")
    print(f"  IL range: {x_axis.min():.0f} to {x_axis.max():.0f}")
    print(f"  XL range: {y_axis.min():.0f} to {y_axis.max():.0f}")
    print(f"  Time range: {t_axis[0]:.0f} to {t_axis[-1]:.0f} ms")

    return velocity, x_axis, y_axis, t_axis


def load_traces_and_headers():
    """Load traces and headers for bin 25."""
    print(f"\nLoading {BIN_NAME}...")

    traces_path = INPUT_DIR / BIN_NAME / "traces.zarr"
    headers_path = INPUT_DIR / BIN_NAME / "headers.parquet"

    z = zarr.open_array(str(traces_path), mode='r')
    traces = np.array(z[:])  # (nt_input, n_traces)

    df = pl.read_parquet(headers_path)

    print(f"  Traces shape: {traces.shape}")
    print(f"  Headers: {len(df)} rows")

    return traces, df


def nmo_correction(trace, t_axis_in, t_axis_out, offset, velocity_trace):
    """
    Apply NMO correction to a single trace.

    t_nmo = sqrt(t0^2 + (offset/v)^2)

    We need to map from t_nmo (input) to t0 (output).
    """
    nt_out = len(t_axis_out)
    corrected = np.zeros(nt_out, dtype=np.float32)

    for i_t0, t0_ms in enumerate(t_axis_out):
        if t0_ms <= 0:
            continue

        # Get velocity at this t0
        v = velocity_trace[i_t0]
        if v <= 0:
            continue

        # Compute NMO time
        t0_s = t0_ms / 1000.0
        offset_m = abs(offset)

        t_nmo_s = np.sqrt(t0_s**2 + (offset_m / v)**2)
        t_nmo_ms = t_nmo_s * 1000.0

        # Check stretch (mute if too much)
        if t0_ms > 0:
            stretch = (t_nmo_ms - t0_ms) / t0_ms
            if stretch > 0.5:  # 50% stretch mute
                continue

        # Interpolate from input trace
        if t_nmo_ms < t_axis_in[-1]:
            # Linear interpolation
            t_idx = t_nmo_ms / (t_axis_in[1] - t_axis_in[0])
            i_low = int(t_idx)
            i_high = i_low + 1

            if i_high < len(trace):
                frac = t_idx - i_low
                corrected[i_t0] = (1 - frac) * trace[i_low] + frac * trace[i_high]

    return corrected


def create_nmo_stack(traces, df, velocity, il_axis, xl_axis, t_axis_vel):
    """Create NMO-corrected stack."""
    print("\nCreating NMO stack...")

    # Input time axis
    nt_in = traces.shape[0]
    dt_in = 2.0  # ms (from headers)
    t_axis_in = np.arange(nt_in) * dt_in

    # Output time axis (matching velocity)
    t_axis_out = t_axis_vel
    nt_out = len(t_axis_out)

    # Initialize output
    stack = np.zeros((N_IL, N_XL, nt_out), dtype=np.float64)
    fold = np.zeros((N_IL, N_XL, nt_out), dtype=np.int32)

    # Get header values
    inlines = df['inline'].to_numpy()
    crosslines = df['crossline'].to_numpy()
    offsets = df['offset'].to_numpy()

    n_traces = len(df)

    # Process each trace
    for i in range(n_traces):
        il = inlines[i]
        xl = crosslines[i]
        offset = offsets[i]

        # Get indices (1-based to 0-based)
        il_idx = il - 1
        xl_idx = xl - 1

        if il_idx < 0 or il_idx >= N_IL or xl_idx < 0 or xl_idx >= N_XL:
            continue

        # Get velocity trace at this IL/XL
        # Velocity is indexed by IL/XL numbers in x_axis/y_axis
        v_il_idx = np.searchsorted(il_axis, il)
        v_xl_idx = np.searchsorted(xl_axis, xl)

        if v_il_idx >= velocity.shape[0]:
            v_il_idx = velocity.shape[0] - 1
        if v_xl_idx >= velocity.shape[1]:
            v_xl_idx = velocity.shape[1] - 1

        vel_trace = velocity[v_il_idx, v_xl_idx, :]

        # Get input trace
        trace = traces[:, i]

        # Apply NMO correction
        corrected = nmo_correction(trace, t_axis_in, t_axis_out, offset, vel_trace)

        # Stack
        stack[il_idx, xl_idx, :] += corrected
        fold[il_idx, xl_idx, :] += (corrected != 0).astype(np.int32)

        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1}/{n_traces} traces ({100*(i+1)/n_traces:.1f}%)")

    # Normalize by fold
    print("  Normalizing by fold...")
    mask = fold > 0
    stack[mask] /= fold[mask]

    print(f"  Stack shape: {stack.shape}")
    print(f"  Max fold: {fold.max()}")
    print(f"  Amplitude range: {stack.min():.4f} to {stack.max():.4f}")

    return stack.astype(np.float32), fold


def save_nmo_stack(stack, fold, t_axis):
    """Save NMO stack to zarr."""
    print("\nSaving NMO stack...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save stack
    stack_path = OUTPUT_DIR / "nmo_stack.zarr"
    z = zarr.open_array(str(stack_path), mode='w',
                        shape=stack.shape, dtype=stack.dtype,
                        chunks=(32, 32, N_T))
    z[:] = stack
    z.attrs['dt_ms'] = DT_MS
    z.attrs['n_il'] = N_IL
    z.attrs['n_xl'] = N_XL
    z.attrs['n_t'] = N_T
    print(f"  Saved: {stack_path}")

    # Save fold
    fold_path = OUTPUT_DIR / "nmo_fold.zarr"
    z_fold = zarr.open_array(str(fold_path), mode='w',
                             shape=fold.shape, dtype=fold.dtype,
                             chunks=(32, 32, N_T))
    z_fold[:] = fold
    print(f"  Saved: {fold_path}")

    return stack_path, fold_path


def create_comparison_images(nmo_stack, nmo_fold):
    """Create comparison images with PSTM."""
    print("\nCreating comparison images...")

    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    # Load PSTM result
    pstm_path = Path("/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/migration_bin_25/migrated_stack.zarr")
    if pstm_path.exists():
        z_pstm = zarr.open_array(str(pstm_path), mode='r')
        pstm_stack = np.array(z_pstm[:])
        print(f"  PSTM shape: {pstm_stack.shape}")
    else:
        print(f"  WARNING: PSTM result not found at {pstm_path}")
        pstm_stack = None

    t_axis = np.arange(N_T) * DT_MS

    # Clip values
    nmo_vmax = np.percentile(np.abs(nmo_stack), 99)
    if pstm_stack is not None:
        pstm_vmax = np.percentile(np.abs(pstm_stack), 99)

    # === Inline sections ===
    il_positions = [128, 256, 384]

    for il_idx in il_positions:
        fig, axes = plt.subplots(1, 2 if pstm_stack is not None else 1, figsize=(16 if pstm_stack is not None else 10, 10))
        if pstm_stack is None:
            axes = [axes]

        fig.suptitle(f'Inline {il_idx+1} - NMO Stack vs PSTM Migration (Bin 25)',
                     fontsize=14, fontweight='bold')

        # NMO
        ax = axes[0]
        section = nmo_stack[il_idx, :, :].T
        ax.imshow(section, aspect='auto', cmap='gray',
                  extent=[1, N_XL, t_axis[-1], 0],
                  vmin=-nmo_vmax, vmax=nmo_vmax, interpolation='bilinear')
        ax.set_xlabel('Crossline')
        ax.set_ylabel('Time (ms)')
        ax.set_title('NMO Stack')

        # PSTM
        if pstm_stack is not None:
            ax = axes[1]
            section = pstm_stack[il_idx, :, :].T
            ax.imshow(section, aspect='auto', cmap='gray',
                      extent=[1, pstm_stack.shape[1], t_axis[-1], 0],
                      vmin=-pstm_vmax, vmax=pstm_vmax, interpolation='bilinear')
            ax.set_xlabel('Crossline')
            ax.set_ylabel('Time (ms)')
            ax.set_title('PSTM Migration')

        plt.tight_layout()
        fig_path = images_dir / f'inline_{il_idx+1:03d}_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {fig_path.name}")

    # === Crossline sections ===
    xl_positions = [107, 214, 320]

    for xl_idx in xl_positions:
        fig, axes = plt.subplots(1, 2 if pstm_stack is not None else 1, figsize=(16 if pstm_stack is not None else 10, 10))
        if pstm_stack is None:
            axes = [axes]

        fig.suptitle(f'Crossline {xl_idx+1} - NMO Stack vs PSTM Migration (Bin 25)',
                     fontsize=14, fontweight='bold')

        # NMO
        ax = axes[0]
        section = nmo_stack[:, xl_idx, :].T
        ax.imshow(section, aspect='auto', cmap='gray',
                  extent=[1, N_IL, t_axis[-1], 0],
                  vmin=-nmo_vmax, vmax=nmo_vmax, interpolation='bilinear')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Time (ms)')
        ax.set_title('NMO Stack')

        # PSTM
        if pstm_stack is not None:
            ax = axes[1]
            section = pstm_stack[:, xl_idx, :].T
            ax.imshow(section, aspect='auto', cmap='gray',
                      extent=[1, pstm_stack.shape[0], t_axis[-1], 0],
                      vmin=-pstm_vmax, vmax=pstm_vmax, interpolation='bilinear')
            ax.set_xlabel('Inline')
            ax.set_ylabel('Time (ms)')
            ax.set_title('PSTM Migration')

        plt.tight_layout()
        fig_path = images_dir / f'crossline_{xl_idx+1:03d}_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {fig_path.name}")

    # === Time slices ===
    time_slices = [500, 800, 1200]

    for t_ms in time_slices:
        t_idx = int(t_ms / DT_MS)

        fig, axes = plt.subplots(1, 2 if pstm_stack is not None else 1, figsize=(16 if pstm_stack is not None else 10, 10))
        if pstm_stack is None:
            axes = [axes]

        fig.suptitle(f'Time Slice @ {t_ms}ms - NMO Stack vs PSTM Migration (Bin 25)',
                     fontsize=14, fontweight='bold')

        # NMO
        ax = axes[0]
        slice_data = nmo_stack[:, :, t_idx]
        local_vmax = np.percentile(np.abs(slice_data), 99)
        if local_vmax == 0:
            local_vmax = 1
        ax.imshow(slice_data.T, aspect='auto', cmap='gray',
                  extent=[1, N_IL, N_XL, 1],
                  vmin=-local_vmax, vmax=local_vmax, interpolation='bilinear', origin='lower')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        ax.set_title('NMO Stack')

        # PSTM
        if pstm_stack is not None:
            ax = axes[1]
            slice_data = pstm_stack[:, :, t_idx]
            local_vmax = np.percentile(np.abs(slice_data), 99)
            if local_vmax == 0:
                local_vmax = 1
            ax.imshow(slice_data.T, aspect='auto', cmap='gray',
                      extent=[1, pstm_stack.shape[0], pstm_stack.shape[1], 1],
                      vmin=-local_vmax, vmax=local_vmax, interpolation='bilinear', origin='lower')
            ax.set_xlabel('Inline')
            ax.set_ylabel('Crossline')
            ax.set_title('PSTM Migration')

        plt.tight_layout()
        fig_path = images_dir / f'timeslice_{t_ms:04d}ms_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {fig_path.name}")

    print(f"\nAll images saved to: {images_dir}")


def main():
    print("=" * 70)
    print("NMO STACK - BIN 25")
    print("=" * 70)

    # Load velocity
    velocity, il_axis, xl_axis, t_axis = load_velocity()

    # Load traces
    traces, df = load_traces_and_headers()

    # Create NMO stack
    stack, fold = create_nmo_stack(traces, df, velocity, il_axis, xl_axis, t_axis)

    # Save results
    save_nmo_stack(stack, fold, t_axis)

    # Create comparison images
    create_comparison_images(stack, fold)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
