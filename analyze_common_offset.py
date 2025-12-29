#!/usr/bin/env python3
"""
Analysis of Computed Common Offset Migration Results.

Provides comprehensive QC and statistics for PSTM common offset gathers:
- Volume statistics (dimensions, amplitude range, RMS)
- Offset bin coverage and fold analysis
- Time-domain amplitude analysis
- Cross-bin consistency checks
- CIG residual moveout estimation
- Visual QC: inline/crossline sections and time slices

Usage:
    python analyze_common_offset.py
    python analyze_common_offset.py --bins 0-10
    python analyze_common_offset.py --output report.txt
    python analyze_common_offset.py --figures output_dir/  # Generate figures
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import zarr

# Optional imports for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# =============================================================================
# Configuration
# =============================================================================

MIGRATION_DIR = Path("/Volumes/AO_DISK/PSTM_common_offset")
INPUT_DIR = Path("/Users/olegadamovich/SeismicData/common_offset_gathers_new")


# =============================================================================
# Analysis Functions
# =============================================================================

def get_available_bins(migration_dir: Path) -> list[int]:
    """Get list of completed migration bins."""
    bins = []
    for d in migration_dir.iterdir():
        if d.is_dir() and d.name.startswith("migration_bin_"):
            try:
                bin_num = int(d.name.replace("migration_bin_", ""))
                zarr_path = d / "migrated_stack.zarr"
                if zarr_path.exists():
                    bins.append(bin_num)
            except ValueError:
                continue
    return sorted(bins)


def analyze_single_bin(bin_num: int, migration_dir: Path, input_dir: Path) -> dict:
    """Analyze a single offset bin migration result."""
    result = {
        'bin_num': bin_num,
        'status': 'not_found',
    }

    # Check migration output
    zarr_path = migration_dir / f"migration_bin_{bin_num:02d}" / "migrated_stack.zarr"
    if not zarr_path.exists():
        return result

    try:
        store = zarr.storage.LocalStore(str(zarr_path))
        z = zarr.open_array(store=store, mode='r')
        data = np.asarray(z)
        attrs = dict(z.attrs)

        result['status'] = 'ok'
        result['shape'] = data.shape
        result['nx'], result['ny'], result['nt'] = data.shape
        result['dtype'] = str(data.dtype)

        # Grid attributes
        result['x_min'] = attrs.get('x_min', 0)
        result['x_max'] = attrs.get('x_max', 0)
        result['y_min'] = attrs.get('y_min', 0)
        result['y_max'] = attrs.get('y_max', 0)
        result['dx'] = attrs.get('dx', 0)
        result['dy'] = attrs.get('dy', 0)
        result['dt_ms'] = attrs.get('dt_ms', 2.0)
        result['t_min_ms'] = attrs.get('t_min_ms', 0)
        result['t_max_ms'] = attrs.get('t_max_ms', 0)

        # File size
        zarr_size = sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file())
        result['size_mb'] = zarr_size / (1024 * 1024)

        # Amplitude statistics
        result['amp_min'] = float(np.min(data))
        result['amp_max'] = float(np.max(data))
        result['amp_mean'] = float(np.mean(data))
        result['amp_std'] = float(np.std(data))
        result['amp_rms'] = float(np.sqrt(np.mean(data**2)))

        # Non-zero statistics (trace activity)
        nonzero_mask = np.abs(data) > 1e-10
        result['nonzero_fraction'] = float(np.sum(nonzero_mask) / data.size)

        # Time-slice amplitude analysis (at 500ms, 1000ms, 1500ms)
        dt_ms = attrs.get('dt_ms', 2.0)
        for t_ms in [500, 1000, 1500]:
            t_idx = int(t_ms / dt_ms)
            if t_idx < data.shape[2]:
                slice_data = data[:, :, t_idx]
                result[f'rms_t{t_ms}'] = float(np.sqrt(np.mean(slice_data**2)))

        # Spatial amplitude variation (check for edge effects)
        center_x = data.shape[0] // 2
        center_y = data.shape[1] // 2
        margin = 50

        if center_x > margin and center_y > margin:
            center_region = data[center_x-margin:center_x+margin,
                                 center_y-margin:center_y+margin, :]
            edge_region = np.concatenate([
                data[:margin, :, :].flatten(),
                data[-margin:, :, :].flatten(),
                data[:, :margin, :].flatten(),
                data[:, -margin:, :].flatten(),
            ])

            result['center_rms'] = float(np.sqrt(np.mean(center_region**2)))
            result['edge_rms'] = float(np.sqrt(np.mean(edge_region**2)))
            result['center_edge_ratio'] = result['center_rms'] / max(result['edge_rms'], 1e-10)

    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        return result

    # Get input trace info
    headers_path = input_dir / f"offset_bin_{bin_num:02d}" / "headers.parquet"
    if headers_path.exists():
        try:
            df = pl.read_parquet(headers_path)
            result['n_input_traces'] = len(df)
            if 'offset' in df.columns and len(df) > 0:
                result['offset_mean'] = float(df['offset'].mean())
                result['offset_min'] = float(df['offset'].min())
                result['offset_max'] = float(df['offset'].max())
                result['offset_std'] = float(df['offset'].std())
        except Exception:
            pass

    return result


def analyze_cig_moveout(migration_dir: Path, bins: list[int],
                        ix: int = None, iy: int = None) -> dict:
    """
    Analyze residual moveout in CIG at a specific location.

    Args:
        migration_dir: Migration output directory
        bins: List of offset bins to analyze
        ix, iy: Grid location (default: center)

    Returns:
        Dictionary with moveout analysis results
    """
    result = {
        'location': (ix, iy),
        'bins_analyzed': [],
        'offsets': [],
        'event_times': {},
        'moveout_ms': {},
    }

    traces = []
    offsets = []
    dt_ms = 2.0

    for bin_num in bins:
        zarr_path = migration_dir / f"migration_bin_{bin_num:02d}" / "migrated_stack.zarr"
        if not zarr_path.exists():
            continue

        try:
            store = zarr.storage.LocalStore(str(zarr_path))
            z = zarr.open_array(store=store, mode='r')

            # Set location to center if not specified
            if ix is None:
                ix = z.shape[0] // 2
            if iy is None:
                iy = z.shape[1] // 2

            result['location'] = (ix, iy)
            dt_ms = z.attrs.get('dt_ms', 2.0)

            trace = z[ix, iy, :]
            traces.append(np.asarray(trace))
            result['bins_analyzed'].append(bin_num)

            # Get offset from input headers
            headers_path = INPUT_DIR / f"offset_bin_{bin_num:02d}" / "headers.parquet"
            if headers_path.exists():
                df = pl.read_parquet(headers_path)
                if 'offset' in df.columns and len(df) > 0:
                    offsets.append(float(df['offset'].mean()))
                else:
                    offsets.append(bin_num * 50 + 25)  # Fallback estimate
            else:
                offsets.append(bin_num * 50 + 25)

        except Exception:
            continue

    if len(traces) < 2:
        result['status'] = 'insufficient_data'
        return result

    result['offsets'] = offsets
    traces = np.array(traces)

    # Find strong events and track moveout
    # Look for peaks in the near-offset trace
    near_trace = traces[0]

    # Find events at different time windows
    time_windows = [
        (200, 400, 'shallow'),
        (500, 700, 'mid'),
        (900, 1100, 'deep'),
    ]

    for t_start, t_end, name in time_windows:
        idx_start = int(t_start / dt_ms)
        idx_end = int(t_end / dt_ms)

        if idx_end > len(near_trace):
            continue

        # Find peak in near-offset
        window = near_trace[idx_start:idx_end]
        peak_idx_local = np.argmax(np.abs(window))
        peak_idx = idx_start + peak_idx_local
        peak_time = peak_idx * dt_ms

        # Track this event across offsets
        event_times = []
        search_window = 50  # samples

        for i, trace in enumerate(traces):
            search_start = max(0, peak_idx - search_window)
            search_end = min(len(trace), peak_idx + search_window)

            local_window = trace[search_start:search_end]
            local_peak = np.argmax(np.abs(local_window))
            event_time = (search_start + local_peak) * dt_ms
            event_times.append(event_time)

        result['event_times'][name] = event_times
        result['moveout_ms'][name] = [t - event_times[0] for t in event_times]

    result['status'] = 'ok'
    return result


# =============================================================================
# Figure Generation Functions
# =============================================================================

def generate_figures_for_bin(
    bin_num: int,
    migration_dir: Path,
    output_dir: Path,
    inline_positions: list[float] = None,
    xline_positions: list[float] = None,
    time_slices_ms: list[float] = None,
    clip_percentile: float = 99,
) -> list[Path]:
    """
    Generate inline, crossline, and time slice figures for a single bin.
    Creates one image per section (separate files).

    Args:
        bin_num: Offset bin number
        migration_dir: Migration output directory
        output_dir: Directory to save figures
        inline_positions: Inline positions as fractions (0-1), default [0.25, 0.5, 0.75]
        xline_positions: Crossline positions as fractions (0-1), default [0.25, 0.5, 0.75]
        time_slices_ms: Time slice positions in ms, default [300, 500, 700]
        clip_percentile: Percentile for amplitude clipping

    Returns:
        List of saved figure paths
    """
    if not HAS_MATPLOTLIB:
        print("  Warning: matplotlib not available, skipping figures")
        return []

    # Defaults
    if inline_positions is None:
        inline_positions = [0.25, 0.5, 0.75]
    if xline_positions is None:
        xline_positions = [0.25, 0.5, 0.75]
    if time_slices_ms is None:
        time_slices_ms = [300, 500, 700]

    zarr_path = migration_dir / f"migration_bin_{bin_num:02d}" / "migrated_stack.zarr"
    if not zarr_path.exists():
        return []

    try:
        store = zarr.storage.LocalStore(str(zarr_path))
        z = zarr.open_array(store=store, mode='r')
        data = np.asarray(z)
        attrs = dict(z.attrs)
    except Exception as e:
        print(f"  Error loading bin {bin_num}: {e}")
        return []

    nx, ny, nt = data.shape
    dt_ms = attrs.get('dt_ms', 2.0)
    t_min_ms = attrs.get('t_min_ms', 0.0)

    # Get offset info
    headers_path = INPUT_DIR / f"offset_bin_{bin_num:02d}" / "headers.parquet"
    offset_m = bin_num * 50 + 25  # Default
    if headers_path.exists():
        try:
            df = pl.read_parquet(headers_path)
            if 'offset' in df.columns and len(df) > 0:
                offset_m = float(df['offset'].mean())
        except:
            pass

    # Calculate clip value
    clip_val = np.percentile(np.abs(data), clip_percentile)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # Time axis for sections
    t_axis = np.arange(nt) * dt_ms + t_min_ms

    # =========================================================================
    # Inline Sections (separate images)
    # =========================================================================
    for i, pos in enumerate(inline_positions):
        il_idx = int(pos * (nx - 1))
        section = data[il_idx, :, :].T  # (nt, ny)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            section,
            aspect='auto',
            cmap='gray',
            vmin=-clip_val,
            vmax=clip_val,
            extent=[0, ny, t_axis[-1], t_axis[0]],
        )
        ax.set_title(f'Bin {bin_num:02d} ({offset_m:.0f}m) - Inline {il_idx + 1}')
        ax.set_xlabel('Crossline')
        ax.set_ylabel('Time (ms)')
        plt.tight_layout()

        fig_path = output_dir / f"bin_{bin_num:02d}_inline_{il_idx:04d}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(fig_path)

    # =========================================================================
    # Crossline Sections (separate images)
    # =========================================================================
    for i, pos in enumerate(xline_positions):
        xl_idx = int(pos * (ny - 1))
        section = data[:, xl_idx, :].T  # (nt, nx)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            section,
            aspect='auto',
            cmap='gray',
            vmin=-clip_val,
            vmax=clip_val,
            extent=[0, nx, t_axis[-1], t_axis[0]],
        )
        ax.set_title(f'Bin {bin_num:02d} ({offset_m:.0f}m) - Crossline {xl_idx + 1}')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Time (ms)')
        plt.tight_layout()

        fig_path = output_dir / f"bin_{bin_num:02d}_xline_{xl_idx:04d}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(fig_path)

    # =========================================================================
    # Time Slices (separate images)
    # =========================================================================
    for i, t_ms in enumerate(time_slices_ms):
        t_idx = int((t_ms - t_min_ms) / dt_ms)
        t_idx = max(0, min(t_idx, nt - 1))
        slice_data = data[:, :, t_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            slice_data.T,
            aspect='auto',
            cmap='gray',
            vmin=-clip_val,
            vmax=clip_val,
            origin='lower',
        )
        ax.set_title(f'Bin {bin_num:02d} ({offset_m:.0f}m) - Time Slice @ {t_ms:.0f} ms')
        ax.set_xlabel('Inline')
        ax.set_ylabel('Crossline')
        plt.tight_layout()

        fig_path = output_dir / f"bin_{bin_num:02d}_tslice_{int(t_ms):04d}ms.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(fig_path)

    return saved_files


def generate_comparison_figures(
    bins: list[int],
    migration_dir: Path,
    output_dir: Path,
    time_slices_ms: list[float] = None,
    clip_percentile: float = 99,
) -> list[Path]:
    """
    Generate comparison figures showing same slice across multiple offset bins.

    Args:
        bins: List of bin numbers to compare
        migration_dir: Migration output directory
        output_dir: Directory to save figures
        time_slices_ms: Time slice positions in ms
        clip_percentile: Percentile for amplitude clipping

    Returns:
        List of saved figure paths
    """
    if not HAS_MATPLOTLIB:
        return []

    if time_slices_ms is None:
        time_slices_ms = [300, 500, 700]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    # Load all bins
    bin_data = {}
    offsets = {}

    for bin_num in bins:
        zarr_path = migration_dir / f"migration_bin_{bin_num:02d}" / "migrated_stack.zarr"
        if not zarr_path.exists():
            continue

        try:
            store = zarr.storage.LocalStore(str(zarr_path))
            z = zarr.open_array(store=store, mode='r')
            bin_data[bin_num] = {
                'data': z,
                'attrs': dict(z.attrs),
            }

            # Get offset
            headers_path = INPUT_DIR / f"offset_bin_{bin_num:02d}" / "headers.parquet"
            if headers_path.exists():
                df = pl.read_parquet(headers_path)
                if 'offset' in df.columns and len(df) > 0:
                    offsets[bin_num] = float(df['offset'].mean())
                else:
                    offsets[bin_num] = bin_num * 50 + 25
            else:
                offsets[bin_num] = bin_num * 50 + 25
        except:
            continue

    if not bin_data:
        return []

    # Get common parameters
    first_bin = list(bin_data.keys())[0]
    attrs = bin_data[first_bin]['attrs']
    dt_ms = attrs.get('dt_ms', 2.0)
    t_min_ms = attrs.get('t_min_ms', 0.0)

    # Generate time slice comparison for each time
    for t_ms in time_slices_ms:
        n_bins = len(bin_data)
        n_cols = min(6, n_bins)
        n_rows = (n_bins + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        fig.suptitle(f'Time Slice @ {t_ms:.0f} ms - All Offset Bins', fontsize=14, fontweight='bold')

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Calculate global clip for this time slice
        all_slices = []
        t_idx = int((t_ms - t_min_ms) / dt_ms)

        for bin_num in sorted(bin_data.keys()):
            z = bin_data[bin_num]['data']
            if t_idx < z.shape[2]:
                slice_data = np.asarray(z[:, :, t_idx])
                all_slices.append(slice_data)

        if all_slices:
            all_data = np.concatenate([s.flatten() for s in all_slices])
            clip_val = np.percentile(np.abs(all_data), clip_percentile)
        else:
            clip_val = 1.0

        # Plot each bin
        for idx, bin_num in enumerate(sorted(bin_data.keys())):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            z = bin_data[bin_num]['data']
            if t_idx < z.shape[2]:
                slice_data = np.asarray(z[:, :, t_idx])

                im = ax.imshow(
                    slice_data.T,
                    aspect='auto',
                    cmap='gray',
                    vmin=-clip_val,
                    vmax=clip_val,
                    origin='lower',
                )
                ax.set_title(f'Bin {bin_num:02d} ({offsets.get(bin_num, 0):.0f}m)')
                ax.set_xlabel('Inline')
                ax.set_ylabel('Crossline')

        # Hide empty axes
        for idx in range(len(bin_data), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        fig_path = output_dir / f"comparison_tslice_{int(t_ms)}ms.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(fig_path)

    # Generate CIG gather figure at center location
    first_z = bin_data[first_bin]['data']
    ix, iy = first_z.shape[0] // 2, first_z.shape[1] // 2
    nt = first_z.shape[2]

    fig, ax = plt.subplots(figsize=(12, 8))

    cig_traces = []
    cig_offsets = []

    for bin_num in sorted(bin_data.keys()):
        z = bin_data[bin_num]['data']
        trace = np.asarray(z[ix, iy, :])
        cig_traces.append(trace)
        cig_offsets.append(offsets.get(bin_num, bin_num * 50 + 25))

    if cig_traces:
        cig_data = np.array(cig_traces).T  # (nt, n_bins)
        clip_val = np.percentile(np.abs(cig_data), clip_percentile)

        t_axis = np.arange(nt) * dt_ms + t_min_ms

        im = ax.imshow(
            cig_data,
            aspect='auto',
            cmap='gray',
            vmin=-clip_val,
            vmax=clip_val,
            extent=[0, len(cig_offsets), t_axis[-1], t_axis[0]],
        )

        ax.set_title(f'CIG at IL={ix+1}, XL={iy+1}')
        ax.set_xlabel('Offset Bin Index')
        ax.set_ylabel('Time (ms)')

        # Add offset labels
        ax.set_xticks(np.arange(len(cig_offsets)) + 0.5)
        if len(cig_offsets) <= 20:
            ax.set_xticklabels([f'{o:.0f}' for o in cig_offsets], rotation=45)
        else:
            # Show every 5th label
            labels = [f'{o:.0f}' if i % 5 == 0 else '' for i, o in enumerate(cig_offsets)]
            ax.set_xticklabels(labels, rotation=45)

        plt.colorbar(im, ax=ax, label='Amplitude')

    plt.tight_layout()
    fig_path = output_dir / "cig_center.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files.append(fig_path)

    return saved_files


def generate_report(bins_analysis: list[dict], cig_analysis: dict = None) -> str:
    """Generate text report from analysis results."""
    lines = []
    lines.append("=" * 80)
    lines.append("COMMON OFFSET MIGRATION ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Summary
    valid_bins = [b for b in bins_analysis if b['status'] == 'ok']
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total bins analyzed: {len(bins_analysis)}")
    lines.append(f"Valid bins: {len(valid_bins)}")
    lines.append(f"Failed/missing: {len(bins_analysis) - len(valid_bins)}")
    lines.append("")

    if valid_bins:
        total_size = sum(b.get('size_mb', 0) for b in valid_bins)
        lines.append(f"Total data size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")

        # Grid info from first bin
        b0 = valid_bins[0]
        lines.append(f"Grid dimensions: {b0['nx']} x {b0['ny']} x {b0['nt']}")
        lines.append(f"Grid spacing: dx={b0['dx']} m, dy={b0['dy']} m, dt={b0['dt_ms']} ms")
        lines.append(f"X range: {b0['x_min']:.1f} - {b0['x_max']:.1f} m")
        lines.append(f"Y range: {b0['y_min']:.1f} - {b0['y_max']:.1f} m")
        lines.append(f"T range: {b0['t_min_ms']:.0f} - {b0['t_max_ms']:.0f} ms")
        lines.append("")

    # Per-bin statistics table
    lines.append("PER-BIN STATISTICS")
    lines.append("-" * 40)

    header = f"{'Bin':>4} {'Offset':>8} {'Traces':>10} {'RMS':>10} {'Size MB':>10} {'Status':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for b in bins_analysis:
        if b['status'] == 'ok':
            lines.append(
                f"{b['bin_num']:4d} "
                f"{b.get('offset_mean', 0):8.0f} "
                f"{b.get('n_input_traces', 0):10,d} "
                f"{b.get('amp_rms', 0):10.2e} "
                f"{b.get('size_mb', 0):10.1f} "
                f"{'OK':>8}"
            )
        else:
            lines.append(
                f"{b['bin_num']:4d} "
                f"{'-':>8} "
                f"{'-':>10} "
                f"{'-':>10} "
                f"{'-':>10} "
                f"{b['status']:>8}"
            )

    lines.append("")

    # Amplitude analysis
    if valid_bins:
        lines.append("AMPLITUDE ANALYSIS")
        lines.append("-" * 40)

        rms_values = [b['amp_rms'] for b in valid_bins]
        lines.append(f"RMS amplitude range: {min(rms_values):.2e} - {max(rms_values):.2e}")
        lines.append(f"RMS amplitude mean: {np.mean(rms_values):.2e}")
        lines.append(f"RMS amplitude std: {np.std(rms_values):.2e}")
        lines.append("")

        # Time-slice RMS
        lines.append("Time-slice RMS by offset:")
        header = f"{'Bin':>4} {'Offset':>8} {'RMS@500':>12} {'RMS@1000':>12} {'RMS@1500':>12}"
        lines.append(header)
        lines.append("-" * len(header))

        for b in valid_bins:
            lines.append(
                f"{b['bin_num']:4d} "
                f"{b.get('offset_mean', 0):8.0f} "
                f"{b.get('rms_t500', 0):12.2e} "
                f"{b.get('rms_t1000', 0):12.2e} "
                f"{b.get('rms_t1500', 0):12.2e}"
            )
        lines.append("")

        # Center vs edge analysis
        lines.append("SPATIAL CONSISTENCY (Center vs Edge)")
        lines.append("-" * 40)
        header = f"{'Bin':>4} {'Offset':>8} {'Center RMS':>12} {'Edge RMS':>12} {'Ratio':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for b in valid_bins:
            if 'center_rms' in b:
                lines.append(
                    f"{b['bin_num']:4d} "
                    f"{b.get('offset_mean', 0):8.0f} "
                    f"{b.get('center_rms', 0):12.2e} "
                    f"{b.get('edge_rms', 0):12.2e} "
                    f"{b.get('center_edge_ratio', 0):8.2f}"
                )
        lines.append("")

    # CIG moveout analysis
    if cig_analysis and cig_analysis.get('status') == 'ok':
        lines.append("CIG RESIDUAL MOVEOUT ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"Location: IL={cig_analysis['location'][0]+1}, XL={cig_analysis['location'][1]+1}")
        lines.append(f"Offset bins analyzed: {cig_analysis['bins_analyzed']}")
        lines.append(f"Offset range: {min(cig_analysis['offsets']):.0f} - {max(cig_analysis['offsets']):.0f} m")
        lines.append("")

        for event_name, moveouts in cig_analysis['moveout_ms'].items():
            lines.append(f"Event '{event_name}':")
            lines.append(f"  Max moveout: {max(moveouts):.1f} ms")
            lines.append(f"  Moveout at far offset: {moveouts[-1]:.1f} ms")

            # Estimate velocity error from moveout
            if len(moveouts) > 1 and cig_analysis['offsets']:
                far_offset = cig_analysis['offsets'][-1]
                far_moveout = moveouts[-1]
                if far_moveout != 0:
                    lines.append(f"  (Residual moveout indicates velocity adjustment may be needed)")
            lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze computed common offset migration results"
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="all",
        help="Bins to analyze: 'all', '0-37', or '0,5,10' (default: all)",
    )
    parser.add_argument(
        "--migration-dir",
        type=Path,
        default=MIGRATION_DIR,
        help="Migration output directory",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Input data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output report file (default: print to console)",
    )
    parser.add_argument(
        "--cig-location",
        type=str,
        default=None,
        help="CIG analysis location as 'ix,iy' (default: center)",
    )
    parser.add_argument(
        "--no-cig",
        action="store_true",
        help="Skip CIG moveout analysis",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=None,
        help="Output directory for figures (generates 3 IL, 3 XL, 3 time slices per bin)",
    )
    parser.add_argument(
        "--time-slices",
        type=str,
        default="300,500,700",
        help="Time slice positions in ms (default: 300,500,700)",
    )
    parser.add_argument(
        "--per-bin-figures",
        action="store_true",
        help="Generate individual section figures for each bin",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Common Offset Migration Analysis")
    print("=" * 70)
    print(f"Migration directory: {args.migration_dir}")
    print()

    # Get available bins
    available_bins = get_available_bins(args.migration_dir)
    print(f"Available bins: {len(available_bins)}")

    if not available_bins:
        print("No completed migration bins found!")
        return 1

    # Parse bin specification
    if args.bins.lower() == "all":
        bins_to_analyze = available_bins
    elif "-" in args.bins and "," not in args.bins:
        start, end = map(int, args.bins.split("-"))
        bins_to_analyze = [b for b in range(start, end + 1) if b in available_bins]
    else:
        bins_to_analyze = [int(b.strip()) for b in args.bins.split(",") if int(b.strip()) in available_bins]

    print(f"Analyzing bins: {bins_to_analyze}")
    print()

    # Analyze each bin
    print("Analyzing bins...")
    bins_analysis = []
    for i, bin_num in enumerate(bins_to_analyze):
        print(f"  [{i+1}/{len(bins_to_analyze)}] Bin {bin_num:02d}...", end=" ", flush=True)
        result = analyze_single_bin(bin_num, args.migration_dir, args.input_dir)
        bins_analysis.append(result)

        if result['status'] == 'ok':
            print(f"OK (RMS: {result['amp_rms']:.2e})")
        else:
            print(f"{result['status']}")

    print()

    # CIG moveout analysis
    cig_analysis = None
    if not args.no_cig and len(bins_to_analyze) >= 3:
        print("Analyzing CIG residual moveout...")

        ix, iy = None, None
        if args.cig_location:
            ix, iy = map(int, args.cig_location.split(","))

        cig_analysis = analyze_cig_moveout(args.migration_dir, bins_to_analyze, ix, iy)

        if cig_analysis.get('status') == 'ok':
            print(f"  Location: IL={cig_analysis['location'][0]+1}, XL={cig_analysis['location'][1]+1}")
            print(f"  Bins analyzed: {len(cig_analysis['bins_analyzed'])}")
            for name, moveouts in cig_analysis['moveout_ms'].items():
                print(f"  {name} event max moveout: {max(moveouts):.1f} ms")
        else:
            print(f"  Status: {cig_analysis.get('status', 'unknown')}")
        print()

    # Generate report
    report = generate_report(bins_analysis, cig_analysis)

    if args.output:
        args.output.write_text(report)
        print(f"Report saved to: {args.output}")
    else:
        print()
        print(report)

    # Generate figures if requested
    if args.figures:
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, cannot generate figures")
        else:
            print()
            print("=" * 70)
            print("Generating Figures")
            print("=" * 70)

            # Parse time slices
            time_slices = [float(t.strip()) for t in args.time_slices.split(",")]
            print(f"Time slices: {time_slices} ms")
            print(f"Output directory: {args.figures}")
            print()

            all_saved_files = []

            # Generate per-bin figures if requested
            if args.per_bin_figures:
                print("Generating per-bin section figures...")
                for i, bin_num in enumerate(bins_to_analyze):
                    print(f"  [{i+1}/{len(bins_to_analyze)}] Bin {bin_num:02d}...", end=" ", flush=True)
                    saved = generate_figures_for_bin(
                        bin_num,
                        args.migration_dir,
                        args.figures,
                        time_slices_ms=time_slices,
                    )
                    if saved:
                        print(f"OK ({len(saved)} files)")
                        all_saved_files.extend(saved)
                    else:
                        print("skipped")
                print()

            # Generate comparison figures
            print("Generating comparison figures...")
            comparison_files = generate_comparison_figures(
                bins_to_analyze,
                args.migration_dir,
                args.figures,
                time_slices_ms=time_slices,
            )
            all_saved_files.extend(comparison_files)

            print()
            print(f"Total figures generated: {len(all_saved_files)}")
            print(f"Saved to: {args.figures}")

            # List generated files
            if all_saved_files:
                print()
                print("Generated files:")
                for f in sorted(all_saved_files):
                    print(f"  {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
