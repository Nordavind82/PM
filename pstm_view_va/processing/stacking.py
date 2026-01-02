"""Stacking functions for creating NMO stacks."""

from typing import List, Tuple, Optional, Callable
import numpy as np
import zarr
import time
from pathlib import Path

from .nmo import apply_nmo_with_velocity_model
from .mute import apply_velocity_mute
from .filters import apply_bandpass_filter, apply_agc
from .velocity_interpolation import (
    interpolate_velocity_along_inline,
    interpolate_velocity_along_crossline,
    generate_velocity_qc_image,
    InterpolatedVelocityModel
)


def compute_stack(offset_bins: List[zarr.Array],
                  offset_values: np.ndarray,
                  t_coords: np.ndarray,
                  velocity_grid,  # VelocityOutputGrid
                  il_range: Tuple[int, int, int],  # (start, end, step)
                  xl_range: Tuple[int, int, int],  # (start, end, step)
                  settings: dict,
                  progress_callback: Optional[Callable[[int, int, str], None]] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute NMO stack over specified IL/XL range.
    Optimized version that reads data in chunks.
    """
    if len(offset_bins) == 0:
        return None, None, None

    n_il_data, n_xl_data, n_time = offset_bins[0].shape
    n_offsets = len(offset_bins)
    dt_ms = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

    # Generate coordinate arrays
    il_coords = np.arange(il_range[0], il_range[1] + 1, il_range[2])
    xl_coords = np.arange(xl_range[0], xl_range[1] + 1, xl_range[2])

    n_il = len(il_coords)
    n_xl = len(xl_coords)

    print(f"[Stacking] Starting: {n_il} inlines x {n_xl} crosslines = {n_il * n_xl} traces")
    print(f"[Stacking] Offsets: {n_offsets}, Time samples: {n_time}")

    # Initialize output stack
    stack = np.zeros((n_il, n_xl, n_time), dtype=np.float32)

    # Get processing parameters
    stretch_percent = settings.get('stretch_percent', 30)
    top_mute_enabled = settings.get('top_mute_enabled', False)
    v_top = settings.get('v_top', 1500)
    bottom_mute_enabled = settings.get('bottom_mute_enabled', False)
    v_bottom = settings.get('v_bottom', 5000)
    apply_bandpass = settings.get('apply_bandpass', False)
    f_low = settings.get('f_low', 5)
    f_high = settings.get('f_high', 80)
    apply_agc_flag = settings.get('apply_agc', False)
    agc_window = settings.get('agc_window', 250)

    start_time = time.time()
    total_traces = n_il * n_xl
    processed = 0

    # Process inline by inline (read entire inline at once for efficiency)
    for i, il in enumerate(il_coords):
        if il < 0 or il >= n_il_data:
            continue

        il_start_time = time.time()

        # Read all offsets for this inline at once (more efficient I/O)
        # Shape: (n_offsets, n_xl_to_process, n_time)
        xl_slice = slice(xl_range[0], min(xl_range[1] + 1, n_xl_data))
        xl_count = xl_slice.stop - xl_slice.start

        # Pre-allocate gather data for this inline
        inline_gathers = np.zeros((xl_count, n_offsets, n_time), dtype=np.float32)

        # Read data for all offsets
        for k, offset_zarr in enumerate(offset_bins):
            inline_gathers[:, k, :] = offset_zarr[il, xl_slice, :]

        # Process each crossline in this inline
        for j_local, xl in enumerate(range(xl_slice.start, xl_slice.stop)):
            # Map to output index
            if xl not in xl_coords.tolist():
                continue
            j = np.where(xl_coords == xl)[0]
            if len(j) == 0:
                continue
            j = j[0]

            # Get velocity function for this location
            vel_func = velocity_grid.get_velocity_at(il, xl)
            if vel_func is None:
                processed += 1
                continue

            # Get gather for this trace
            gather = inline_gathers[j_local, :, :]  # (n_offsets, n_time)

            # Apply processing chain
            if apply_bandpass:
                gather = apply_bandpass_filter(gather, dt_ms, f_low, f_high)

            if apply_agc_flag:
                gather = apply_agc(gather, agc_window, dt_ms)

            if top_mute_enabled or bottom_mute_enabled:
                v_t = v_top if top_mute_enabled else None
                v_b = v_bottom if bottom_mute_enabled else None
                gather = apply_velocity_mute(gather, offset_values, t_coords, v_t, v_b)

            # Apply NMO correction with stretch mute
            gather_nmo = apply_nmo_with_velocity_model(
                gather, offset_values, t_coords, vel_func,
                inverse=False, stretch_mute_percent=stretch_percent,
                vel_t_coords=velocity_grid.t_coords
            )

            # Stack (mean over offsets)
            stack[i, j, :] = np.nanmean(gather_nmo, axis=0)
            processed += 1

        # Progress reporting
        elapsed = time.time() - start_time
        il_time = time.time() - il_start_time
        traces_in_il = xl_count

        if i > 0:
            eta = (elapsed / (i + 1)) * (n_il - i - 1)
            eta_str = f"{eta:.0f}s" if eta < 60 else f"{eta/60:.1f}min"
        else:
            eta_str = "calculating..."

        print(f"[Stacking] IL {il} ({i+1}/{n_il}) - {traces_in_il} traces in {il_time:.1f}s - ETA: {eta_str}")

        if progress_callback:
            progress_callback(processed, total_traces, f"IL={il} ({i+1}/{n_il})")

    elapsed = time.time() - start_time
    print(f"[Stacking] Complete: {processed} traces in {elapsed:.1f}s ({processed/elapsed:.0f} traces/s)")

    if progress_callback:
        progress_callback(total_traces, total_traces, "Stacking complete")

    return stack, il_coords, xl_coords


def compute_inline_stack(offset_bins: List[zarr.Array],
                         offset_values: np.ndarray,
                         t_coords: np.ndarray,
                         velocity_grid,
                         il: int,
                         settings: dict,
                         progress_callback: Optional[Callable[[int, int, str], None]] = None,
                         qc_dir: Optional[Path] = None,
                         stack_name: str = "stack",
                         initial_velocity_grid=None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute NMO stack for a single inline (all crosslines).

    Processing sequence:
    1. Inverse NMO with initial velocity (remove existing NMO from data)
    2. Apply processing (bandpass, AGC, mutes)
    3. Forward NMO with selected velocity
    4. Stack

    Args:
        offset_bins: List of zarr arrays for each offset bin
        offset_values: Array of offset values
        t_coords: Time coordinates
        velocity_grid: VelocityOutputGrid with selected velocity picks
        il: Inline number to stack
        settings: Processing settings dict
        progress_callback: Optional progress callback
        qc_dir: Directory to save QC images (None to skip)
        stack_name: Name for QC image files
        initial_velocity_grid: VelocityOutputGrid with initial velocities (for inverse NMO)
    """
    if len(offset_bins) == 0:
        return None, None

    n_il_data, n_xl_data, n_time = offset_bins[0].shape
    n_offsets = len(offset_bins)
    dt_ms = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

    if il < 0 or il >= n_il_data:
        return None, None

    print(f"[Stacking] Inline {il}: {n_xl_data} crosslines x {n_offsets} offsets")
    start_time = time.time()

    # Step 1: Interpolate SELECTED velocities for all crosslines along this inline
    print(f"[Stacking] Interpolating selected velocities along inline {il}...")
    interp_start = time.time()

    interp_vels, xl_coords = interpolate_velocity_along_inline(
        velocity_grid, il, (0, n_xl_data - 1), method='linear'
    )

    if interp_vels is None:
        print(f"[Stacking] ERROR: Failed to interpolate velocities for IL={il}")
        return None, None

    print(f"[Stacking] Selected velocity interpolation done in {time.time() - interp_start:.1f}s")

    # Create interpolated velocity model for lookups
    interp_vel_model = InterpolatedVelocityModel(
        interp_vels, xl_coords, velocity_grid.t_coords, 'inline'
    )

    # Step 2: Interpolate INITIAL velocities (for inverse NMO)
    interp_initial_vels = None
    interp_initial_model = None
    if initial_velocity_grid is not None and initial_velocity_grid.has_velocity():
        print(f"[Stacking] Interpolating initial velocities for inverse NMO...")
        interp_initial_vels, _ = interpolate_velocity_along_inline(
            initial_velocity_grid, il, (0, n_xl_data - 1), method='linear'
        )
        if interp_initial_vels is not None:
            interp_initial_model = InterpolatedVelocityModel(
                interp_initial_vels, xl_coords, initial_velocity_grid.t_coords, 'inline'
            )
            print(f"[Stacking] Initial velocity interpolation done")

    # Step 3: Generate QC image of interpolated velocities
    if qc_dir is not None:
        # Find pick locations for marking on QC image
        pick_xls = []
        for j, xl in enumerate(velocity_grid.xl_coords):
            il_idx = np.argmin(np.abs(velocity_grid.il_coords - il))
            if not np.all(np.isnan(velocity_grid.velocities[il_idx, j, :])):
                pick_xls.append(int(xl))

        qc_path = qc_dir / f"{stack_name}_velocity_IL{il}.png"
        generate_velocity_qc_image(
            interp_vels, velocity_grid.t_coords, xl_coords,
            x_label="Crossline",
            title=f"Interpolated Velocity - Inline {il} ({stack_name})",
            save_path=qc_path,
            pick_locations=pick_xls
        )

    # Get processing parameters
    stretch_percent = settings.get('stretch_percent', 30)
    top_mute_enabled = settings.get('top_mute_enabled', False)
    v_top = settings.get('v_top', 1500)
    bottom_mute_enabled = settings.get('bottom_mute_enabled', False)
    v_bottom = settings.get('v_bottom', 5000)
    apply_bandpass = settings.get('apply_bandpass', False)
    f_low = settings.get('f_low', 5)
    f_high = settings.get('f_high', 80)
    apply_agc_flag = settings.get('apply_agc', False)
    agc_window = settings.get('agc_window', 250)

    # Step 4: Read all data for this inline at once
    print(f"[Stacking] Reading data...")
    read_start = time.time()
    inline_data = np.zeros((n_offsets, n_xl_data, n_time), dtype=np.float32)
    for k, offset_zarr in enumerate(offset_bins):
        inline_data[k, :, :] = offset_zarr[il, :, :]
    print(f"[Stacking] Data read in {time.time() - read_start:.1f}s")

    # Initialize output
    stack = np.zeros((n_xl_data, n_time), dtype=np.float32)

    # Step 5: Process each crossline
    print(f"[Stacking] Processing sequence: Inverse NMO -> Processing -> Forward NMO -> Stack")
    for xl in range(n_xl_data):
        if xl % 50 == 0:
            elapsed = time.time() - start_time
            if xl > 0:
                eta = (elapsed / xl) * (n_xl_data - xl)
                print(f"[Stacking] XL {xl}/{n_xl_data} - ETA: {eta:.0f}s")
            if progress_callback:
                progress_callback(xl, n_xl_data, f"XL={xl}")

        # Get selected velocity
        vel_func = interp_vel_model.get_velocity_at(il, xl)
        if vel_func is None:
            continue

        # Get gather
        gather = inline_data[:, xl, :].copy()  # (n_offsets, n_time)

        # === PROCESSING SEQUENCE ===

        # 1. INVERSE NMO with initial velocity (remove existing NMO)
        if interp_initial_model is not None:
            initial_vel_func = interp_initial_model.get_velocity_at(il, xl)
            if initial_vel_func is not None:
                gather = apply_nmo_with_velocity_model(
                    gather, offset_values, t_coords, initial_vel_func,
                    inverse=True, stretch_mute_percent=100,  # No stretch mute for inverse
                    vel_t_coords=initial_velocity_grid.t_coords
                )

        # 2. Apply processing (on un-NMO'd data)
        if apply_bandpass:
            gather = apply_bandpass_filter(gather, dt_ms, f_low, f_high)

        if apply_agc_flag:
            gather = apply_agc(gather, agc_window, dt_ms)

        if top_mute_enabled or bottom_mute_enabled:
            v_t = v_top if top_mute_enabled else None
            v_b = v_bottom if bottom_mute_enabled else None
            gather = apply_velocity_mute(gather, offset_values, t_coords, v_t, v_b)

        # 3. FORWARD NMO with selected velocity
        gather_nmo = apply_nmo_with_velocity_model(
            gather, offset_values, t_coords, vel_func,
            inverse=False, stretch_mute_percent=stretch_percent,
            vel_t_coords=velocity_grid.t_coords
        )

        # 4. Stack (mean over offsets)
        stack[xl, :] = np.nanmean(gather_nmo, axis=0)

    elapsed = time.time() - start_time
    print(f"[Stacking] Complete: {n_xl_data} traces in {elapsed:.1f}s ({n_xl_data/elapsed:.0f} traces/s)")

    if progress_callback:
        progress_callback(n_xl_data, n_xl_data, "Complete")

    return stack, xl_coords


def compute_crossline_stack(offset_bins: List[zarr.Array],
                            offset_values: np.ndarray,
                            t_coords: np.ndarray,
                            velocity_grid,
                            xl: int,
                            settings: dict,
                            progress_callback: Optional[Callable[[int, int, str], None]] = None,
                            qc_dir: Optional[Path] = None,
                            stack_name: str = "stack",
                            initial_velocity_grid=None
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute NMO stack for a single crossline (all inlines).

    Processing sequence:
    1. Inverse NMO with initial velocity (remove existing NMO from data)
    2. Apply processing (bandpass, AGC, mutes)
    3. Forward NMO with selected velocity
    4. Stack

    Args:
        offset_bins: List of zarr arrays for each offset bin
        offset_values: Array of offset values
        t_coords: Time coordinates
        velocity_grid: VelocityOutputGrid with selected velocity picks
        xl: Crossline number to stack
        settings: Processing settings dict
        progress_callback: Optional progress callback
        qc_dir: Directory to save QC images (None to skip)
        stack_name: Name for QC image files
        initial_velocity_grid: VelocityOutputGrid with initial velocities (for inverse NMO)
    """
    if len(offset_bins) == 0:
        return None, None

    n_il_data, n_xl_data, n_time = offset_bins[0].shape
    n_offsets = len(offset_bins)
    dt_ms = t_coords[1] - t_coords[0] if len(t_coords) > 1 else 2.0

    if xl < 0 or xl >= n_xl_data:
        return None, None

    print(f"[Stacking] Crossline {xl}: {n_il_data} inlines x {n_offsets} offsets")
    start_time = time.time()

    # Step 1: Interpolate SELECTED velocities for all inlines along this crossline
    print(f"[Stacking] Interpolating selected velocities along crossline {xl}...")
    interp_start = time.time()

    interp_vels, il_coords = interpolate_velocity_along_crossline(
        velocity_grid, xl, (0, n_il_data - 1), method='linear'
    )

    if interp_vels is None:
        print(f"[Stacking] ERROR: Failed to interpolate velocities for XL={xl}")
        return None, None

    print(f"[Stacking] Selected velocity interpolation done in {time.time() - interp_start:.1f}s")

    # Create interpolated velocity model for lookups
    interp_vel_model = InterpolatedVelocityModel(
        interp_vels, il_coords, velocity_grid.t_coords, 'crossline'
    )

    # Step 2: Interpolate INITIAL velocities (for inverse NMO)
    interp_initial_vels = None
    interp_initial_model = None
    if initial_velocity_grid is not None and initial_velocity_grid.has_velocity():
        print(f"[Stacking] Interpolating initial velocities for inverse NMO...")
        interp_initial_vels, _ = interpolate_velocity_along_crossline(
            initial_velocity_grid, xl, (0, n_il_data - 1), method='linear'
        )
        if interp_initial_vels is not None:
            interp_initial_model = InterpolatedVelocityModel(
                interp_initial_vels, il_coords, initial_velocity_grid.t_coords, 'crossline'
            )
            print(f"[Stacking] Initial velocity interpolation done")

    # Step 3: Generate QC image of interpolated velocities
    if qc_dir is not None:
        # Find pick locations for marking on QC image
        pick_ils = []
        for i, il in enumerate(velocity_grid.il_coords):
            xl_idx = np.argmin(np.abs(velocity_grid.xl_coords - xl))
            if not np.all(np.isnan(velocity_grid.velocities[i, xl_idx, :])):
                pick_ils.append(int(il))

        qc_path = qc_dir / f"{stack_name}_velocity_XL{xl}.png"
        generate_velocity_qc_image(
            interp_vels, velocity_grid.t_coords, il_coords,
            x_label="Inline",
            title=f"Interpolated Velocity - Crossline {xl} ({stack_name})",
            save_path=qc_path,
            pick_locations=pick_ils
        )

    # Get processing parameters
    stretch_percent = settings.get('stretch_percent', 30)
    top_mute_enabled = settings.get('top_mute_enabled', False)
    v_top = settings.get('v_top', 1500)
    bottom_mute_enabled = settings.get('bottom_mute_enabled', False)
    v_bottom = settings.get('v_bottom', 5000)
    apply_bandpass = settings.get('apply_bandpass', False)
    f_low = settings.get('f_low', 5)
    f_high = settings.get('f_high', 80)
    apply_agc_flag = settings.get('apply_agc', False)
    agc_window = settings.get('agc_window', 250)

    # Step 4: Read all data for this crossline at once
    print(f"[Stacking] Reading data...")
    read_start = time.time()
    xline_data = np.zeros((n_offsets, n_il_data, n_time), dtype=np.float32)
    for k, offset_zarr in enumerate(offset_bins):
        xline_data[k, :, :] = offset_zarr[:, xl, :]
    print(f"[Stacking] Data read in {time.time() - read_start:.1f}s")

    # Initialize output
    stack = np.zeros((n_il_data, n_time), dtype=np.float32)

    # Step 5: Process each inline
    print(f"[Stacking] Processing sequence: Inverse NMO -> Processing -> Forward NMO -> Stack")
    for il in range(n_il_data):
        if il % 50 == 0:
            elapsed = time.time() - start_time
            if il > 0:
                eta = (elapsed / il) * (n_il_data - il)
                print(f"[Stacking] IL {il}/{n_il_data} - ETA: {eta:.0f}s")
            if progress_callback:
                progress_callback(il, n_il_data, f"IL={il}")

        # Get selected velocity
        vel_func = interp_vel_model.get_velocity_at(il, xl)
        if vel_func is None:
            continue

        # Get gather
        gather = xline_data[:, il, :].copy()  # (n_offsets, n_time)

        # === PROCESSING SEQUENCE ===

        # 1. INVERSE NMO with initial velocity (remove existing NMO)
        if interp_initial_model is not None:
            initial_vel_func = interp_initial_model.get_velocity_at(il, xl)
            if initial_vel_func is not None:
                gather = apply_nmo_with_velocity_model(
                    gather, offset_values, t_coords, initial_vel_func,
                    inverse=True, stretch_mute_percent=100,  # No stretch mute for inverse
                    vel_t_coords=initial_velocity_grid.t_coords
                )

        # 2. Apply processing (on un-NMO'd data)
        if apply_bandpass:
            gather = apply_bandpass_filter(gather, dt_ms, f_low, f_high)

        if apply_agc_flag:
            gather = apply_agc(gather, agc_window, dt_ms)

        if top_mute_enabled or bottom_mute_enabled:
            v_t = v_top if top_mute_enabled else None
            v_b = v_bottom if bottom_mute_enabled else None
            gather = apply_velocity_mute(gather, offset_values, t_coords, v_t, v_b)

        # 3. FORWARD NMO with selected velocity
        gather_nmo = apply_nmo_with_velocity_model(
            gather, offset_values, t_coords, vel_func,
            inverse=False, stretch_mute_percent=stretch_percent,
            vel_t_coords=velocity_grid.t_coords
        )

        # 4. Stack (mean over offsets)
        stack[il, :] = np.nanmean(gather_nmo, axis=0)

    elapsed = time.time() - start_time
    print(f"[Stacking] Complete: {n_il_data} traces in {elapsed:.1f}s ({n_il_data/elapsed:.0f} traces/s)")

    if progress_callback:
        progress_callback(n_il_data, n_il_data, "Complete")

    return stack, il_coords
