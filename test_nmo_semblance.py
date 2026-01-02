"""Test script to debug NMO and semblance pipeline with synthetic data."""

import numpy as np
import matplotlib.pyplot as plt

# Import our modules
from pstm_view_va.processing.nmo import apply_nmo_correction, apply_nmo_with_velocity_model
from pstm_view_va.processing.semblance import compute_semblance_fast


def create_synthetic_gather(offsets, t_coords, velocity, t0_event=500.0):
    """
    Create synthetic gather with a single hyperbolic event.

    Args:
        offsets: Array of offset values in meters
        t_coords: Array of time values in ms
        velocity: NMO velocity in m/s
        t0_event: Zero-offset time of the event in ms

    Returns:
        gather: (n_offsets, n_times) array
    """
    n_offsets = len(offsets)
    n_times = len(t_coords)
    dt = t_coords[1] - t_coords[0]

    gather = np.zeros((n_offsets, n_times), dtype=np.float32)

    # Create hyperbolic event: t(x) = sqrt(t0^2 + x^2/v^2)
    for i, offset in enumerate(offsets):
        t_hyp = np.sqrt(t0_event**2 + (offset / velocity * 1000)**2)  # in ms

        # Find sample index
        idx = int(t_hyp / dt)
        if 0 <= idx < n_times:
            # Put a Ricker wavelet centered at the hyperbola
            for j in range(-10, 11):
                if 0 <= idx + j < n_times:
                    # Simple Ricker-like pulse
                    gather[i, idx + j] = np.exp(-0.5 * (j/3)**2) * np.cos(2 * np.pi * j / 10)

    return gather


def test_forward_nmo():
    """Test forward NMO correction."""
    print("=" * 60)
    print("TEST 1: Forward NMO Correction")
    print("=" * 60)

    # Setup
    offsets = np.arange(100, 2001, 100, dtype=np.float32)  # 100 to 2000m
    t_coords = np.arange(0, 2001, 2, dtype=np.float32)  # 0 to 2000ms, 2ms sampling
    velocity = 2500.0  # m/s
    t0_event = 800.0  # ms

    print(f"Offsets: {offsets[0]} to {offsets[-1]} m, {len(offsets)} traces")
    print(f"Times: {t_coords[0]} to {t_coords[-1]} ms, dt={t_coords[1]-t_coords[0]} ms")
    print(f"True velocity: {velocity} m/s")
    print(f"Event t0: {t0_event} ms")

    # Create synthetic hyperbolic gather
    gather = create_synthetic_gather(offsets, t_coords, velocity, t0_event)
    print(f"Gather shape: {gather.shape}")

    # Apply forward NMO with correct velocity - should flatten
    gather_nmo = apply_nmo_correction(gather, offsets, t_coords, velocity, inverse=False)

    # Check if event is flat after NMO
    # Find max amplitude position for each trace
    max_positions = np.argmax(np.abs(gather_nmo), axis=1)
    max_times = t_coords[max_positions]

    print(f"\nAfter forward NMO with v={velocity} m/s:")
    print(f"  Event times: min={max_times.min():.1f}, max={max_times.max():.1f}, spread={max_times.max()-max_times.min():.1f} ms")
    print(f"  Expected: all traces at t0={t0_event} ms")

    if max_times.max() - max_times.min() < 10:  # Within 10ms
        print("  ✓ PASS: Event is flattened")
    else:
        print("  ✗ FAIL: Event not flattened properly")

    return gather, gather_nmo, offsets, t_coords, velocity


def test_inverse_nmo():
    """Test inverse NMO - should create hyperbola from flat event."""
    print("\n" + "=" * 60)
    print("TEST 2: Inverse NMO Correction")
    print("=" * 60)

    # Setup - start with FLAT event (like PSTM migrated data)
    offsets = np.arange(100, 2001, 100, dtype=np.float32)
    t_coords = np.arange(0, 2001, 2, dtype=np.float32)
    velocity = 2500.0
    t0_event = 800.0
    dt = t_coords[1] - t_coords[0]

    # Create flat gather (all traces have event at same time)
    n_offsets = len(offsets)
    n_times = len(t_coords)
    flat_gather = np.zeros((n_offsets, n_times), dtype=np.float32)

    idx_event = int(t0_event / dt)
    for i in range(n_offsets):
        for j in range(-10, 11):
            if 0 <= idx_event + j < n_times:
                flat_gather[i, idx_event + j] = np.exp(-0.5 * (j/3)**2) * np.cos(2 * np.pi * j / 10)

    print(f"Created flat gather with event at t={t0_event} ms")

    # Apply INVERSE NMO - should create hyperbola
    gather_inv_nmo = apply_nmo_correction(flat_gather, offsets, t_coords, velocity, inverse=True)

    # Check event positions - should follow hyperbola
    max_positions = np.argmax(np.abs(gather_inv_nmo), axis=1)
    max_times = t_coords[max_positions]

    # Expected hyperbola times
    expected_times = np.sqrt(t0_event**2 + (offsets / velocity * 1000)**2)

    print(f"\nAfter inverse NMO with v={velocity} m/s:")
    print(f"  Near offset ({offsets[0]}m): actual={max_times[0]:.1f} ms, expected={expected_times[0]:.1f} ms")
    print(f"  Far offset ({offsets[-1]}m): actual={max_times[-1]:.1f} ms, expected={expected_times[-1]:.1f} ms")

    # Check a few offsets
    errors = np.abs(max_times - expected_times)
    print(f"  Max error: {errors.max():.1f} ms")

    if errors.max() < 20:  # Within 20ms (accounting for wavelet width)
        print("  ✓ PASS: Inverse NMO creates correct hyperbola")
    else:
        print("  ✗ FAIL: Inverse NMO not working correctly")

    return flat_gather, gather_inv_nmo


def test_semblance_on_hyperbola():
    """Test semblance computation on hyperbolic data."""
    print("\n" + "=" * 60)
    print("TEST 3: Semblance on Hyperbolic Data")
    print("=" * 60)

    # Create hyperbolic gather
    offsets = np.arange(100, 2001, 100, dtype=np.float32)
    t_coords = np.arange(0, 2001, 2, dtype=np.float32)
    true_velocity = 2500.0
    t0_event = 800.0

    gather = create_synthetic_gather(offsets, t_coords, true_velocity, t0_event)

    # Compute semblance
    v_min, v_max, v_step = 1500, 4000, 50
    semblance, velocities = compute_semblance_fast(
        gather, offsets, t_coords, v_min, v_max, v_step, window_samples=5
    )

    print(f"Semblance shape: {semblance.shape}")
    print(f"Velocities: {velocities[0]} to {velocities[-1]} m/s, {len(velocities)} values")

    # Find max semblance at the event time
    t0_idx = int(t0_event / (t_coords[1] - t_coords[0]))
    sem_at_t0 = semblance[:, t0_idx]
    max_vel_idx = np.argmax(sem_at_t0)
    picked_velocity = velocities[max_vel_idx]

    print(f"\nAt t0={t0_event} ms:")
    print(f"  True velocity: {true_velocity} m/s")
    print(f"  Picked velocity (max semblance): {picked_velocity} m/s")
    print(f"  Error: {abs(picked_velocity - true_velocity)} m/s")

    if abs(picked_velocity - true_velocity) <= v_step:
        print("  ✓ PASS: Semblance picks correct velocity")
    else:
        print("  ✗ FAIL: Semblance picks wrong velocity")

    return semblance, velocities, gather, t0_idx


def test_inverse_nmo_then_semblance():
    """
    Test the full pipeline:
    1. Start with FLAT gather (PSTM-like)
    2. Apply inverse NMO with known velocity
    3. Compute semblance
    4. Check if semblance peak matches the inverse NMO velocity
    """
    print("\n" + "=" * 60)
    print("TEST 4: Inverse NMO then Semblance (PSTM workflow)")
    print("=" * 60)

    # Setup - FLAT gather (simulating PSTM output)
    offsets = np.arange(100, 2001, 100, dtype=np.float32)
    t_coords = np.arange(0, 2001, 2, dtype=np.float32)
    migration_velocity = 2500.0  # The velocity used for "migration"
    t0_event = 800.0
    dt = t_coords[1] - t_coords[0]

    # Create flat gather
    n_offsets = len(offsets)
    n_times = len(t_coords)
    flat_gather = np.zeros((n_offsets, n_times), dtype=np.float32)

    idx_event = int(t0_event / dt)
    for i in range(n_offsets):
        for j in range(-10, 11):
            if 0 <= idx_event + j < n_times:
                flat_gather[i, idx_event + j] = np.exp(-0.5 * (j/3)**2) * np.cos(2 * np.pi * j / 10)

    print(f"Step 1: Created flat gather (PSTM-like) with event at t={t0_event} ms")

    # Step 2: Apply inverse NMO with migration velocity
    print(f"Step 2: Apply inverse NMO with v={migration_velocity} m/s")
    gather_inv = apply_nmo_correction(flat_gather, offsets, t_coords, migration_velocity, inverse=True)

    # Step 3: Compute semblance
    print(f"Step 3: Compute semblance")
    v_min, v_max, v_step = 1500, 4000, 50
    semblance, velocities = compute_semblance_fast(
        gather_inv, offsets, t_coords, v_min, v_max, v_step, window_samples=5
    )

    # Step 4: Find max semblance at event time
    t0_idx = int(t0_event / dt)
    sem_at_t0 = semblance[:, t0_idx]
    max_vel_idx = np.argmax(sem_at_t0)
    picked_velocity = velocities[max_vel_idx]

    print(f"\nStep 4: Check semblance peak at t0={t0_event} ms:")
    print(f"  Inverse NMO velocity: {migration_velocity} m/s")
    print(f"  Picked velocity (max semblance): {picked_velocity} m/s")
    print(f"  Error: {abs(picked_velocity - migration_velocity)} m/s")

    if abs(picked_velocity - migration_velocity) <= v_step:
        print("  ✓ PASS: Semblance peak matches inverse NMO velocity")
    else:
        print("  ✗ FAIL: Semblance peak does NOT match inverse NMO velocity")

    return flat_gather, gather_inv, semblance, velocities, t0_idx


def test_velocity_model_nmo():
    """Test NMO with velocity model (time-varying velocity)."""
    print("\n" + "=" * 60)
    print("TEST 5: NMO with Velocity Model (different sampling)")
    print("=" * 60)

    # Gather with fine sampling
    offsets = np.arange(100, 2001, 100, dtype=np.float32)
    t_coords = np.arange(0, 2001, 2, dtype=np.float32)  # 2ms sampling, 1001 samples
    t0_event = 800.0
    dt = t_coords[1] - t_coords[0]

    # Velocity model with coarse sampling (like 100ms output grid)
    vel_t_coords = np.arange(0, 4001, 100, dtype=np.float32)  # 100ms sampling, 41 samples
    velocity_at_t0 = 2500.0
    # Create velocity function that varies with depth
    velocity_func = 1800.0 + 1.0 * vel_t_coords  # Linear increase: 1800 + t (m/s)
    # At t=800ms, velocity should be ~2600 m/s

    print(f"Gather: {len(t_coords)} samples, dt={dt} ms, range 0-{t_coords[-1]} ms")
    print(f"Velocity model: {len(velocity_func)} samples, dt=100 ms, range 0-{vel_t_coords[-1]} ms")
    print(f"Velocity at t={t0_event}ms: {np.interp(t0_event, vel_t_coords, velocity_func):.0f} m/s")

    # Create flat gather
    n_offsets = len(offsets)
    n_times = len(t_coords)
    flat_gather = np.zeros((n_offsets, n_times), dtype=np.float32)

    idx_event = int(t0_event / dt)
    for i in range(n_offsets):
        for j in range(-10, 11):
            if 0 <= idx_event + j < n_times:
                flat_gather[i, idx_event + j] = np.exp(-0.5 * (j/3)**2) * np.cos(2 * np.pi * j / 10)

    # Apply inverse NMO WITH vel_t_coords
    print("\nTest A: apply_nmo_with_velocity_model WITH vel_t_coords")
    gather_inv_correct = apply_nmo_with_velocity_model(
        flat_gather, offsets, t_coords, velocity_func,
        inverse=True, vel_t_coords=vel_t_coords
    )

    # Apply inverse NMO WITHOUT vel_t_coords (old buggy way)
    print("Test B: apply_nmo_with_velocity_model WITHOUT vel_t_coords (buggy)")
    gather_inv_buggy = apply_nmo_with_velocity_model(
        flat_gather, offsets, t_coords, velocity_func,
        inverse=True, vel_t_coords=None
    )

    # Compute semblance on both
    v_min, v_max, v_step = 1500, 4000, 50

    sem_correct, velocities = compute_semblance_fast(
        gather_inv_correct, offsets, t_coords, v_min, v_max, v_step, window_samples=5
    )
    sem_buggy, _ = compute_semblance_fast(
        gather_inv_buggy, offsets, t_coords, v_min, v_max, v_step, window_samples=5
    )

    # Check semblance peaks
    t0_idx = int(t0_event / dt)
    expected_vel = np.interp(t0_event, vel_t_coords, velocity_func)

    picked_correct = velocities[np.argmax(sem_correct[:, t0_idx])]
    picked_buggy = velocities[np.argmax(sem_buggy[:, t0_idx])]

    print(f"\nResults at t0={t0_event} ms:")
    print(f"  Expected velocity: {expected_vel:.0f} m/s")
    print(f"  WITH vel_t_coords: picked {picked_correct:.0f} m/s, error {abs(picked_correct-expected_vel):.0f} m/s")
    print(f"  WITHOUT vel_t_coords: picked {picked_buggy:.0f} m/s, error {abs(picked_buggy-expected_vel):.0f} m/s")

    if abs(picked_correct - expected_vel) <= v_step:
        print("  ✓ Test A PASS")
    else:
        print("  ✗ Test A FAIL")

    return gather_inv_correct, gather_inv_buggy, sem_correct, sem_buggy, velocities


def test_visualization_coordinates():
    """Test that visualization coordinates match data coordinates."""
    print("\n" + "=" * 60)
    print("TEST 6: Visualization Coordinate Check")
    print("=" * 60)

    # Simulate what happens in the viewer
    v_min, v_max, v_step = 1500, 4000, 50
    velocities = np.arange(v_min, v_max + v_step, v_step, dtype=np.float32)
    t_coords = np.arange(0, 2001, 2, dtype=np.float32)

    # Create fake semblance with peak at known location
    true_vel = 2500.0
    true_t = 800.0

    n_vel = len(velocities)
    n_time = len(t_coords)
    semblance = np.zeros((n_vel, n_time), dtype=np.float32)

    # Put a peak at the known location
    vel_idx = np.argmin(np.abs(velocities - true_vel))
    t_idx = np.argmin(np.abs(t_coords - true_t))
    semblance[vel_idx, t_idx] = 1.0

    print(f"Created semblance with peak at v={velocities[vel_idx]} m/s, t={t_coords[t_idx]} ms")
    print(f"Peak at indices: vel_idx={vel_idx}, t_idx={t_idx}")
    print(f"Semblance shape: {semblance.shape} (n_vel, n_time)")

    # In the viewer, semblance is transposed: data = semblance.T
    data = semblance.T
    print(f"After transpose for display: {data.shape} (n_time, n_vel)")

    # Axis setup in viewer:
    # x_axis = velocities[0] to velocities[-1]
    # y_axis = t_coords[0] to t_coords[-1]
    x_min, x_max = velocities[0], velocities[-1]
    y_min, y_max = t_coords[0], t_coords[-1]

    print(f"X-axis (velocity): {x_min} to {x_max}")
    print(f"Y-axis (time): {y_min} to {y_max}")

    # The image drawing maps:
    # pixel (0, 0) -> top-left of image
    # For semblance.T with shape (n_time, n_vel):
    #   row 0 = time 0
    #   col 0 = velocity v_min

    # To draw velocity overlay at (vel, time):
    # x_data = vel, y_data = time
    # These should map to correct pixel position

    h, w = data.shape  # h=n_time, w=n_vel

    # Pixel of peak in transposed data
    peak_row, peak_col = np.unravel_index(np.argmax(data), data.shape)
    print(f"Peak in transposed data at row={peak_row}, col={peak_col}")

    # What velocity/time does this pixel represent?
    # col -> velocity: v = v_min + col * (v_max - v_min) / (w - 1)
    # row -> time: t = t_min + row * (t_max - t_min) / (h - 1)
    pixel_vel = x_min + peak_col * (x_max - x_min) / (w - 1) if w > 1 else x_min
    pixel_time = y_min + peak_row * (y_max - y_min) / (h - 1) if h > 1 else y_min

    print(f"Pixel maps to: v={pixel_vel:.0f} m/s, t={pixel_time:.0f} ms")
    print(f"Expected: v={true_vel:.0f} m/s, t={true_t:.0f} ms")

    if abs(pixel_vel - true_vel) < v_step and abs(pixel_time - true_t) < 10:
        print("✓ PASS: Coordinates match")
    else:
        print("✗ FAIL: Coordinate mismatch!")


def plot_results(gather, gather_nmo, semblance, velocities, t_coords, offsets, t0_idx):
    """Plot results for visual inspection."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    # Original gather
    ax = axes[0]
    extent = [offsets[0], offsets[-1], t_coords[-1], t_coords[0]]
    ax.imshow(gather.T, aspect='auto', cmap='seismic', extent=extent,
              vmin=-np.percentile(np.abs(gather), 99), vmax=np.percentile(np.abs(gather), 99))
    ax.set_xlabel('Offset (m)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Original Gather')

    # NMO corrected gather
    ax = axes[1]
    ax.imshow(gather_nmo.T, aspect='auto', cmap='seismic', extent=extent,
              vmin=-np.percentile(np.abs(gather_nmo), 99), vmax=np.percentile(np.abs(gather_nmo), 99))
    ax.set_xlabel('Offset (m)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('After NMO')

    # Semblance
    ax = axes[2]
    extent_sem = [velocities[0], velocities[-1], t_coords[-1], t_coords[0]]
    ax.imshow(semblance.T, aspect='auto', cmap='viridis', extent=extent_sem)
    ax.axhline(y=t_coords[t0_idx], color='r', linestyle='--', label=f't0={t_coords[t0_idx]}ms')

    # Mark max at t0
    sem_at_t0 = semblance[:, t0_idx]
    max_vel = velocities[np.argmax(sem_at_t0)]
    ax.plot(max_vel, t_coords[t0_idx], 'ro', markersize=10, label=f'v_max={max_vel}m/s')

    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Semblance')
    ax.legend()

    plt.tight_layout()
    plt.savefig('/Users/olegadamovich/pstm/test_nmo_semblance_result.png', dpi=150)
    print("\nPlot saved to test_nmo_semblance_result.png")
    plt.close()


def test_canvas_coordinate_mapping():
    """Test the exact coordinate mapping used in canvas overlay."""
    print("\n" + "=" * 60)
    print("TEST 7a: Canvas Coordinate Mapping")
    print("=" * 60)

    # Simulate canvas setup
    # Semblance axes
    v_min, v_max = 1500.0, 4000.0  # X-axis
    t_min, t_max = 0.0, 2000.0     # Y-axis

    # Canvas view (should match axes)
    view_x_min, view_x_max = v_min, v_max
    view_y_min, view_y_max = t_min, t_max

    # Velocity overlay with coarse sampling (100ms)
    vel_t_coords = np.arange(0, 4001, 100, dtype=np.float32)
    vel_func = 2000.0 + 0.5 * vel_t_coords  # v(t) = 2000 + 0.5*t

    print(f"View: x=[{view_x_min}, {view_x_max}], y=[{view_y_min}, {view_y_max}]")
    print(f"Velocity at t=0: {vel_func[0]:.0f} m/s")
    print(f"Velocity at t=800: {np.interp(800, vel_t_coords, vel_func):.0f} m/s")
    print(f"Velocity at t=2000: {np.interp(2000, vel_t_coords, vel_func):.0f} m/s")

    # Simulate _data_to_screen mapping
    def data_to_screen(x_data, y_data, view_x_min, view_x_max, view_y_min, view_y_max, screen_w=800, screen_h=600):
        """Simulate canvas coordinate mapping."""
        x_range = view_x_max - view_x_min
        y_range = view_y_max - view_y_min

        if x_range > 0:
            x_norm = (x_data - view_x_min) / x_range
        else:
            x_norm = 0.5

        if y_range > 0:
            y_norm = (y_data - view_y_min) / y_range
        else:
            y_norm = 0.5

        x_screen = x_norm * screen_w
        y_screen = y_norm * screen_h

        return x_screen, y_screen

    # Test: where does velocity v=2400 at t=800 map to?
    t_test = 800.0
    v_test = np.interp(t_test, vel_t_coords, vel_func)  # Should be 2400

    x_screen, y_screen = data_to_screen(v_test, t_test, view_x_min, view_x_max, view_y_min, view_y_max)
    print(f"\nVelocity point (v={v_test:.0f}, t={t_test:.0f}):")
    print(f"  Maps to screen: ({x_screen:.1f}, {y_screen:.1f})")

    # Verify: what data coordinates does this screen position represent?
    # Inverse mapping
    x_norm = x_screen / 800
    y_norm = y_screen / 600
    x_data_back = view_x_min + x_norm * (view_x_max - view_x_min)
    y_data_back = view_y_min + y_norm * (view_y_max - view_y_min)
    print(f"  Back to data: (v={x_data_back:.0f}, t={y_data_back:.0f})")

    # Now test semblance coordinate
    # If semblance peak is at velocity index 18 out of 51 velocities
    # And semblance shape is (51, 1001) transposed to (1001, 51)
    n_vel = 51
    n_time = 1001
    velocities = np.arange(1500, 4001, 50, dtype=np.float32)  # 51 values

    # Find which velocity index corresponds to v=2400
    v_target = 2400
    v_idx = np.argmin(np.abs(velocities - v_target))
    print(f"\nSemblance peak at v={v_target} m/s:")
    print(f"  Velocity index: {v_idx} (actual velocity: {velocities[v_idx]:.0f} m/s)")

    # In transposed semblance array (n_time, n_vel), this is column v_idx
    # The image is drawn with:
    #   x_scale = w / (data_x_max - data_x_min) = 51 / 2500 = 0.0204
    #   Pixel x = (v - v_min) * x_scale = (2400 - 1500) * 0.0204 = 18.36

    # So pixel column 18 corresponds to velocity 2400
    # This pixel is drawn at screen position based on view mapping

    print(f"\n  Semblance pixel column for v={v_target}: {(v_target - v_min) * n_vel / (v_max - v_min):.1f}")

    # Check if there's an off-by-half issue
    # The image draws pixels centered at integer positions
    # The overlay draws lines at exact data coordinates

    print("\n  Potential issue: image pixels vs overlay points")
    print("  Image pixel 0 represents velocity range [v_min, v_min + delta_v)")
    print("  Overlay draws at exact velocity value")

    # Let's check the exact mapping
    delta_v = (v_max - v_min) / n_vel
    print(f"  Delta_v per pixel: {delta_v:.1f} m/s")
    print(f"  Pixel 0 center: {v_min + delta_v/2:.1f} m/s")
    print(f"  Pixel {v_idx} center: {v_min + (v_idx + 0.5) * delta_v:.1f} m/s")


def test_full_workflow_with_vel_grid():
    """
    Test the EXACT workflow used in the viewer:
    1. Load velocity onto grid with different sampling
    2. Apply inverse NMO with initial_vel_grid
    3. Compute semblance
    4. Compare semblance peak with velocity overlay position
    """
    print("\n" + "=" * 60)
    print("TEST 7: Full Workflow with VelocityOutputGrid")
    print("=" * 60)

    from pstm_view_va.windows.velocity_analysis import VelocityOutputGrid

    # Setup - simulate loading velocity
    # Seismic data coordinates (2ms sampling)
    t_coords = np.arange(0, 2001, 2, dtype=np.float32)  # 1001 samples
    offsets = np.arange(100, 2001, 100, dtype=np.float32)

    # Create a velocity grid with 100ms output sampling
    vel_grid = VelocityOutputGrid()
    vel_grid.setup_grid(
        il_start=0, il_end=10, il_step=10,
        xl_start=0, xl_end=10, xl_step=10,
        output_dt_ms=100, t_min=0, t_max=4000
    )

    print(f"vel_grid.t_coords: {vel_grid.t_coords[0]} to {vel_grid.t_coords[-1]} ms, {len(vel_grid.t_coords)} samples")
    print(f"Seismic t_coords: {t_coords[0]} to {t_coords[-1]} ms, {len(t_coords)} samples")

    # Simulate a velocity model - velocity increases with time
    # v(t) = 2000 + 0.5*t
    source_vel = 2000.0 + 0.5 * vel_grid.t_coords  # At 100ms sampling
    vel_grid.velocities = np.zeros((2, 2, len(vel_grid.t_coords)), dtype=np.float32)
    vel_grid.velocities[0, 0, :] = source_vel

    print(f"Velocity at t=0: {source_vel[0]:.0f} m/s")
    print(f"Velocity at t=800: {np.interp(800, vel_grid.t_coords, source_vel):.0f} m/s")
    print(f"Velocity at t=2000: {np.interp(2000, vel_grid.t_coords, source_vel):.0f} m/s")

    # Get velocity function at grid point
    vel_func = vel_grid.get_velocity_at(0, 0)
    print(f"vel_func shape: {vel_func.shape}")
    print(f"vel_func values at indices 0,8,20: {vel_func[0]:.0f}, {vel_func[8]:.0f}, {vel_func[20]:.0f} m/s")

    # Create flat gather (PSTM output)
    t0_event = 800.0  # Event at 800ms
    dt = t_coords[1] - t_coords[0]
    n_offsets = len(offsets)
    n_times = len(t_coords)
    flat_gather = np.zeros((n_offsets, n_times), dtype=np.float32)

    idx_event = int(t0_event / dt)
    for i in range(n_offsets):
        for j in range(-10, 11):
            if 0 <= idx_event + j < n_times:
                flat_gather[i, idx_event + j] = np.exp(-0.5 * (j/3)**2) * np.cos(2 * np.pi * j / 10)

    # Expected velocity at t=800ms
    expected_vel = np.interp(t0_event, vel_grid.t_coords, vel_func)
    print(f"\nExpected velocity at t={t0_event}ms: {expected_vel:.0f} m/s")

    # Apply inverse NMO WITH vel_t_coords (correct way)
    print("\nApplying inverse NMO with vel_t_coords...")
    gather_inv = apply_nmo_with_velocity_model(
        flat_gather, offsets, t_coords, vel_func,
        inverse=True, vel_t_coords=vel_grid.t_coords
    )

    # Compute semblance
    v_min, v_max, v_step = 1500, 4000, 50
    semblance, velocities = compute_semblance_fast(
        gather_inv, offsets, t_coords, v_min, v_max, v_step, window_samples=5
    )

    # Find semblance peak at t=800ms
    t0_idx = int(t0_event / dt)
    sem_at_t0 = semblance[:, t0_idx]
    picked_vel = velocities[np.argmax(sem_at_t0)]

    print(f"\nResults:")
    print(f"  Velocity from vel_grid at t={t0_event}ms: {expected_vel:.0f} m/s")
    print(f"  Semblance peak at t={t0_event}ms: {picked_vel:.0f} m/s")
    print(f"  Difference: {abs(picked_vel - expected_vel):.0f} m/s")

    # Now check what the overlay would show
    # The overlay uses vel_func directly with vel_grid.t_coords
    # At t=800ms, we need to interpolate
    overlay_vel_at_t0 = np.interp(t0_event, vel_grid.t_coords, vel_func)
    print(f"\n  Overlay velocity at t={t0_event}ms: {overlay_vel_at_t0:.0f} m/s")

    if abs(picked_vel - overlay_vel_at_t0) <= v_step:
        print("  ✓ PASS: Semblance peak matches overlay velocity")
    else:
        print(f"  ✗ FAIL: Mismatch of {abs(picked_vel - overlay_vel_at_t0):.0f} m/s")

    # Also test what happens at different times
    print("\n  Checking alignment at multiple times:")
    for t_check in [400, 800, 1200, 1600]:
        t_idx = int(t_check / dt)
        if t_idx < semblance.shape[1]:
            sem_peak = velocities[np.argmax(semblance[:, t_idx])]
            overlay_vel = np.interp(t_check, vel_grid.t_coords, vel_func)
            diff = abs(sem_peak - overlay_vel)
            status = "✓" if diff <= v_step else "✗"
            print(f"    t={t_check}ms: semblance={sem_peak:.0f}, overlay={overlay_vel:.0f}, diff={diff:.0f} {status}")


def test_il_xl_coordinate_extraction():
    """Test that velocity extraction uses coordinates, not indices."""
    print("\n" + "=" * 60)
    print("TEST 8: IL/XL Coordinate vs Index Extraction")
    print("=" * 60)

    from pstm_view_va.io.velocity import extract_velocity_function

    # Create a 3D velocity model with specific IL/XL coordinates
    # IL coords: [50, 60, 70, 80, 90] (5 values, indices 0-4)
    # XL coords: [200, 210, 220, 230, 240] (5 values, indices 0-4)
    # Time coords: [0, 100, 200, 300, 400] (5 values)
    il_coords = np.array([50, 60, 70, 80, 90])
    xl_coords = np.array([200, 210, 220, 230, 240])
    t_coords = np.array([0, 100, 200, 300, 400], dtype=np.float32)

    # Create velocity model: velocity = 2000 + 100*il_idx + 10*xl_idx
    # So at IL=60 (idx=1), XL=220 (idx=2): v = 2000 + 100*1 + 10*2 = 2120
    vel_model = np.zeros((5, 5, 5), dtype=np.float32)
    for i in range(5):
        for j in range(5):
            vel_model[i, j, :] = 2000 + 100*i + 10*j

    print(f"Velocity model shape: {vel_model.shape}")
    print(f"IL coords: {il_coords}")
    print(f"XL coords: {xl_coords}")
    print(f"Velocity at IL=60 (idx=1), XL=220 (idx=2): {vel_model[1, 2, 0]:.0f} m/s")

    # Test with correct coordinate arrays
    vel_func_correct = extract_velocity_function(
        vel_model, il=60, xl=220, t_coords=t_coords,
        il_coords=il_coords, xl_coords=xl_coords
    )

    # Test without coordinate arrays (buggy behavior)
    # This would try to access vel_model[60, 220, :] which would be out of bounds
    # or if we clamped indices, it would get vel_model[4, 4, :] = 2040
    vel_func_buggy = extract_velocity_function(
        vel_model, il=60, xl=220, t_coords=t_coords,
        il_coords=None, xl_coords=None
    )

    expected_vel = 2120  # 2000 + 100*1 + 10*2
    correct_vel = vel_func_correct[0] if vel_func_correct is not None else None
    buggy_vel = vel_func_buggy[0] if vel_func_buggy is not None else None

    print(f"\nTest: Extract velocity at IL=60, XL=220")
    print(f"  Expected velocity: {expected_vel:.0f} m/s")
    print(f"  WITH il_coords/xl_coords: {correct_vel:.0f} m/s")
    print(f"  WITHOUT il_coords/xl_coords: {buggy_vel:.0f} m/s")

    if correct_vel == expected_vel:
        print("  ✓ PASS: Correct extraction with coordinate arrays")
    else:
        print(f"  ✗ FAIL: Got {correct_vel:.0f} instead of {expected_vel:.0f}")

    # Also test edge case: IL/XL that need interpolation to nearest
    vel_func_interp = extract_velocity_function(
        vel_model, il=65, xl=215, t_coords=t_coords,
        il_coords=il_coords, xl_coords=xl_coords
    )
    # 65 is between 60 (idx=1) and 70 (idx=2) -> nearest is 60 or 70
    # 215 is between 210 (idx=1) and 220 (idx=2) -> nearest is 210 or 220
    interp_vel = vel_func_interp[0] if vel_func_interp is not None else None
    print(f"\n  Extraction at IL=65, XL=215 (between grid points):")
    print(f"    Got: {interp_vel:.0f} m/s")
    print(f"    (Should be nearest neighbor to one of the grid values)")


if __name__ == "__main__":
    print("NMO and Semblance Pipeline Tests")
    print("=" * 60)

    # Run tests
    test_forward_nmo()
    test_inverse_nmo()
    sem_result = test_semblance_on_hyperbola()
    test_inverse_nmo_then_semblance()
    test_velocity_model_nmo()
    test_visualization_coordinates()
    test_canvas_coordinate_mapping()
    test_full_workflow_with_vel_grid()
    test_il_xl_coordinate_extraction()

    # Plot the semblance test results
    semblance, velocities, gather, t0_idx = sem_result
    t_coords = np.arange(0, 2001, 2, dtype=np.float32)
    offsets = np.arange(100, 2001, 100, dtype=np.float32)
    gather_nmo = apply_nmo_correction(gather, offsets, t_coords, 2500.0, inverse=False)
    plot_results(gather, gather_nmo, semblance, velocities, t_coords, offsets, t0_idx)

    print("\n" + "=" * 60)
    print("All tests completed!")
