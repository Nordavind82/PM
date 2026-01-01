#!/usr/bin/env python3
"""
Diagnostic script to check coordinate systems in PSTM migration.
"""

import numpy as np
import zarr
from pathlib import Path

print("=" * 70)
print("PSTM COORDINATE SYSTEM DIAGNOSTIC")
print("=" * 70)

# 1. Check velocity model axes
print("\n1. VELOCITY MODEL AXES")
print("-" * 50)
vel_path = '/Users/olegadamovich/SeismicData/common_offset_20m/velocity_pstm.zarr'
v = zarr.open_array(vel_path, mode='r')
v_attrs = dict(v.attrs)

x_axis = np.array(v_attrs['x_axis'])
y_axis = np.array(v_attrs['y_axis'])
print(f"   x_axis (IL): {x_axis[0]:.1f} to {x_axis[-1]:.1f} (n={len(x_axis)})")
print(f"   y_axis (XL): {y_axis[0]:.1f} to {y_axis[-1]:.1f} (n={len(y_axis)})")
print(f"   These are IL/XL numbers, NOT UTM coordinates!")

# 2. Check output grid corners
print("\n2. OUTPUT GRID CORNERS (from run_pstm_all_offsets.py)")
print("-" * 50)
GRID_CORNERS = {
    'c1': (618813.59, 5116498.50),  # Origin (IL=1, XL=1)
    'c2': (627094.02, 5106803.16),  # Inline end (IL=511, XL=1)
    'c3': (631143.35, 5110261.43),  # Far corner (IL=511, XL=427)
    'c4': (622862.92, 5119956.77),  # Crossline end (IL=1, XL=427)
}
for name, (x, y) in GRID_CORNERS.items():
    print(f"   {name}: X={x:.2f}, Y={y:.2f}")

# 3. Compute grid vectors
print("\n3. GRID VECTORS")
print("-" * 50)
c1 = np.array(GRID_CORNERS['c1'])
c2 = np.array(GRID_CORNERS['c2'])
c4 = np.array(GRID_CORNERS['c4'])

il_vec = (c2 - c1) / 510
xl_vec = (c4 - c1) / 426

print(f"   IL vector: dX={il_vec[0]:.4f}, dY={il_vec[1]:.4f} per IL")
print(f"   XL vector: dX={xl_vec[0]:.4f}, dY={xl_vec[1]:.4f} per XL")
print(f"   |IL| = {np.linalg.norm(il_vec):.2f} m (expected 25m)")
print(f"   |XL| = {np.linalg.norm(xl_vec):.2f} m (expected 12.5m)")

# 4. Sample output grid point
print("\n4. SAMPLE OUTPUT GRID POINT")
print("-" * 50)
# IL=256, XL=214 (center of grid)
il_idx = 255  # 0-based
xl_idx = 213  # 0-based
utm_x = c1[0] + (il_idx) * il_vec[0] + (xl_idx) * xl_vec[0]
utm_y = c1[1] + (il_idx) * il_vec[1] + (xl_idx) * xl_vec[1]
print(f"   Grid index: IL={il_idx+1}, XL={xl_idx+1}")
print(f"   UTM: X={utm_x:.2f}, Y={utm_y:.2f}")

# 5. Check velocity at this point
print("\n5. VELOCITY AT CENTER POINT")
print("-" * 50)
vel = np.array(v[:])
# For IL/XL indexed velocity, we use grid indices directly
vel_il = il_idx  # 0-based index for IL=256
vel_xl = xl_idx  # 0-based index for XL=214
print(f"   Velocity indices: [{vel_il}, {vel_xl}, :]")
print(f"   t=0ms: {vel[vel_il, vel_xl, 0]:.0f} m/s")
print(f"   t=500ms: {vel[vel_il, vel_xl, 250]:.0f} m/s")
print(f"   t=1000ms: {vel[vel_il, vel_xl, 500]:.0f} m/s")
print(f"   t=1500ms: {vel[vel_il, vel_xl, 750]:.0f} m/s")

# 6. Check a migrated stack
print("\n6. MIGRATED STACK ANALYSIS")
print("-" * 50)
stack_path = Path('/Users/olegadamovich/SeismicData/PSTM_common_offset_20m/migration_bin_10/migrated_stack.zarr')
if stack_path.exists():
    z = zarr.open_array(str(stack_path), mode='r')
    data = np.array(z[:])
    attrs = dict(z.attrs)

    print(f"   Shape: {data.shape}")
    print(f"   x_min: {attrs.get('x_min', 'N/A')}")
    print(f"   x_max: {attrs.get('x_max', 'N/A')}")
    print(f"   y_min: {attrs.get('y_min', 'N/A')}")
    print(f"   y_max: {attrs.get('y_max', 'N/A')}")

    # Check data values at center
    print(f"\n   Data at center (IL=256, XL=214):")
    center_trace = data[255, 213, :]
    print(f"   Non-zero samples: {np.count_nonzero(center_trace)}")
    print(f"   RMS amplitude: {np.sqrt(np.mean(center_trace**2)):.6f}")
    print(f"   Max amplitude: {np.abs(center_trace).max():.6f}")

# 7. Potential issues summary
print("\n7. POTENTIAL ISSUES")
print("-" * 50)
print("""
   A. Velocity Model Axes:
      - Velocity uses IL/XL indices (1-511, 1-427) as axes
      - Migration code should detect this and use IL/XL grids for sampling
      - VERIFY: Check that create_velocity_manager() detects IL/XL indexing

   B. Grid Rotation:
      - Output grid is rotated ~50 degrees from N-S
      - IL direction: azimuth -49.5 degrees from East
      - XL direction: azimuth +40.5 degrees from East
      - VERIFY: Check that rotated grid coordinates are passed to kernel

   C. Coordinate Consistency:
      - Trace coordinates should be in UTM
      - Output coordinates should be in UTM
      - Velocity should be sampled at IL/XL indices
      - VERIFY: All coordinates are consistent

   D. Grid Spacing:
      - Expected: dx=25m (IL), dy=12.5m (XL)
      - Actual IL spacing: 25.00m (correct)
      - Actual XL spacing: 12.50m (correct)
""")

print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
