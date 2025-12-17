# PSTM Kernel Optimization Analysis

## Executive Summary

**Key Finding**: Numba CPU with linear interpolation is currently the **fastest** kernel option, outperforming MLX GPU by 7.3x.

| Kernel | Time (s) | vs Baseline | Status |
|--------|----------|-------------|--------|
| **Numba CPU (linear)** | 1.59 | 1.00x | **FASTEST** |
| MLX GPU (Optimized) | 11.57 | 0.14x | 7.3x slower |
| Numba CPU (sinc8) | 12.94 | 0.12x | Interp bottleneck |
| MLX GPU (Basic) | 2538 | 0.00x | Python loop issue |

---

## Section 1: Current Performance Breakdown

### Numba CPU with Linear Interpolation

| Component | % of Time | Operations |
|-----------|-----------|------------|
| DSR travel time (2× sqrt) | 50% | pillar × trace × sample |
| Output accumulation | 23% | Memory writes |
| Distance calculation | 14% | pillar × trace |
| Weight computations | 6% | Taper, spreading, obliquity |
| Linear interpolation | 3% | 2 reads + lerp |
| Sample index calc | 5% | Arithmetic |

### MLX GPU Components (Vectorized)

| Operation | % of Time | Notes |
|-----------|-----------|-------|
| Accumulation (sum) | 25.6% | Memory bandwidth limited |
| DSR travel time | 24.1% | 2× sqrt per contribution |
| Gather/interpolation | 23.9% | Random memory access |
| Aperture masking | 8.9% | Boolean operations |
| Distance calculation | 8.6% | Vector math |
| Other | 9.0% | Index calc, array conversion |

---

## Section 2: Why MLX is Currently Slower

### 1. Python Loop Overhead
- MLX "Basic" kernel: Python `for` loop over traces → **2500s**
- MLX "Optimized" kernel: Chunked but still has Python loops → **11.5s**
- Numba: Compiles to native code with `prange` parallelization → **1.6s**

### 2. GPU Kernel Launch Overhead
- Each MLX array operation = separate GPU kernel launch
- Small operations don't amortize the launch cost
- Numba compiles entire migration into single fused kernel

### 3. Memory Access Patterns
- Trace interpolation requires random memory access (gather)
- GPU prefers coalesced (sequential) memory access
- CPU cache handles irregular access patterns better for small tiles

---

## Section 3: Optimization Proposals

### Phase 1: Quick Wins (Low Complexity, High Impact)

| Optimization | Benefit | Implementation | Risk |
|--------------|---------|----------------|------|
| Use linear interpolation | 5-10x vs sinc8 | Settings change | Slight quality reduction |
| Pre-filter traces by aperture | ~40% | KD-tree query before kernel | None |
| Output decimation for preview | ~4x | Process every Nth sample | User controls |

**Expected improvement**: ~50% speedup with no algorithm changes

### Phase 2: Numba Optimization (Medium Complexity)

| Optimization | Benefit | Implementation | Risk |
|--------------|---------|----------------|------|
| Vectorize DSR over time | ~30% | Broadcast sqrt over all t | Memory increase |
| Fast sqrt approximation | ~20% on DSR | Quake algorithm | ~0.1% accuracy |
| Tile-local trace sorting | ~15% | Sort by midpoint distance | Pre-processing time |

**Expected improvement**: Additional 30-40% speedup

### Phase 3: MLX Rewrite (High Complexity)

| Optimization | Benefit | Implementation | Risk |
|--------------|---------|----------------|------|
| Full trace vectorization | 5-10x for MLX | Eliminate Python loops | Memory explosion |
| Custom Metal kernel | 2-3x over MLX | Native GPU shader | Platform-specific |
| Hybrid CPU+GPU | ~50% utilization | Async pipeline | Synchronization |

**Expected**: MLX competitive with or faster than Numba

---

## Section 4: Detailed Optimization Descriptions

### 4.1 Vectorize DSR Over Time Samples (Numba)

**Current code** (simplified):
```python
for iot in range(nt):
    t0_s = t_axis[iot] / 1000.0
    vrms = vrms_1d[iot]
    t_src = sqrt(t0_half_sq + ds2 / v_sq)
    t_rec = sqrt(t0_half_sq + dr2 / v_sq)
```

**Proposed**:
```python
# Pre-compute distance² once
ds2 = (ox - sx)**2 + (oy - sy)**2
dr2 = (ox - rx)**2 + (oy - ry)**2

# Vectorize over all time samples
t0_half_sq = (t_axis / 2000.0)**2  # (nt,)
v_sq = vrms_1d**2  # (nt,)

# Broadcast: ds2 is scalar, t0_half_sq and v_sq are (nt,)
t_src = sqrt(t0_half_sq + ds2 / v_sq)  # (nt,)
t_rec = sqrt(t0_half_sq + dr2 / v_sq)  # (nt,)
t_travel = t_src + t_rec  # (nt,)
```

**Benefit**: Eliminates inner loop, enables SIMD vectorization

### 4.2 Fast Sqrt Approximation

**Quake III fast inverse sqrt** (adapted):
```python
@njit(fastmath=True)
def fast_sqrt(x):
    # Fast approximation using bit manipulation
    # Accuracy: ~0.1% error
    y = x
    i = y.view(np.int32)
    i = 0x1fbd1df5 + (i >> 1)  # Magic constant
    y = i.view(np.float32)
    y = 0.5 * (y + x / y)  # One Newton-Raphson iteration
    return y
```

### 4.3 Pre-filter Traces by Aperture

```python
# Before kernel call
from scipy.spatial import cKDTree

tree = cKDTree(midpoints)  # Already built for spatial index
tile_center = (tile.center_x, tile.center_y)
trace_indices = tree.query_ball_point(tile_center, r=max_aperture * 1.1)
filtered_traces = traces[trace_indices]
```

### 4.4 MLX Full Vectorization

**Current** (Python loop over traces):
```python
for i in range(n_traces):
    sx = source_x[i]
    # ... process one trace
```

**Proposed** (full vectorization):
```python
# Shape: (n_pillars, n_traces, n_times)
t_travel = compute_dsr_vectorized(
    ox[:, None, None],  # (n_pillars, 1, 1)
    oy[:, None, None],
    sx[None, :, None],  # (1, n_traces, 1)
    sy[None, :, None],
    rx[None, :, None],
    ry[None, :, None],
    t_axis[None, None, :],  # (1, 1, n_times)
    vrms[None, None, :],
)  # Result: (n_pillars, n_traces, n_times)

# Accumulate all at once
contribution = amp_interp * weights * valid_mask
image = mx.sum(contribution, axis=1)  # Sum over traces
```

---

## Section 5: Estimated Timeline & Impact

| Phase | Duration | Cumulative Speedup | Tile Time (from 139s) |
|-------|----------|-------------------|----------------------|
| Current | - | 1x | 139s |
| Phase 1 | 1-2 days | ~2x | ~70s |
| Phase 2 | 3-5 days | ~3x | ~45s |
| Phase 3 | 1-2 weeks | ~4-5x | ~30s |

---

## Files Created

- `scripts/profile_kernel.py` - Component-level profiling
- `scripts/profile_kernel_direct.py` - Direct kernel timing
- `scripts/compare_kernels.py` - CPU vs GPU comparison
- `docs/kernel_profiling_results.md` - Detailed profiling data
- `docs/kernel_optimization_analysis.md` - This document

---

## Conclusion

1. **Stick with Numba CPU** for now - it's faster than current MLX
2. **Use linear interpolation** as default for 5-10x speedup
3. **Implement Phase 1 quick wins** for immediate 50% improvement
4. **Consider MLX rewrite only if** full vectorization can be achieved
