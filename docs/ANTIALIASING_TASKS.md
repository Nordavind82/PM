# Anti-Aliasing Implementation Tasks

## Overview

Implementation of anti-aliasing filter for Kirchhoff PSTM migration.
Estimated complexity: Medium-High
Priority: High (currently not functional)

---

## Phase 1: Core Anti-Aliasing Implementation

### Task 1.1: Create Anti-Aliasing Filter Module
**File:** `pstm/algorithm/antialiasing.py`
**Priority:** Critical

- [ ] Create `AntiAliasingFilter` class
  - [ ] Implement `__init__` with config parameters
  - [ ] Implement `compute_local_dip()` - calculate dt/dx and dt/dy
  - [ ] Implement `compute_max_frequency()` - f_max from dip angle
  - [ ] Implement `compute_aa_weight()` - triangle/cosine filter response
  - [ ] Add unit tests for dip calculation
  - [ ] Add unit tests for frequency calculation

### Task 1.2: Implement Low-Pass Filter Utilities
**File:** `pstm/algorithm/filters.py`
**Priority:** High

- [ ] Implement `design_lowpass_filter()` - create filter coefficients
- [ ] Implement `apply_lowpass_filter()` - filter trace in frequency domain
- [ ] Implement `create_filter_bank()` - pre-compute multiple filters
- [ ] Implement `apply_filter_bank()` - efficiently filter traces
- [ ] Optimize with Numba where possible
- [ ] Add unit tests

### Task 1.3: Update Configuration Models
**File:** `pstm/config/models.py`
**Priority:** Medium

- [ ] Verify `AntiAliasingConfig` has all needed fields:
  - [ ] `enabled: bool`
  - [ ] `method: AntiAliasingMethod`
  - [ ] `num_filters: int`
  - [ ] `min_frequency_hz: float`
  - [ ] `max_frequency_hz: float`
  - [ ] `dominant_frequency_hz: float` (for analytical method)
  - [ ] `per_tile_filtering: bool`
- [ ] Add validation for frequency ranges
- [ ] Add `AntiAliasingMethod.ANALYTICAL` enum value

---

## Phase 2: Kernel Integration

### Task 2.1: Update Numba CPU Kernel
**File:** `pstm/kernels/numba_cpu_optimized.py`
**Priority:** Critical

- [ ] Add `aa_config` parameter to `OptimizedNumbaKernel.__init__()`
- [ ] Implement `_prefilter_traces()` method for filter bank approach
- [ ] Modify `migrate_tile()` to accept AA parameters
- [ ] Implement AA logic in inner loop:
  - [ ] Compute local dip for each trace-output pair
  - [ ] Calculate f_max from dip
  - [ ] **Option A (Analytical):** Apply weight based on dominant frequency
  - [ ] **Option B (Filter Bank):** Select from pre-filtered traces
- [ ] Create Numba-optimized helper functions:
  - [ ] `_compute_dip_numba()`
  - [ ] `_compute_aa_weight_numba()`
  - [ ] `_interpolate_filtered_trace_numba()`
- [ ] Add performance benchmarks

### Task 2.2: Update MLX Metal Kernel (Optional)
**File:** `pstm/kernels/mlx_metal.py`
**Priority:** Low (if MLX is slow anyway)

- [ ] Port AA logic to MLX operations
- [ ] Handle filter bank in GPU memory
- [ ] Benchmark GPU AA performance

### Task 2.3: Create Kernel Base Class Update
**File:** `pstm/kernels/base.py`
**Priority:** Medium

- [ ] Add `aa_config` to `BaseKernel` interface
- [ ] Add abstract method `supports_antialiasing() -> bool`
- [ ] Update `TraceBlock` to optionally hold filtered traces

---

## Phase 3: Executor Integration

### Task 3.1: Update Migration Executor
**File:** `pstm/pipeline/executor.py`
**Priority:** High

- [ ] Pass `aa_config` from `MigrationConfig` to kernel
- [ ] Implement per-tile trace filtering if `per_tile_filtering=True`:
  - [ ] Filter traces after loading for tile
  - [ ] Pass filtered traces to kernel
  - [ ] Clean up filtered traces after tile
- [ ] Add AA-related logging:
  - [ ] Log AA method and parameters
  - [ ] Log memory usage for filter bank
  - [ ] Log filter computation time
- [ ] Handle memory limits with filter bank

### Task 3.2: Add AA Diagnostics Output
**File:** `pstm/pipeline/executor.py`
**Priority:** Low

- [ ] Optionally output f_max volume
- [ ] Output AA weight statistics per tile
- [ ] Add to migration summary report

---

## Phase 4: GUI Integration

### Task 4.1: Update Algorithm Step UI
**File:** `pstm/gui/steps/algorithm_step.py`
**Priority:** Medium

- [ ] Add "Anti-Aliasing" collapsible section
- [ ] Add checkbox: "Enable Anti-Aliasing"
- [ ] Add dropdown: "Method" (Triangle, Analytical, Cosine)
- [ ] Add spinbox: "Number of Filters" (4-64, default 16)
- [ ] Add spinbox: "Min Frequency (Hz)" (1-20, default 5)
- [ ] Add spinbox: "Max Frequency (Hz)" (auto-detect from Nyquist)
- [ ] Add checkbox: "Per-tile filtering" (memory optimization)
- [ ] Add tooltip explanations for each option
- [ ] Wire up to `state.algorithm.anti_aliasing`

### Task 4.2: Update State Management
**File:** `pstm/gui/state.py`
**Priority:** Medium

- [ ] Ensure `AlgorithmState` includes `AntiAliasingState`
- [ ] Update `build_migration_config()` to convert AA state to config
- [ ] Add validation for AA parameters

### Task 4.3: Update Visualization Step
**File:** `pstm/gui/steps/visualization_step.py`
**Priority:** Low

- [ ] Add AA diagnostic tab (optional)
- [ ] Show f_max map overlay
- [ ] Show AA weight histogram

---

## Phase 5: Testing

### Task 5.1: Unit Tests
**Files:** `tests/test_antialiasing.py`
**Priority:** High

- [ ] Test dip calculation correctness
- [ ] Test f_max calculation for known dips
- [ ] Test filter response shapes
- [ ] Test weight calculation
- [ ] Test edge cases (zero dip, vertical, horizontal)

### Task 5.2: Integration Tests
**Files:** `tests/test_migration_aa.py`
**Priority:** High

- [ ] Test migration with AA enabled vs disabled
- [ ] Verify output differences exist
- [ ] Test with synthetic dipping reflector
- [ ] Test memory usage with filter bank
- [ ] Test per-tile filtering

### Task 5.3: Benchmark Tests
**Files:** `benchmarks/benchmark_aa.py`
**Priority:** Medium

- [ ] Benchmark AA overhead for analytical method
- [ ] Benchmark AA overhead for filter bank (8, 16, 32 filters)
- [ ] Compare quality vs performance tradeoffs
- [ ] Document results

### Task 5.4: Visual QC Tests
**Priority:** Medium

- [ ] Create synthetic with aliased steep dips
- [ ] Run with/without AA
- [ ] Generate comparison figures
- [ ] Document expected behavior

---

## Phase 6: Documentation

### Task 6.1: User Documentation
**Priority:** Low

- [ ] Add AA section to user guide
- [ ] Explain when to use AA
- [ ] Explain parameter selection
- [ ] Add troubleshooting section

### Task 6.2: Developer Documentation
**Priority:** Low

- [ ] Document AA algorithm details
- [ ] Document kernel integration points
- [ ] Add code examples

---

## Implementation Order (Recommended)

```
Week 1: Phase 1 (Core implementation)
├── Task 1.1: AntiAliasingFilter class
├── Task 1.2: Filter utilities
└── Task 1.3: Config updates

Week 2: Phase 2 (Kernel integration)
├── Task 2.1: Numba kernel (analytical method first)
├── Task 2.3: Base class updates
└── Task 5.1: Unit tests

Week 3: Phase 3 + 4 (Integration)
├── Task 3.1: Executor integration
├── Task 4.1: Algorithm step UI
├── Task 4.2: State management
└── Task 5.2: Integration tests

Week 4: Polish
├── Task 2.1: Filter bank method (if needed)
├── Task 5.3: Benchmarks
├── Task 5.4: Visual QC
└── Phase 6: Documentation
```

---

## Quick Start Implementation (Minimum Viable)

For fastest path to working AA, implement only:

1. **Analytical weight method** (no filter bank, minimal memory)
2. **Numba kernel only** (skip MLX for now)
3. **Basic UI toggle** (enable/disable + method selection)

This gives ~80% of the benefit with ~20% of the work.

### Minimal Code Changes

```python
# In numba_cpu_optimized.py, modify inner loop:

# Add after traveltime calculation:
if aa_enabled:
    # Compute local dip
    if t_mig > 1e-6:
        dip_x = (mx - ox) / (vel**2 * t_mig / 2)
        dip_y = (my - oy) / (vel**2 * t_mig / 2)
        sin_theta = sqrt(min(1.0, (dip_x * vel / 2)**2 + (dip_y * vel / 2)**2))

        if sin_theta > 0.01:
            f_max = vel / (4 * max(dx, dy) * sin_theta)
            f_max = min(f_max, nyquist)

            # Triangle filter weight
            aa_weight = max(0.0, 1.0 - dominant_freq / f_max)
        else:
            aa_weight = 1.0
    else:
        aa_weight = 1.0

    amplitude *= aa_weight
```

---

## Success Criteria

- [ ] AA produces measurably different output than no-AA
- [ ] Steep dip artifacts visibly reduced in test data
- [ ] Performance overhead < 20% for analytical method
- [ ] Memory overhead < 2x for filter bank with 16 filters
- [ ] UI properly enables/disables AA
- [ ] All tests pass
