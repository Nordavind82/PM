# Metal Kernel Optimization Roadmap

## Overview

This document outlines a systematic approach to optimizing the PSTM Metal kernels on Apple Silicon. Each phase introduces new optimizations with corresponding tests to validate correctness and measure performance improvements.

## File Structure

All optimized implementations will be created as **new files** alongside existing ones:

```
pstm/
├── kernels/
│   ├── metal_compiled.py          # Current (baseline)
│   ├── metal_optimized_v1.py      # Phase 1 optimizations
│   ├── metal_optimized_v2.py      # Phase 2 optimizations
│   └── metal_optimized_v3.py      # Phase 3 optimizations
├── metal/
│   └── shaders/
│       ├── pstm_migration.metal           # Current (baseline)
│       ├── pstm_migration_v1.metal        # Phase 1 shaders
│       ├── pstm_migration_v2.metal        # Phase 2 shaders
│       └── pstm_migration_v3.metal        # Phase 3 shaders
└── tests/
    └── benchmarks/
        ├── test_metal_correctness.py      # Correctness validation
        ├── benchmark_metal_versions.py    # Performance comparison
        └── generate_test_datasets.py      # Synthetic data generator
```

---

## Testing Infrastructure

### Test Dataset Sizes

| Size | Survey | Traces | Memory | Use Case |
|------|--------|--------|--------|----------|
| Tiny | 20x20 midpoints, 2 offsets | ~800 | ~6 MB | Quick iteration, debugging |
| Small | 40x40 midpoints, 4 offsets | ~6,400 | ~50 MB | Unit tests, correctness |
| Medium | 80x80 midpoints, 8 offsets | ~51,200 | ~400 MB | Performance benchmarks |
| Large | 160x160 midpoints, 16 offsets | ~410,000 | ~3.2 GB | Stress tests, real-world |

### Dataset Generation Script

Create `tests/benchmarks/generate_test_datasets.py`:

```python
"""
Generate standardized test datasets for Metal kernel benchmarking.
Uses synthetic_gather_2.py example_4 pattern.
"""

from pathlib import Path
from pstm.synthetic import (
    create_simple_synthetic,
    export_to_zarr_parquet,
)

DATASETS = {
    "tiny": {
        "survey_extent": 500.0,
        "grid_spacing": 25.0,
        "offsets": [200, 400],
        "n_samples": 501,
    },
    "small": {
        "survey_extent": 1000.0,
        "grid_spacing": 25.0,
        "offsets": [200, 400, 600, 800],
        "n_samples": 1001,
    },
    "medium": {
        "survey_extent": 2000.0,
        "grid_spacing": 25.0,
        "offsets": [200, 400, 600, 800, 1000, 1200, 1400, 1600],
        "n_samples": 1501,
    },
    "large": {
        "survey_extent": 4000.0,
        "grid_spacing": 25.0,
        "offsets": list(range(200, 3400, 200)),  # 16 offsets
        "n_samples": 2001,
    },
}

def generate_dataset(name: str, output_base: Path):
    """Generate a single test dataset."""
    cfg = DATASETS[name]
    result = create_simple_synthetic(
        diffractor_x=cfg["survey_extent"] / 2,
        diffractor_y=cfg["survey_extent"] / 2,
        diffractor_z=800.0,
        survey_extent=cfg["survey_extent"],
        grid_spacing=cfg["grid_spacing"],
        offsets=cfg["offsets"],
        azimuths=[0, 90],
        velocity=2500.0,
        n_samples=cfg["n_samples"],
        dt_ms=2.0,
        wavelet_freq=25.0,
        noise_level=0.05,
    )

    output_dir = output_base / f"benchmark_{name}"
    export_to_zarr_parquet(result, output_dir)
    return output_dir
```

### Correctness Validation

Create `tests/benchmarks/test_metal_correctness.py`:

```python
"""
Validate that optimized kernels produce identical results to baseline.
"""

import numpy as np

def validate_migration_result(baseline_image, optimized_image, rtol=1e-4, atol=1e-6):
    """Compare migration outputs within tolerance."""

    # Check shapes match
    assert baseline_image.shape == optimized_image.shape, \
        f"Shape mismatch: {baseline_image.shape} vs {optimized_image.shape}"

    # Check values within tolerance
    max_diff = np.max(np.abs(baseline_image - optimized_image))
    rel_diff = max_diff / (np.max(np.abs(baseline_image)) + 1e-10)

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")

    assert np.allclose(baseline_image, optimized_image, rtol=rtol, atol=atol), \
        f"Results differ beyond tolerance: max_diff={max_diff:.2e}"

    return True
```

### Performance Benchmark Framework

Create `tests/benchmarks/benchmark_metal_versions.py`:

```python
"""
Benchmark different Metal kernel versions.
"""

import time
import numpy as np
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    version: str
    dataset: str
    n_traces: int
    n_output_samples: int
    total_time_s: float
    kernel_time_s: float
    buffer_time_s: float
    throughput_mtrace_s: float  # Million traces per second

def run_benchmark(kernel_class, dataset_path, n_warmup=2, n_runs=5):
    """Run benchmark with warmup and multiple iterations."""

    # Warmup runs
    for _ in range(n_warmup):
        run_migration(kernel_class, dataset_path)

    # Timed runs
    times = []
    for _ in range(n_runs):
        result = run_migration(kernel_class, dataset_path)
        times.append(result.compute_time_s)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }
```

---

## Phase 1: Memory Management & Python Binding (Target: 1.5-2x speedup)

### Task 1.1: Buffer Pool Implementation

**File:** `pstm/kernels/metal_optimized_v1.py`

**Objective:** Eliminate per-tile buffer allocation overhead

**Sub-tasks:**

- [ ] **1.1.1** Create `BufferPool` class
  - Pre-allocate buffers at initialization based on max tile size
  - Track buffer usage with reference counting
  - Support multiple buffer sizes (small, medium, large)

- [ ] **1.1.2** Implement buffer size estimation
  - Calculate max buffer size from config (max_aperture, tile_size)
  - Add 20% headroom for edge cases

- [ ] **1.1.3** Replace `_create_buffer` with pool allocation
  - Modify `migrate_tile` to request buffers from pool
  - Return buffers to pool after kernel completion

- [ ] **1.1.4** Add buffer reuse for static data
  - Velocity, time axis, apertures don't change between tiles
  - Create once at initialization, reuse

**Test Protocol:**
```bash
# Generate test data
python tests/benchmarks/generate_test_datasets.py --size small

# Run correctness test
python -m pytest tests/benchmarks/test_metal_correctness.py \
    --baseline metal_compiled --optimized metal_optimized_v1

# Run performance benchmark
python tests/benchmarks/benchmark_metal_versions.py \
    --versions baseline,v1 --dataset small --runs 10
```

**Expected Metrics:**
- Buffer allocation time: <1ms (from ~10-50ms)
- Overall speedup: 1.2-1.4x

---

### Task 1.2: Double Buffering for Overlap

**File:** `pstm/kernels/metal_optimized_v1.py` (extend)

**Objective:** Overlap CPU data preparation with GPU execution

**Sub-tasks:**

- [ ] **1.2.1** Create two buffer sets (A and B)
  - While GPU processes tile N with buffer set A
  - CPU prepares tile N+1 in buffer set B

- [ ] **1.2.2** Implement async command buffer submission
  - Use `addCompletedHandler` instead of `waitUntilCompleted`
  - Track in-flight command buffers

- [ ] **1.2.3** Add synchronization primitives
  - Fence/semaphore for buffer handoff
  - Ensure correct ordering

- [ ] **1.2.4** Handle edge cases
  - First tile (no overlap possible)
  - Last tile (drain pipeline)
  - Error handling with async execution

**Test Protocol:**
```bash
# Test with medium dataset (more tiles = more overlap opportunity)
python tests/benchmarks/benchmark_metal_versions.py \
    --versions v1_no_overlap,v1_overlap --dataset medium --runs 5

# Verify no race conditions
python tests/benchmarks/test_metal_correctness.py \
    --baseline metal_compiled --optimized metal_optimized_v1 \
    --dataset medium --repeat 10
```

**Expected Metrics:**
- CPU-GPU overlap: 30-50% of buffer prep time hidden
- Additional speedup: 1.1-1.2x (cumulative: 1.3-1.5x)

---

### Task 1.3: Eliminate Python/NumPy Copies

**File:** `pstm/kernels/metal_optimized_v1.py` (extend)

**Objective:** Use shared memory to avoid CPU-GPU copies

**Sub-tasks:**

- [ ] **1.3.1** Use `MTLResourceStorageModeShared` with direct pointer access
  - Map Metal buffer memory directly
  - Write numpy data directly to mapped pointer

- [ ] **1.3.2** Implement zero-copy buffer creation
  - Use `newBufferWithBytesNoCopy_` where possible
  - Handle memory alignment requirements

- [ ] **1.3.3** Profile memory bandwidth
  - Add timing for buffer creation vs data copy
  - Identify remaining bottlenecks

**Test Protocol:**
```bash
# Profile memory operations
python tests/benchmarks/profile_memory_ops.py --version v1

# Verify data integrity after zero-copy
python tests/benchmarks/test_metal_correctness.py \
    --baseline metal_compiled --optimized metal_optimized_v1 \
    --dataset small --check-intermediate-buffers
```

**Expected Metrics:**
- Buffer creation time: <0.5ms (from <1ms)
- Memory bandwidth utilization: >80% of theoretical

---

### Task 1.4: Velocity as 3D Texture

**Files:**
- `pstm/metal/shaders/pstm_migration_v1.metal`
- `pstm/kernels/metal_optimized_v1.py`

**Objective:** Use hardware texture interpolation for velocity

**Sub-tasks:**

- [ ] **1.4.1** Create `MTLTexture` for velocity model
  - 3D texture with `.r32Float` or `.r16Float` format
  - Set up texture sampler with trilinear filtering

- [ ] **1.4.2** Modify shader to use texture sampling
  ```metal
  // Replace:
  float velocity = vrms[velo_idx];
  // With:
  float velocity = velocity_texture.sample(sampler, float3(x, y, t)).r;
  ```

- [ ] **1.4.3** Add FP16 velocity option
  - Convert velocity to half-precision on CPU
  - Verify accuracy is sufficient (typically <0.1% error)

- [ ] **1.4.4** Handle 1D velocity case
  - 1D texture for V(t) only
  - Automatic broadcast in shader

**Test Protocol:**
```bash
# Test with 3D velocity model
python tests/benchmarks/test_metal_correctness.py \
    --baseline metal_compiled --optimized metal_optimized_v1 \
    --dataset medium --velocity-model cube_3d

# Compare texture vs buffer performance
python tests/benchmarks/benchmark_velocity_access.py \
    --methods buffer,texture_fp32,texture_fp16
```

**Expected Metrics:**
- Velocity lookup time: 50% reduction (hardware interpolation is free)
- Memory bandwidth: 50% reduction with FP16

---

## Phase 1 Completion Checkpoint

**Validation Steps:**

1. [ ] All Phase 1 tasks completed
2. [ ] Correctness tests pass on all dataset sizes
3. [ ] Performance improvement: 1.5-2x over baseline
4. [ ] No memory leaks (run with Instruments)
5. [ ] Documentation updated

**Benchmark Command:**
```bash
python tests/benchmarks/benchmark_metal_versions.py \
    --versions baseline,v1 \
    --datasets tiny,small,medium \
    --runs 10 \
    --output results/phase1_benchmark.json
```

---

## Phase 2: Shader Optimizations (Target: 3-4x cumulative speedup)

### Task 2.1: Threadgroup Memory for Trace Geometry

**File:** `pstm/metal/shaders/pstm_migration_v2.metal`

**Objective:** Cache trace geometry in fast threadgroup memory

**Sub-tasks:**

- [ ] **2.1.1** Design threadgroup work distribution
  - Each threadgroup processes a "patch" of output points (e.g., 4x4x16)
  - Determine optimal patch size for occupancy

- [ ] **2.1.2** Implement cooperative geometry loading
  ```metal
  threadgroup float tg_source_x[TRACE_BLOCK_SIZE];
  threadgroup float tg_source_y[TRACE_BLOCK_SIZE];
  // ... other geometry

  // Cooperative load
  for (uint i = tid; i < n_traces_in_block; i += tg_size) {
      tg_source_x[i] = source_x[block_start + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ```

- [ ] **2.1.3** Implement trace blocking
  - Process traces in blocks that fit in threadgroup memory
  - Accumulate partial results per block

- [ ] **2.1.4** Tune threadgroup size
  - Test different configurations: 64, 128, 256, 512 threads
  - Balance occupancy vs shared memory usage

**Test Protocol:**
```bash
# Correctness with new shader
python tests/benchmarks/test_metal_correctness.py \
    --baseline metal_optimized_v1 --optimized metal_optimized_v2 \
    --dataset small

# Profile threadgroup utilization
xcrun metal-profiler --capture \
    python tests/benchmarks/run_single_tile.py --version v2

# Benchmark different threadgroup sizes
python tests/benchmarks/benchmark_threadgroup_size.py \
    --sizes 64,128,256,512 --dataset medium
```

**Expected Metrics:**
- Global memory reads: 8x reduction for geometry
- Threadgroup occupancy: >75%
- Speedup: 2-3x over Phase 1

---

### Task 2.2: Simdgroup Operations

**File:** `pstm/metal/shaders/pstm_migration_v2.metal` (extend)

**Objective:** Use simdgroup primitives for efficient data sharing and reduction

**Sub-tasks:**

- [ ] **2.2.1** Implement simd_shuffle for neighbor access
  ```metal
  // Share data within simdgroup without memory
  float neighbor_amp = simd_shuffle(amplitude, lane_id ^ 1);
  ```

- [ ] **2.2.2** Use simd_sum for accumulation
  ```metal
  // Reduce across simdgroup
  float local_sum = ...;
  float simd_total = simd_sum(local_sum);
  ```

- [ ] **2.2.3** Implement simd_ballot for early exit
  ```metal
  // Check if any thread in simdgroup has work
  simd_vote::vote_t any_in_aperture = simd_ballot(in_aperture);
  if (any_in_aperture == 0) return;
  ```

- [ ] **2.2.4** Optimize fold accumulation
  - Use simd_sum instead of atomic operations
  - Single atomic per simdgroup instead of per thread

**Test Protocol:**
```bash
# Verify simd operations don't affect results
python tests/benchmarks/test_metal_correctness.py \
    --baseline metal_optimized_v2_no_simd --optimized metal_optimized_v2 \
    --dataset small --rtol 1e-5

# Profile simd utilization
xcrun metal-profiler --capture --simd-stats \
    python tests/benchmarks/run_single_tile.py --version v2
```

**Expected Metrics:**
- Atomic operation reduction: 32x (one per simdgroup)
- Additional speedup: 1.2-1.5x

---

### Task 2.3: Vectorized Trace Data Access

**File:** `pstm/metal/shaders/pstm_migration_v2.metal` (extend)

**Objective:** Use SIMD vector loads for trace amplitudes

**Sub-tasks:**

- [ ] **2.3.1** Align trace data for vector loads
  - Ensure trace start addresses are 16-byte aligned
  - Pad trace length to multiple of 4

- [ ] **2.3.2** Implement float4 trace loading
  ```metal
  // Load 4 consecutive samples at once
  float4 samples = *reinterpret_cast<device const float4*>(trace + idx);
  ```

- [ ] **2.3.3** Vectorized interpolation
  - Load float4 containing both interpolation points
  - Use vector operations for interpolation

- [ ] **2.3.4** Handle boundary conditions
  - First/last samples need special handling
  - Avoid out-of-bounds vector reads

**Test Protocol:**
```bash
# Verify vectorized access is correct
python tests/benchmarks/test_metal_correctness.py \
    --baseline metal_optimized_v2_scalar --optimized metal_optimized_v2 \
    --dataset small --check-interpolation

# Measure memory bandwidth improvement
python tests/benchmarks/benchmark_memory_bandwidth.py --version v2
```

**Expected Metrics:**
- Memory bandwidth efficiency: 90%+ of theoretical
- Trace load time: 2-4x improvement

---

### Task 2.4: Loop Unrolling and Instruction Scheduling

**File:** `pstm/metal/shaders/pstm_migration_v2.metal` (extend)

**Objective:** Maximize instruction-level parallelism

**Sub-tasks:**

- [ ] **2.4.1** Unroll inner trace loop
  ```metal
  // Process 4 traces per iteration
  #pragma unroll 4
  for (int tr = 0; tr < n_traces; tr += 4) {
      // Process traces tr, tr+1, tr+2, tr+3
  }
  ```

- [ ] **2.4.2** Interleave independent operations
  - Start next trace's aperture check while current interpolates
  - Overlap memory loads with ALU operations

- [ ] **2.4.3** Precompute loop-invariant values
  - Move constant calculations outside loops
  - Cache repeated subexpressions

- [ ] **2.4.4** Use `[[unroll]]` attribute where beneficial
  - Test different unroll factors
  - Profile with Metal System Trace

**Test Protocol:**
```bash
# Profile instruction mix
xcrun metal-profiler --capture --instruction-stats \
    python tests/benchmarks/run_single_tile.py --version v2

# Compare unroll factors
python tests/benchmarks/benchmark_unroll_factors.py \
    --factors 1,2,4,8 --dataset medium
```

**Expected Metrics:**
- ALU utilization: >80%
- Memory stall cycles: <20% of total

---

## Phase 2 Completion Checkpoint

**Validation Steps:**

1. [ ] All Phase 2 tasks completed
2. [ ] Correctness tests pass with rtol=1e-5
3. [ ] Cumulative speedup: 3-4x over baseline
4. [ ] GPU occupancy: >75%
5. [ ] No register spilling

**Benchmark Command:**
```bash
python tests/benchmarks/benchmark_metal_versions.py \
    --versions baseline,v1,v2 \
    --datasets tiny,small,medium,large \
    --runs 10 \
    --output results/phase2_benchmark.json
```

---

## Phase 3: Algorithmic Optimizations (Target: 5-10x cumulative speedup)

### Task 3.1: Traveltime Table Precomputation

**Files:**
- `pstm/algorithm/traveltime_tables.py`
- `pstm/metal/shaders/pstm_migration_v3.metal`

**Objective:** Replace DSR computation with texture lookup

**Sub-tasks:**

- [ ] **3.1.1** Design traveltime table structure
  - Table dimensions: (offset, t0) for 1D velocity
  - Table dimensions: (x, y, offset, t0) for 3D velocity
  - Determine required resolution

- [ ] **3.1.2** Implement CPU table generation
  ```python
  def precompute_traveltime_table(
      velocity_model,
      offset_range,
      t0_range,
      resolution
  ) -> np.ndarray:
      """Precompute t_travel(offset, t0) table."""
  ```

- [ ] **3.1.3** Store table as Metal texture
  - 2D texture for 1D velocity
  - 3D/Array texture for 3D velocity
  - FP16 precision (sufficient for traveltimes)

- [ ] **3.1.4** Modify shader to use table lookup
  ```metal
  // Replace DSR computation:
  float t_travel = traveltime_table.sample(
      sampler,
      float2(offset / max_offset, t0 / max_t0)
  ).r;
  ```

- [ ] **3.1.5** Handle VTI anisotropy tables
  - Extend table to 3D: (offset, t0, eta) or
  - Precompute separate tables per eta value

**Test Protocol:**
```bash
# Test table accuracy vs exact DSR
python tests/benchmarks/test_traveltime_table_accuracy.py \
    --resolutions 256,512,1024 --velocity-models constant,linear,3d

# Compare table vs compute performance
python tests/benchmarks/benchmark_traveltime_methods.py \
    --methods dsr_compute,table_fp32,table_fp16
```

**Expected Metrics:**
- Traveltime accuracy: <0.1% relative error
- Compute reduction: 2x (eliminates sqrt operations)

---

### Task 3.2: Persistent Kernel Pattern

**Files:**
- `pstm/kernels/metal_optimized_v3.py`
- `pstm/metal/shaders/pstm_migration_v3.metal`

**Objective:** Eliminate per-tile kernel launch overhead

**Sub-tasks:**

- [ ] **3.2.1** Design work queue structure
  ```metal
  struct TileWorkItem {
      uint tile_id;
      uint x_start, x_end;
      uint y_start, y_end;
      // ... other tile info
  };
  ```

- [ ] **3.2.2** Implement GPU-side work distribution
  ```metal
  kernel void persistent_migrate(
      device TileWorkItem* work_queue [[buffer(0)]],
      device atomic_uint* queue_head [[buffer(1)]],
      // ... other buffers
  ) {
      while (true) {
          // Atomically claim next tile
          uint tile_idx = atomic_fetch_add(queue_head, 1);
          if (tile_idx >= n_tiles) break;

          // Process tile
          process_tile(work_queue[tile_idx], ...);
      }
  }
  ```

- [ ] **3.2.3** Implement indirect command buffers
  - Use `MTLIndirectCommandBuffer` for GPU-driven dispatch
  - CPU enqueues work, GPU consumes asynchronously

- [ ] **3.2.4** Handle dynamic work balancing
  - Some tiles have more traces than others
  - Work stealing between threadgroups

**Test Protocol:**
```bash
# Test persistent kernel correctness
python tests/benchmarks/test_metal_correctness.py \
    --baseline metal_optimized_v2 --optimized metal_optimized_v3 \
    --dataset medium

# Profile kernel launch overhead
python tests/benchmarks/benchmark_launch_overhead.py \
    --versions v2_per_tile,v3_persistent --n_tiles 100,500,1000
```

**Expected Metrics:**
- Kernel launch overhead: <100us total (from ~100us per tile)
- GPU idle time between tiles: ~0

---

### Task 3.3: Hierarchical/Adaptive Migration

**Files:**
- `pstm/algorithm/adaptive_migration.py`
- `pstm/metal/shaders/pstm_migration_v3.metal`

**Objective:** Skip computation for low-amplitude regions

**Sub-tasks:**

- [ ] **3.3.1** Implement coarse-pass migration
  - 4x downsampled output grid
  - Compute approximate amplitude map

- [ ] **3.3.2** Design amplitude threshold mask
  ```python
  # After coarse pass
  significant_mask = coarse_image > threshold
  # Dilate mask to include neighbors
  significant_mask = morphological_dilate(significant_mask, radius=2)
  ```

- [ ] **3.3.3** Implement fine-pass with mask
  - Only compute output points where mask is True
  - Use indirect dispatch for sparse output

- [ ] **3.3.4** Adaptive time sampling
  - Coarser sampling at deep times (lower frequency content)
  - Finer sampling at shallow times
  - Combine with time-variant approach

**Test Protocol:**
```bash
# Test adaptive migration preserves amplitude
python tests/benchmarks/test_adaptive_accuracy.py \
    --thresholds 0.001,0.01,0.1 --dataset medium

# Benchmark adaptive vs full migration
python tests/benchmarks/benchmark_adaptive.py \
    --dataset large --output results/adaptive_benchmark.json
```

**Expected Metrics:**
- Computation reduction: 2-5x (depends on data sparsity)
- Amplitude accuracy: >99% correlation with full migration

---

### Task 3.4: Multi-Resolution Output

**Files:**
- `pstm/algorithm/multiresolution.py`
- `pstm/metal/shaders/pstm_migration_v3.metal`

**Objective:** Generate multiple resolution outputs efficiently

**Sub-tasks:**

- [ ] **3.4.1** Design pyramid structure
  - Level 0: Full resolution (dx, dy, dt)
  - Level 1: 2x downsampled
  - Level 2: 4x downsampled

- [ ] **3.4.2** Implement single-pass multi-resolution
  - Accumulate to all levels simultaneously
  - Use simd operations for downsampling

- [ ] **3.4.3** Progressive refinement option
  - Start with coarse, refine on demand
  - Useful for interactive QC

**Test Protocol:**
```bash
# Verify multi-resolution consistency
python tests/benchmarks/test_multiresolution.py --dataset medium

# Benchmark multi-res vs separate runs
python tests/benchmarks/benchmark_multiresolution.py \
    --levels 1,2,3 --dataset medium
```

---

## Phase 3 Completion Checkpoint

**Validation Steps:**

1. [ ] All Phase 3 tasks completed
2. [ ] Correctness tests pass on all datasets
3. [ ] Cumulative speedup: 5-10x over baseline
4. [ ] Memory usage within bounds
5. [ ] All kernel variants documented

**Final Benchmark Command:**
```bash
python tests/benchmarks/benchmark_metal_versions.py \
    --versions baseline,v1,v2,v3 \
    --datasets tiny,small,medium,large \
    --runs 10 \
    --output results/final_benchmark.json \
    --generate-report
```

---

## Appendix A: Testing Checklist

### Per-Task Testing

For each task, complete:

- [ ] Unit test for new functionality
- [ ] Correctness test vs previous version
- [ ] Performance benchmark (min 5 runs)
- [ ] Memory leak check (Instruments)
- [ ] Edge case testing (empty tiles, max aperture, etc.)

### Integration Testing

After each phase:

- [ ] Full migration on medium dataset
- [ ] Compare output image to baseline (visual + numeric)
- [ ] Run with GUI to verify integration
- [ ] Test all kernel types (straight ray, curved ray, VTI)

### Performance Regression Testing

- [ ] Add benchmark to CI/CD
- [ ] Alert if performance drops >5%
- [ ] Track metrics over time

---

## Appendix B: Profiling Commands

### Metal System Trace
```bash
xcrun xctrace record --template 'Metal System Trace' \
    --output trace.trace \
    --launch -- python run_migration.py
```

### GPU Counters
```bash
xcrun metal-profiler --capture \
    --counters occupancy,memory_bandwidth,alu_utilization \
    python run_migration.py
```

### Memory Profiling
```bash
# Check for leaks
leaks --atExit -- python run_migration.py

# Track allocations
MallocStackLogging=1 python run_migration.py
```

---

## Appendix C: Expected Results Summary

| Version | Speedup vs Baseline | Key Improvements |
|---------|--------------------|--------------------|
| Baseline | 1.0x | Current implementation |
| V1 | 1.5-2.0x | Buffer pooling, textures, async |
| V2 | 3.0-4.0x | Threadgroup memory, simd ops |
| V3 | 5.0-10.0x | Traveltime tables, persistent kernel |

---

## Appendix D: Rollback Plan

If any optimization causes issues:

1. Each version is independent - can use any version
2. Factory pattern allows runtime selection:
   ```python
   kernel = create_kernel("metal_v2")  # or "metal_v1", "metal_compiled"
   ```
3. Config option to force specific version:
   ```yaml
   execution:
     metal_version: "v2"  # Override auto-selection
   ```
