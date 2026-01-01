# PSTM Optimization Plan

**Target Platform:** Mac M4 Max (48GB RAM)
**Benchmark Dataset:** Offset Bin 10 (819,045 traces, 500-549m offset)
**Baseline Script:** `run_pstm_all_offsets.py --bins 10`
**Output Grid:** 511 x 427 x 1001 samples (2ms interval, 0-2000ms)

---

## OPTIMIZATION RESULTS SUMMARY

| Test | Runtime | vs Baseline | Status |
|------|---------|-------------|--------|
| **Baseline (512×512 tiles)** | 679.7s | - | Reference |
| Memory 32GB→44GB | 685.9s | +0.9% | No improvement |
| Tile 256×256 | 617.3s | **-9.2%** | Improved |
| Tile 128×128 | 608.4s | **-10.5%** | Improved |
| Tile 64×64 | 606.8s | **-10.7%** | Diminishing returns |
| Threadgroup 8×8×16 | 661.4s | +8.7% (slower) | Worse |
| Threadgroup 16×16×4 | 794.1s | +30.5% (slower) | Much worse |
| Optimized (128×128 tiles) | 591.3s | **-13.0%** | Improved |
| + Async Prefetching | 587.2s | **-13.6%** | Small gain |
| **SIMD Batching (float4)** | - | Already default | ✓ |
| Multi-bin Parallel (2 workers) | ~1190s/bin | **2x slower** | ✗ Contention |
| **+ UMA Zero-Copy Buffers** | **576.7s** | **-15.2%** | **BEST** |

### Recommended Configuration:
```python
TILE_NX = 128
TILE_NY = 128
# Threadgroup: 4×4×64 (original - optimal for deep data)
# Async prefetching: Enabled (modest improvement)
# SIMD batching: Already default (pstm_migrate_3d_simd kernel)
# Multi-bin parallel: NOT recommended (GPU contention)
```

### Key Findings:
1. **Tile size matters:** 128×128 provides 10.5% speedup over single tile
2. **Memory allocation:** No impact when using single tile per grid section
3. **Threadgroup dimensions:** Time-prioritized (4×4×64) is optimal for deep seismic data
4. **SIMD batching:** Already implemented as default (float4 trace batching)
5. **Async prefetching:** Modest ~0.7% improvement (GPU time dominates I/O)
6. **Multi-bin parallel:** NOT effective on single GPU - causes 2x slowdown per bin due to contention
7. **UMA Zero-Copy:** Using memoryview.cast('B') instead of tobytes() eliminates one buffer copy
8. **Quality preserved:** All optimizations produce identical output

### Total Improvement: **15.2%** (679.7s → 576.7s)

---

## Benchmark Protocol

Each optimization will follow this testing protocol:

1. **Clean start:** Delete existing `migration_bin_10` output
2. **Run migration:** `python run_pstm_all_offsets.py --bins 10`
3. **Record metrics:**
   - Total runtime (seconds)
   - Peak memory usage (GB)
   - GPU utilization (if available)
4. **Generate QC images:** Run visualization script
5. **Compare outputs:** Verify quality vs baseline

### Baseline Metrics (to be captured)

```
Bin 10 Baseline:
- Traces: 819,045
- Runtime: ___ seconds
- Peak Memory: ___ GB
- Output Shape: (511, 427, 1001)
```

---

## Phase 1: Baseline Establishment

### Task 1.1: Capture Baseline Performance

**Objective:** Establish baseline timing and quality metrics for offset bin 10

**Steps:**
1. Delete existing migration_bin_10 output
2. Run: `python run_pstm_all_offsets.py --bins 10`
3. Record execution time from output
4. Capture memory usage during execution
5. Generate QC visualizations

**Deliverables:**
- [ ] Baseline runtime recorded
- [ ] `baseline_qc/` folder with inline, crossline, time slice images
- [ ] Memory profile captured

**Commands:**
```bash
# Clean previous output
rm -rf /Users/olegadamovich/SeismicData/PSTM_common_offset/migration_bin_10

# Run with timing
time python run_pstm_all_offsets.py --bins 10

# Generate QC images (update visualize_migrated.py for bin 10 first)
python visualize_migrated.py
```

---

### Task 1.2: Create Benchmark Visualization Script

**Objective:** Create reusable QC visualization for any bin

**Steps:**
1. Modify `visualize_migrated.py` to accept bin number as argument
2. Generate standardized output: inline 256, crossline 214, time slices at 500ms, 1000ms, 1500ms
3. Add fold map visualization
4. Save to `optimization_qc/baseline/` folder

**Deliverables:**
- [ ] `visualize_benchmark.py` script
- [ ] 3 inline slices (100, 256, 410)
- [ ] 3 crossline slices (85, 214, 340)
- [ ] 3 time slices (500ms, 1000ms, 1500ms)
- [ ] Fold map image
- [ ] Combined summary figure

---

## Phase 2: Configuration-Only Optimizations

These require no code changes, only configuration adjustments.

### Task 2.1: Increase Memory Allocation

**Objective:** Utilize available 48GB RAM more effectively

**Current:** `max_memory_gb=32.0` in `run_pstm_all_offsets.py:334`

**Change:** Increase to `max_memory_gb=44.0`

**Expected Impact:**
- Larger trace cache (currently 15% of max_memory = 4.8GB → 6.6GB)
- Potentially larger tiles
- Expected speedup: 5-10%

**Test Protocol:**
1. Modify ResourceConfig in run_pstm_all_offsets.py
2. Run benchmark
3. Compare runtime and memory usage

**Deliverables:**
- [ ] Runtime comparison (baseline vs 44GB)
- [ ] Memory usage profile
- [ ] QC images in `optimization_qc/memory_44gb/`

**Pros:**
- Simple change
- Better cache utilization
- No quality impact

**Cons:**
- Less memory for other applications
- Diminishing returns beyond ~80% system RAM

---

### Task 2.2: Optimize Tile Size

**Objective:** Test different tile configurations for optimal GPU utilization

**Current:** `TILE_NX=512, TILE_NY=512` (single tile covers entire grid)

**Test Configurations:**
| Config | TILE_NX | TILE_NY | Tiles | Expected Impact |
|--------|---------|---------|-------|-----------------|
| A (baseline) | 512 | 512 | 1 | Current baseline |
| B | 256 | 256 | 4 | More parallelism |
| C | 128 | 128 | 16 | Finer granularity |
| D | 64 | 64 | 64 | Maximum parallelism |

**Test Protocol:**
1. For each configuration:
   - Clean output
   - Run benchmark
   - Record timing
   - Generate QC images
2. Compare all configurations

**Deliverables:**
- [ ] Runtime table for all configurations
- [ ] QC images in `optimization_qc/tile_NxN/` for each
- [ ] Recommendation with justification

**Pros:**
- Smaller tiles may enable better checkpointing
- More even work distribution

**Cons:**
- Smaller tiles may increase overhead
- Need testing to find optimal

---

### Task 2.3: Test Trace-Centric vs Tile-Based

**Objective:** Determine optimal processing strategy for bin 10

**Background:** Current code uses tile-based by default. Trace-centric may be faster for high aperture overlap.

**Test Protocol:**
1. Force tile-based (current default)
2. Force trace-centric (requires code modification flag)
3. Compare timing and memory usage

**Deliverables:**
- [ ] Timing comparison
- [ ] Memory usage comparison
- [ ] QC comparison to verify identical output

---

## Phase 3: I/O Optimizations

### Task 3.1: Increase LRU Cache Size

**Objective:** Reduce redundant Zarr reads with larger trace cache

**Current:** Cache is 15% of max_memory (~4.8GB with 32GB max)

**Test Configurations:**
| Config | Cache Size | Expected Hit Rate |
|--------|-----------|-------------------|
| Baseline | 15% (4.8GB) | ~35% |
| Test A | 20% (8.8GB) | ~45% |
| Test B | 25% (11GB) | ~50% |

**Location:** `pstm/data/trace_cache.py` or executor initialization

**Test Protocol:**
1. Modify cache size percentage
2. Run benchmark with cache hit/miss logging
3. Record hit rate and timing

**Deliverables:**
- [ ] Cache hit rate for each configuration
- [ ] Runtime comparison
- [ ] Recommendation

**Pros:**
- Higher hit rate = fewer disk reads
- Simple configuration change

**Cons:**
- Uses more RAM
- Diminishing returns at high hit rates

---

### Task 3.2: Implement Async Trace Prefetching

**Objective:** Overlap I/O with GPU computation

**Current:** Sequential: Load traces → GPU migrate → Load next traces

**Proposed:** Parallel: GPU migrates tile N while loading traces for tile N+1

**Implementation Location:** `pstm/pipeline/executor.py`

**Test Protocol:**
1. Implement prefetching with `concurrent.futures.ThreadPoolExecutor`
2. Benchmark with prefetching enabled/disabled
3. Monitor I/O wait time vs GPU time

**Deliverables:**
- [ ] Modified executor with prefetching
- [ ] Runtime comparison
- [ ] I/O vs compute time breakdown
- [ ] QC images in `optimization_qc/prefetch/`

**Pros:**
- Hides I/O latency behind GPU work
- Significant speedup if I/O-bound

**Cons:**
- Increases peak memory (2 tile's data in memory)
- Adds complexity

---

## Phase 4: GPU Kernel Optimizations

### Task 4.1: Optimize Threadgroup Dimensions

**Objective:** Improve GPU occupancy with better thread configuration

**Current:** Default threadgroup size (likely 4x4x16 = 256)

**Test Configurations:**
| Config | Threadgroup | Threads | Expected Impact |
|--------|-------------|---------|-----------------|
| A (default) | 4x4x16 | 256 | Baseline |
| B | 8x8x8 | 512 | More spatial parallelism |
| C | 8x8x16 | 1024 | Maximum threads |
| D | 16x16x4 | 1024 | Maximum spatial |

**Location:** `pstm/kernels/metal_compiled.py`

**Test Protocol:**
1. Modify threadgroup dimensions in Metal dispatch
2. Run benchmark for each configuration
3. Monitor GPU utilization if possible

**Deliverables:**
- [ ] Runtime for each configuration
- [ ] GPU utilization metrics (if available)
- [ ] QC images to verify correctness
- [ ] Optimal configuration recommendation

**Pros:**
- Higher occupancy masks memory latency
- No quality impact

**Cons:**
- Larger threadgroups use more resources
- May reduce parallelism if too large

---

### Task 4.2: SIMD Trace Batching

**Objective:** Process multiple traces per SIMD operation

**Current:** Sequential trace loop in kernel

**Proposed:** Vectorize trace processing in batches of 4-8

**Location:** `pstm/metal/shaders/pstm_migration.metal`

**Complexity:** HIGH - requires shader rewrite

**Test Protocol:**
1. Implement SIMD batching in Metal shader
2. Verify correctness with synthetic test
3. Benchmark on real data

**Deliverables:**
- [ ] Modified Metal shader
- [ ] Correctness test results
- [ ] Runtime comparison
- [ ] QC images in `optimization_qc/simd/`

**Pros:**
- 20-30% speedup expected for large trace counts
- Better GPU utilization

**Cons:**
- Complex implementation
- Risk of bugs
- Memory alignment requirements

---

### Task 4.3: Shared Memory Velocity Caching

**Objective:** Load velocity to threadgroup memory for faster access

**Current:** Each thread reads velocity from global memory

**Proposed:** Load velocity once per threadgroup, share among threads

**Location:** `pstm/metal/shaders/pstm_migration.metal`

**Test Protocol:**
1. Add threadgroup memory for velocity array
2. Load at start of threadgroup
3. Use shared memory in inner loop

**Deliverables:**
- [ ] Modified Metal shader
- [ ] Runtime comparison
- [ ] Memory bandwidth analysis

**Pros:**
- Threadgroup memory 10x faster
- Simple modification

**Cons:**
- Limited threadgroup memory (32KB)
- Requires barrier synchronization

---

## Phase 5: Algorithm Optimizations

### Task 5.1: Implement Depth-Adaptive Aperture

**Objective:** Use smaller aperture at shallow depths

**Current:** Fixed aperture (2000m) for all depths

**Proposed:**
- 0-500ms: 1000m aperture
- 500-1000ms: 1500m aperture
- 1000-2000ms: 2000m aperture

**Expected Impact:**
- 20-30% fewer traces processed at shallow depths
- Quality maintained (shallow reflections need less aperture)

**Location:** `pstm/config/models.py` (new config), `pstm/pipeline/executor.py`

**Test Protocol:**
1. Implement depth-variant aperture
2. Benchmark runtime improvement
3. QC to verify no quality loss at shallow depths

**Deliverables:**
- [ ] Depth-adaptive aperture implementation
- [ ] Runtime comparison
- [ ] Shallow depth QC comparison (time slices at 200ms, 400ms)
- [ ] Images in `optimization_qc/adaptive_aperture/`

**Pros:**
- Physically motivated
- Significant speedup at shallow depths

**Cons:**
- May miss steep shallow events (unlikely in most cases)
- Additional complexity

---

### Task 5.2: Implement Mute Curve

**Objective:** Skip samples above first arrival

**Current:** All 1001 samples processed

**Proposed:** Apply offset-dependent mute to zero pre-arrival samples

**Expected Impact:**
- 10-20% fewer samples processed (offset-dependent)
- Improved quality (no noise above first arrival)

**Test Protocol:**
1. Design mute curve based on velocity model
2. Implement in trace loading
3. Benchmark and QC

**Deliverables:**
- [ ] Mute curve design
- [ ] Implementation
- [ ] Runtime comparison
- [ ] QC comparison showing noise reduction

---

## Phase 6: Architecture Optimizations

### Task 6.1: Multi-Bin Parallel Processing

**Objective:** Process multiple offset bins concurrently

**Current:** Sequential bin processing

**Proposed:** Use multiprocessing to run 2-3 bins in parallel

**Considerations:**
- 48GB RAM / 3 bins = ~16GB per bin (sufficient)
- I/O contention may be limiting factor

**Test Protocol:**
1. Implement multiprocessing wrapper
2. Test with bins 10, 11, 12 in parallel
3. Compare total time vs 3x sequential

**Deliverables:**
- [ ] Multiprocessing implementation
- [ ] Total runtime comparison
- [ ] Resource utilization analysis
- [ ] Recommendation for optimal parallelism

**Pros:**
- Near-linear speedup possible
- Bins are independent

**Cons:**
- Memory usage multiplied
- I/O contention risk

---

### Task 6.2: Metal Command Buffer Pipelining

**Objective:** Overlap GPU command submission and execution

**Current:** `waitUntilCompleted()` blocks after each tile

**Proposed:** Use completion handlers to pipeline tiles

**Location:** `pstm/kernels/metal_compiled.py`

**Test Protocol:**
1. Implement non-blocking command submission
2. Use completion handlers for tile accumulation
3. Benchmark with 2-3 tiles in flight

**Deliverables:**
- [ ] Pipelined Metal implementation
- [ ] Runtime comparison
- [ ] GPU utilization improvement

**Pros:**
- Hides command submission latency
- Better GPU utilization

**Cons:**
- Complex synchronization
- Risk of race conditions

---

## Phase 7: Quality Improvements

### Task 7.1: Full 3D Velocity Interpolation

**Objective:** Use full 3D velocity cube instead of center pillar only

**Current:** Uses velocity from tile center only (ignores lateral variations)

**Proposed:** Pass full velocity cube to GPU, interpolate per output point

**Impact:**
- 10-15% slowdown expected
- Significant quality improvement in complex geology

**Location:** `pstm/kernels/metal_compiled.py`, `pstm/metal/shaders/pstm_migration.metal`

**Test Protocol:**
1. Implement 3D velocity passing to GPU
2. Add trilinear interpolation in shader
3. QC to verify improved imaging
4. Benchmark slowdown

**Deliverables:**
- [ ] 3D velocity implementation
- [ ] QC comparison showing improvement
- [ ] Performance impact analysis
- [ ] Images in `optimization_qc/velocity_3d/`

---

## Summary: Priority Matrix

| Priority | Task | Expected Gain | Effort | Risk |
|----------|------|---------------|--------|------|
| **1** | Task 1.1: Baseline | Required | Low | None |
| **2** | Task 2.1: Memory 44GB | 5-10% | Low | None |
| **3** | Task 2.2: Tile Size | 5-15% | Low | None |
| **4** | Task 3.1: Cache Size | 5-15% | Low | None |
| **5** | Task 4.1: Threadgroup | 5-15% | Medium | Low |
| **6** | Task 3.2: Prefetching | 15-25% | Medium | Low |
| **7** | Task 5.1: Adaptive Aperture | 15-25% | Medium | Medium |
| **8** | Task 4.2: SIMD Batching | 20-30% | High | Medium |
| **9** | Task 6.1: Multi-Bin | 50-100%* | Medium | Low |
| **10** | Task 7.1: 3D Velocity | -10% speed | High | Low |

*Multi-bin speedup applies to processing multiple bins, not single bin

---

## Execution Order

### Week 1: Foundation
1. Task 1.1: Baseline establishment
2. Task 1.2: Visualization script
3. Task 2.1: Memory optimization
4. Task 2.2: Tile size testing

### Week 2: I/O & Cache
5. Task 3.1: Cache size optimization
6. Task 3.2: Prefetching implementation
7. Task 2.3: Trace-centric testing

### Week 3: GPU
8. Task 4.1: Threadgroup optimization
9. Task 4.3: Shared memory velocity
10. Task 4.2: SIMD batching (if time permits)

### Week 4: Algorithm & Architecture
11. Task 5.1: Adaptive aperture
12. Task 6.1: Multi-bin processing
13. Task 7.1: 3D velocity (quality improvement)

---

## QC Output Structure

```
optimization_qc/
├── baseline/
│   ├── inline_256.png
│   ├── crossline_214.png
│   ├── timeslice_500ms.png
│   ├── timeslice_1000ms.png
│   ├── timeslice_1500ms.png
│   ├── fold_map.png
│   └── summary.png
├── memory_44gb/
│   └── (same structure)
├── tile_256x256/
│   └── (same structure)
├── cache_20pct/
│   └── (same structure)
├── prefetch/
│   └── (same structure)
├── threadgroup_8x8x16/
│   └── (same structure)
└── results_summary.md
```

---

## Metrics Tracking Template

For each optimization test, record:

```markdown
## Test: [Optimization Name]
Date: YYYY-MM-DD
Baseline Runtime: ___ seconds

### Configuration
- Parameter changed: ___
- Old value: ___
- New value: ___

### Results
- Runtime: ___ seconds
- Speedup: ___x (___%)
- Peak Memory: ___ GB
- Cache Hit Rate: ___% (if applicable)

### Quality Assessment
- [ ] Inline slices match baseline
- [ ] Crossline slices match baseline
- [ ] Time slices match baseline
- [ ] Fold map unchanged

### Notes
(Any observations, issues, or recommendations)
```

---

## Appendix: Current Script Configuration

From `run_pstm_all_offsets.py`:

```python
# Grid parameters
DX = 25.0   # Inline bin size (m)
DY = 12.5   # Crossline bin size (m)
DT_MS = 2.0 # Time sample interval (ms)
T_MIN_MS = 0.0
T_MAX_MS = 2000.0

# Algorithm parameters
MAX_APERTURE_M = 2000.0
MIN_APERTURE_M = 500.0
MAX_DIP_DEGREES = 65.0

# Tile size
TILE_NX = 512
TILE_NY = 512

# Resource allocation
max_memory_gb = 32.0
backend = ComputeBackend.METAL_COMPILED
```

---

## Appendix: Offset Bin 10 Characteristics

```
Path: /Users/olegadamovich/SeismicData/common_offset_gathers_new/offset_bin_10/
Traces: 819,045
Offset Range: 500-549 m
Data Size: 4.5 GB
Output Shape: (511, 427, 1001)
Output Chunks: (64, 64, 1001)
```

This represents a medium-sized offset bin suitable for benchmarking. Results should scale to other bins based on trace count.
