# Time-Variant Sampling Implementation Tasks

## Overview

Implementation of time-dependent sample rate optimization for PSTM migration.
User specifies (time, frequency) table in UI, kernel uses coarser sampling at depth.

---

## Phase 1: Core Algorithm

### Task 1.1: Create TimeVariantSampling Module
**File:** `pstm/algorithm/time_variant.py`

- [ ] Create `FrequencyTimeTable` dataclass
  - [ ] Fields: `times: list[float]`, `frequencies: list[float]`
  - [ ] Validation: times monotonically increasing, frequencies > 0
  - [ ] Method: `to_list() -> list[tuple[float, float]]`
  - [ ] Method: `from_list(pairs: list[tuple[float, float]])`

- [ ] Implement `interpolate_fmax(t_ms: float, table: FrequencyTimeTable) -> float`
  - [ ] Use scipy.interpolate.PchipInterpolator for smooth monotonic interpolation
  - [ ] Clamp to table bounds at edges
  - [ ] Return frequency in Hz

- [ ] Create `TimeWindow` dataclass
  - [ ] Fields: `t_start_ms`, `t_end_ms`, `dt_effective_ms`, `downsample_factor`
  - [ ] Fields: `sample_start`, `sample_end`, `n_samples`
  - [ ] Property: `duration_ms`

- [ ] Implement `compute_time_windows()` function
  - [ ] Input: `t_min`, `t_max`, `base_dt`, `freq_table`
  - [ ] Output: `list[TimeWindow]`
  - [ ] Logic: divide time axis where downsample factor changes
  - [ ] Constraint: downsample_factor must be power of 2 (1, 2, 4, 8)

- [ ] Implement `estimate_speedup()` function
  - [ ] Compare uniform samples vs time-variant samples
  - [ ] Return speedup factor

### Task 1.2: Create Configuration Model
**File:** `pstm/config/models.py`

- [ ] Add `TimeVariantConfig` class
  ```python
  class TimeVariantConfig(BaseConfig):
      enabled: bool = False
      frequency_table: list[tuple[float, float]] = Field(default_factory=list)
      min_downsample_factor: int = 1
      max_downsample_factor: int = 8
  ```

- [ ] Add `time_variant: TimeVariantConfig` to `AlgorithmConfig`

- [ ] Add validation for frequency_table
  - [ ] At least 2 points
  - [ ] Times must be increasing
  - [ ] Frequencies must be positive

### Task 1.3: Create Output Resampling Utility
**File:** `pstm/algorithm/time_variant.py`

- [ ] Implement `resample_to_uniform()` function
  - [ ] Input: time-variant image, windows, target_dt
  - [ ] Output: uniformly sampled image
  - [ ] Use sinc interpolation for quality
  - [ ] Handle window boundaries smoothly

- [ ] Implement `create_output_sample_map()` function
  - [ ] Maps compute sample indices to output sample indices
  - [ ] Pre-computed for kernel efficiency

### Task 1.4: Unit Tests
**File:** `tests/test_time_variant.py`

- [ ] Test `interpolate_fmax()` with known values
- [ ] Test `compute_time_windows()` output structure
- [ ] Test downsample factors are powers of 2
- [ ] Test `resample_to_uniform()` preserves low frequencies
- [ ] Test edge cases (single window, max downsampling)

---

## Phase 2: Metal Kernel Integration

### Task 2.1: Update Metal Shader Structs
**File:** `pstm/metal/shaders/pstm_migration.metal`

- [ ] Add `TimeWindow` struct
  ```metal
  struct TimeWindow {
      float t_start_ms;
      float t_end_ms;
      float dt_effective_ms;
      int downsample_factor;
      int sample_offset;
      int n_samples;
  };
  ```

- [ ] Add constants for max windows
  ```metal
  constant int MAX_TIME_WINDOWS = 8;
  ```

### Task 2.2: Create Time-Variant Migration Kernel
**File:** `pstm/metal/shaders/pstm_migration.metal`

- [ ] Add new kernel function `pstm_migrate_time_variant`
  - [ ] Additional buffer: `constant TimeWindow* windows`
  - [ ] Additional buffer: `constant int& n_windows`

- [ ] Implement window-based loop structure
  - [ ] Outer loop over windows
  - [ ] Inner loop over samples within window at effective dt
  - [ ] Compute correct output index from window offset

- [ ] Implement downsampled trace interpolation
  - [ ] Standard linear interp for factor=1
  - [ ] Wider kernel for factor>1 (anti-aliasing)

- [ ] Add SIMD variant `pstm_migrate_time_variant_simd`

### Task 2.3: Update Python Metal Wrapper
**File:** `pstm/kernels/metal_compiled.py`

- [ ] Add `TimeWindowParams` ctypes struct matching Metal
  ```python
  class TimeWindowParams(ctypes.Structure):
      _fields_ = [
          ("t_start_ms", ctypes.c_float),
          ("t_end_ms", ctypes.c_float),
          ("dt_effective_ms", ctypes.c_float),
          ("downsample_factor", ctypes.c_int),
          ("sample_offset", ctypes.c_int),
          ("n_samples", ctypes.c_int),
      ]
  ```

- [ ] Update `CompiledMetalKernel.__init__()`
  - [ ] Add `time_variant_config` parameter
  - [ ] Store kernel function name variant

- [ ] Update `migrate_tile()` method
  - [ ] Accept `time_variant_windows: list[TimeWindow] | None`
  - [ ] Create windows buffer if time-variant enabled
  - [ ] Select correct kernel function
  - [ ] Handle output array sizing

- [ ] Add `_create_windows_buffer()` method
  - [ ] Convert Python TimeWindow list to Metal buffer
  - [ ] Pack into contiguous memory

- [ ] Update `_create_buffers()` to handle variable output size

### Task 2.4: Rebuild Metal Library
**File:** `scripts/build_metal.sh`

- [ ] Verify new kernel compiles without errors
- [ ] Check for symbol conflicts
- [ ] Update version/build info

### Task 2.5: Kernel Integration Tests
**File:** `tests/test_metal_time_variant.py`

- [ ] Test kernel loads with time-variant function
- [ ] Test single window (equivalent to uniform)
- [ ] Test multiple windows produce correct output shape
- [ ] Compare output quality vs uniform sampling
- [ ] Benchmark speedup measurement

---

## Phase 3: Executor Integration

### Task 3.1: Update KernelConfig
**File:** `pstm/kernels/base.py`

- [ ] Add time-variant fields to `KernelConfig`
  ```python
  time_variant_enabled: bool = False
  time_variant_windows: list[TimeWindow] | None = None
  ```

### Task 3.2: Update Migration Executor
**File:** `pstm/pipeline/executor.py`

- [ ] Import time-variant module

- [ ] Update `_initialize()` method
  - [ ] Compute time windows from config if enabled
  - [ ] Log window breakdown
  - [ ] Adjust output array allocation

- [ ] Update `_process_tile()` method
  - [ ] Pass windows to kernel
  - [ ] Handle time-variant output accumulation

- [ ] Update `_finalize()` method
  - [ ] Resample to uniform dt if time-variant was used
  - [ ] Log resampling info

- [ ] Add time-variant metrics to `ExecutionMetrics`
  - [ ] `time_variant_speedup: float`
  - [ ] `n_windows: int`

### Task 3.3: Update Kernel Factory
**File:** `pstm/kernels/factory.py`

- [ ] Pass time-variant config to kernel creation
- [ ] Verify kernel supports time-variant mode

---

## Phase 4: UI Integration

### Task 4.1: Create TimeVariantState
**File:** `pstm/gui/state.py`

- [ ] Add `TimeVariantState` dataclass
  ```python
  @dataclass
  class TimeVariantState:
      enabled: bool = False
      frequency_table: list[tuple[float, float]] = field(default_factory=lambda: [
          (0, 80),
          (1000, 50),
          (2500, 30),
          (5000, 20),
      ])
  ```

- [ ] Add `time_variant: TimeVariantState` to `AlgorithmState`

- [ ] Update `build_migration_config()` to include time-variant config

### Task 4.2: Create FrequencyTimeTable Widget
**File:** `pstm/gui/widgets/frequency_time_table.py`

- [ ] Create `FrequencyTimeTableWidget(QTableWidget)`
  - [ ] Columns: "Time (ms)", "Max Frequency (Hz)", "Effective dt"
  - [ ] Editable first two columns
  - [ ] Read-only third column (computed)

- [ ] Implement `set_data(table: list[tuple[float, float]])`

- [ ] Implement `get_data() -> list[tuple[float, float]]`

- [ ] Implement `_update_effective_dt()`
  - [ ] Compute and display dt for each row
  - [ ] Show downsample factor (1x, 2x, 4x, 8x)

- [ ] Add row management
  - [ ] "Add Row" button - inserts row with interpolated values
  - [ ] "Remove Row" button - removes selected row
  - [ ] Minimum 2 rows enforced

- [ ] Add validation
  - [ ] Times must be increasing
  - [ ] Frequencies must be positive
  - [ ] Visual feedback for invalid entries

- [ ] Emit `tableChanged` signal on edits

### Task 4.3: Create FrequencyTimeGraph Widget
**File:** `pstm/gui/widgets/frequency_time_graph.py`

- [ ] Create `FrequencyTimeGraphWidget(QWidget)`
  - [ ] Embed matplotlib figure
  - [ ] Fixed size ~400x200 pixels

- [ ] Implement `update_plot(table, t_max)`
  - [ ] Plot points from table
  - [ ] Plot interpolated curve
  - [ ] X-axis: Time (ms)
  - [ ] Y-axis: Frequency (Hz)
  - [ ] Grid lines

- [ ] Add visual indicators
  - [ ] Shaded regions for each window
  - [ ] Different colors per downsample factor

### Task 4.4: Create TimeVariantStep Wizard Page
**File:** `pstm/gui/steps/time_variant_step.py`

- [ ] Create `TimeVariantStep(BaseStep)`

- [ ] Layout structure:
  ```
  - Enable checkbox at top
  - Frequency-time table (left)
  - Preview graph (right)
  - Info panel at bottom (speedup estimate, warnings)
  ```

- [ ] Implement `_init_ui()`
  - [ ] Create enable checkbox
  - [ ] Create table widget
  - [ ] Create graph widget
  - [ ] Create info label
  - [ ] Add/Remove row buttons

- [ ] Implement `_connect_signals()`
  - [ ] Enable checkbox -> toggle table/graph enabled
  - [ ] Table changed -> update graph, update info
  - [ ] State changes -> update widgets

- [ ] Implement `load_from_state()`
  - [ ] Populate enable checkbox
  - [ ] Populate table from state

- [ ] Implement `save_to_state()`
  - [ ] Save enable state
  - [ ] Save table data

- [ ] Implement `validate() -> tuple[bool, str]`
  - [ ] Check table has valid data
  - [ ] Check times span output range
  - [ ] Return warnings if needed

- [ ] Implement `_update_info_panel()`
  - [ ] Compute estimated speedup
  - [ ] Show window breakdown
  - [ ] Show memory estimate

### Task 4.5: Integrate into Main Window
**File:** `pstm/gui/main_window.py`

- [ ] Import `TimeVariantStep`

- [ ] Add step to wizard after AlgorithmStep
  ```python
  self.steps = [
      InputStep,
      SurveyStep,
      OutputGridStep,
      VelocityStep,
      DataSelectionStep,
      AlgorithmStep,
      TimeVariantStep,  # NEW
      ExecutionStep,
      VisualizationStep,
  ]
  ```

- [ ] Update step navigation

### Task 4.6: Update Execution Step Display
**File:** `pstm/gui/steps/execution_step.py`

- [ ] Show time-variant info in config summary
  - [ ] "Time-Variant Sampling: Enabled/Disabled"
  - [ ] Number of windows
  - [ ] Estimated speedup

- [ ] Update progress display for window info

---

## Phase 5: Testing & Validation

### Task 5.1: Synthetic Data Tests
**File:** `tests/test_time_variant_integration.py`

- [ ] Create test with known frequency content at different times
- [ ] Compare uniform vs time-variant migration output
- [ ] Verify frequencies below local f_max are preserved
- [ ] Verify no artifacts at window boundaries

### Task 5.2: Performance Benchmarks
**File:** `benchmarks/benchmark_time_variant.py`

- [ ] Benchmark with varying number of windows
- [ ] Benchmark with different downsample factors
- [ ] Measure actual vs estimated speedup
- [ ] Memory usage comparison

- [ ] Create results table:
  ```
  Config              Time (s)    Speedup    RMS Diff
  Uniform (2ms)       10.5        1.0x       -
  2 windows           5.2         2.0x       0.1%
  4 windows           3.1         3.4x       0.3%
  8 windows           2.4         4.4x       0.5%
  ```

### Task 5.3: Visual QC Script
**File:** `scripts/qc_time_variant.py`

- [ ] Run migration with uniform and time-variant
- [ ] Generate comparison images (inline, crossline, timeslice)
- [ ] Generate difference images
- [ ] Compute and display statistics

### Task 5.4: UI Testing
- [ ] Test table editing (add/remove/modify rows)
- [ ] Test graph updates in real-time
- [ ] Test validation messages
- [ ] Test state save/load
- [ ] Test with various output time ranges

---

## File Summary

### New Files
```
pstm/algorithm/time_variant.py
pstm/gui/widgets/frequency_time_table.py
pstm/gui/widgets/frequency_time_graph.py
pstm/gui/steps/time_variant_step.py
tests/test_time_variant.py
tests/test_time_variant_integration.py
tests/test_metal_time_variant.py
benchmarks/benchmark_time_variant.py
scripts/qc_time_variant.py
```

### Modified Files
```
pstm/config/models.py
pstm/kernels/base.py
pstm/kernels/metal_compiled.py
pstm/kernels/factory.py
pstm/pipeline/executor.py
pstm/gui/state.py
pstm/gui/main_window.py
pstm/gui/steps/execution_step.py
pstm/metal/shaders/pstm_migration.metal
scripts/build_metal.sh
```

---

## Implementation Order

```
Phase 1: Core Algorithm
├── Task 1.1: TimeVariantSampling module
├── Task 1.2: Configuration model
├── Task 1.3: Output resampling
└── Task 1.4: Unit tests

Phase 2: Metal Kernel
├── Task 2.1: Metal structs
├── Task 2.2: Time-variant kernel
├── Task 2.3: Python wrapper
├── Task 2.4: Rebuild library
└── Task 2.5: Integration tests

Phase 3: Executor
├── Task 3.1: KernelConfig update
├── Task 3.2: Executor update
└── Task 3.3: Factory update

Phase 4: UI
├── Task 4.1: State model
├── Task 4.2: Table widget
├── Task 4.3: Graph widget
├── Task 4.4: Wizard step
├── Task 4.5: Main window integration
└── Task 4.6: Execution step update

Phase 5: Validation
├── Task 5.1: Synthetic tests
├── Task 5.2: Benchmarks
├── Task 5.3: QC script
└── Task 5.4: UI testing
```

---

## Success Criteria

- [ ] Time-variant migration produces equivalent results to uniform (< 1% RMS diff)
- [ ] Measured speedup within 20% of estimated speedup
- [ ] No artifacts at window boundaries
- [ ] UI table is intuitive and validates input
- [ ] Graph updates in real-time with table changes
- [ ] Full round-trip: UI -> config -> kernel -> output -> visualization
