# Time-Variant Sampling Rate Design

## Overview

Time-dependent sample rate optimization for Kirchhoff PSTM migration. This feature allows the migration kernel to use coarser sampling at deeper times where high frequencies are naturally attenuated, significantly reducing computation while maintaining image quality.

## Motivation

In seismic data:
- **Shallow times (0-500ms)**: High frequencies present (up to 80-100 Hz)
- **Medium times (500-2000ms)**: Dominant frequencies decrease (30-60 Hz)
- **Deep times (2000ms+)**: Low frequencies dominate (10-30 Hz)

The Nyquist criterion states: `dt_max = 1 / (2 * f_max)`

| Time (ms) | Typical f_max (Hz) | Required dt (ms) | Speedup vs 2ms |
|-----------|-------------------|------------------|----------------|
| 0-500     | 80                | 6.25             | 3.1x           |
| 500-1500  | 50                | 10.0             | 5.0x           |
| 1500-3000 | 30                | 16.7             | 8.3x           |
| 3000+     | 20                | 25.0             | 12.5x          |

**Potential speedup: 3-10x** for typical datasets.

---

## Algorithm Design

### 1. User Input: Frequency-Time Table

User specifies pairs of `(t0_ms, f_max_hz)`:

```
Time (ms)    Max Frequency (Hz)
---------    ------------------
0            80
500          60
1500         40
3000         25
5000         15
```

### 2. Interpolation

Smooth interpolation of f_max along time axis:

```python
def interpolate_fmax(t_ms: float, table: list[tuple[float, float]]) -> float:
    """
    Interpolate maximum frequency at given time.

    Uses monotonic cubic spline to ensure:
    - Smooth transitions (no abrupt frequency jumps)
    - Monotonically decreasing (physically realistic)
    """
    times = [t for t, f in table]
    freqs = [f for t, f in table]

    # Clamp to table bounds
    if t_ms <= times[0]:
        return freqs[0]
    if t_ms >= times[-1]:
        return freqs[-1]

    # Monotonic cubic interpolation
    return pchip_interpolate(times, freqs, t_ms)
```

### 3. Compute Effective Sample Rate per Time Window

Divide output time axis into windows, each with optimal dt:

```python
def compute_time_windows(
    t_min_ms: float,
    t_max_ms: float,
    base_dt_ms: float,
    freq_table: list[tuple[float, float]],
) -> list[TimeWindow]:
    """
    Compute time windows with optimal sampling.

    Returns list of (t_start, t_end, dt_effective, downsample_factor)
    """
    windows = []
    t = t_min_ms

    while t < t_max_ms:
        # Get f_max at this time
        f_max = interpolate_fmax(t, freq_table)

        # Nyquist: dt_max = 1000 / (2 * f_max)  [ms]
        dt_nyquist = 1000.0 / (2.0 * f_max)

        # Downsample factor (must be integer, power of 2 preferred)
        factor = max(1, int(dt_nyquist / base_dt_ms))
        factor = nearest_power_of_2(factor)  # 1, 2, 4, 8, ...

        dt_effective = base_dt_ms * factor

        # Find window extent (where factor remains valid)
        t_end = find_window_end(t, factor, freq_table, base_dt_ms)

        windows.append(TimeWindow(
            t_start=t,
            t_end=t_end,
            dt_effective=dt_effective,
            downsample_factor=factor,
            f_max=f_max,
        ))

        t = t_end

    return windows
```

### 4. Migration Kernel Modification

#### 4.1 Pre-compute Window Parameters

```cpp
struct TimeWindow {
    float t_start_ms;
    float t_end_ms;
    float dt_effective_ms;
    int downsample_factor;
    int sample_start;    // Index in output array
    int sample_end;      // Index in output array
    int n_samples;       // Samples in this window
};

struct TimeVariantParams {
    int n_windows;
    TimeWindow windows[MAX_WINDOWS];  // Typically 4-8 windows

    // Pre-computed output sample mapping
    int total_compute_samples;  // Sum of samples across windows
    int* output_sample_map;     // Maps compute sample -> output sample
};
```

#### 4.2 Modified Kirchhoff Summation

```cpp
// For each output (x, y) position:
for (int win = 0; win < params.n_windows; win++) {
    TimeWindow& w = params.windows[win];

    // Loop over samples in this window at effective dt
    for (int it_win = 0; it_win < w.n_samples; it_win++) {
        // Actual output time
        float t_out_ms = w.t_start_ms + it_win * w.dt_effective_ms;
        int it_out = (int)(t_out_ms / base_dt_ms);  // Output sample index

        // Kirchhoff summation at this time
        float sum = 0.0f;
        for (int tr = 0; tr < n_traces; tr++) {
            // Compute traveltime
            float t_travel = compute_traveltime(x, y, t_out_ms, trace[tr], velocity);

            // Interpolate trace amplitude
            float amp = interpolate_trace(trace[tr], t_travel, w.downsample_factor);

            // Apply corrections (spreading, obliquity, AA)
            amp *= compute_corrections(x, y, t_out_ms, trace[tr]);

            sum += amp;
        }

        // Store in output
        output[ix][iy][it_out] = sum;
    }
}
```

#### 4.3 Trace Interpolation with Downsampling

```cpp
inline float interpolate_trace_downsampled(
    const float* trace,
    float t_ms,
    float dt_trace_ms,
    int downsample_factor
) {
    // For downsampled windows, use wider interpolation kernel
    // to properly anti-alias the trace before sampling

    if (downsample_factor == 1) {
        // Standard linear interpolation
        return linear_interp(trace, t_ms, dt_trace_ms);
    }

    // Sinc interpolation with anti-aliasing filter
    // Filter cutoff = f_nyquist / downsample_factor
    float cutoff = 0.5f / downsample_factor;
    return sinc_interp_aa(trace, t_ms, dt_trace_ms, cutoff);
}
```

### 5. Output Resampling

Final output is resampled to uniform dt specified in output configuration:

```python
def resample_to_uniform(
    time_variant_image: np.ndarray,
    windows: list[TimeWindow],
    output_dt_ms: float,
    output_nt: int,
) -> np.ndarray:
    """
    Resample time-variant result to uniform sampling.

    Uses sinc interpolation to preserve frequencies up to
    the local Nyquist at each time.
    """
    output = np.zeros((nx, ny, output_nt))

    for ix in range(nx):
        for iy in range(ny):
            for it in range(output_nt):
                t_ms = it * output_dt_ms

                # Find which window this time falls in
                win = find_window(t_ms, windows)

                # Interpolate from window's sampling
                output[ix, iy, it] = interpolate_from_window(
                    time_variant_image[ix, iy],
                    t_ms,
                    win,
                )

    return output
```

---

## Metal Shader Implementation

### Kernel Structure

```metal
struct TimeWindow {
    float t_start_ms;
    float t_end_ms;
    float dt_effective_ms;
    int downsample_factor;
    int sample_offset;  // Offset in output array for this window
    int n_samples;
};

kernel void pstm_migrate_time_variant(
    // Input buffers
    device const float* amplitudes [[buffer(0)]],
    device const float* source_x [[buffer(1)]],
    // ... other trace geometry ...

    // Output buffer (time-variant sampling)
    device atomic_float* image [[buffer(7)]],

    // Time windows
    constant TimeWindow* windows [[buffer(16)]],
    constant int& n_windows [[buffer(17)]],

    // Standard params
    constant MigrationParams& params [[buffer(15)]],

    uint3 gid [[thread_position_in_grid]]
) {
    int ix = gid.x;
    int iy = gid.y;
    int win_idx = gid.z / MAX_SAMPLES_PER_WINDOW;
    int it_local = gid.z % MAX_SAMPLES_PER_WINDOW;

    if (win_idx >= n_windows) return;

    TimeWindow win = windows[win_idx];
    if (it_local >= win.n_samples) return;

    float t_out_ms = win.t_start_ms + it_local * win.dt_effective_ms;

    // Kirchhoff summation with time-variant parameters
    float sum = 0.0f;

    for (int tr = 0; tr < params.n_traces; tr++) {
        // ... migration loop with downsampled trace interpolation ...
    }

    // Atomic add to output
    int out_idx = ix * params.ny * params.nt + iy * params.nt + win.sample_offset + it_local;
    atomic_fetch_add_explicit(&image[out_idx], sum, memory_order_relaxed);
}
```

---

## UI Wizard Design

### New Tab: "Time-Variant Processing"

Located after "Algorithm" tab, before "Execution" tab.

#### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Time-Variant Sampling Configuration                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [✓] Enable Time-Variant Sampling                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Frequency-Time Table                                    │   │
│  │  ─────────────────────────────────────────────────────  │   │
│  │  Time (ms)    Max Frequency (Hz)    Effective dt (ms)   │   │
│  │  ──────────   ──────────────────    ─────────────────   │   │
│  │  [    0  ]    [      80      ]      2.0 (1x)            │   │
│  │  [  500  ]    [      60      ]      2.0 (1x)            │   │
│  │  [ 1500  ]    [      40      ]      4.0 (2x)            │   │
│  │  [ 3000  ]    [      25      ]      8.0 (4x)            │   │
│  │  [ 5000  ]    [      15      ]      16.0 (8x)           │   │
│  │                                                          │   │
│  │  [+ Add Row]  [- Remove Row]  [Auto-Detect from Data]   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Preview Graph                         │   │
│  │     80 ┤ ●                                               │   │
│  │        │  ╲                                              │   │
│  │     60 ┤   ●                                             │   │
│  │  f_max │    ╲                                            │   │
│  │  (Hz)  │     ╲                                           │   │
│  │     40 ┤      ●                                          │   │
│  │        │       ╲                                         │   │
│  │     20 ┤        ●─────●                                  │   │
│  │        └────────────────────────────────────────────     │   │
│  │          0    1000   2000   3000   4000   5000           │   │
│  │                      Time (ms)                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Estimated Performance Gain: ~4.2x speedup                      │
│  Memory Reduction: ~65%                                         │
│                                                                 │
│  ⚠ Output will be resampled to uniform dt=2.0ms as specified   │
│    in Output Grid configuration.                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### UI Components

#### 1. Enable Checkbox
```python
self.enable_tv_sampling = QCheckBox("Enable Time-Variant Sampling")
self.enable_tv_sampling.setToolTip(
    "Use coarser time sampling at depth where high frequencies "
    "are naturally attenuated. Can provide 3-10x speedup."
)
```

#### 2. Frequency-Time Table
```python
class FrequencyTimeTable(QTableWidget):
    """Editable table for time-frequency pairs."""

    columns = ["Time (ms)", "Max Frequency (Hz)", "Effective dt"]

    def __init__(self):
        super().__init__()
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(self.columns)

        # Default values
        self.set_default_table()

    def set_default_table(self):
        defaults = [
            (0, 80),
            (500, 60),
            (1500, 40),
            (3000, 25),
            (5000, 15),
        ]
        self.setRowCount(len(defaults))
        for i, (t, f) in enumerate(defaults):
            self.set_row(i, t, f)

    def get_table(self) -> list[tuple[float, float]]:
        """Return list of (time_ms, freq_hz) pairs."""
        result = []
        for i in range(self.rowCount()):
            t = float(self.item(i, 0).text())
            f = float(self.item(i, 1).text())
            result.append((t, f))
        return sorted(result, key=lambda x: x[0])
```

#### 3. Preview Graph
```python
class FrequencyTimeGraph(QWidget):
    """Interactive graph showing frequency vs time curve."""

    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

    def update_plot(self, table: list[tuple[float, float]], output_t_max: float):
        self.ax.clear()

        times = [t for t, f in table]
        freqs = [f for t, f in table]

        # Plot points
        self.ax.plot(times, freqs, 'o-', color='blue', markersize=8)

        # Interpolated curve
        t_interp = np.linspace(0, output_t_max, 200)
        f_interp = [interpolate_fmax(t, table) for t in t_interp]
        self.ax.plot(t_interp, f_interp, '--', color='lightblue', alpha=0.7)

        # Annotations
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Max Frequency (Hz)")
        self.ax.set_title("Time-Variant Frequency Limit")
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()
```

#### 4. Auto-Detect Button
```python
def auto_detect_frequencies(self):
    """
    Analyze input data spectrum to suggest frequency table.

    Computes amplitude spectrum in sliding windows and
    estimates f_max at each time.
    """
    # Load sample traces
    traces = self.state.load_sample_traces(n=1000)

    # Compute spectrogram
    window_ms = 500
    hop_ms = 250

    freq_estimates = []
    for t_center in range(0, max_time, hop_ms):
        t_start = max(0, t_center - window_ms // 2)
        t_end = min(max_time, t_center + window_ms // 2)

        # Extract window
        window_data = traces[:, t_start:t_end]

        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(window_data, axis=1)).mean(axis=0)
        freqs = np.fft.rfftfreq(window_data.shape[1], d=dt_s)

        # Find frequency where 95% of energy is below
        cumsum = np.cumsum(spectrum**2)
        f95 = freqs[np.searchsorted(cumsum, 0.95 * cumsum[-1])]

        freq_estimates.append((t_center, f95))

    # Simplify to key points
    simplified = simplify_frequency_curve(freq_estimates)

    self.table.set_from_list(simplified)
```

---

## State Model

### New State Class

```python
@dataclass
class TimeVariantSamplingState:
    """State for time-variant sampling configuration."""

    enabled: bool = False
    frequency_table: list[tuple[float, float]] = field(default_factory=lambda: [
        (0, 80),
        (500, 60),
        (1500, 40),
        (3000, 25),
        (5000, 15),
    ])

    def get_windows(self, t_min: float, t_max: float, base_dt: float) -> list[TimeWindow]:
        """Compute time windows for migration."""
        if not self.enabled:
            return [TimeWindow(t_min, t_max, base_dt, 1)]
        return compute_time_windows(t_min, t_max, base_dt, self.frequency_table)

    def estimate_speedup(self, t_max: float, base_dt: float) -> float:
        """Estimate performance gain from time-variant sampling."""
        windows = self.get_windows(0, t_max, base_dt)

        # Total samples with time-variant
        tv_samples = sum(w.n_samples for w in windows)

        # Total samples with uniform
        uniform_samples = int(t_max / base_dt)

        return uniform_samples / tv_samples if tv_samples > 0 else 1.0
```

### Config Model

```python
class TimeVariantConfig(BaseConfig):
    """Configuration for time-variant sampling."""

    enabled: bool = Field(default=False)
    frequency_table: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0, 80), (1000, 50), (3000, 25)],
        description="List of (time_ms, max_freq_hz) pairs"
    )
    interpolation_method: Literal["linear", "pchip", "cubic"] = Field(
        default="pchip",
        description="Interpolation method for frequency curve"
    )
    min_downsample_factor: int = Field(default=1, ge=1)
    max_downsample_factor: int = Field(default=8, ge=1, le=16)
```

---

## Implementation Tasks

### Phase 1: Core Algorithm (Python Prototype)

- [ ] **Task 1.1**: Create `TimeVariantSampling` class in `pstm/algorithm/time_variant.py`
  - [ ] Implement frequency interpolation
  - [ ] Implement window computation
  - [ ] Add unit tests

- [ ] **Task 1.2**: Create time-variant trace interpolation utilities
  - [ ] Implement sinc interpolation with AA filter
  - [ ] Implement downsampled trace reading
  - [ ] Add benchmarks

- [ ] **Task 1.3**: Implement output resampling
  - [ ] Uniform resampling from time-variant grid
  - [ ] Quality validation tests

### Phase 2: Metal Kernel Integration

- [ ] **Task 2.1**: Update Metal shader structs
  - [ ] Add `TimeWindow` struct
  - [ ] Add `TimeVariantParams` constant buffer

- [ ] **Task 2.2**: Implement time-variant migration kernel
  - [ ] Window-based parallelization
  - [ ] Downsampled trace interpolation
  - [ ] Atomic output accumulation

- [ ] **Task 2.3**: Update `CompiledMetalKernel` Python wrapper
  - [ ] Pass time window buffers
  - [ ] Handle output resampling

### Phase 3: UI Integration

- [ ] **Task 3.1**: Create `TimeVariantStep` wizard page
  - [ ] Enable checkbox
  - [ ] Frequency-time table widget
  - [ ] Preview graph

- [ ] **Task 3.2**: Add auto-detect functionality
  - [ ] Spectrum analysis
  - [ ] Frequency curve simplification

- [ ] **Task 3.3**: Update state management
  - [ ] Add `TimeVariantSamplingState`
  - [ ] Wire to `MigrationConfig`

- [ ] **Task 3.4**: Update execution step
  - [ ] Show estimated speedup
  - [ ] Display window breakdown

### Phase 4: Testing & Validation

- [ ] **Task 4.1**: Synthetic data tests
  - [ ] Compare uniform vs time-variant output
  - [ ] Verify frequency preservation

- [ ] **Task 4.2**: Performance benchmarks
  - [ ] Measure actual speedup
  - [ ] Memory usage comparison

- [ ] **Task 4.3**: Real data validation
  - [ ] Visual QC comparison
  - [ ] Amplitude spectrum analysis

---

## Success Criteria

1. **Performance**: 3-5x speedup on typical data with time-variant sampling
2. **Quality**: < 1% RMS difference from uniform sampling at frequencies below local f_max
3. **UI**: Intuitive table editor with real-time preview
4. **Auto-detect**: Reasonable frequency estimates from data analysis
5. **Compatibility**: Works with all correction options (AA, spreading, obliquity)

---

## References

- Stolt, R.H., 1978, Migration by Fourier transform: Geophysics
- Yilmaz, O., 2001, Seismic Data Analysis (Chapter on efficiency optimization)
- Gray, S.H., 1992, Frequency-selective design of the Kirchhoff migration operator
