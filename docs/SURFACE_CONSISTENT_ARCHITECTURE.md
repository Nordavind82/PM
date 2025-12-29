# Surface-Consistent Seismic Processing Architecture

## Executive Summary

This design leverages your existing PSTM infrastructure - the PyQt6 wizard UI pattern, compiled Metal shader architecture, Zarr/Parquet data handling, and tile-based memory management - to create a high-performance surface-consistent deconvolution and amplitude correction system optimized for M4 Max GPU.

---

## 1. Mathematical Foundations

### 1.1 Classical Surface-Consistent Model (Taner & Koehler, 1981)

In the log/Fourier domain, a seismic trace can be decomposed:

```
ln|S(f,i,j,m,h)| = ln|W_s(f,i)| + ln|W_r(f,j)| + ln|W_m(f,m)| + ln|W_h(f,h)| + ln|R(f,m)|
```

Where:
- `S(f,i,j,m,h)` = observed spectrum at frequency f
- `W_s(f,i)` = source wavelet for shot i
- `W_r(f,j)` = receiver response for receiver j
- `W_m(f,m)` = midpoint (structural) term
- `W_h(f,h)` = offset-dependent term
- `R(f,m)` = reflectivity (what we want to preserve)

This creates a **linear system** at each frequency:
```
d = G·m
```
where `d` = log amplitudes, `G` = design matrix (0s and 1s), `m` = factor amplitudes

### 1.2 Modern Robust Formulations

**L1-Norm Solution (for outlier robustness):**
```
minimize ||Gm - d||_1
```
This handles anomalous traces better than least-squares.

**Reweighted L1 (IRLS) for sparse solutions:**
```python
for iteration in range(max_iter):
    W = diag(1 / (|residual| + epsilon))
    m = solve(G.T @ W @ G, G.T @ W @ d)
```

**L1-L2 Hybrid (GMC Penalty):**
```
minimize ||Gm - d||_2 + λ·GMC(m)
```
GMC = Generalized Minimax Concave penalty, better than L1 for sparse recovery.

### 1.3 Phase Estimation (Mixed-Phase Wavelet)

Classical surface-consistent assumes minimum phase, which is often wrong. Modern approaches use **higher-order statistics**:

**Fourth-Order Cumulant Matching** (Lazear 1993):
- Autocorrelation (2nd order) has no phase information
- Trispectrum (4th order) contains phase
- Match 4th-order statistics to estimate mixed-phase wavelets

**Bispectrum Method:**
```
Phase(f) = arg(B(f1, f2)) where f = f1 + f2
```

**All-Pass Factorization** (Misra & Sacchi 2006):
```
W(z) = W_min(z) · A(z)
```
where A(z) is an all-pass filter preserving amplitude but modifying phase.

---

## 2. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SURFACE-CONSISTENT PROCESSING                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │
│   │   INPUT     │───▶│   SPECTRA   │───▶│  DECOMPOSE  │───▶│    QC     │ │
│   │   STEP      │    │   ESTIMATE  │    │   FACTORS   │    │   MAPS    │ │
│   └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘ │
│         │                  │                  │                  │        │
│         ▼                  ▼                  ▼                  ▼        │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │
│   │  Zarr/Pqt   │    │  Metal GPU  │    │  Metal GPU  │    │  PyQtG-   │ │
│   │  Headers    │    │  FFT Kernel │    │  Solver     │    │  raph     │ │
│   └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘ │
│                                                                           │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│   │  EQUALIZER  │───▶│   FILTER    │───▶│ PRODUCTION  │                  │
│   │   DESIGN    │    │   QC TEST   │    │   APPLY     │                  │
│   └─────────────┘    └─────────────┘    └─────────────┘                  │
│         │                  │                  │                           │
│         ▼                  ▼                  ▼                           │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│   │  Real-time  │    │  Metal GPU  │    │  Metal GPU  │                  │
│   │  EQ Sliders │    │  Decon      │    │  Batch      │                  │
│   └─────────────┘    └─────────────┘    └─────────────┘                  │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Configuration Models

```python
# pstm/config/surface_consistent.py

from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal

class FactorType(str, Enum):
    """Surface-consistent decomposition factors."""
    SOURCE = "source"
    RECEIVER = "receiver"
    MIDPOINT = "midpoint"
    OFFSET = "offset"
    OFFSET_BIN = "offset_bin"
    AZIMUTH_SECTOR = "azimuth_sector"

class SolverMethod(str, Enum):
    """Decomposition solver method."""
    LEAST_SQUARES = "least_squares"      # Fast, not robust
    L1_IRLS = "l1_irls"                  # Robust to outliers
    L1_L2_GMC = "l1_l2_gmc"              # Best sparse recovery
    CONJUGATE_GRADIENT = "cg"            # Memory efficient
    GPU_DIRECT = "gpu_direct"            # Metal compute solver

class PhaseEstimationMethod(str, Enum):
    """Phase estimation approach."""
    MINIMUM_PHASE = "minimum_phase"
    BISPECTRUM = "bispectrum"
    FOURTH_ORDER_CUMULANT = "fourth_order_cumulant"
    MIXED_PHASE_ITERATIVE = "mixed_phase_iterative"

class SpectraEstimationConfig(BaseModel):
    """Step 1: Spectra/amplitude estimation configuration."""
    # Analysis windows
    windows: list[tuple[float, float]] = Field(
        default=[(0, 1000), (1000, 2500), (2500, 5000)],
        description="Time windows for spectral estimation (ms)"
    )
    window_taper: Literal["hanning", "hamming", "blackman", "tukey"] = "tukey"
    taper_fraction: float = Field(default=0.1, ge=0.0, le=0.5)

    # FFT parameters
    fft_size: int = Field(default=1024, description="FFT size (power of 2)")
    overlap_percent: float = Field(default=50.0, ge=0, le=90)

    # Frequency range
    freq_min_hz: float = Field(default=5.0, ge=0)
    freq_max_hz: float = Field(default=120.0, le=500)

    # Output
    output_metric: Literal["amplitude", "power", "db"] = "amplitude"
    smoothing_octaves: float = Field(default=0.25, ge=0, le=1)

class DecompositionConfig(BaseModel):
    """Step 2: Surface-consistent decomposition configuration."""
    # Factor selection
    factors: list[FactorType] = Field(
        default=[FactorType.SOURCE, FactorType.RECEIVER],
        description="Factors to decompose"
    )

    # Factor header mapping
    source_header: str = Field(default="FFID", description="Header for source ID")
    receiver_header: str = Field(default="CHAN", description="Header for receiver ID")
    midpoint_header: str = Field(default="CDP", description="Header for midpoint bin")
    offset_header: str = Field(default="OFFSET", description="Header for offset")

    # Offset binning (if OFFSET_BIN factor used)
    offset_bins: list[float] = Field(
        default=[0, 500, 1000, 2000, 3000, 5000],
        description="Offset bin edges (m)"
    )

    # Solver configuration
    solver: SolverMethod = SolverMethod.L1_IRLS
    max_iterations: int = Field(default=50, ge=1, le=500)
    convergence_threshold: float = Field(default=1e-4, ge=1e-8)
    regularization_lambda: float = Field(default=0.01, ge=0)

    # Phase estimation
    phase_method: PhaseEstimationMethod = PhaseEstimationMethod.MINIMUM_PHASE
    estimate_phase: bool = Field(default=True)

class EqualizerBand(BaseModel):
    """Single equalizer frequency band."""
    center_freq_hz: float
    gain_db: float = 0.0
    q_factor: float = 1.0  # Bandwidth control

class SeismicEqualizerConfig(BaseModel):
    """Step 4: Seismic Equalizer configuration."""
    enabled: bool = False

    # Preset bands (like audio equalizer)
    bands: list[EqualizerBand] = Field(
        default=[
            EqualizerBand(center_freq_hz=8, gain_db=0),
            EqualizerBand(center_freq_hz=16, gain_db=0),
            EqualizerBand(center_freq_hz=25, gain_db=0),
            EqualizerBand(center_freq_hz=40, gain_db=0),
            EqualizerBand(center_freq_hz=60, gain_db=0),
            EqualizerBand(center_freq_hz=80, gain_db=0),
            EqualizerBand(center_freq_hz=100, gain_db=0),
            EqualizerBand(center_freq_hz=125, gain_db=0),
        ],
        description="Equalizer bands (8-band parametric EQ)"
    )

    # Additional filters
    high_pass_freq_hz: float | None = None
    low_pass_freq_hz: float | None = None
    notch_filters: list[tuple[float, float]] = []  # (freq, bandwidth)

    # Preview settings
    preview_inline: int | None = None
    preview_crossline: int | None = None

class ApplicationConfig(BaseModel):
    """Step 5: Production application configuration."""
    # What to apply
    apply_amplitude: bool = True
    apply_phase: bool = False
    apply_equalizer: bool = True

    # Output scaling
    output_scaling: Literal["preserve", "rms_normalize", "agc"] = "preserve"
    agc_window_ms: float = 500.0

    # QC output
    write_before_after_qc: bool = True
    qc_inlines: list[int] = []
    qc_crosslines: list[int] = []

class SurfaceConsistentConfig(BaseModel):
    """Master configuration for surface-consistent processing."""
    spectra: SpectraEstimationConfig = SpectraEstimationConfig()
    decomposition: DecompositionConfig = DecompositionConfig()
    equalizer: SeismicEqualizerConfig = SeismicEqualizerConfig()
    application: ApplicationConfig = ApplicationConfig()

    # Execution
    use_gpu: bool = True
    tile_size_traces: int = Field(default=50000, description="Traces per GPU batch")
    checkpoint_interval: int = 1000
```

---

## 4. Metal Shader Architecture

### 4.1 Spectra Estimation Kernel

```metal
// pstm/metal/shaders/surface_consistent_spectra.metal

#include <metal_stdlib>
using namespace metal;

struct SpectraParams {
    int n_traces;
    int n_samples;
    int fft_size;
    int n_windows;
    float dt_ms;
    float t_start_ms;
    float freq_min_hz;
    float freq_max_hz;
    int taper_type;       // 0=hanning, 1=hamming, 2=blackman, 3=tukey
    float taper_fraction;
};

struct TimeWindow {
    float t_start_ms;
    float t_end_ms;
    int sample_start;
    int sample_count;
};

// Taper functions
inline float apply_taper(float sample_pos, int n_samples, int taper_type, float taper_frac) {
    float x = sample_pos / float(n_samples - 1);

    switch(taper_type) {
        case 0: // Hanning
            return 0.5f * (1.0f - cos(2.0f * M_PI_F * x));
        case 1: // Hamming
            return 0.54f - 0.46f * cos(2.0f * M_PI_F * x);
        case 2: // Blackman
            return 0.42f - 0.5f * cos(2.0f * M_PI_F * x) + 0.08f * cos(4.0f * M_PI_F * x);
        case 3: // Tukey
            if (x < taper_frac / 2.0f)
                return 0.5f * (1.0f + cos(2.0f * M_PI_F * (x / taper_frac - 0.5f)));
            else if (x > 1.0f - taper_frac / 2.0f)
                return 0.5f * (1.0f + cos(2.0f * M_PI_F * ((x - 1.0f) / taper_frac + 0.5f)));
            else
                return 1.0f;
        default:
            return 1.0f;
    }
}

// FFT butterfly (Cooley-Tukey radix-2)
// Note: For production, use Metal Performance Shaders MPSFFTDescriptor
kernel void compute_spectra_windowed(
    device const float* traces [[buffer(0)]],           // [n_traces, n_samples]
    device float2* spectra_out [[buffer(1)]],           // [n_traces, n_windows, fft_size/2+1] complex
    device float* amplitude_out [[buffer(2)]],          // [n_traces, n_windows, fft_size/2+1] real
    constant TimeWindow* windows [[buffer(3)]],
    constant SpectraParams& params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]               // (trace_idx, window_idx)
) {
    int trace_idx = gid.x;
    int window_idx = gid.y;

    if (trace_idx >= params.n_traces || window_idx >= params.n_windows) return;

    TimeWindow win = windows[window_idx];

    // Extract and taper window
    device const float* trace = traces + trace_idx * params.n_samples;

    // Local storage for FFT (threadgroup memory for larger FFTs)
    float2 fft_data[1024];  // Assuming max FFT size 1024

    for (int i = 0; i < params.fft_size; i++) {
        int sample_idx = win.sample_start + i;
        float val = 0.0f;
        if (sample_idx >= 0 && sample_idx < params.n_samples && i < win.sample_count) {
            val = trace[sample_idx];
            val *= apply_taper(float(i), win.sample_count, params.taper_type, params.taper_fraction);
        }
        fft_data[i] = float2(val, 0.0f);
    }

    // In-place FFT (simplified - use MPS for production)
    // ... FFT implementation ...

    // Write amplitude spectrum
    int out_offset = trace_idx * params.n_windows * (params.fft_size / 2 + 1)
                   + window_idx * (params.fft_size / 2 + 1);

    for (int k = 0; k <= params.fft_size / 2; k++) {
        float2 c = fft_data[k];
        float amp = sqrt(c.x * c.x + c.y * c.y);
        amplitude_out[out_offset + k] = amp;
        spectra_out[out_offset + k] = c;
    }
}
```

### 4.2 Surface-Consistent Decomposition Kernel

```metal
// pstm/metal/shaders/surface_consistent_decompose.metal

struct DecomposeParams {
    int n_equations;          // Total observations
    int n_source_factors;     // Unique sources
    int n_receiver_factors;   // Unique receivers
    int n_offset_factors;     // Offset bins
    int n_freq_bins;          // Frequency bins to process
    float regularization;
    int solver_type;          // 0=LS, 1=IRLS, 2=CG
    int max_iterations;
    float convergence_tol;
};

// Sparse matrix row for surface-consistent system
struct SCRow {
    int source_idx;           // -1 if not used
    int receiver_idx;
    int offset_idx;
    int midpoint_idx;
    float observation;        // log(amplitude)
    float weight;             // For IRLS
};

// IRLS (Iteratively Reweighted Least Squares) for L1 norm
kernel void solve_surface_consistent_irls(
    device const SCRow* rows [[buffer(0)]],
    device float* source_factors [[buffer(1)]],
    device float* receiver_factors [[buffer(2)]],
    device float* offset_factors [[buffer(3)]],
    device float* residuals [[buffer(4)]],
    constant DecomposeParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]  // frequency bin index
) {
    if (gid >= params.n_freq_bins) return;

    int freq_offset = gid * params.n_equations;

    // Initialize factors to zero
    // ...

    float prev_objective = 1e30f;

    for (int iter = 0; iter < params.max_iterations; iter++) {
        // Compute residuals
        float total_residual = 0.0f;
        for (int i = 0; i < params.n_equations; i++) {
            SCRow row = rows[freq_offset + i];
            float predicted = 0.0f;

            if (row.source_idx >= 0)
                predicted += source_factors[gid * params.n_source_factors + row.source_idx];
            if (row.receiver_idx >= 0)
                predicted += receiver_factors[gid * params.n_receiver_factors + row.receiver_idx];
            if (row.offset_idx >= 0)
                predicted += offset_factors[gid * params.n_offset_factors + row.offset_idx];

            float r = row.observation - predicted;
            residuals[freq_offset + i] = r;

            // Update weight for IRLS (L1)
            row.weight = 1.0f / (abs(r) + 1e-6f);

            total_residual += abs(r);
        }

        // Check convergence
        if (abs(prev_objective - total_residual) / prev_objective < params.convergence_tol) {
            break;
        }
        prev_objective = total_residual;

        // Weighted normal equations solve
        // G' W G m = G' W d
        // Using conjugate gradient for efficiency
        // ...
    }
}
```

### 4.3 Real-Time Equalizer Preview Kernel

```metal
// pstm/metal/shaders/seismic_equalizer.metal

struct EqualizerBand {
    float center_freq_hz;
    float gain_linear;      // 10^(gain_db/20)
    float q_factor;
    float bandwidth_hz;     // Derived from Q
};

struct EqualizerParams {
    int n_traces;
    int n_samples;
    float dt_ms;
    int n_bands;
    float high_pass_hz;     // 0 = disabled
    float low_pass_hz;      // 0 = disabled
};

// Biquad coefficients for parametric EQ band
struct BiquadCoeffs {
    float b0, b1, b2;
    float a1, a2;
};

// Design peaking EQ filter (peakingEQ from Audio EQ Cookbook)
inline BiquadCoeffs design_peaking_eq(float fc, float fs, float gain_db, float Q) {
    float A = pow(10.0f, gain_db / 40.0f);
    float w0 = 2.0f * M_PI_F * fc / fs;
    float cos_w0 = cos(w0);
    float sin_w0 = sin(w0);
    float alpha = sin_w0 / (2.0f * Q);

    BiquadCoeffs c;
    float a0 = 1.0f + alpha / A;
    c.b0 = (1.0f + alpha * A) / a0;
    c.b1 = (-2.0f * cos_w0) / a0;
    c.b2 = (1.0f - alpha * A) / a0;
    c.a1 = (-2.0f * cos_w0) / a0;
    c.a2 = (1.0f - alpha / A) / a0;

    return c;
}

// Apply equalizer in time domain (low latency for preview)
kernel void apply_equalizer_realtime(
    device const float* traces_in [[buffer(0)]],
    device float* traces_out [[buffer(1)]],
    constant EqualizerBand* bands [[buffer(2)]],
    constant EqualizerParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]  // trace index
) {
    if (gid >= params.n_traces) return;

    device const float* in_trace = traces_in + gid * params.n_samples;
    device float* out_trace = traces_out + gid * params.n_samples;

    float fs = 1000.0f / params.dt_ms;  // Sample rate in Hz

    // Apply each EQ band as cascaded biquads
    // First copy input to output
    for (int i = 0; i < params.n_samples; i++) {
        out_trace[i] = in_trace[i];
    }

    // Apply each band
    for (int b = 0; b < params.n_bands; b++) {
        EqualizerBand band = bands[b];
        if (abs(band.gain_linear - 1.0f) < 0.001f) continue;  // Skip unity gain

        float gain_db = 20.0f * log10(band.gain_linear);
        BiquadCoeffs c = design_peaking_eq(band.center_freq_hz, fs, gain_db, band.q_factor);

        // Biquad filter (Direct Form II Transposed)
        float z1 = 0.0f, z2 = 0.0f;
        for (int i = 0; i < params.n_samples; i++) {
            float x = out_trace[i];
            float y = c.b0 * x + z1;
            z1 = c.b1 * x - c.a1 * y + z2;
            z2 = c.b2 * x - c.a2 * y;
            out_trace[i] = y;
        }
    }
}

// Fast preview using frequency domain (for larger datasets)
kernel void apply_equalizer_fft(
    device float2* spectra [[buffer(0)]],           // In-place
    constant EqualizerBand* bands [[buffer(1)]],
    constant EqualizerParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]           // (trace_idx, freq_bin)
) {
    int trace_idx = gid.x;
    int freq_bin = gid.y;
    int fft_size = params.n_samples;  // Assuming power of 2

    if (trace_idx >= params.n_traces || freq_bin > fft_size / 2) return;

    float fs = 1000.0f / params.dt_ms;
    float freq = float(freq_bin) * fs / float(fft_size);

    // Compute combined gain at this frequency
    float total_gain = 1.0f;

    for (int b = 0; b < params.n_bands; b++) {
        EqualizerBand band = bands[b];
        float fc = band.center_freq_hz;
        float Q = band.q_factor;

        // Parametric EQ frequency response magnitude
        float ratio = freq / fc;
        float bw = fc / Q;
        float gain_at_freq = band.gain_linear;

        // Simplified Gaussian-shaped response for preview speed
        float x = (freq - fc) / (bw * 0.5f);
        float shape = exp(-0.5f * x * x);
        total_gain *= (1.0f - shape) + shape * gain_at_freq;
    }

    // High-pass filter
    if (params.high_pass_hz > 0.0f && freq < params.high_pass_hz) {
        float ratio = freq / params.high_pass_hz;
        total_gain *= ratio * ratio;  // 2nd order rolloff
    }

    // Low-pass filter
    if (params.low_pass_hz > 0.0f && freq > params.low_pass_hz) {
        float ratio = params.low_pass_hz / freq;
        total_gain *= ratio * ratio;
    }

    // Apply gain
    int idx = trace_idx * (fft_size / 2 + 1) + freq_bin;
    spectra[idx] *= total_gain;
}
```

### 4.4 Deconvolution Filter Application

```metal
// pstm/metal/shaders/surface_consistent_apply.metal

struct ApplyParams {
    int n_traces;
    int n_samples;
    int fft_size;
    float dt_ms;
    int apply_amplitude;
    int apply_phase;
    float water_level_db;    // Prevent division by zero
};

// Apply surface-consistent correction filters
kernel void apply_sc_correction(
    device float2* trace_spectra [[buffer(0)]],     // [n_traces, fft_size/2+1] in-place
    device const float* source_amp [[buffer(1)]],   // [n_sources, n_freq] amplitude correction
    device const float* source_phase [[buffer(2)]],
    device const float* receiver_amp [[buffer(3)]],
    device const float* receiver_phase [[buffer(4)]],
    device const int* trace_source_idx [[buffer(5)]],
    device const int* trace_receiver_idx [[buffer(6)]],
    constant ApplyParams& params [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]           // (trace_idx, freq_bin)
) {
    int trace_idx = gid.x;
    int freq_bin = gid.y;

    if (trace_idx >= params.n_traces || freq_bin > params.fft_size / 2) return;

    int src_idx = trace_source_idx[trace_idx];
    int rcv_idx = trace_receiver_idx[trace_idx];

    // Get correction factors
    float amp_correction = 1.0f;
    float phase_correction = 0.0f;

    if (params.apply_amplitude) {
        float src_amp = source_amp[src_idx * (params.fft_size / 2 + 1) + freq_bin];
        float rcv_amp = receiver_amp[rcv_idx * (params.fft_size / 2 + 1) + freq_bin];

        // Water level to prevent instability
        float water = pow(10.0f, params.water_level_db / 20.0f);
        src_amp = max(src_amp, water);
        rcv_amp = max(rcv_amp, water);

        amp_correction = 1.0f / (src_amp * rcv_amp);
    }

    if (params.apply_phase) {
        phase_correction = -(source_phase[src_idx * (params.fft_size / 2 + 1) + freq_bin]
                           + receiver_phase[rcv_idx * (params.fft_size / 2 + 1) + freq_bin]);
    }

    // Apply correction
    int idx = trace_idx * (params.fft_size / 2 + 1) + freq_bin;
    float2 spec = trace_spectra[idx];

    // Convert to polar, apply, convert back
    float mag = length(spec) * amp_correction;
    float phase = atan2(spec.y, spec.x) + phase_correction;

    trace_spectra[idx] = float2(mag * cos(phase), mag * sin(phase));
}
```

---

## 5. UI/UX Design

### 5.1 Wizard Step Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Surface-Consistent Processing Wizard                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1. Input] → [2. Spectra] → [3. Decompose] → [4. QC Maps] →       │
│                                                                     │
│  → [5. Equalizer] → [6. Filter Test] → [7. Production]             │
│                                                                     │
│  ═══════════════════════════════════════════════════════════════   │
│                                                                     │
│                    Current Step Content Here                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Step 1: Input Configuration

```
┌────────────────────────────┬────────────────────────────────────────┐
│ INPUT DATA                 │ HEADER MAPPING                         │
├────────────────────────────┼────────────────────────────────────────┤
│                            │                                        │
│ Traces (Zarr):             │ Factor Headers:                        │
│ [________________________] │                                        │
│ [Browse...]                │ Source ID:    [FFID     ▼]             │
│                            │ Receiver ID:  [CHAN     ▼]             │
│ Headers (Parquet):         │ Midpoint:     [CDP      ▼]             │
│ [________________________] │ Offset:       [OFFSET   ▼]             │
│ [Browse...]                │ Azimuth:      [AZIMUTH  ▼]             │
│                            │                                        │
│ ℹ️ 2,450,000 traces         │ ┌──────────────────────────────────┐   │
│ ℹ️ 3000 samples @ 2ms       │ │ Detected:                        │   │
│ ℹ️ 1,245 shots              │ │ • 1,245 unique sources           │   │
│ ℹ️ 2,400 receivers          │ │ • 2,400 unique receivers         │   │
│                            │ │ • 15,432 CDPs                    │   │
│                            │ │ • Offset: 150-6,200m             │   │
│                            │ └──────────────────────────────────┘   │
└────────────────────────────┴────────────────────────────────────────┘
```

### 5.3 Step 2: Spectra Estimation

```
┌────────────────────────────────────────┬────────────────────────────┐
│ ANALYSIS WINDOWS                       │ PARAMETERS                 │
├────────────────────────────────────────┼────────────────────────────┤
│                                        │                            │
│ Time Windows (ms):                     │ FFT Size:    [1024    ▼]   │
│ ┌────────┬────────┬─────────┐          │                            │
│ │ Start  │ End    │ [X]     │          │ Window:      [Tukey   ▼]   │
│ ├────────┼────────┼─────────┤          │ Taper:       [0.1_____]   │
│ │ 0      │ 1000   │ [del]   │          │                            │
│ │ 1000   │ 2500   │ [del]   │          │ Overlap:     [50%     ▼]   │
│ │ 2500   │ 5000   │ [del]   │          │                            │
│ └────────┴────────┴─────────┘          │ Freq Range:                │
│ [+ Add Window]                         │ [5____] - [120__] Hz       │
│                                        │                            │
│ ┌──────────────────────────────────────┼────────────────────────────┤
│ │ SPECTRA PREVIEW                      │ COMPUTE                    │
│ │                                      │                            │
│ │  [Amplitude spectrum visualization]  │ ⚡ GPU Accelerated         │
│ │                                      │                            │
│ │  ^                                   │ Est. Time: ~45 sec         │
│ │  |    ___                            │ Memory: ~2.1 GB            │
│ │  |   /   \___                        │                            │
│ │  |  /        \___                    │ [Compute Spectra]          │
│ │  |_/             \_____              │                            │
│ │  +-------------------->              │ Progress: ████████░░ 78%   │
│ │  5  25  50  75  100 Hz               │                            │
│ └──────────────────────────────────────┴────────────────────────────┘
```

### 5.4 Step 3: Factor Decomposition

```
┌────────────────────────────────────────┬────────────────────────────┐
│ DECOMPOSITION FACTORS                  │ SOLVER SETTINGS            │
├────────────────────────────────────────┼────────────────────────────┤
│                                        │                            │
│ [✓] Source (FFID)                      │ Method:                    │
│ [✓] Receiver (CHAN)                    │ ○ Least Squares (fast)     │
│ [ ] Midpoint (CDP)                     │ ● L1 IRLS (robust)         │
│ [ ] Offset                             │ ○ L1-L2 GMC (sparse)       │
│ [✓] Offset Bins                        │ ○ GPU Conjugate Gradient   │
│     Bins: [0,500,1000,2000,4000,6000]  │                            │
│ [ ] Azimuth Sectors                    │ Max Iterations: [50___]    │
│                                        │ Convergence:    [1e-4_]    │
│ ──────────────────────────────────────│ Regularization: [0.01_]    │
│                                        │                            │
│ PHASE ESTIMATION                       │ ──────────────────────────│
│                                        │                            │
│ [✓] Estimate phase                     │ SYSTEM SIZE                │
│                                        │                            │
│ Method:                                │ Equations:    2,450,000    │
│ ○ Minimum Phase (classical)            │ Unknowns:                  │
│ ○ Bispectrum (modern)                  │  - Sources:     1,245      │
│ ● 4th Order Cumulant                   │  - Receivers:   2,400      │
│ ○ Mixed-Phase Iterative                │  - Offset bins:     6      │
│                                        │                            │
│                                        │ Sparsity: 99.85%           │
│                                        │                            │
│                                        │ [Run Decomposition]        │
└────────────────────────────────────────┴────────────────────────────┘
```

### 5.5 Step 4: QC Maps

```
┌─────────────────────────────────────────────────────────────────────┐
│ QC MAPS - Surface Consistent Factors                                │
├───────────────────────────────────┬─────────────────────────────────┤
│                                   │                                 │
│  SOURCE AMPLITUDE @ 40 Hz         │  RECEIVER AMPLITUDE @ 40 Hz     │
│  ┌─────────────────────────────┐  │  ┌─────────────────────────────┐│
│  │                             │  │  │                             ││
│  │     ████  ████              │  │  │  ░░░░░░░░░░░░░░░░░░░░░░░░  ││
│  │   ██████████████            │  │  │  ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░  ││
│  │  ████████████████           │  │  │  ░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░  ││
│  │ ██████████████████          │  │  │  ░▒▒▓████████████▓▒▒░    ││
│  │  ████████████████           │  │  │  ░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░    ││
│  │   ██████████████            │  │  │  ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░    ││
│  │     ████  ████              │  │  │  ░░░░░░░░░░░░░░░░░░░░    ││
│  │                             │  │  │                             ││
│  └─────────────────────────────┘  │  └─────────────────────────────┘│
│  [Export] [Histogram]             │  [Export] [Histogram]           │
│                                   │                                 │
├───────────────────────────────────┴─────────────────────────────────┤
│ FREQUENCY SELECTOR                                                  │
│ [5 Hz]──────●────────────────────────────────────────────[120 Hz]   │
│             40 Hz                                                   │
│                                                                     │
│ OFFSET BINS COMPARISON                                              │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │    ^                                                          │   │
│ │ dB │  ─── 0-500m   ─── 1000-2000m   ─── 4000-6000m           │   │
│ │  0 │____.___.___.___.___                                      │   │
│ │-10 │     \  \   /  /                                          │   │
│ │-20 │      \_\__/__/                                           │   │
│ │    +─────────────────────────────────────────────────> Hz     │   │
│ └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.6 Step 5: Seismic Equalizer (Key Feature)

```
┌─────────────────────────────────────────────────────────────────────┐
│ SEISMIC EQUALIZER                                         [Bypass] │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ PARAMETRIC EQ BANDS                                                │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │      8Hz   16Hz   25Hz   40Hz   60Hz   80Hz  100Hz  125Hz     │   │
│ │       │      │      │      │      │      │      │      │      │   │
│ │  +12 ─┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼─     │   │
│ │       │      │      │      │      │      │      │      │      │   │
│ │   +6 ─┼──────┼──────┼────▲─┼──────┼──────┼──────┼──────┼─     │   │
│ │       │      │      │   /│\│      │      │      │      │      │   │
│ │    0 ─●──────●──────●──/ │ \●─────●──────●──────●──────●─     │   │
│ │       │      │      │ /  │  \     │      │      │      │      │   │
│ │   -6 ─┼──────┼──────┼/   │   ────┼──────┼──────┼──────┼─     │   │
│ │       │      │      │    │        │      │      │      │      │   │
│ │  -12 ─┼──────┼──────┼────┼────────┼──────┼──────┼──────┼─     │   │
│ │      [0]    [0]    [0]   [+4] [+2]  [-2]  [0]    [0]  dB      │   │
│ └───────────────────────────────────────────────────────────────┘   │
│                                                                     │
│ FILTERS                           │ PRESETS                         │
│ ┌─────────────────────────────────┼─────────────────────────────┐   │
│ │ High-Pass: [○ Off] [● 8 Hz  ▼] │ [Deep Enhancement]           │   │
│ │ Low-Pass:  [● Off] [○ 100 Hz▼] │ [High Frequency Boost]       │   │
│ │ Notch:     [+ Add 50Hz notch]  │ [Balanced]                   │   │
│ │                                 │ [Custom...]  [Save Preset]   │   │
│ └─────────────────────────────────┴─────────────────────────────┘   │
│                                                                     │
│ RESPONSE CURVE                                                      │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │  +12│                    ╱╲                                   │   │
│ │   +6│              _____╱  ╲_____                             │   │
│ │    0│_____________╱              ╲___________________________  │   │
│ │   -6│                                                         │   │
│ │  -12│                                                         │   │
│ │     └────────────────────────────────────────────────────────│   │
│ │      5    10   20   30  40 50 60 80 100   150   Hz            │   │
│ └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.7 Step 6: Filter Design & QC Test

```
┌─────────────────────────────────────────────────────────────────────┐
│ FILTER DESIGN & QC TEST                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ SELECT QC LINES                       │ DECONVOLUTION SETTINGS      │
│ ┌─────────────────────────────────────┼─────────────────────────┐   │
│ │ Survey Map (click to select)        │                         │   │
│ │ ┌─────────────────────────────────┐ │ Apply:                  │   │
│ │ │    ╔═══════════════════════╗    │ │ [✓] Source amplitude    │   │
│ │ │    ║░░░░░░░░░░░░░░░░░░░░░░░║    │ │ [✓] Receiver amplitude  │   │
│ │ │    ║░░░░░░░░░░░░░░░░░░░░░░░║    │ │ [ ] Source phase        │   │
│ │ │    ║░░░░░▓▓▓▓▓░░░░░░░░░░░░░║◀──IL│ │ [ ] Receiver phase      │   │
│ │ │    ║░░░░░░░░░░░░░░░░░░░░░░░║    │ │ [✓] Offset correction   │   │
│ │ │    ║░░░░░░░░░░│░░░░░░░░░░░░║    │ │ [✓] Apply equalizer     │   │
│ │ │    ╚═══════════════════════╝    │ │                         │   │
│ │ └─────────────▲───────────────────┘ │ Water Level: [-40_ dB]  │   │
│ │               XL 1250               │                         │   │
│ │                                     │ White Noise:  [0.1_%]   │   │
│ │ Selected: IL 425, XL 1250           │                         │   │
│ └─────────────────────────────────────┴─────────────────────────┘   │
│                                                                     │
│ [▶ Run Test on Selected Lines]                                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ BEFORE / AFTER COMPARISON                                           │
│ ┌───────────────────────────────┬───────────────────────────────┐   │
│ │ BEFORE (IL 425)               │ AFTER (IL 425)                │   │
│ │ ┌───────────────────────────┐ │ ┌───────────────────────────┐ │   │
│ │ │ ▓▓▓▓▓░░░░░░░░░▓▓▓▓▓▓▓▓▓▓ │ │ │ ████████████████████████ │ │   │
│ │ │ ░░░░▓▓▓▓░░░░░▓▓▓░░░░░░░░ │ │ │ ████████████████████████ │ │   │
│ │ │ ▓▓▓░░░░▓▓▓▓▓░░░░▓▓▓▓▓▓▓▓ │ │ │ ████████████████████████ │ │   │
│ │ │ ░░░▓▓▓░░░░░▓▓▓▓░░░░░░░░░ │ │ │ ████████████████████████ │ │   │
│ │ │ ▓▓░░░░▓▓▓░░░░░░▓▓▓▓▓▓▓▓▓ │ │ │ ████████████████████████ │ │   │
│ │ └───────────────────────────┘ │ └───────────────────────────┘ │   │
│ │ [Wiggle] [Variable Density]   │ [Wiggle] [Variable Density]   │   │
│ └───────────────────────────────┴───────────────────────────────┘   │
│                                                                     │
│ AMPLITUDE SPECTRA COMPARISON                                        │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │     ^  ─── Before   ─── After   ─── Target                    │   │
│ │  dB │                                                         │   │
│ │   0 │_____                                                    │   │
│ │ -10 │     \____    ____                                       │   │
│ │ -20 │          \__/    \____                                  │   │
│ │ -30 │                       \____                             │   │
│ │     └────────────────────────────────────────────────> Hz     │   │
│ └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.8 Step 7: Production Application

```
┌─────────────────────────────────────────────────────────────────────┐
│ PRODUCTION APPLICATION                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ OUTPUT CONFIGURATION                  │ EXECUTION                   │
│ ┌─────────────────────────────────────┼─────────────────────────┐   │
│ │ Output Path:                        │                         │   │
│ │ [/data/output/sc_corrected.zarr___] │ Backend:                │   │
│ │ [Browse...]                         │ [● GPU Metal] [○ CPU]   │   │
│ │                                     │                         │   │
│ │ Format: [Zarr     ▼]                │ Batch Size: [50000_]    │   │
│ │                                     │ traces                  │   │
│ │ [✓] Write QC inline/crosslines      │                         │   │
│ │ [✓] Write amplitude difference map  │ Checkpoint: [✓] Every   │   │
│ │ [✓] Write processing log            │ [1000__] traces         │   │
│ │                                     │                         │   │
│ │ Post-Processing:                    │ ─────────────────────── │   │
│ │ ○ Preserve amplitudes               │                         │   │
│ │ ● RMS Normalize                     │ ESTIMATES               │   │
│ │ ○ AGC (window: [500_] ms)           │                         │   │
│ └─────────────────────────────────────│ Traces:     2,450,000   │   │
│                                       │ Output:     ~45 GB      │   │
│ SUMMARY                               │ Est. Time:  ~12 min     │   │
│ ┌─────────────────────────────────────│ GPU Memory: ~4.2 GB     │   │
│ │ Processing Chain:                   │                         │   │
│ │ 1. Source amplitude correction ✓    │ [▶ START PROCESSING]    │   │
│ │ 2. Receiver amplitude correction ✓  │                         │   │
│ │ 3. Offset bin correction ✓          │ ─────────────────────── │   │
│ │ 4. Seismic equalizer ✓              │                         │   │
│ │ 5. RMS normalization                │ PROGRESS                │   │
│ └─────────────────────────────────────│ ████████████░░░░ 75%    │   │
│                                       │ 1,837,500 / 2,450,000   │   │
│                                       │ Rate: 205k traces/sec   │   │
│                                       │ ETA: 2:58               │   │
│                                       │                         │   │
│                                       │ [Pause] [Cancel]        │   │
└───────────────────────────────────────┴─────────────────────────────┘
```

---

## 6. Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW DIAGRAM                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  INPUT                     PROCESSING                         OUTPUT      │
│  ─────                     ──────────                         ──────      │
│                                                                           │
│  ┌──────────┐   Chunk     ┌────────────────┐                             │
│  │  Zarr    │────────────▶│ Spectra Kernel │                             │
│  │  Traces  │   (50K)     │    (GPU)       │                             │
│  └──────────┘             └───────┬────────┘                             │
│       │                           │                                       │
│       │                           ▼                                       │
│  ┌──────────┐             ┌────────────────┐    ┌──────────────┐         │
│  │ Parquet  │────────────▶│ Factor Index   │───▶│ Spectra DB   │         │
│  │ Headers  │  (source,   │ Build          │    │ (per factor) │         │
│  └──────────┘   receiver) └────────────────┘    └──────┬───────┘         │
│                                                        │                  │
│                                                        ▼                  │
│                           ┌────────────────────────────────────┐         │
│                           │      DECOMPOSITION ENGINE          │         │
│                           │  ┌─────────────────────────────┐   │         │
│                           │  │ For each frequency bin:     │   │         │
│                           │  │   Build sparse matrix G     │   │         │
│                           │  │   Solve: min ||Gm - d||_p   │   │         │
│                           │  │   Store factors per freq    │   │         │
│                           │  └─────────────────────────────┘   │         │
│                           └──────────────┬─────────────────────┘         │
│                                          │                                │
│                                          ▼                                │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                    FACTOR STORAGE (Zarr)                          │   │
│  │  source_amp[n_sources, n_freq]      receiver_amp[n_receivers,     │   │
│  │  source_phase[n_sources, n_freq]    n_freq]                       │   │
│  │  offset_amp[n_offset_bins, n_freq]  receiver_phase[...]           │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                          │                                │
│                                          ▼                                │
│                           ┌────────────────────────────────┐             │
│                           │    APPLICATION ENGINE          │             │
│                           │                                │             │
│  ┌──────────┐   Stream    │  ┌───────────────────────┐     │  ┌───────┐ │
│  │  Input   │────────────▶│  │ For each trace chunk: │     │─▶│Output │ │
│  │  Zarr    │             │  │   1. Load factors     │     │  │ Zarr  │ │
│  └──────────┘             │  │   2. FFT trace        │     │  └───────┘ │
│                           │  │   3. Apply correction │     │             │
│                           │  │   4. Apply EQ         │     │             │
│                           │  │   5. IFFT             │     │             │
│                           │  │   6. Write output     │     │             │
│                           │  └───────────────────────┘     │             │
│                           └────────────────────────────────┘             │
│                                                                           │
│  MEMORY MANAGEMENT:                                                       │
│  ─────────────────                                                       │
│  • Tile/chunk processing for datasets >> RAM                             │
│  • Memory-mapped intermediate results                                     │
│  • GPU buffer pooling and reuse                                          │
│  • Streaming I/O with double buffering                                   │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Python Kernel Wrapper

```python
# pstm/kernels/surface_consistent_metal.py

from __future__ import annotations

import ctypes
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from numpy.typing import NDArray

try:
    import Metal
    import Foundation
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


@dataclass
class SpectraResult:
    """Result from spectra estimation."""
    amplitudes: NDArray[np.float32]      # [n_traces, n_windows, n_freq]
    phases: NDArray[np.float32]          # [n_traces, n_windows, n_freq]
    frequencies: NDArray[np.float32]     # [n_freq]
    windows: list[tuple[float, float]]   # [(start_ms, end_ms), ...]


@dataclass
class DecompositionResult:
    """Result from surface-consistent decomposition."""
    source_amplitudes: NDArray[np.float32]    # [n_sources, n_freq]
    source_phases: NDArray[np.float32]
    receiver_amplitudes: NDArray[np.float32]  # [n_receivers, n_freq]
    receiver_phases: NDArray[np.float32]
    offset_amplitudes: NDArray[np.float32] | None
    residuals: NDArray[np.float32]
    convergence_history: list[float]


class SurfaceConsistentMetalKernel:
    """
    GPU-accelerated surface-consistent processing using Metal compute shaders.
    """

    METALLIB_PATH = Path(__file__).parent.parent / "metal" / "surface_consistent.metallib"

    def __init__(self):
        if not METAL_AVAILABLE:
            raise RuntimeError("Metal not available")

        self._device = Metal.MTLCreateSystemDefaultDevice()
        self._command_queue = self._device.newCommandQueue()
        self._library = None
        self._pipelines = {}
        self._load_library()

    def _load_library(self):
        """Load compiled Metal library."""
        if not self.METALLIB_PATH.exists():
            raise RuntimeError(f"Metal library not found: {self.METALLIB_PATH}")

        error = None
        self._library, error = self._device.newLibraryWithURL_error_(
            Foundation.NSURL.fileURLWithPath_(str(self.METALLIB_PATH)),
            None
        )
        if error:
            raise RuntimeError(f"Failed to load Metal library: {error}")

        # Pre-create pipeline states for each kernel
        kernel_names = [
            "compute_spectra_windowed",
            "solve_surface_consistent_irls",
            "apply_equalizer_realtime",
            "apply_equalizer_fft",
            "apply_sc_correction",
        ]

        for name in kernel_names:
            func = self._library.newFunctionWithName_(name)
            if func:
                pipeline, error = self._device.newComputePipelineStateWithFunction_error_(
                    func, None
                )
                if not error:
                    self._pipelines[name] = pipeline

    def estimate_spectra(
        self,
        traces: NDArray[np.float32],
        windows: list[tuple[float, float]],
        dt_ms: float,
        fft_size: int = 1024,
        taper_type: str = "tukey",
        taper_fraction: float = 0.1,
    ) -> SpectraResult:
        """
        Estimate amplitude and phase spectra for all traces in user-defined windows.

        Optimized for M4 Max GPU with parallel FFT computation.
        """
        n_traces, n_samples = traces.shape
        n_windows = len(windows)
        n_freq = fft_size // 2 + 1

        # For production, use Metal Performance Shaders (MPS) for FFT
        # MPS provides highly optimized FFT implementation
        # from metalperformanceshaders import MPSFFTDescriptor, MPSRealToComplexFFT

        # ... implementation using MPS FFT ...

        # Placeholder return
        return SpectraResult(
            amplitudes=np.zeros((n_traces, n_windows, n_freq), dtype=np.float32),
            phases=np.zeros((n_traces, n_windows, n_freq), dtype=np.float32),
            frequencies=np.fft.rfftfreq(fft_size, dt_ms / 1000),
            windows=windows,
        )

    def decompose_factors(
        self,
        spectra: SpectraResult,
        source_indices: NDArray[np.int32],
        receiver_indices: NDArray[np.int32],
        offset_bin_indices: NDArray[np.int32] | None = None,
        solver: str = "l1_irls",
        max_iterations: int = 50,
        regularization: float = 0.01,
    ) -> DecompositionResult:
        """
        Decompose spectra into surface-consistent factors using GPU-accelerated solver.
        """
        n_traces = spectra.amplitudes.shape[0]
        n_freq = spectra.amplitudes.shape[2]
        n_sources = source_indices.max() + 1
        n_receivers = receiver_indices.max() + 1

        # Build sparse system and solve on GPU
        # ... implementation ...

        return DecompositionResult(
            source_amplitudes=np.zeros((n_sources, n_freq), dtype=np.float32),
            source_phases=np.zeros((n_sources, n_freq), dtype=np.float32),
            receiver_amplitudes=np.zeros((n_receivers, n_freq), dtype=np.float32),
            receiver_phases=np.zeros((n_receivers, n_freq), dtype=np.float32),
            offset_amplitudes=None,
            residuals=np.zeros(n_traces, dtype=np.float32),
            convergence_history=[],
        )

    def apply_equalizer_preview(
        self,
        traces: NDArray[np.float32],
        eq_bands: list[dict],
        dt_ms: float,
    ) -> NDArray[np.float32]:
        """
        Apply equalizer with real-time preview (low latency).

        Uses time-domain biquad filters for instant feedback.
        """
        # ... GPU implementation ...
        return traces.copy()

    def apply_corrections(
        self,
        traces: NDArray[np.float32],
        source_indices: NDArray[np.int32],
        receiver_indices: NDArray[np.int32],
        decomposition: DecompositionResult,
        dt_ms: float,
        apply_amplitude: bool = True,
        apply_phase: bool = False,
        water_level_db: float = -40.0,
    ) -> NDArray[np.float32]:
        """
        Apply surface-consistent corrections to trace data.
        """
        # ... GPU implementation ...
        return traces.copy()
```

---

## 8. Key Performance Optimizations

### 8.1 M4 Max GPU Utilization

```python
# Optimal threadgroup sizes for M4 Max
THREADGROUP_SIZE_1D = 256      # For 1D operations
THREADGROUP_SIZE_2D = (16, 16) # For 2D operations
THREADGROUP_SIZE_3D = (8, 8, 4) # For 3D operations

# M4 Max has 40 GPU cores, ~128 execution units
# Target occupancy: keep all SIMDs busy
MAX_THREADS_PER_THREADGROUP = 1024
SIMD_WIDTH = 32  # Apple GPU SIMD width
```

### 8.2 Memory Hierarchy

```python
# Buffer allocation strategy
class GPUBufferPool:
    """Reusable GPU buffer pool to minimize allocation overhead."""

    def __init__(self, device):
        self._device = device
        self._pools = {}  # size -> [available_buffers]

    def get_buffer(self, size: int, mode=Metal.MTLResourceStorageModeShared):
        """Get buffer from pool or create new."""
        # Round up to power of 2 for efficient pooling
        pool_size = 1 << (size - 1).bit_length()

        if pool_size in self._pools and self._pools[pool_size]:
            return self._pools[pool_size].pop()

        return self._device.newBufferWithLength_options_(pool_size, mode)

    def return_buffer(self, buffer):
        """Return buffer to pool for reuse."""
        size = buffer.length()
        if size not in self._pools:
            self._pools[size] = []
        self._pools[size].append(buffer)
```

### 8.3 Double-Buffered Streaming

```python
async def process_with_streaming(
    input_zarr: zarr.Array,
    output_zarr: zarr.Array,
    kernel: SurfaceConsistentMetalKernel,
    chunk_size: int = 50000,
):
    """
    Process large dataset with double-buffered GPU/CPU overlap.
    """
    n_traces = input_zarr.shape[0]

    # Two buffers for ping-pong
    buffer_a = np.empty((chunk_size, input_zarr.shape[1]), dtype=np.float32)
    buffer_b = np.empty((chunk_size, input_zarr.shape[1]), dtype=np.float32)

    current_buffer = buffer_a
    next_buffer = buffer_b

    for start in range(0, n_traces, chunk_size):
        end = min(start + chunk_size, n_traces)
        actual_size = end - start

        # Start async read for next chunk while processing current
        if start + chunk_size < n_traces:
            asyncio.create_task(
                async_read_zarr(input_zarr, start + chunk_size, next_buffer)
            )

        # Process current chunk on GPU
        result = kernel.apply_corrections(
            current_buffer[:actual_size],
            # ... other params ...
        )

        # Write result (can also be async)
        output_zarr[start:end] = result

        # Swap buffers
        current_buffer, next_buffer = next_buffer, current_buffer
```

---

## 9. Implementation Roadmap

### Phase 1: Core Infrastructure
- [ ] Configuration models (`pstm/config/surface_consistent.py`)
- [ ] Data structures and factor indexing
- [ ] Basic Metal shader compilation setup

### Phase 2: Spectra Estimation
- [ ] Metal FFT kernel using MPS
- [ ] Windowed spectra computation
- [ ] GPU buffer management

### Phase 3: Decomposition Engine
- [ ] Sparse matrix builder
- [ ] IRLS solver (L1 norm)
- [ ] Phase estimation (minimum phase + bispectrum)

### Phase 4: UI Steps 1-4
- [ ] Input step with header mapping
- [ ] Spectra estimation step
- [ ] Factor decomposition step
- [ ] QC maps visualization

### Phase 5: Seismic Equalizer
- [ ] Parametric EQ UI with sliders
- [ ] Real-time preview kernel
- [ ] Filter presets system

### Phase 6: Filter Test & Production
- [ ] Before/after comparison viewer
- [ ] Production application kernel
- [ ] Checkpoint/resume support

### Phase 7: Polish & Optimization
- [ ] Performance profiling
- [ ] Memory optimization
- [ ] Documentation

---

## 10. References

### Classical Surface-Consistent Processing
- [Taner & Koehler, 1981 - Surface Consistent Corrections](https://library.seg.org/doi/10.1190/1.1441133)
- [Cambois & Stoffa, 1992 - Log/Fourier Domain](https://pubs.geoscienceworld.org/seg/geophysics/article/57/6/823/72743)
- [SEG Wiki - Surface-Consistent Deconvolution](https://wiki.seg.org/wiki/Surface-consistent_deconvolution)

### Modern Robust Methods
- [L1 Norm Trust Region](https://ui.adsabs.harvard.edu/abs/2014JGE....11d5010C)
- [Reweighted L1-Norm Sparse Constraint](https://ieeexplore.ieee.org/document/9758817/)
- [GMC Penalty for Robust Compensation](https://academic.oup.com/jge/article/20/5/1054/7259153)

### Phase Estimation
- [Mixed-Phase Wavelet Estimation](https://csegrecorder.com/articles/view/mixed-phase-wavelet-estimation-a-case-study)
- [Bispectrum Method (EAGE 2020)](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202010410)
- [Homomorphic Wavelet Estimation](https://arxiv.org/abs/1205.3752)

### GPU Seismic Processing
- [NVIDIA GPU Gems - Seismic Imaging](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-38-imaging-earths-subsurface-using-cuda)
- [Apple Metal Documentation](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)

### Recent Research (2024)
- [A New Perspective of Surface Consistent Deconvolution - SEG IMAGE 2024](https://library.seg.org/doi/10.1190/image2024-4095553.1)
- [Surface-Consistent Amplitude Correction via Waveform Modeling](https://www.earthdoc.org/content/papers/10.3997/2214-4609.201801108)

---

## Summary

This architecture provides a complete blueprint for implementing surface-consistent seismic processing with:

1. **Modern math**: L1-norm robust solvers, higher-order statistics for phase
2. **GPU acceleration**: Metal compute shaders optimized for M4 Max
3. **Memory efficiency**: Tiled processing for datasets exceeding RAM
4. **Intuitive UI**: Audio-style equalizer for spectral shaping with real-time preview
5. **Production quality**: Checkpoint/resume, QC outputs, logging
