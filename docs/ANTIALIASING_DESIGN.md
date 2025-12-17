# Anti-Aliasing Filter Design for Kirchhoff PSTM

## 1. Problem Statement

In Kirchhoff migration, we sum seismic amplitudes along diffraction curves (traveltime surfaces). At steep dips, the spatial sampling of the input data becomes insufficient to properly sample the migration operator, causing **aliasing artifacts** - false events that appear as "smiles" or noise in the migrated image.

### The Aliasing Condition

For a given dip angle θ, the maximum unaliased frequency is:

```
f_max = v / (4 * Δx * sin(θ))
```

Where:
- `v` = local velocity (m/s)
- `Δx` = midpoint spacing (m)
- `θ` = local dip angle of the diffraction curve

At steep dips (large θ), f_max decreases, meaning high frequencies will alias.

## 2. Anti-Aliasing Strategy

### 2.1 Triangle Filter Method (Recommended)

The triangle filter is a computationally efficient approach that:
1. Computes the **local dip** at each output point for each input trace
2. Determines the **maximum unaliased frequency** based on that dip
3. Applies a **low-pass filter** that smoothly attenuates frequencies above f_max

#### Local Dip Calculation

For PSTM, the traveltime from source S to output point P to receiver R is:

```
t(x_m, h) = sqrt(t0² + 4h²/v²)
```

Where:
- `t0` = two-way vertical time at output point
- `h` = half-offset
- `x_m` = midpoint position
- `v` = RMS velocity

The local dip (dt/dx_m) at the output point with respect to midpoint position:

```
dt/dx_m = (x_m - x_out) / (v² * t / 2)
```

Where `t` is the total traveltime and `x_out` is the output point's x-coordinate.

The dip angle:
```
sin(θ) = |dt/dx_m| * v / 2
```

#### Filter Response

The triangle filter response for frequency f:

```
W(f) = max(0, 1 - f/f_max)    for f <= f_max
W(f) = 0                       for f > f_max
```

This creates a linear taper from full amplitude at DC to zero at f_max.

### 2.2 Implementation in Time Domain

Rather than filtering each trace in the frequency domain (expensive), we use **amplitude weighting** based on the instantaneous frequency content:

**Method 1: Analytical Weighting (Fast)**
```python
# For each trace contribution at output point:
dip = compute_local_dip(output_point, trace_geometry, velocity, traveltime)
f_max = velocity / (4 * dx * abs(sin(dip_angle)))
f_max = min(f_max, nyquist_frequency)

# Weight based on dominant frequency of wavelet
weight = max(0, 1 - f_dominant / f_max)
amplitude_contribution *= weight
```

**Method 2: Multi-Filter Bank (More Accurate)**
Pre-compute filtered versions of each trace at multiple cutoff frequencies:
```python
# Pre-filter traces at N frequency bands
filtered_traces = []
for f_cut in [10, 20, 30, 40, 50, 60, 70, 80]:  # Hz
    filtered_traces.append(lowpass_filter(trace, f_cut))

# During migration, select appropriate filtered trace based on local f_max
idx = find_filter_index(f_max, filter_frequencies)
amplitude = interpolate_filtered_traces(filtered_traces, idx, sample_time)
```

### 2.3 2D vs 3D Considerations

For 3D PSTM, we have dip in both inline and crossline directions:

```
sin²(θ_total) = sin²(θ_inline) + sin²(θ_crossline)
```

The maximum frequency must satisfy both directions:
```
f_max = v / (4 * max(Δx*|sin(θ_x)|, Δy*|sin(θ_y)|))
```

## 3. Detailed Algorithm

### 3.1 Pre-Migration Setup

```python
class AntiAliasingFilter:
    def __init__(self, config: AntiAliasingConfig, dt_ms: float, dx: float, dy: float):
        self.enabled = config.enabled
        self.method = config.method
        self.num_filters = config.num_filters  # e.g., 32
        self.f_min = config.min_frequency_hz   # e.g., 5 Hz
        self.f_max = config.max_frequency_hz   # e.g., 80 Hz (Nyquist)
        self.dt = dt_ms / 1000.0
        self.dx = dx
        self.dy = dy

        # Pre-compute filter bank frequencies
        self.filter_freqs = np.linspace(self.f_min, self.f_max, self.num_filters)

    def prefilter_traces(self, traces: np.ndarray) -> np.ndarray:
        """Create multi-frequency filtered versions of input traces."""
        n_traces, n_samples = traces.shape
        filtered = np.zeros((self.num_filters, n_traces, n_samples))

        for i, f_cut in enumerate(self.filter_freqs):
            filtered[i] = apply_lowpass(traces, f_cut, self.dt)

        return filtered
```

### 3.2 Per-Sample Migration with AA

```python
def migrate_sample_with_aa(
    output_x, output_y, output_t,
    trace_midpoint_x, trace_midpoint_y,
    trace_offset, velocity,
    filtered_traces,  # (num_filters, n_samples)
    filter_freqs,
    dx, dy
):
    # 1. Compute traveltime
    t_mig = compute_traveltime(output_x, output_y, output_t,
                                trace_midpoint_x, trace_midpoint_y,
                                trace_offset, velocity)

    # 2. Compute local dip in x and y directions
    dip_x = compute_dip_x(output_x, trace_midpoint_x, velocity, t_mig)
    dip_y = compute_dip_y(output_y, trace_midpoint_y, velocity, t_mig)

    # 3. Compute maximum unaliased frequency
    sin_theta_x = abs(dip_x * velocity / 2)
    sin_theta_y = abs(dip_y * velocity / 2)

    # Avoid division by zero for zero dip
    sin_theta_x = max(sin_theta_x, 0.01)
    sin_theta_y = max(sin_theta_y, 0.01)

    f_max_x = velocity / (4 * dx * sin_theta_x)
    f_max_y = velocity / (4 * dy * sin_theta_y)
    f_max = min(f_max_x, f_max_y, nyquist)

    # 4. Select and interpolate from filter bank
    filter_idx = np.searchsorted(filter_freqs, f_max)
    filter_idx = np.clip(filter_idx, 0, len(filter_freqs) - 1)

    # 5. Get amplitude from appropriate filtered trace
    sample_idx = t_mig / dt
    amplitude = interpolate_trace(filtered_traces[filter_idx], sample_idx)

    return amplitude
```

### 3.3 Optimized Numba Implementation

```python
@numba.njit(parallel=True, fastmath=True)
def migrate_tile_with_aa(
    output_image,      # (nx, ny, nt)
    traces,            # (n_traces, n_samples)
    filtered_traces,   # (n_filters, n_traces, n_samples)
    filter_freqs,      # (n_filters,)
    midpoint_x,        # (n_traces,)
    midpoint_y,        # (n_traces,)
    offset,            # (n_traces,)
    output_x,          # (nx,)
    output_y,          # (ny,)
    output_t,          # (nt,)
    velocity,          # (nt,) or scalar
    dx, dy, dt,
    aperture_max,
):
    nx, ny, nt = output_image.shape
    n_traces = traces.shape[0]
    n_filters = len(filter_freqs)
    nyquist = 0.5 / dt

    for ix in numba.prange(nx):
        for iy in range(ny):
            ox = output_x[ix]
            oy = output_y[iy]

            for it in range(nt):
                ot = output_t[it]
                vel = velocity[it] if velocity.ndim > 0 else velocity

                accum = 0.0

                for itrace in range(n_traces):
                    mx = midpoint_x[itrace]
                    my = midpoint_y[itrace]
                    h = offset[itrace] / 2.0

                    # Check aperture
                    dist = sqrt((ox - mx)**2 + (oy - my)**2)
                    if dist > aperture_max:
                        continue

                    # Compute traveltime
                    t0 = ot  # Two-way vertical time
                    t_mig = sqrt(t0**2 + 4*h**2/vel**2)

                    # Compute local dip
                    if t_mig > 1e-6:
                        dip_x = (mx - ox) / (vel**2 * t_mig / 2)
                        dip_y = (my - oy) / (vel**2 * t_mig / 2)
                    else:
                        dip_x = 0.0
                        dip_y = 0.0

                    # Compute f_max from dip
                    sin_theta_x = min(abs(dip_x * vel / 2), 1.0)
                    sin_theta_y = min(abs(dip_y * vel / 2), 1.0)

                    if sin_theta_x > 0.01:
                        f_max_x = vel / (4 * dx * sin_theta_x)
                    else:
                        f_max_x = nyquist

                    if sin_theta_y > 0.01:
                        f_max_y = vel / (4 * dy * sin_theta_y)
                    else:
                        f_max_y = nyquist

                    f_max = min(f_max_x, f_max_y, nyquist)

                    # Find filter index
                    filter_idx = 0
                    for fi in range(n_filters):
                        if filter_freqs[fi] <= f_max:
                            filter_idx = fi
                        else:
                            break

                    # Interpolate from filtered trace
                    sample_f = t_mig / dt
                    sample_i = int(sample_f)
                    frac = sample_f - sample_i

                    if 0 <= sample_i < traces.shape[1] - 1:
                        amp = (1-frac) * filtered_traces[filter_idx, itrace, sample_i] + \
                              frac * filtered_traces[filter_idx, itrace, sample_i + 1]
                        accum += amp

                output_image[ix, iy, it] = accum
```

## 4. Memory Considerations

### Filter Bank Memory

For the multi-filter approach:
```
Memory = n_filters × n_traces × n_samples × 4 bytes (float32)

Example:
- 32 filters × 100,000 traces × 2000 samples × 4 bytes = 25.6 GB
```

This is too much! Solutions:

**Option A: On-the-fly filtering per tile**
- Filter only traces needed for current tile
- Memory: n_filters × traces_per_tile × n_samples

**Option B: Analytical weight method**
- No pre-filtering needed
- Apply weight based on dominant frequency estimate
- Memory: Same as original (just traces)

**Option C: Hybrid approach**
- Use fewer filters (8-16 instead of 32)
- Apply per-tile
- Interpolate between adjacent filters

## 5. Configuration

```python
class AntiAliasingConfig:
    enabled: bool = False
    method: AntiAliasingMethod = AntiAliasingMethod.TRIANGLE

    # Filter bank parameters
    num_filters: int = 16           # Number of frequency bands
    min_frequency_hz: float = 5.0   # Lowest filter cutoff
    max_frequency_hz: float = 80.0  # Highest filter cutoff (usually Nyquist)

    # Performance tuning
    per_tile_filtering: bool = True  # Filter traces per-tile to save memory

    # Advanced options
    dominant_frequency_hz: float = 30.0  # For analytical weight method
    taper_type: str = "triangle"         # "triangle", "cosine", "gaussian"
```

## 6. UI Integration

### Algorithm Step
Add anti-aliasing panel:
```
[x] Enable Anti-Aliasing
    Method: [Triangle Filter ▼]
    Number of filters: [16]
    Min frequency (Hz): [5.0]
    Max frequency (Hz): [Auto from data]
    [ ] Show AA diagnostics in output
```

### Visualization Step
Add AA diagnostic view:
- Show f_max map at selected time slice
- Overlay AA weight distribution
- Show before/after comparison for QC

## 7. Testing Strategy

1. **Synthetic dipping reflector test**
   - Create synthetic with known dip
   - Compare aliased vs anti-aliased migration
   - Verify correct f_max calculation

2. **Point diffractor test**
   - Steep dips at far offsets should be attenuated
   - Compare hyperbola tails with/without AA

3. **Real data test**
   - Complex structure with steep dips
   - Visual comparison of aliasing artifacts

## 8. Performance Estimates

| Method | Memory Overhead | Compute Overhead |
|--------|-----------------|------------------|
| No AA (baseline) | 0% | 0% |
| Analytical weight | ~0% | +5-10% |
| Filter bank (16) | +1600% per tile | +50-100% |
| Filter bank (8) | +800% per tile | +30-50% |

Recommendation: Start with **analytical weight method** for minimal overhead, add filter bank as advanced option.
