# Advanced Migration Kernels: Curved Ray & Anisotropic Eta

## Executive Summary

This document outlines the implementation plan for two advanced migration kernel variants:
1. **Curved Ray Migration** - accounts for velocity gradients causing ray bending
2. **Anisotropic VTI Migration** - handles vertically transversely isotropic media using the eta parameter

Both kernels will be implemented as Metal compute shaders following the existing `metal_compiled` architecture.

---

## Part 1: Physics Background

### 1.1 Current Implementation: Straight Ray DSR

The current PSTM kernel uses the **Double Square Root (DSR)** equation assuming straight ray paths:

```
t_total = sqrt(t_0^2 + (x_s - x_m)^2/V^2) + sqrt(t_0^2 + (x_r - x_m)^2/V^2)
```

Where:
- `t_0` = zero-offset two-way time to image point
- `x_s`, `x_r` = source and receiver positions
- `x_m` = midpoint (image point) position
- `V` = RMS velocity (assumed constant or slowly varying)

**Limitation**: Assumes rays travel in straight lines, which is only valid for:
- Constant velocity media
- Small offsets relative to depth
- Weak velocity gradients

### 1.2 Curved Ray Migration

**Physical Basis**: In real Earth, velocity typically increases with depth (compaction, pressure). This causes seismic rays to bend according to Snell's Law, following curved paths rather than straight lines.

**V(z) Linear Gradient Model**: `V(z) = V_0 + k * z`
- `V_0` = velocity at surface (m/s)
- `k` = velocity gradient (1/s or m/s per m)
- Rays follow **circular arcs** in this model

**Curved Ray Traveltime Equation** (for linear V(z)):

For a ray from (0, 0) to (x, z) in a medium with V = V_0 + k*z:
```
t = (1/k) * ln[(V_0 + k*z + sqrt((V_0 + k*z)^2 + k^2*x^2)) / (V_0 + sqrt(V_0^2 + k^2*x^2))]
```

Or equivalently using ray parameter p:
```
t = (2/k) * arcsinh(k * x / (2 * V_0 * cos(theta)))
```

**Benefits**:
- More accurate imaging at far offsets (offset > depth)
- Better velocity analysis in areas with strong gradients
- Improved depth conversion accuracy
- Essential for deep targets with long-offset data

**References**:
- Slotnick (1959) - Original curved ray theory
- [A Practical Approach of Curved Ray Prestack Kirchhoff Time Migration on GPGPU](https://link.springer.com/chapter/10.1007/978-3-642-03644-6_13)

### 1.3 Anisotropic VTI Migration (Eta Parameter)

**Physical Basis**: Many sedimentary rocks (shales, layered sequences) exhibit **Vertical Transverse Isotropy (VTI)** - velocities differ between vertical and horizontal propagation directions.

**Thomsen Parameters** (for weak anisotropy):
- `ε (epsilon)` = fractional difference between horizontal and vertical P-wave velocity
- `δ (delta)` = controls near-vertical propagation and NMO velocity
- `γ (gamma)` = S-wave anisotropy (not used in P-wave processing)

**The Eta Parameter** (Alkhalifah & Tsvankin, 1995):
```
η = (ε - δ) / (1 + 2δ)
```

For weak anisotropy: `η ≈ ε - δ`

**Key Insight**: Time processing (NMO, DMO, PSTM) depends only on:
1. `V_nmo` - NMO velocity for horizontal reflector
2. `η` - anisotropy parameter

This means we do NOT need to know V_0 (vertical velocity) for time imaging!

**VTI Traveltime Equation** (Alkhalifah, 1998):
```
t²(x) = t_0² + x²/V_nmo² - (2η * x⁴) / [V_nmo² * (t_0² * V_nmo² + (1 + 2η) * x²)]
```

This is the **non-hyperbolic moveout equation** accounting for:
- Standard hyperbolic moveout (first two terms)
- Fourth-order correction for anisotropy (third term)

**For PSTM**, the DSR equation becomes:
```
t_total = sqrt(t_0² + h_s²/V² - C_η(h_s)) + sqrt(t_0² + h_r²/V² - C_η(h_r))
```

Where `C_η` is the anisotropic correction term.

**Benefits**:
- Correct focusing in anisotropic sedimentary basins
- Better flat-spot imaging
- Improved AVO/AVA fidelity
- Essential for shale-dominated sequences

**References**:
- [Alkhalifah & Tsvankin (1995) - Velocity analysis for transversely isotropic media](https://pubs.geoscienceworld.org/seg/geophysics/article-abstract/60/5/1550/106912/Velocity-analysis-for-transversely-isotropic-media)
- [Alkhalifah (1997) - VTI eikonal equation](https://library.seg.org/doi/abs/10.1190/1.1443888)

---

## Part 2: Code Design & Naming Conventions

### 2.1 New Enums and Types

```python
# In pstm/config/models.py

class MigrationKernelType(str, Enum):
    """Type of migration traveltime computation."""
    STRAIGHT_RAY = "straight_ray"      # Current DSR (default)
    CURVED_RAY = "curved_ray"          # V(z) gradient model
    ANISOTROPIC_VTI = "anisotropic_vti"  # Alkhalifah-Tsvankin eta
    CURVED_RAY_VTI = "curved_ray_vti"  # Combined (future)
```

### 2.2 New Configuration Classes

```python
# In pstm/config/models.py

class CurvedRayConfig(BaseConfig):
    """Configuration for curved ray migration."""

    enabled: bool = Field(
        default=False,
        description="Enable curved ray traveltime computation"
    )
    gradient_source: Literal["from_velocity", "manual"] = Field(
        default="from_velocity",
        description="Source of velocity gradient"
    )
    # Manual gradient specification
    v0_m_s: PositiveFloat | None = Field(
        default=None,
        description="Surface velocity V_0 (m/s)"
    )
    k_per_s: float | None = Field(
        default=None,
        description="Velocity gradient k (1/s), typically 0.3-0.6"
    )
    # From velocity model
    gradient_estimation_window_ms: PositiveFloat = Field(
        default=500.0,
        description="Time window for gradient estimation (ms)"
    )


class AnisotropyVTIConfig(BaseConfig):
    """Configuration for VTI anisotropic migration."""

    enabled: bool = Field(
        default=False,
        description="Enable VTI anisotropic migration"
    )
    eta_source: Literal["constant", "table_1d", "cube_3d"] = Field(
        default="constant",
        description="Source of eta values"
    )
    # Constant eta
    eta_constant: float = Field(
        default=0.1,
        ge=-0.5,
        le=0.5,
        description="Constant eta value (typical range: 0.05-0.2)"
    )
    # 1D eta table: (time_ms, eta) pairs
    eta_table: list[tuple[float, float]] | None = Field(
        default=None,
        description="1D eta vs time table"
    )
    # 3D eta cube
    eta_cube_path: Path | None = Field(
        default=None,
        description="Path to 3D eta cube (Zarr)"
    )
    # Estimation parameters
    estimation_offset_min_m: PositiveFloat = Field(
        default=2000.0,
        description="Minimum offset for eta estimation (far offsets)"
    )
```

### 2.3 Updated AlgorithmConfig

```python
class AlgorithmConfig(BaseConfig):
    """Configuration for migration algorithm parameters."""

    # Kernel type selection
    kernel_type: MigrationKernelType = Field(
        default=MigrationKernelType.STRAIGHT_RAY,
        description="Migration kernel traveltime model"
    )

    # Existing fields...
    interpolation: InterpolationMethod = ...
    aperture: ApertureConfig = ...
    anti_aliasing: AntiAliasingConfig = ...
    amplitude: AmplitudeConfig = ...
    mute: MuteConfig = ...
    time_variant: TimeVariantConfig = ...

    # New fields
    curved_ray: CurvedRayConfig = Field(default_factory=CurvedRayConfig)
    anisotropy_vti: AnisotropyVTIConfig = Field(default_factory=AnisotropyVTIConfig)
```

### 2.4 Metal Shader Naming

```
pstm/metal/shaders/
├── pstm_migration.metal          # Current straight-ray kernel
├── pstm_curved_ray.metal         # NEW: Curved ray kernel
├── pstm_anisotropic_vti.metal    # NEW: VTI anisotropic kernel
└── pstm_common.metal             # Shared utilities (interpolation, etc.)
```

### 2.5 Kernel Class Hierarchy

```
pstm/kernels/
├── base.py                       # KernelConfig, MigrationKernel protocol
├── metal_compiled.py             # Current CompiledMetalKernel
├── metal_curved_ray.py           # NEW: CurvedRayMetalKernel
├── metal_anisotropic.py          # NEW: AnisotropicVTIMetalKernel
└── factory.py                    # Updated to handle kernel type selection
```

### 2.6 GUI State Classes

```python
# In pstm/gui/state.py

@dataclass
class CurvedRayState:
    """Curved ray configuration state."""
    enabled: bool = False
    gradient_source: str = "from_velocity"
    v0_m_s: float = 1500.0
    k_per_s: float = 0.5
    gradient_estimation_window_ms: float = 500.0


@dataclass
class AnisotropyVTIState:
    """VTI anisotropy configuration state."""
    enabled: bool = False
    eta_source: str = "constant"
    eta_constant: float = 0.1
    eta_table: list = field(default_factory=lambda: [
        (0.0, 0.05),
        (2000.0, 0.10),
        (4000.0, 0.15),
    ])
    eta_cube_path: str = ""


@dataclass
class AlgorithmState:
    """Migration algorithm parameters."""
    # Existing fields...

    # New fields
    kernel_type: str = "straight_ray"
    curved_ray: CurvedRayState = field(default_factory=CurvedRayState)
    anisotropy_vti: AnisotropyVTIState = field(default_factory=AnisotropyVTIState)
```

---

## Part 3: Implementation Phases

### Phase 1: Core Algorithm & Data Structures (1.1-1.6)

| Task | Description |
|------|-------------|
| 1.1 | Add `MigrationKernelType` enum to `pstm/config/models.py` |
| 1.2 | Create `CurvedRayConfig` class with validation |
| 1.3 | Create `AnisotropyVTIConfig` class with validation |
| 1.4 | Add new fields to `AlgorithmConfig` |
| 1.5 | Create `pstm/algorithm/curved_ray.py` with traveltime functions |
| 1.6 | Create `pstm/algorithm/anisotropy_vti.py` with VTI traveltime functions |

**Key Functions in `curved_ray.py`**:
```python
def curved_ray_traveltime(x: float, z: float, v0: float, k: float) -> float:
    """Compute traveltime for curved ray in V(z) = V0 + k*z medium."""

def estimate_gradient_from_velocity(vrms: NDArray, t_ms: NDArray) -> tuple[float, float]:
    """Estimate V0 and k from RMS velocity function."""

def straight_to_curved_correction(t_straight: float, offset: float, v0: float, k: float) -> float:
    """Compute correction factor from straight to curved ray."""
```

**Key Functions in `anisotropy_vti.py`**:
```python
def vti_traveltime_correction(t0: float, x: float, v_nmo: float, eta: float) -> float:
    """Compute VTI anisotropic traveltime correction term."""

def vti_dsr_traveltime(t0: float, h_s: float, h_r: float, v: float, eta: float) -> float:
    """DSR traveltime with VTI correction."""

def interpolate_eta(t_ms: float, eta_table: list) -> float:
    """Interpolate eta from 1D table."""
```

### Phase 2: Metal Shaders (2.1-2.4)

| Task | Description |
|------|-------------|
| 2.1 | Create `pstm_common.metal` with shared interpolation functions |
| 2.2 | Create `pstm_curved_ray.metal` kernel |
| 2.3 | Create `pstm_anisotropic_vti.metal` kernel |
| 2.4 | Update `scripts/build_metal.sh` to compile new shaders |

**Curved Ray Metal Kernel Structure**:
```metal
// pstm_curved_ray.metal

struct CurvedRayParams {
    float v0;           // Surface velocity
    float k;            // Gradient (1/s)
    float dx, dy, dt_ms;
    // ... other params
};

// Curved ray traveltime computation
inline float curved_ray_time(float x, float z, float v0, float k) {
    // Circular arc ray path in V(z) = v0 + k*z
    float v_z = v0 + k * z;
    float term1 = v_z + sqrt(v_z * v_z + k * k * x * x);
    float term2 = v0 + sqrt(v0 * v0 + k * k * x * x);
    return (1.0f / k) * log(term1 / term2);
}

kernel void pstm_migrate_curved_ray(
    device const float* traces [[buffer(0)]],
    device float* output [[buffer(1)]],
    // ... other buffers
    constant CurvedRayParams& params [[buffer(N)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Similar structure to existing kernel but using curved_ray_time()
}
```

**VTI Anisotropic Metal Kernel Structure**:
```metal
// pstm_anisotropic_vti.metal

struct VTIParams {
    float eta;          // Anisotropy parameter (or index into eta array)
    int eta_is_1d;      // Whether eta varies with time
    float dx, dy, dt_ms;
    // ... other params
};

// VTI correction term
inline float vti_correction(float t0, float x, float v, float eta) {
    float t0v = t0 * v;
    float x2 = x * x;
    float denom = t0v * t0v + (1.0f + 2.0f * eta) * x2;
    return (2.0f * eta * x2 * x2) / (v * v * denom);
}

// VTI-corrected traveltime
inline float vti_traveltime(float t0, float x, float v, float eta) {
    float t2_hyper = t0 * t0 + x * x / (v * v);
    float correction = vti_correction(t0, x, v, eta);
    return sqrt(max(t2_hyper - correction, 0.0f));
}

kernel void pstm_migrate_vti(
    device const float* traces [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* eta_array [[buffer(N)]],  // 1D or 3D eta
    constant VTIParams& params [[buffer(M)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // DSR with VTI corrections on both source and receiver legs
}
```

### Phase 3: Python Kernel Wrappers (3.1-3.4)

| Task | Description |
|------|-------------|
| 3.1 | Create `pstm/kernels/metal_curved_ray.py` wrapper class |
| 3.2 | Create `pstm/kernels/metal_anisotropic.py` wrapper class |
| 3.3 | Update `pstm/kernels/base.py` with new config fields |
| 3.4 | Update `pstm/kernels/factory.py` for kernel type selection |

**Updated KernelConfig**:
```python
@dataclass
class KernelConfig:
    # Existing fields...
    max_aperture_m: float | None = None
    min_aperture_m: float | None = None
    # ...

    # New fields for advanced kernels
    kernel_type: str = "straight_ray"

    # Curved ray parameters
    curved_ray_enabled: bool = False
    curved_ray_v0: float = 1500.0
    curved_ray_k: float = 0.5

    # VTI anisotropy parameters
    vti_enabled: bool = False
    vti_eta_constant: float = 0.0
    vti_eta_array: NDArray | None = None  # 1D or 3D
    vti_eta_is_1d: bool = True
```

**Updated Factory**:
```python
def create_kernel(backend: ComputeBackend | str, kernel_type: str = "straight_ray") -> MigrationKernel:
    """Create kernel with specified backend and type."""

    if kernel_type == "curved_ray":
        if backend == ComputeBackend.METAL_COMPILED:
            from pstm.kernels.metal_curved_ray import CurvedRayMetalKernel
            return CurvedRayMetalKernel()
    elif kernel_type == "anisotropic_vti":
        if backend == ComputeBackend.METAL_COMPILED:
            from pstm.kernels.metal_anisotropic import AnisotropicVTIMetalKernel
            return AnisotropicVTIMetalKernel()

    # Default: straight ray
    return _create_standard_kernel(backend)
```

### Phase 4: Executor Integration (4.1-4.3)

| Task | Description |
|------|-------------|
| 4.1 | Update `pstm/pipeline/executor.py` to read kernel type from config |
| 4.2 | Add gradient estimation logic for curved ray |
| 4.3 | Add eta loading/interpolation for VTI |

**Key Changes in Executor**:
```python
def _migrate_single_bin(self, ...):
    # Determine kernel type
    kernel_type = self.config.algorithm.kernel_type.value

    # Build kernel config based on type
    if kernel_type == "curved_ray":
        cr = self.config.algorithm.curved_ray
        if cr.gradient_source == "from_velocity":
            v0, k = estimate_gradient_from_velocity(velocity.vrms, velocity.t_axis_ms)
        else:
            v0, k = cr.v0_m_s, cr.k_per_s

        kernel_config.curved_ray_enabled = True
        kernel_config.curved_ray_v0 = v0
        kernel_config.curved_ray_k = k

    elif kernel_type == "anisotropic_vti":
        vti = self.config.algorithm.anisotropy_vti
        kernel_config.vti_enabled = True

        if vti.eta_source == "constant":
            kernel_config.vti_eta_constant = vti.eta_constant
        elif vti.eta_source == "table_1d":
            # Interpolate to output time axis
            kernel_config.vti_eta_array = interpolate_eta_table(vti.eta_table, grid.t_axis_ms)
            kernel_config.vti_eta_is_1d = True
        elif vti.eta_source == "cube_3d":
            # Load and resample 3D eta cube
            kernel_config.vti_eta_array = load_eta_cube(vti.eta_cube_path, grid)
            kernel_config.vti_eta_is_1d = False
```

### Phase 5: GUI Integration (5.1-5.6)

| Task | Description |
|------|-------------|
| 5.1 | Add `CurvedRayState` and `AnisotropyVTIState` to `pstm/gui/state.py` |
| 5.2 | Add kernel type selector to algorithm step |
| 5.3 | Create curved ray parameters section in algorithm step |
| 5.4 | Create VTI parameters section with eta table editor |
| 5.5 | Update `build_migration_config()` to pass new parameters |
| 5.6 | Update execution step preflight check |

**UI Layout in Algorithm Step**:
```
┌─────────────────────────────────────────────────┐
│ Traveltime Model                                │
├─────────────────────────────────────────────────┤
│ ○ Straight Ray (Standard DSR)                   │
│ ○ Curved Ray (V(z) Gradient)                    │
│ ○ Anisotropic VTI (Eta Parameter)               │
├─────────────────────────────────────────────────┤
│                                                 │
│ [Curved Ray Section - shown when selected]      │
│ ┌─────────────────────────────────────────────┐ │
│ │ Gradient Source: [From Velocity ▼]          │ │
│ │                                             │ │
│ │ Manual Parameters (optional):               │ │
│ │   V₀ (surface): [1500.0] m/s               │ │
│ │   k (gradient): [0.50] 1/s                 │ │
│ │                                             │ │
│ │ Estimated: V(z) = 1500 + 0.5z m/s          │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ [VTI Anisotropy Section - shown when selected]  │
│ ┌─────────────────────────────────────────────┐ │
│ │ Eta Source: [1D Table ▼]                    │ │
│ │                                             │ │
│ │ ┌──────────┬────────────┐                   │ │
│ │ │ Time(ms) │ Eta        │                   │ │
│ │ ├──────────┼────────────┤                   │ │
│ │ │ 0        │ 0.05       │                   │ │
│ │ │ 2000     │ 0.10       │                   │ │
│ │ │ 4000     │ 0.15       │                   │ │
│ │ └──────────┴────────────┘                   │ │
│ │ [Add] [Remove] [Reset]                      │ │
│ │                                             │ │
│ │ ℹ️ Typical η: 0.05-0.20 for shales          │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### Phase 6: Testing & Validation (6.1-6.5)

| Task | Description |
|------|-------------|
| 6.1 | Create unit tests for curved ray traveltime functions |
| 6.2 | Create unit tests for VTI traveltime functions |
| 6.3 | Create synthetic test with known geometry for curved ray |
| 6.4 | Create synthetic test with known anisotropy for VTI |
| 6.5 | Benchmark performance comparison vs straight ray |

**Test Cases**:

1. **Curved Ray Validation**:
   - Known V(z) = 2000 + 0.5z model
   - Point diffractor at known depth
   - Compare with analytical solution
   - Verify correct ray bending at far offsets

2. **VTI Validation**:
   - Constant η = 0.1 model
   - Flat reflector with long offsets
   - Verify fourth-order moveout correction
   - Compare focused image vs isotropic

---

## Part 4: File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `pstm/algorithm/curved_ray.py` | Curved ray traveltime algorithms |
| `pstm/algorithm/anisotropy_vti.py` | VTI anisotropy algorithms |
| `pstm/metal/shaders/pstm_common.metal` | Shared Metal utilities |
| `pstm/metal/shaders/pstm_curved_ray.metal` | Curved ray Metal kernel |
| `pstm/metal/shaders/pstm_anisotropic_vti.metal` | VTI Metal kernel |
| `pstm/kernels/metal_curved_ray.py` | Curved ray kernel wrapper |
| `pstm/kernels/metal_anisotropic.py` | VTI kernel wrapper |
| `tests/test_curved_ray.py` | Curved ray unit tests |
| `tests/test_anisotropy_vti.py` | VTI unit tests |

### Modified Files

| File | Changes |
|------|---------|
| `pstm/config/models.py` | Add `MigrationKernelType`, `CurvedRayConfig`, `AnisotropyVTIConfig` |
| `pstm/kernels/base.py` | Add new fields to `KernelConfig` |
| `pstm/kernels/factory.py` | Handle kernel type selection |
| `pstm/pipeline/executor.py` | Pass kernel type and parameters |
| `pstm/gui/state.py` | Add `CurvedRayState`, `AnisotropyVTIState` |
| `pstm/gui/steps/algorithm_step.py` | Add kernel type UI sections |
| `pstm/gui/steps/execution_step.py` | Update preflight check |
| `scripts/build_metal.sh` | Compile new shaders |

---

## Part 5: Risk Assessment & Mitigations

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Curved ray singularity at k=0 | Crash | Add epsilon guard, fallback to straight ray |
| VTI correction instability at far offsets | Artifacts | Clamp correction term, add max offset limit |
| Eta estimation from noisy data | Poor focusing | Provide manual override, smoothing options |
| Performance regression | Slower processing | Profile early, optimize critical paths |
| GPU memory for 3D eta cube | OOM | Tile-based loading, lazy evaluation |

### Validation Strategy

1. **Analytical Tests**: Compare with published formulas
2. **Synthetic Tests**: Known geometry with computed traveltimes
3. **Benchmark Tests**: Compare straight ray vs curved/VTI on same data
4. **Real Data Tests**: Validate on field data with known anisotropy

---

## Part 6: Estimated Effort

| Phase | Effort |
|-------|--------|
| Phase 1: Core Algorithm | 2 days |
| Phase 2: Metal Shaders | 3 days |
| Phase 3: Python Wrappers | 2 days |
| Phase 4: Executor Integration | 1 day |
| Phase 5: GUI Integration | 2 days |
| Phase 6: Testing | 2 days |
| **Total** | **~12 days** |

---

## References

1. Alkhalifah, T., & Tsvankin, I. (1995). [Velocity analysis for transversely isotropic media](https://pubs.geoscienceworld.org/seg/geophysics/article-abstract/60/5/1550/106912/Velocity-analysis-for-transversely-isotropic-media). Geophysics, 60(5), 1550-1566.

2. Alkhalifah, T. (1997). Velocity analysis using nonhyperbolic moveout in transversely isotropic media. Geophysics, 62(6), 1839-1854.

3. [The basic components of residual migration in VTI media using anisotropy continuation](https://link.springer.com/article/10.1007/s13202-011-0006-6). Journal of Petroleum Exploration and Production Technology.

4. [A Practical Approach of Curved Ray Prestack Kirchhoff Time Migration on GPGPU](https://link.springer.com/chapter/10.1007/978-3-642-03644-6_13). Springer.

5. [Computing Prestack Kirchhoff Time Migration on General Purpose GPU](https://ieeexplore.ieee.org/document/7231787/). IEEE Conference.

6. [Prestack Kirchhoff time migration for complex media](https://sepwww.stanford.edu/data/media/public/docs/sep97/tariq4.pdf). Stanford Exploration Project.
