# PSTM GPU Architecture Design for Apple Silicon

## Executive Summary

This document outlines a new GPU compute architecture for PSTM that maximizes Apple Silicon (M4 Max) capabilities through:
1. **Metal Compute Shaders** written in C++ with Metal Shading Language
2. **Zero-copy UMA integration** eliminating memory transfers
3. **Proper GPU parallelization** across traces, pillars, and time samples
4. **Python bindings via pybind11** for seamless integration

**Expected Performance**: 10-50x speedup over current MLX implementation, potentially 3-5x over optimized Numba CPU.

---

## Part 1: Apple Silicon Architecture Analysis

### M4 Max GPU Specifications

| Feature | Specification | Implication |
|---------|---------------|-------------|
| GPU Cores | 40 | 40 parallel compute units |
| Execution Units | ~5,120 | Massive parallelism available |
| Memory Bandwidth | 546 GB/s | Not memory-bound for compute |
| Unified Memory | Up to 128 GB | GPU can access all RAM |
| FP32 Performance | ~14 TFLOPS | Theoretical peak |
| FP16 Performance | ~28 TFLOPS | 2x throughput with half precision |
| Threadgroup Size | 1024 threads | Max threads per workgroup |
| SIMD Width | 32 | Threads execute in lockstep |

### Unified Memory Architecture (UMA) Advantages

```
Traditional GPU (NVIDIA/AMD):
┌─────────────┐     PCIe      ┌─────────────┐
│  CPU RAM    │ ──────────────│  GPU VRAM   │
│  (System)   │   ~32 GB/s    │  (Discrete) │
└─────────────┘               └─────────────┘
     ↑ Explicit copy required between memories

Apple Silicon UMA:
┌─────────────────────────────────────────────┐
│              Unified Memory                  │
│         (Shared CPU + GPU Access)           │
│              546 GB/s bandwidth             │
└─────────────────────────────────────────────┘
     ↑ Zero-copy: Both processors see same memory
```

**Key Insight**: On Apple Silicon, data should NEVER be copied. CPU prepares pointers, GPU accesses same memory directly.

---

## Part 2: Current Implementation Problems

### MLX Python Loop Problem

```python
# Current MLX (SLOW - Python loop overhead)
for i in range(n_traces):           # Python loop = GIL + interpreter
    for it in range(n_times):       # Nested Python loop
        # GPU operation             # Tiny GPU kernel launch
        t_travel = compute_dsr(...) # GPU idle between iterations
```

**Problem**: Each Python iteration:
- Acquires/releases GIL
- Launches separate GPU kernel (~10-50μs overhead)
- GPU utilization: <5%

### What We Need

```
# Ideal GPU execution (FAST - single kernel)
GPU Kernel Launch (single dispatch):
├── Thread Block 0: pillars[0:32], all traces, all times
├── Thread Block 1: pillars[32:64], all traces, all times
├── Thread Block 2: pillars[64:96], all traces, all times
└── ... (1024 thread blocks for 32x32 tile)
    └── Each thread: processes subset of traces × times
```

---

## Part 3: Proposed Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python Application                           │
│                    (pstm.kernels.metal)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ pybind11 bindings
┌─────────────────────────────────────────────────────────────────┐
│                   C++ Metal Interface                            │
│                 (libpstm_metal.dylib)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ MetalDevice │  │ MetalBuffer │  │ MetalComputePipeline   │  │
│  │  Manager    │  │   Manager   │  │      Manager           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Metal API (metal-cpp)
┌─────────────────────────────────────────────────────────────────┐
│                    Metal Compute Shaders                         │
│                   (pstm_kernels.metallib)                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ migrate_tile_   │  │ precompute_     │  │ accumulate_     │  │
│  │ kernel          │  │ travel_times    │  │ contributions   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Direct memory access
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Memory (UMA)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Input Traces │  │ Output Image │  │ Velocity Model       │   │
│  │ (read-only)  │  │ (read-write) │  │ (read-only)          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Metal Compute Shader (MSL)

```metal
// pstm_kernels.metal - Metal Shading Language

struct MigrationParams {
    uint n_traces;
    uint n_samples_in;
    uint nx, ny, nt;
    float dt_s;
    float t_start_s;
    float max_aperture;
    float taper_fraction;
    // ... other params
};

kernel void migrate_tile(
    // Input buffers (read-only, zero-copy from CPU)
    device const float* amplitudes      [[buffer(0)]],  // (n_traces, n_samples)
    device const float* source_x        [[buffer(1)]],
    device const float* source_y        [[buffer(2)]],
    device const float* receiver_x      [[buffer(3)]],
    device const float* receiver_y      [[buffer(4)]],
    device const float* midpoint_x      [[buffer(5)]],
    device const float* midpoint_y      [[buffer(6)]],

    // Output buffers (read-write)
    device atomic_float* image          [[buffer(7)]],  // (nx, ny, nt)
    device atomic_uint* fold            [[buffer(8)]],  // (nx, ny)

    // Grid coordinates
    device const float* ox_coords       [[buffer(9)]],
    device const float* oy_coords       [[buffer(10)]],
    device const float* t0_s_arr        [[buffer(11)]],
    device const float* inv_v_sq        [[buffer(12)]],
    device const float* apertures       [[buffer(13)]],

    // Parameters
    constant MigrationParams& params    [[buffer(14)]],

    // Thread identification
    uint3 thread_pos                    [[thread_position_in_grid]],
    uint3 threads_per_grid              [[threads_per_grid]]
) {
    // Thread assignment strategy:
    // - thread_pos.x = pillar index (0 to nx*ny-1)
    // - thread_pos.y = trace chunk index
    // - thread_pos.z = time chunk index

    uint pillar_idx = thread_pos.x;
    if (pillar_idx >= params.nx * params.ny) return;

    uint ix = pillar_idx / params.ny;
    uint iy = pillar_idx % params.ny;
    float ox = ox_coords[ix];
    float oy = oy_coords[iy];

    // Process traces in chunks (coalesced memory access)
    uint traces_per_thread = (params.n_traces + threads_per_grid.y - 1) / threads_per_grid.y;
    uint trace_start = thread_pos.y * traces_per_thread;
    uint trace_end = min(trace_start + traces_per_thread, params.n_traces);

    // Local accumulator (avoid atomic contention)
    float local_accum[MAX_TIME_CHUNK];
    for (uint t = 0; t < MAX_TIME_CHUNK; t++) local_accum[t] = 0.0f;

    for (uint it = trace_start; it < trace_end; it++) {
        float mx = midpoint_x[it];
        float my = midpoint_y[it];

        float dist_sq = (ox - mx) * (ox - mx) + (oy - my) * (oy - my);
        float dist = sqrt(dist_sq);

        if (dist > params.max_aperture) continue;

        // Pre-compute distance² terms (invariant over time)
        float sx = source_x[it];
        float sy = source_y[it];
        float rx = receiver_x[it];
        float ry = receiver_y[it];

        float ds2 = (ox - sx) * (ox - sx) + (oy - sy) * (oy - sy);
        float dr2 = (ox - rx) * (ox - rx) + (oy - ry) * (oy - ry);

        // Loop over time samples
        for (uint iot = 0; iot < params.nt; iot++) {
            float aperture = apertures[iot];
            if (dist > aperture) continue;

            // DSR travel time
            float t0_half_sq = t0_s_arr[iot] * t0_s_arr[iot] * 0.25f;
            float t_src = sqrt(t0_half_sq + ds2 * inv_v_sq[iot]);
            float t_rec = sqrt(t0_half_sq + dr2 * inv_v_sq[iot]);
            float t_total = t_src + t_rec;

            // Sample index
            float t_sample = (t_total - params.t_start_s) / params.dt_s;
            if (t_sample < 0.0f || t_sample >= params.n_samples_in - 1) continue;

            // Linear interpolation
            uint i0 = (uint)t_sample;
            float frac = t_sample - (float)i0;
            uint trace_offset = it * params.n_samples_in;
            float amp = amplitudes[trace_offset + i0] * (1.0f - frac)
                      + amplitudes[trace_offset + i0 + 1] * frac;

            // Accumulate locally
            local_accum[iot % MAX_TIME_CHUNK] += amp;
        }
    }

    // Atomic accumulate to global memory (once per thread)
    for (uint t = 0; t < params.nt; t++) {
        uint out_idx = ix * params.ny * params.nt + iy * params.nt + t;
        atomic_fetch_add_explicit(&image[out_idx], local_accum[t % MAX_TIME_CHUNK],
                                  memory_order_relaxed);
    }
}
```

#### 2. C++ Metal Interface

```cpp
// metal_kernel.hpp

#pragma once
#include <Metal/Metal.hpp>
#include <vector>

namespace pstm {

class MetalMigrationKernel {
public:
    MetalMigrationKernel();
    ~MetalMigrationKernel();

    // Initialize with Metal device
    bool initialize();

    // Migrate tile - main entry point
    // All arrays are numpy arrays (zero-copy via buffer protocol)
    void migrate_tile(
        // Input arrays (numpy compatible, zero-copy)
        const float* amplitudes, size_t n_traces, size_t n_samples_in,
        const float* source_x,
        const float* source_y,
        const float* receiver_x,
        const float* receiver_y,
        const float* midpoint_x,
        const float* midpoint_y,

        // Output arrays (numpy compatible, zero-copy)
        float* image, size_t nx, size_t ny, size_t nt,
        int32_t* fold,

        // Grid coordinates
        const float* ox_coords,
        const float* oy_coords,
        const float* t0_s_arr,
        const float* inv_v_sq,
        const float* apertures,

        // Parameters
        float dt_s,
        float t_start_s,
        float max_aperture,
        float taper_fraction,
        bool apply_spreading,
        bool apply_obliquity
    );

    // Synchronize GPU operations
    void synchronize();

private:
    MTL::Device* m_device;
    MTL::CommandQueue* m_command_queue;
    MTL::ComputePipelineState* m_pipeline;
    MTL::Library* m_library;

    // Create buffer from numpy array (zero-copy on UMA)
    MTL::Buffer* create_buffer_from_ptr(const void* ptr, size_t size, bool read_only);
};

} // namespace pstm
```

#### 3. Python Bindings (pybind11)

```cpp
// metal_bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "metal_kernel.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pstm_metal_cpp, m) {
    m.doc() = "PSTM Metal GPU kernel";

    py::class_<pstm::MetalMigrationKernel>(m, "MetalKernel")
        .def(py::init<>())
        .def("initialize", &pstm::MetalMigrationKernel::initialize)
        .def("migrate_tile", [](
            pstm::MetalMigrationKernel& self,
            py::array_t<float, py::array::c_style> amplitudes,
            py::array_t<float, py::array::c_style> source_x,
            py::array_t<float, py::array::c_style> source_y,
            py::array_t<float, py::array::c_style> receiver_x,
            py::array_t<float, py::array::c_style> receiver_y,
            py::array_t<float, py::array::c_style> midpoint_x,
            py::array_t<float, py::array::c_style> midpoint_y,
            py::array_t<float, py::array::c_style> image,
            py::array_t<int32_t, py::array::c_style> fold,
            py::array_t<float, py::array::c_style> ox_coords,
            py::array_t<float, py::array::c_style> oy_coords,
            py::array_t<float, py::array::c_style> t0_s_arr,
            py::array_t<float, py::array::c_style> inv_v_sq,
            py::array_t<float, py::array::c_style> apertures,
            float dt_s, float t_start_s, float max_aperture,
            float taper_fraction, bool apply_spreading, bool apply_obliquity
        ) {
            // Get buffer info (zero-copy access to numpy data)
            auto amp_buf = amplitudes.request();
            auto img_buf = image.request();
            // ... etc

            self.migrate_tile(
                static_cast<float*>(amp_buf.ptr),
                amp_buf.shape[0], amp_buf.shape[1],
                // ... pass all pointers
            );
        })
        .def("synchronize", &pstm::MetalMigrationKernel::synchronize);
}
```

---

## Part 4: Parallelization Strategy

### Thread Hierarchy Design

```
Grid Dimensions: (n_pillars, trace_chunks, time_chunks)
                 (1024,      256,          1)

Threadgroup Size: (32, 1, 1)  // 32 threads per SIMD group

Total Threads: 1024 × 256 = 262,144 parallel execution units

Thread Assignment:
┌─────────────────────────────────────────────────────────────────┐
│ Thread (pillar=0, chunk=0):                                      │
│   - Output point: (ox[0], oy[0])                                │
│   - Traces: 0 to n_traces/256                                   │
│   - Times: all nt samples                                       │
│   - Accumulates to local buffer, then atomic add                │
├─────────────────────────────────────────────────────────────────┤
│ Thread (pillar=0, chunk=1):                                      │
│   - Output point: (ox[0], oy[0])  (same pillar)                 │
│   - Traces: n_traces/256 to 2*n_traces/256                      │
│   - Times: all nt samples                                       │
│   - Accumulates to local buffer, then atomic add                │
├─────────────────────────────────────────────────────────────────┤
│ ... (262,144 threads total)                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Access Optimization

```
Problem: Random trace access kills GPU performance

Solution: Trace chunking with threadgroup memory

Threadgroup Memory Layout (shared within SIMD group):
┌────────────────────────────────────────────┐
│ shared_amplitudes[32][256]                 │  // 32 traces × 256 samples
│ shared_geometry[32][6]                     │  // sx,sy,rx,ry,mx,my
└────────────────────────────────────────────┘

Algorithm:
1. Load 32 traces into threadgroup memory (coalesced)
2. All 32 threads process same 32 traces (broadcast)
3. Repeat for next 32 traces
4. Minimizes global memory access
```

### Atomic Accumulation Strategy

```
Problem: Many threads writing to same output location

Naive approach (SLOW):
  atomic_add(image[ix][iy][it], contribution);  // Contention!

Optimized approach:
  1. Each thread maintains local accumulator array
  2. Process all assigned traces
  3. Single atomic add at end

// Per-thread local accumulator
threadgroup float local_sum[32][MAX_TIMES];  // Thread-private

// Process all traces
for (trace : my_traces) {
    for (time : all_times) {
        local_sum[thread_id][time] += contribution;
    }
}

// Single atomic write per thread per time sample
for (time : all_times) {
    atomic_add(image[pillar][time], local_sum[thread_id][time]);
}
```

---

## Part 5: Build System Design

### Project Structure

```
pstm/
├── pstm/
│   └── kernels/
│       ├── metal_cpp/                    # C++ Metal implementation
│       │   ├── CMakeLists.txt
│       │   ├── include/
│       │   │   └── metal_kernel.hpp
│       │   ├── src/
│       │   │   ├── metal_kernel.cpp
│       │   │   └── metal_bindings.cpp
│       │   └── shaders/
│       │       └── pstm_kernels.metal
│       └── metal_native.py               # Python wrapper
├── setup.py                              # Build with scikit-build
└── pyproject.toml
```

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.18)
project(pstm_metal_cpp LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Metal framework (macOS only)
if(APPLE)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
endif()

# Compile Metal shaders to metallib
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pstm_kernels.metallib
    COMMAND xcrun -sdk macosx metal
            -c ${CMAKE_CURRENT_SOURCE_DIR}/shaders/pstm_kernels.metal
            -o ${CMAKE_CURRENT_BINARY_DIR}/pstm_kernels.air
    COMMAND xcrun -sdk macosx metallib
            ${CMAKE_CURRENT_BINARY_DIR}/pstm_kernels.air
            -o ${CMAKE_CURRENT_BINARY_DIR}/pstm_kernels.metallib
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/pstm_kernels.metal
    COMMENT "Compiling Metal shaders"
)

# Create pybind11 module
pybind11_add_module(pstm_metal_cpp
    src/metal_kernel.cpp
    src/metal_bindings.cpp
)

target_include_directories(pstm_metal_cpp PRIVATE
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../metal-cpp  # Apple's metal-cpp headers
)

target_link_libraries(pstm_metal_cpp PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
)

# Embed metallib path
target_compile_definitions(pstm_metal_cpp PRIVATE
    METALLIB_PATH="${CMAKE_CURRENT_BINARY_DIR}/pstm_kernels.metallib"
)

# Add dependency on metallib compilation
add_custom_target(metal_shaders DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/pstm_kernels.metallib
)
add_dependencies(pstm_metal_cpp metal_shaders)
```

---

## Part 6: Performance Projections

### Theoretical Analysis

```
PSTM Kernel Operations per Tile:
- Pillars: 1,024 (32×32)
- Traces in aperture: ~100,000
- Time samples: 500
- Operations per contribution:
  - 2 sqrt (DSR): 20 cycles
  - 1 interpolation: 4 memory reads + 3 FLOPs
  - Weight computation: ~10 FLOPs

Total FLOPs per tile:
  1,024 × 100,000 × 500 × ~40 FLOPs = 2 × 10^12 FLOPs

M4 Max theoretical peak: 14 TFLOPS
Achievable (50% efficiency): 7 TFLOPS

Projected time: 2 × 10^12 / 7 × 10^12 = 0.29 seconds per tile

Current MLX time: ~11 seconds per tile
Projected speedup: 11 / 0.29 = ~38x
```

### Realistic Expectations

| Implementation | Time per Tile | vs Current MLX | vs Optimized Numba |
|----------------|---------------|----------------|-------------------|
| Current MLX | 11.5s | 1.0x | 0.13x |
| Optimized Numba | 1.5s | 7.7x | 1.0x |
| **Metal C++ (projected)** | **0.3-0.5s** | **23-38x** | **3-5x** |

---

## Part 7: Implementation Phases

### Phase 1: Foundation (1 week)
- [ ] Set up metal-cpp build environment
- [ ] Create basic pybind11 bindings
- [ ] Implement simple test kernel (vector add)
- [ ] Verify zero-copy memory sharing

### Phase 2: Core Kernel (1-2 weeks)
- [ ] Implement migrate_tile Metal shader
- [ ] Handle atomic accumulation
- [ ] Implement linear interpolation
- [ ] Basic correctness testing

### Phase 3: Optimization (1 week)
- [ ] Threadgroup memory optimization
- [ ] Memory coalescing for trace access
- [ ] Reduce atomic contention
- [ ] Profile with Metal System Trace

### Phase 4: Integration (3-5 days)
- [ ] Create MetalNativeKernel Python class
- [ ] Register in kernel factory
- [ ] Add fallback to Numba if Metal unavailable
- [ ] Integration testing

### Phase 5: Advanced Features (optional)
- [ ] Half-precision (FP16) option
- [ ] Multi-tile pipelining
- [ ] Async CPU preparation while GPU processes

---

## Part 8: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Atomic contention limits speedup | Medium | High | Local accumulation, reduce atomics |
| Memory bandwidth bottleneck | Low | Medium | Threadgroup memory, coalescing |
| Build complexity on different Macs | Medium | Medium | Pre-built wheels, fallback to Numba |
| Correctness issues with FP32 | Low | High | Extensive testing, FP64 reference |
| Metal API learning curve | Medium | Medium | Use metal-cpp (C++ wrapper) |

---

## Part 9: Alternatives Considered

### 1. MLX with @mx.compile
- **Pros**: Pure Python, simpler build
- **Cons**: Still limited by MLX's graph compilation, no custom kernels
- **Verdict**: May get 2-3x improvement, not 10-50x

### 2. PyTorch MPS Backend
- **Pros**: Mature, well-tested
- **Cons**: No custom kernels, generic ops only
- **Verdict**: Not suitable for DSR computation

### 3. OpenCL
- **Pros**: Cross-platform
- **Cons**: Deprecated on macOS, poor Metal integration
- **Verdict**: Not recommended for Apple Silicon

### 4. CUDA via Rosetta (not possible)
- **Verdict**: CUDA not supported on Apple Silicon

### Recommendation: **Metal C++ with pybind11**
- Native Apple Silicon support
- Maximum performance potential
- Full control over GPU execution
- Zero-copy UMA integration

---

## Appendix: Quick Start Commands

```bash
# Install dependencies
brew install cmake
pip install pybind11 scikit-build-core

# Download metal-cpp
curl -L https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14.2_iOS17.2.zip -o metal-cpp.zip
unzip metal-cpp.zip -d external/

# Build
cd pstm/kernels/metal_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Test
python -c "import pstm_metal_cpp; k = pstm_metal_cpp.MetalKernel(); k.initialize()"
```
