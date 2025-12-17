# Metal C++ Kernel Implementation Tasks

## Overview

Tangible task list for implementing a high-performance Metal C++ migration kernel with full Python/UI integration.

**Goal**: Replace MLX GPU kernel with native Metal C++ for 3-5x speedup over optimized Numba CPU.

---

## Phase 1: Build Environment Setup

### 1.1 Project Structure
- [ ] Create `pstm/metal/` directory for C++ source
- [ ] Create `pstm/metal/src/` for C++ implementation files
- [ ] Create `pstm/metal/include/` for header files
- [ ] Create `pstm/metal/shaders/` for Metal shader files (.metal)
- [ ] Create `pstm/metal/python/` for pybind11 bindings

### 1.2 Dependencies
- [ ] Download metal-cpp headers from Apple Developer (header-only, no build needed)
- [ ] Add pybind11 as submodule or pip dependency (`pybind11[global]`)
- [ ] Verify Xcode Command Line Tools installed (`xcode-select --install`)
- [ ] Verify Metal SDK available (included with Xcode)

### 1.3 CMake Configuration
- [ ] Create `pstm/metal/CMakeLists.txt` - main build file
- [ ] Configure C++17 standard requirement
- [ ] Find and link Metal framework (`-framework Metal -framework Foundation`)
- [ ] Find and link MetalKit framework
- [ ] Configure pybind11 module build
- [ ] Set up shader compilation (`.metal` → `.metallib`)
- [ ] Configure output to `pstm/metal/lib/` directory

### 1.4 Build Scripts
- [ ] Create `scripts/build_metal.sh` - one-command build script
- [ ] Create `scripts/clean_metal.sh` - clean build artifacts
- [ ] Add build instructions to README or docs
- [ ] Test build on clean system

**Deliverable**: `pstm_metal.cpython-*.so` Python extension module builds successfully.

---

## Phase 2: Metal Shader Development

### 2.1 Core Data Structures (MSL)
- [ ] Define `TraceData` struct (source/receiver coords, amplitudes pointer)
- [ ] Define `MigrationParams` struct (aperture, dip, taper settings)
- [ ] Define `TileConfig` struct (output grid coordinates, dimensions)
- [ ] Define `VelocityModel` struct (1D vrms array, time axis)

### 2.2 Migration Kernel (`migrate_tile.metal`)
- [ ] Implement DSR travel time calculation function
- [ ] Implement linear interpolation function for trace sampling
- [ ] Implement aperture check and taper weight calculation
- [ ] Implement obliquity correction (optional, configurable)
- [ ] Implement spherical spreading correction (optional, configurable)
- [ ] Implement main kernel function with thread indexing
- [ ] Use atomic_fetch_add for image accumulation (thread safety)
- [ ] Handle edge cases (out-of-bounds samples, zero velocity)

### 2.3 Optimizations
- [ ] Use threadgroup shared memory for velocity model caching
- [ ] Implement trace chunking (process 32 traces per thread batch)
- [ ] Use SIMD group functions where beneficial
- [ ] Add fast math compiler flags (`-ffast-math`)
- [ ] Profile with Metal System Trace, optimize hotspots

### 2.4 Shader Compilation
- [ ] Create shader compilation target in CMake
- [ ] Generate `.metallib` from `.metal` sources
- [ ] Embed metallib in Python package or load at runtime
- [ ] Handle shader compilation errors gracefully

**Deliverable**: `migrate_tile.metallib` compiled shader library.

---

## Phase 3: C++ Host Code

### 3.1 Metal Device Management (`metal_device.h/cpp`)
- [ ] Create `MetalDevice` class - singleton for device access
- [ ] Implement device discovery (prefer discrete GPU if available)
- [ ] Create command queue management
- [ ] Implement error handling and logging
- [ ] Add device capability queries (max threads, memory, etc.)

### 3.2 Buffer Management (`metal_buffers.h/cpp`)
- [ ] Create `MetalBuffer` template class for typed GPU buffers
- [ ] Implement zero-copy buffer creation from numpy arrays (shared memory)
- [ ] Implement managed buffer mode (for non-contiguous data)
- [ ] Add buffer synchronization helpers
- [ ] Implement buffer pool for reuse (avoid allocation overhead)

### 3.3 Kernel Wrapper (`migration_kernel.h/cpp`)
- [ ] Create `MigrationKernel` class
- [ ] Load compiled metallib at initialization
- [ ] Create compute pipeline state from kernel function
- [ ] Implement `migrate_tile()` method:
  - Accept trace data, output tile, velocity model
  - Set up argument buffers
  - Dispatch compute threads
  - Wait for completion
  - Return timing metrics
- [ ] Implement async version with completion callback
- [ ] Add kernel configuration (threadgroup size, etc.)

### 3.4 Error Handling
- [ ] Define custom exception types for Metal errors
- [ ] Wrap all Metal API calls with error checking
- [ ] Provide meaningful error messages for common failures
- [ ] Handle GPU timeout/hang gracefully

**Deliverable**: C++ library that can run migration on GPU.

---

## Phase 4: Python Bindings (pybind11)

### 4.1 Module Definition (`bindings.cpp`)
- [ ] Create pybind11 module `pstm_metal`
- [ ] Expose `MetalDevice` class (or just functions)
- [ ] Expose `is_available()` function - check Metal support
- [ ] Expose `get_device_info()` - return device name, memory, etc.

### 4.2 Kernel Bindings
- [ ] Expose `migrate_tile()` function with numpy array arguments:
  ```python
  def migrate_tile(
      amplitudes: np.ndarray,      # (n_traces, n_samples) float32
      source_x: np.ndarray,        # (n_traces,) float64
      source_y: np.ndarray,
      receiver_x: np.ndarray,
      receiver_y: np.ndarray,
      midpoint_x: np.ndarray,
      midpoint_y: np.ndarray,
      image: np.ndarray,           # (nx, ny, nt) float64, OUTPUT
      fold: np.ndarray,            # (nx, ny) int32, OUTPUT
      x_coords: np.ndarray,
      y_coords: np.ndarray,
      t_coords_ms: np.ndarray,
      vrms: np.ndarray,
      config: dict,                # Migration parameters
  ) -> dict:                       # Timing metrics
  ```
- [ ] Implement zero-copy numpy buffer access (pybind11 buffer protocol)
- [ ] Handle non-contiguous arrays (copy if needed, warn user)
- [ ] Return metrics dict (kernel_time_ms, traces_processed, etc.)

### 4.3 Async Support
- [ ] Expose async version `migrate_tile_async()` returning future
- [ ] Implement callback mechanism for progress updates
- [ ] Allow cancellation of in-progress migration

### 4.4 Testing
- [ ] Create `tests/test_metal_bindings.py`
- [ ] Test basic import and availability check
- [ ] Test buffer passing (contiguous and non-contiguous)
- [ ] Test error handling (invalid inputs)
- [ ] Compare results with Numba kernel (numerical accuracy)

**Deliverable**: `import pstm_metal` works, `pstm_metal.migrate_tile()` callable from Python.

---

## Phase 5: Kernel Integration

### 5.1 Create Metal Kernel Class (`pstm/kernels/metal_cpp.py`)
- [ ] Create `MetalCppKernel` class extending `MigrationKernel` base
- [ ] Implement `name` property → "Metal C++"
- [ ] Implement `initialize(config)` - load C++ module, verify GPU
- [ ] Implement `migrate_tile(traces, output, velocity)`:
  - Extract numpy arrays from dataclasses
  - Call `pstm_metal.migrate_tile()`
  - Return `KernelMetrics` with timing info
- [ ] Implement `synchronize()` - ensure GPU work complete
- [ ] Implement `cleanup()` - release GPU resources
- [ ] Handle import error gracefully (module not built)

### 5.2 Register in Factory (`pstm/kernels/factory.py`)
- [ ] `ComputeBackend.METAL_CPP` already exists in enum ✓
- [ ] Register `MetalCppKernel` in `_register_backends()`
- [ ] Update `select_best_backend()` priority:
  ```python
  priority = [
      ComputeBackend.METAL_CPP,     # Fastest on Apple Silicon
      ComputeBackend.NUMBA_CPU,
      ComputeBackend.MLX_METAL,     # Fallback GPU
      ComputeBackend.NUMPY,
  ]
  ```
- [ ] Handle case where Metal module not available

### 5.3 Config Integration
- [ ] `METAL_CPP = "metal_cpp"` already in enum ✓
- [ ] Update `parse_backend()` in `pstm/config/backends.py` if needed
- [ ] Add Metal-specific config options if needed (threadgroup size, etc.)

### 5.4 Testing
- [ ] Add Metal kernel to `scripts/benchmark_optimized.py`
- [ ] Verify identical output to Numba kernel (within floating point tolerance)
- [ ] Benchmark against Numba CPU baseline
- [ ] Test with various data sizes (small, medium, large)

**Deliverable**: `--backend metal_cpp` works in CLI, auto-selected on Apple Silicon.

---

## Phase 6: UI Wizard Integration

### 6.1 Backend Selection UI (`pstm/ui/wizard/`)
- [ ] Update backend dropdown/selector to include "Metal GPU (Native)"
- [ ] Show Metal option only when `pstm_metal` module available
- [ ] Display GPU device info when Metal selected (device name, memory)
- [ ] Add tooltip explaining Metal vs MLX vs CPU options

### 6.2 Progress Reporting
- [ ] Implement progress callback from Metal kernel to UI
- [ ] Update progress bar during migration
- [ ] Show current tile being processed
- [ ] Display real-time traces/second metric

### 6.3 Error Handling UI
- [ ] Show user-friendly message if Metal not available
- [ ] Suggest fallback to Numba CPU
- [ ] Handle GPU timeout with retry option
- [ ] Display Metal compilation errors clearly

### 6.4 Settings Panel
- [ ] Add Metal-specific settings section (if needed):
  - Threadgroup size (advanced)
  - Buffer mode (shared vs managed)
  - Enable/disable specific optimizations
- [ ] Save Metal preferences to user config

### 6.5 Status Display
- [ ] Show "Using Metal GPU" indicator during processing
- [ ] Display GPU memory usage if available
- [ ] Show kernel compilation status on first run

**Deliverable**: User can select and monitor Metal GPU in wizard UI.

---

## Phase 7: Testing & Validation

### 7.1 Unit Tests
- [ ] `tests/test_metal_device.py` - device detection, capabilities
- [ ] `tests/test_metal_buffers.py` - buffer creation, zero-copy
- [ ] `tests/test_metal_kernel.py` - kernel execution, basic correctness
- [ ] `tests/test_metal_integration.py` - full pipeline test

### 7.2 Numerical Validation
- [ ] Compare Metal output to Numba reference (tolerance < 1e-5)
- [ ] Test edge cases:
  - Empty trace block
  - Single trace
  - Very large aperture (all traces contribute)
  - Very small aperture (few traces contribute)
  - 1D vs 2D velocity model
- [ ] Verify fold counts match between backends

### 7.3 Performance Tests
- [ ] Benchmark suite with various configurations
- [ ] Memory usage profiling
- [ ] GPU utilization monitoring
- [ ] Thermal throttling detection

### 7.4 Stress Tests
- [ ] Large dataset (1M+ traces)
- [ ] Extended run (multiple tiles, hours of processing)
- [ ] Memory pressure (near GPU memory limit)
- [ ] Concurrent processing test

**Deliverable**: All tests pass, numerical accuracy verified.

---

## Phase 8: Documentation & Polish

### 8.1 User Documentation
- [ ] Add Metal GPU section to user guide
- [ ] Document system requirements (macOS version, Apple Silicon)
- [ ] Explain backend selection and when to use Metal
- [ ] Troubleshooting guide for common Metal issues

### 8.2 Developer Documentation
- [ ] Document C++ code structure and design
- [ ] Explain shader optimization techniques used
- [ ] Document build process and dependencies
- [ ] Add inline code comments for complex sections

### 8.3 Build/Install Documentation
- [ ] Update `pyproject.toml` with optional Metal dependency
- [ ] Create installation instructions for Metal support
- [ ] Document how to rebuild Metal module after code changes
- [ ] Add CI/CD configuration for Metal builds (if possible)

### 8.4 Final Polish
- [ ] Code review and cleanup
- [ ] Remove debug logging
- [ ] Optimize error messages
- [ ] Version bump and changelog update

**Deliverable**: Complete, documented Metal GPU support.

---

## Task Summary

| Phase | Tasks | Priority | Estimated Effort |
|-------|-------|----------|------------------|
| 1. Build Environment | 14 | Critical | Foundation |
| 2. Metal Shader | 14 | Critical | Core work |
| 3. C++ Host Code | 14 | Critical | Core work |
| 4. Python Bindings | 12 | Critical | Integration |
| 5. Kernel Integration | 12 | High | Integration |
| 6. UI Integration | 11 | Medium | User experience |
| 7. Testing | 14 | High | Quality |
| 8. Documentation | 12 | Medium | Completeness |
| **Total** | **103** | | |

---

## Dependencies Graph

```
Phase 1 (Build) ─────┬──────────────────────────────────────┐
                     │                                      │
                     ▼                                      ▼
Phase 2 (Shader) ────┴──► Phase 3 (C++ Host) ──► Phase 4 (Python Bindings)
                                                            │
                                                            ▼
                                            Phase 5 (Kernel Integration)
                                                            │
                                            ┌───────────────┼───────────────┐
                                            ▼               ▼               ▼
                                    Phase 6 (UI)    Phase 7 (Testing)    Phase 8 (Docs)
```

---

## Quick Start Commands

```bash
# 1. Set up build environment
cd pstm/metal
mkdir build && cd build
cmake ..
make -j8

# 2. Test the module
python -c "import pstm_metal; print(pstm_metal.is_available())"

# 3. Run benchmark
python scripts/benchmark_optimized.py --backend metal_cpp

# 4. Run tests
pytest tests/test_metal*.py -v
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Metal API complexity | Start with minimal kernel, iterate |
| Numerical differences | Compare with Numba at each step |
| Build system issues | Test on clean macOS install |
| Performance not meeting target | Profile early, optimize iteratively |
| Memory issues with large data | Implement chunked processing |
| UI thread blocking | Use async kernel dispatch |
