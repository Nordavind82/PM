# PSTM Migration Pipeline - Complete Code Flow Analysis

**Generated:** 2024-12-15
**Purpose:** Trace data flow from UI wizard to kernel execution, identify issues

---

## Executive Summary

This document traces the complete execution path when a user clicks "Run Migration" in the PSTM wizard. Key findings:

- **4 backend conversions** (string→enum→string→enum) - inefficient
- **Duplicate tile calculation logic** in 2 places
- **Silent fallbacks** that can mask user settings
- **Hardcoded magic numbers** scattered throughout

---

## Complete Execution Path

### PHASE 1: User Initiates Migration (UI Layer)

#### 1.1 User Clicks "Run Migration" Button
- **File:** `pstm/gui/main_window.py:343`
- **Class:** `ActionBar`
- **Code:** `self.run_btn.clicked.connect(self.run_clicked.emit)`

#### 1.2 Main Window Receives Click Signal
- **File:** `pstm/gui/main_window.py:451-467`
- **Class:** `PSTMWizardWindow`
- **Method:** `_run_migration()`
- **Actions:**
  - Validates all steps
  - Shows confirmation dialog
  - Calls `self._execute_migration()`

#### 1.3 Navigation to Execution Step
- **File:** `pstm/gui/main_window.py:469-479`
- **Method:** `_execute_migration()`
- **Actions:**
  - Navigates to step index 6
  - Calls `execution_step._start_migration()`

---

### PHASE 2: Execution Step Preparation

#### 2.1 ExecutionStep._start_migration()
- **File:** `pstm/gui/steps/execution_step.py:593-617`
- **Actions:**
  1. Runs preflight checks (Line 598)
  2. Validates no issues (Line 602)
  3. Shows confirmation dialog (Lines 609-614)
  4. Calls `self._execute_migration()`

#### 2.2 ExecutionStep.on_leave() - Save UI State
- **File:** `pstm/gui/steps/execution_step.py:954-1003`
- **Critical Data Saved:**

```
Backend Selection (Lines 974-985):
├── combo_index = self.backend_combo.currentIndex()
├── selected_backend = backends[combo_index]  # "auto", "numba_cpu", etc.
└── state.backend = selected_backend

Tiling (Lines 990-992):
├── state.auto_tile_size = self.auto_tile_check.isChecked()
├── state.tile_nx = self.tile_nx_spin.value()
└── state.tile_ny = self.tile_ny_spin.value()

Resources (Lines 987-988):
├── state.max_memory_gb = self.max_memory_spin.value()
└── state.n_threads = self.n_threads_spin.value()
```

#### 2.3 ExecutionStep._execute_migration()
- **File:** `pstm/gui/steps/execution_step.py:619-673`
- **Actions:**
  1. `on_leave()` - save UI state (Line 626)
  2. `config = self.controller.build_migration_config()` (Line 631)
  3. `dialog = MigrationProgressDialog(config)` (Line 645)
  4. `dialog.start_migration()` (Line 649)
  5. `dialog.exec()` - modal blocking (Line 653)

---

### PHASE 3: Config Building (State → Config)

#### 3.1 WizardController.build_migration_config()
- **File:** `pstm/gui/state.py:690-839`

**Backend Enum Mapping (Lines 791-802):**
```python
backend_map = {
    "auto": ComputeBackend.AUTO,
    "numba_cpu": ComputeBackend.NUMBA_CPU,
    "mlx_metal": ComputeBackend.MLX_METAL,
    "numpy": ComputeBackend.NUMPY,
}
backend_str = state.execution.backend  # String from UI
backend = backend_map.get(backend_str, ComputeBackend.AUTO)  # ⚠️ FALLBACK
```

**Execution Config (Lines 805-820):**
```python
execution_config = ExecutionConfig(
    resources=ResourceConfig(
        backend=backend,  # Enum here
        max_memory_gb=state.execution.max_memory_gb,
        num_workers=state.execution.n_threads or None,
    ),
    tiling=TilingConfig(
        auto_tile_size=state.execution.auto_tile_size,
        tile_nx=state.execution.tile_nx,
        tile_ny=state.execution.tile_ny,
    ),
    checkpoint=CheckpointConfig(...),
)
```

---

### PHASE 4: Worker Initialization (Threading)

#### 4.1 MigrationProgressDialog.start_migration()
- **File:** `pstm/gui/migration_dialog.py:223-250`
- **Actions:**
  1. Start heartbeat timer (500ms interval)
  2. Create worker: `MigrationWorker(self.config)`
  3. Connect signals (progress, phase, success, error)
  4. Start thread: `self._worker.start()`

#### 4.2 MigrationWorker.run()
- **File:** `pstm/gui/migration_worker.py:91-151`
- **Actions:**
  1. Import executor (Lines 98-106)
  2. Create executor (Lines 120-123):
     ```python
     self._executor = MigrationExecutor(
         self.config,
         progress_callback=self._on_progress
     )
     ```
  3. Run migration (Line 129):
     ```python
     success = self._executor.run(resume=self.resume)
     ```

---

### PHASE 5: Executor Initialization

#### 5.1 MigrationExecutor.__init__()
- **File:** `pstm/pipeline/executor.py:271-327`
- **Actions:**
  - Store config and callback
  - Setup debug logging
  - Log system info including backend requested

#### 5.2 MigrationExecutor.run()
- **File:** `pstm/pipeline/executor.py:328-413`
- **Four Phases:**
  1. `_initialize()` - Open data, create kernel
  2. `_plan(resume)` - Create tile plan
  3. `_migrate()` - Process all tiles
  4. `_finalize()` - Normalize and write output

---

### PHASE 6: Initialization (_initialize)

#### 6.1 MigrationExecutor._initialize()
- **File:** `pstm/pipeline/executor.py:484-581`

**Steps:**
```
1. Open Trace Reader (Lines 490-497)
   └── ZarrTraceReader(config.input.traces_path, ...)

2. Open Header Manager (Lines 500-507)
   └── ParquetHeaderManager(config.input.headers_path, ...)

3. Build Spatial Index (Lines 509-521)
   ├── If index_path exists: SpatialIndex.load(...)
   └── Else: SpatialIndex.build(trace_indices, midpoint_x, midpoint_y)

4. Load Velocity Model (Lines 523-534)
   └── create_velocity_manager(config.velocity, config.output.grid)

5. CREATE KERNEL (Lines 536-559) ⚠️ CRITICAL
   ├── backend = config.execution.resources.backend.value  # Enum → String!
   ├── self._kernel = create_kernel(backend)  # String input
   └── self._kernel.initialize(kernel_config)
```

---

### PHASE 7: Kernel Creation (Factory)

#### 7.1 create_kernel(backend)
- **File:** `pstm/kernels/factory.py:100-150`

**String to Enum Conversion (Lines 117-127):**
```python
if isinstance(backend, str):
    backend_map = {
        "auto": ComputeBackend.AUTO,
        "numpy": ComputeBackend.NUMPY,
        "numba_cpu": ComputeBackend.NUMBA_CPU,
        "mlx_metal": ComputeBackend.MLX_METAL,
    }
    backend = backend_map.get(backend.lower(), ComputeBackend.AUTO)  # ⚠️ FALLBACK
```

**Auto Selection (Lines 153-182):**
```python
def select_best_backend() -> ComputeBackend:
    priority = [
        ComputeBackend.NUMBA_CPU,   # First choice
        ComputeBackend.MLX_METAL,   # Second choice
        ComputeBackend.NUMPY,       # Fallback
    ]
    for backend in priority:
        if backend in _BACKEND_REGISTRY:
            return backend
```

---

### PHASE 8: Tile Planning

#### 8.1 TilePlanner.plan()
- **File:** `pstm/pipeline/tile_planner.py:126-162`

```python
if self.tiling_config.auto_tile_size:
    tile_nx, tile_ny = self._auto_tile_size()
else:
    tile_nx = self.tiling_config.tile_nx or self.output_grid.nx  # ⚠️ FALLBACK
    tile_ny = self.tiling_config.tile_ny or self.output_grid.ny  # ⚠️ FALLBACK
```

**Auto Tile Size Logic (Lines 164-210):**
```python
if nt > 1000:
    max_tile = 32   # Deep data
elif nt > 500:
    max_tile = 48   # Medium depth
else:
    max_tile = 64   # Shallow data
```

---

### PHASE 9: Migration Loop

#### 9.1 MigrationExecutor._migrate()
- **File:** `pstm/pipeline/executor.py:625-789`

```python
for tile in iter_tiles(self._tile_plan, completed):
    # Process tile
    metrics, trace_count, data_mb, indices = self._process_tile(...)

    # Update metrics
    self.metrics.n_tiles_completed += 1
    self.metrics.n_output_bins_completed += tile.nx * tile.ny

    # Report progress
    self._report_progress(...)
```

#### 9.2 MigrationExecutor._process_tile()
- **File:** `pstm/pipeline/executor.py:791-976`

**Key Operations:**
1. Query traces in aperture
2. Load trace data
3. Load geometry
4. Get velocity slice
5. **Execute kernel** (Line 912):
   ```python
   metrics = self._kernel.migrate_tile(traces, output_tile, velocity, kernel_config)
   ```
6. Accumulate to output

---

## Critical Issues Found

### Issue 1: BACKEND CONVERSION CHAIN (4 conversions!)

```
UI Combo Index
    ↓ (execution_step.py:975-985)
String "numba_cpu"
    ↓ (state.py:799)
Enum ComputeBackend.NUMBA_CPU
    ↓ (executor.py:539)
String "numba_cpu" (.value)
    ↓ (factory.py:126)
Enum ComputeBackend.NUMBA_CPU
```

**Impact:** Inefficient, error-prone, multiple fallback points

---

### Issue 2: DUPLICATE BACKEND MAPPING

**Location 1:** `pstm/gui/state.py:791-802`
**Location 2:** `pstm/kernels/factory.py:119-126`

Both define identical mapping - maintenance risk if changed in one place.

---

### Issue 3: SILENT FALLBACKS

| Location | Code | Fallback | Risk |
|----------|------|----------|------|
| state.py:799 | `backend_map.get(backend_str, ComputeBackend.AUTO)` | AUTO | User setting ignored |
| factory.py:126 | `backend_map.get(backend.lower(), ComputeBackend.AUTO)` | AUTO | Double fallback |
| tile_planner.py:137 | `tile_nx or self.output_grid.nx` | Full grid | No tiling |
| execution_step.py:419 | `self.max_memory_spin.setValue(8.0)` | 8GB | Under-allocation |

---

### Issue 4: DUPLICATE TILE CALCULATION LOGIC

**Location 1:** `pstm/pipeline/tile_planner.py:164-210`
**Location 2:** `pstm/gui/steps/execution_step.py:447-470`

Same logic duplicated:
```python
if nt > 1000:
    max_tile = 32
elif nt > 500:
    max_tile = 48
else:
    max_tile = 64
```

---

### Issue 5: HARDCODED MAGIC NUMBERS

| Value | Location | Purpose |
|-------|----------|---------|
| 32, 48, 64 | tile_planner.py:188-196 | Max tile sizes |
| 1000, 500 | tile_planner.py:188-192 | Time depth thresholds |
| 8.0 | execution_step.py:182 | Default memory GB |
| 10 | execution_step.py:268 | Checkpoint interval tiles |
| 300 | execution_step.py:269 | Checkpoint interval seconds |
| 0.75 | execution_step.py:415 | Memory usage factor |
| 64 | execution_step.py:415 | Memory cap GB |
| 4 | tile_planner.py:177 | Memory division factor |

---

### Issue 6: NO ORPHAN CODE FOUND

All defined functions are called. Code is connected.

---

## Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
├─────────────────────────────────────────────────────────────────┤
│  [Run Migration Button]                                          │
│         │                                                        │
│         ▼                                                        │
│  ExecutionStep._start_migration()                                │
│         │                                                        │
│         ├── _run_preflight()                                     │
│         │                                                        │
│         ▼                                                        │
│  ExecutionStep._execute_migration()                              │
│         │                                                        │
│         ├── on_leave() ─────────────► WizardState               │
│         │                              (backend="numba_cpu")     │
│         ▼                                                        │
│  WizardController.build_migration_config()                       │
│         │                                                        │
│         ├── backend_map.get() ──────► ComputeBackend.NUMBA_CPU  │
│         │                              (FALLBACK: AUTO)          │
│         ▼                                                        │
│  MigrationConfig                                                 │
│         │                                                        │
└─────────┼───────────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────────┐
│         ▼              THREADING LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  MigrationProgressDialog                                         │
│         │                                                        │
│         ├── MigrationWorker(config)                              │
│         │                                                        │
│         ▼                                                        │
│  MigrationWorker.run()  [QThread]                                │
│         │                                                        │
└─────────┼───────────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────────┐
│         ▼              EXECUTOR LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  MigrationExecutor.__init__(config)                              │
│         │                                                        │
│         ▼                                                        │
│  MigrationExecutor.run()                                         │
│         │                                                        │
│         ├── _initialize()                                        │
│         │       ├── ZarrTraceReader                              │
│         │       ├── ParquetHeaderManager                         │
│         │       ├── SpatialIndex                                 │
│         │       ├── VelocityManager                              │
│         │       └── create_kernel(backend.value) ◄── String!    │
│         │                    │                                   │
│         │                    ▼                                   │
│         │            ┌─────────────────┐                         │
│         │            │ KERNEL FACTORY  │                         │
│         │            ├─────────────────┤                         │
│         │            │ backend_map.get │ ◄── FALLBACK: AUTO     │
│         │            │ select_best()   │                         │
│         │            │ NumbaKernel()   │                         │
│         │            └─────────────────┘                         │
│         │                                                        │
│         ├── _plan()                                              │
│         │       └── TilePlanner.plan()                           │
│         │                                                        │
│         ├── _migrate()                                           │
│         │       └── for tile: _process_tile()                    │
│         │                └── kernel.migrate_tile() ◄── COMPUTE  │
│         │                                                        │
│         └── _finalize()                                          │
│                 └── Write output files                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary Table

| # | Layer | File | Class.Method | Line | Input → Output |
|---|-------|------|--------------|------|----------------|
| 1 | UI | execution_step.py | ExecutionStep._start_migration | 593 | Click → preflight |
| 2 | UI | execution_step.py | ExecutionStep.on_leave | 954 | Widgets → State |
| 3 | Config | state.py | WizardController.build_migration_config | 690 | State → Config |
| 4 | Dialog | migration_dialog.py | MigrationProgressDialog.start_migration | 223 | Config → Worker |
| 5 | Thread | migration_worker.py | MigrationWorker.run | 91 | Config → Executor |
| 6 | Executor | executor.py | MigrationExecutor._initialize | 484 | Config → Readers |
| 7 | Factory | factory.py | create_kernel | 100 | String → Kernel |
| 8 | Executor | executor.py | MigrationExecutor._plan | 583 | Config → TilePlan |
| 9 | Executor | executor.py | MigrationExecutor._migrate | 625 | TilePlan → Output |
| 10 | Kernel | numba_cpu.py | NumbaKernel.migrate_tile | 574 | Traces → Image |

---

## Recommendations

### High Priority

1. **Eliminate backend conversion chain** - Keep enum throughout after initial UI selection
2. **Centralize backend mapping** - Single source of truth in config module
3. **Add explicit validation** - Fail fast instead of silent fallbacks
4. **Consolidate tile calculation** - Single function shared by UI and planner

### Medium Priority

5. **Move magic numbers to config** - Centralize in settings.py
6. **Add preflight backend check** - Show actual backend that will be used
7. **Log all fallbacks** - Make silent defaults visible

### Low Priority

8. **Add type hints** - Better IDE support and documentation
9. **Unit test critical paths** - Especially backend selection
