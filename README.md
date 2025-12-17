# PSTM - 3D Prestack Kirchhoff Time Migration

High-performance seismic migration software optimized for Apple Silicon (M4 Max).

## Features

- **Wizard-driven interface**: Step-by-step configuration with validation
- **Memory-efficient**: Tiled processing with memory-mapped I/O
- **Modular kernels**: Numba CPU (default), MLX GPU (optional), Metal C++ (future)
- **Modern data formats**: Zarr + Parquet for efficient storage and access
- **Resumable**: Checkpoint/resume support for long-running jobs
- **Comprehensive QC**: Fold maps, geometry analysis, output verification

## Installation

```bash
# Basic installation
pip install -e .

# With GPU support (Apple Silicon)
pip install -e ".[gpu]"

# Full installation with all optional dependencies
pip install -e ".[all]"
```

## Quick Start

```bash
# Launch wizard interface
pstm wizard

# Run from configuration file
pstm run config.json

# Validate configuration
pstm validate config.json
```

## Requirements

- Python 3.11+
- macOS with Apple Silicon (M1/M2/M3/M4) recommended
- 16GB+ RAM (32GB+ recommended for large surveys)

## Documentation

See the `docs/` directory for:
- User Guide
- Algorithm Documentation  
- Configuration Reference
- API Documentation

## License

MIT License
