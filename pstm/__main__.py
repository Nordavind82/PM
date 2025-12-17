"""
PSTM Command Line Interface.

Entry point for the migration application.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pstm import __version__
from pstm.utils.logging import (
    console,
    print_banner,
    print_error,
    print_info,
    print_section,
    print_success,
    setup_logging,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="pstm",
        description="3D Prestack Kirchhoff Time Migration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pstm wizard                     Launch interactive wizard
  pstm run config.json            Run migration from config file
  pstm validate config.json       Validate configuration file
  pstm info input.zarr            Show information about input data
        """,
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-essential output",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write logs to file",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -------------------------------------------------------------------------
    # wizard command
    # -------------------------------------------------------------------------
    wizard_parser = subparsers.add_parser(
        "wizard",
        help="Launch interactive configuration wizard",
    )
    wizard_parser.add_argument(
        "--config",
        type=Path,
        help="Load existing configuration file",
    )

    # -------------------------------------------------------------------------
    # run command
    # -------------------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run migration from configuration file",
    )
    run_parser.add_argument(
        "config",
        type=Path,
        help="Path to configuration JSON file",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    run_parser.add_argument(
        "--backend",
        choices=["auto", "numpy", "numba_cpu", "mlx_metal"],
        default="auto",
        help="Compute backend (default: auto)",
    )

    # -------------------------------------------------------------------------
    # validate command
    # -------------------------------------------------------------------------
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration file",
    )
    validate_parser.add_argument(
        "config",
        type=Path,
        help="Path to configuration JSON file",
    )
    validate_parser.add_argument(
        "--check-data",
        action="store_true",
        help="Also validate that input data files are readable",
    )

    # -------------------------------------------------------------------------
    # info command
    # -------------------------------------------------------------------------
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about input data",
    )
    info_parser.add_argument(
        "path",
        type=Path,
        help="Path to Zarr, Parquet, or SEG-Y file",
    )

    # -------------------------------------------------------------------------
    # benchmark command
    # -------------------------------------------------------------------------
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark compute backends",
    )
    bench_parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Benchmark size (default: small)",
    )

    # -------------------------------------------------------------------------
    # create-config command
    # -------------------------------------------------------------------------
    create_parser_cmd = subparsers.add_parser(
        "create-config",
        help="Create a template configuration file",
    )
    create_parser_cmd.add_argument(
        "output",
        type=Path,
        help="Output configuration file path",
    )
    create_parser_cmd.add_argument(
        "--minimal",
        action="store_true",
        help="Create minimal configuration",
    )

    return parser


def cmd_wizard(args: argparse.Namespace) -> int:
    """Launch the interactive wizard."""
    print_section("Migration Wizard")
    
    try:
        from pstm.wizard import check_textual_available, run_wizard
        
        if not check_textual_available():
            print_error("Wizard requires textual package")
            print_info("Install with: pip install textual")
            return 1
        
        # Run the wizard
        run_wizard()
        return 0
        
    except ImportError as e:
        print_error(f"Wizard requires textual package: {e}")
        print_info("Install with: pip install textual")
        return 1
    except Exception as e:
        print_error(f"Wizard failed: {e}")
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Run migration from configuration file."""
    from pstm.config import MigrationConfig

    config_path = args.config

    if not config_path.exists():
        print_error(f"Configuration file not found: {config_path}")
        return 1

    print_section("Loading Configuration")

    try:
        config = MigrationConfig.from_json(config_path)
        print_success(f"Loaded configuration: {config.name}")
    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        return 1

    # Print summary
    summary = config.get_summary()
    for key, value in summary.items():
        print_info(f"  {key}: {value}")

    if args.dry_run:
        print_section("Dry Run Complete")
        print_success("Configuration is valid.")
        return 0

    print_section("Starting Migration")
    
    try:
        from pstm.pipeline.executor import MigrationExecutor
        
        # Create executor
        executor = MigrationExecutor(config)
        
        # Run migration (with resume if checkpoint exists)
        resume = getattr(args, 'resume', False)
        success = executor.run(resume=resume)
        
        if success:
            print_success("Migration completed successfully!")
            return 0
        else:
            print_error("Migration failed or was interrupted")
            return 1
            
    except Exception as e:
        print_error(f"Migration failed: {e}")
        logger.exception("Migration error details")
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate configuration file."""
    from pstm.config import MigrationConfig

    config_path = args.config

    if not config_path.exists():
        print_error(f"Configuration file not found: {config_path}")
        return 1

    print_section("Validating Configuration")

    try:
        config = MigrationConfig.from_json(config_path)
        print_success("Configuration syntax is valid.")
    except Exception as e:
        print_error(f"Configuration validation failed: {e}")
        return 1

    # Optionally check data files
    if args.check_data:
        print_info("Checking input data files...")

        # Check traces
        if not config.input.traces_path.exists():
            print_error(f"Traces file not found: {config.input.traces_path}")
            return 1
        print_success(f"Traces file exists: {config.input.traces_path}")

        # Check headers
        if not config.input.headers_path.exists():
            print_error(f"Headers file not found: {config.input.headers_path}")
            return 1
        print_success(f"Headers file exists: {config.input.headers_path}")

        # Check velocity (if 3D cube)
        if config.velocity.velocity_path:
            if not config.velocity.velocity_path.exists():
                print_error(f"Velocity file not found: {config.velocity.velocity_path}")
                return 1
            print_success(f"Velocity file exists: {config.velocity.velocity_path}")

    print_section("Validation Complete")
    print_success("All checks passed.")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show information about input data."""
    import zarr

    path = args.path

    if not path.exists():
        print_error(f"File not found: {path}")
        return 1

    print_section(f"Information: {path.name}")

    # Determine file type
    if path.suffix == ".zarr" or (path.is_dir() and (path / ".zarray").exists()):
        # Zarr array
        try:
            z = zarr.open(str(path), mode="r")
            print_info(f"Type: Zarr Array")
            print_info(f"Shape: {z.shape}")
            print_info(f"Dtype: {z.dtype}")
            print_info(f"Chunks: {z.chunks}")
            if hasattr(z, "compressor") and z.compressor:
                print_info(f"Compressor: {z.compressor}")

            # Print attributes
            if z.attrs:
                print_info("Attributes:")
                for key, value in z.attrs.items():
                    print_info(f"  {key}: {value}")

        except Exception as e:
            print_error(f"Failed to read Zarr file: {e}")
            return 1

    elif path.suffix == ".parquet":
        # Parquet file
        try:
            import polars as pl

            df = pl.scan_parquet(path)
            schema = df.collect_schema()

            print_info(f"Type: Parquet File")
            print_info(f"Columns: {len(schema)}")
            print_info("Schema:")
            for name, dtype in schema.items():
                print_info(f"  {name}: {dtype}")

            # Get row count
            count = df.select(pl.len()).collect().item()
            print_info(f"Rows: {count:,}")

        except Exception as e:
            print_error(f"Failed to read Parquet file: {e}")
            return 1

    else:
        print_error(f"Unsupported file type: {path.suffix}")
        return 1

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Benchmark compute backends."""
    print_section("Compute Backend Benchmark")
    
    # Parse size
    size = args.size.lower()
    if size == "small":
        n_traces, n_samples = 500, 1000
    elif size == "medium":
        n_traces, n_samples = 1000, 2000
    elif size == "large":
        n_traces, n_samples = 2000, 3000
    else:
        print_error(f"Unknown size: {args.size}. Use small/medium/large")
        return 1
    
    print_info(f"Test size: {n_traces} traces × {n_samples} samples")
    
    try:
        from pstm.kernels.factory import benchmark_all_backends, get_available_backends
        
        # Show available backends
        available = get_available_backends()
        print_info(f"Available backends: {[b.value for b in available]}")
        
        # Run benchmark
        print_section("Running Benchmarks...")
        results = benchmark_all_backends(n_traces=n_traces, n_samples=n_samples)
        
        # Display results - results is dict[backend, samples_per_second]
        print_section("Results")
        print_info(f"{'Backend':<15} {'Samples/s':<15} {'Status'}")
        print_info("-" * 45)
        
        for backend in sorted(results.keys(), key=lambda b: results[b], reverse=True):
            samples_per_sec = results[backend]
            print_info(f"{backend.value:<15} {samples_per_sec:<15.2e} OK")
        
        # Recommend best backend
        if results:
            best_backend = max(results, key=results.get)
            print_section("Recommendation")
            print_success(f"Fastest backend: {best_backend.value}")
        else:
            print_error("No backends successfully benchmarked")
        
        return 0
        
    except Exception as e:
        print_error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_create_config(args: argparse.Namespace) -> int:
    """Create a template configuration file."""
    from pstm.config import create_minimal_config

    output_path = args.output

    print_section("Creating Configuration Template")

    # Create a minimal example config
    try:
        config = create_minimal_config(
            traces_path="/path/to/traces.zarr",
            headers_path="/path/to/headers.parquet",
            output_dir="/path/to/output",
            velocity=2000.0,  # Constant velocity
            x_range=(0, 10000),
            y_range=(0, 10000),
            t_range_ms=(0, 4000),
        )

        config.name = "example_migration"
        config.description = "Example migration configuration - edit paths before use"

        config.to_json(output_path)
        print_success(f"Created configuration template: {output_path}")
        print_info("Edit the file to set correct paths and parameters.")

    except Exception as e:
        print_error(f"Failed to create configuration: {e}")
        return 1

    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else ("WARNING" if args.quiet else "INFO")
    setup_logging(level=log_level, log_file=args.log_file)

    # Print banner unless quiet
    if not args.quiet:
        print_banner()

    # Dispatch to command handler
    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "wizard": cmd_wizard,
        "run": cmd_run,
        "validate": cmd_validate,
        "info": cmd_info,
        "benchmark": cmd_benchmark,
        "create-config": cmd_create_config,
    }

    handler = commands.get(args.command)
    if handler is None:
        print_error(f"Unknown command: {args.command}")
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print_info("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if args.verbose:
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
