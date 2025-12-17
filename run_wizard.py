#!/usr/bin/env python3
"""
PSTM Migration Wizard - Entry Point

Launch the PyQt6 wizard interface for PSTM configuration and execution.
"""

import logging
import sys


def main():
    """Main entry point for the PSTM wizard."""
    # Setup debug logging for all pstm modules
    from pstm.utils.logging import setup_logging
    setup_logging(level="DEBUG", show_path=True)

    # Also configure the debug logger
    debug_logger = logging.getLogger("pstm.migration.debug")
    debug_logger.setLevel(logging.DEBUG)

    # Add console handler if not already present
    if not debug_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] %(levelname)-8s %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        debug_logger.addHandler(handler)

    from pstm.gui import run_wizard

    # Check for config file argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    return run_wizard(config_path)


if __name__ == "__main__":
    sys.exit(main())
