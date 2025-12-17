"""
PSTM Migration Wizard - Entry Point

Run with: python -m pstm.gui
"""

import sys
from pstm.gui import run_wizard


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PSTM Migration Wizard - 3D Prestack Kirchhoff Time Migration"
    )
    parser.add_argument(
        "config", 
        nargs="?",
        help="Path to configuration file to load"
    )
    
    args = parser.parse_args()
    
    sys.exit(run_wizard(args.config))


if __name__ == "__main__":
    main()
