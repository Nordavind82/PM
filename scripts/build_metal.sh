#!/bin/bash
# Build the Metal C++ kernel for PSTM
#
# Usage:
#   ./scripts/build_metal.sh [clean]
#
# Requirements:
#   - macOS with Apple Silicon or AMD GPU
#   - Xcode Command Line Tools (xcode-select --install)
#   - CMake (brew install cmake)
#   - pybind11 (pip install pybind11)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
METAL_DIR="$PROJECT_DIR/pstm/metal"
BUILD_DIR="$METAL_DIR/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== PSTM Metal Kernel Build ===${NC}"
echo "Project: $PROJECT_DIR"
echo "Metal:   $METAL_DIR"
echo ""

# Check for clean flag
if [ "$1" = "clean" ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
    echo "Done."
    exit 0
fi

# Check requirements
echo "Checking requirements..."

# Check for macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}Error: This script only works on macOS${NC}"
    exit 1
fi

# Check for Xcode Command Line Tools
if ! xcode-select -p &>/dev/null; then
    echo -e "${RED}Error: Xcode Command Line Tools not installed${NC}"
    echo "Install with: xcode-select --install"
    exit 1
fi
echo "  Xcode Command Line Tools: OK"

# Check for CMake
if ! command -v cmake &>/dev/null; then
    echo -e "${RED}Error: CMake not found${NC}"
    echo "Install with: brew install cmake"
    exit 1
fi
echo "  CMake: $(cmake --version | head -1)"

# Check for pybind11
if ! python3 -c "import pybind11" &>/dev/null; then
    echo -e "${YELLOW}Warning: pybind11 not found, installing...${NC}"
    pip install pybind11
fi
echo "  pybind11: OK"

# Check for Metal
if ! xcrun -sdk macosx metal --version &>/dev/null; then
    echo -e "${RED}Error: Metal compiler not found${NC}"
    echo ""
    echo "The Metal shader compiler requires the full Xcode installation."
    echo "Command Line Tools alone are not sufficient."
    echo ""
    echo "To install Xcode:"
    echo "  1. Open App Store and search for 'Xcode'"
    echo "  2. Install Xcode (free, ~12GB download)"
    echo "  3. Run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
    echo "  4. Run this script again"
    exit 1
fi
echo "  Metal: $(xcrun -sdk macosx metal --version 2>&1 | head -1)"

echo ""
echo -e "${GREEN}Building Metal kernel...${NC}"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$(which python3)"

# Build
echo "Compiling..."
make -j$(sysctl -n hw.ncpu)

# Check for output
if [ -f "$BUILD_DIR/pstm_metal"*.so ]; then
    echo ""
    echo -e "${GREEN}Build successful!${NC}"
    echo ""
    echo "Module location:"
    ls -la "$BUILD_DIR"/pstm_metal*.so
    echo ""
    echo "Shader location:"
    ls -la "$BUILD_DIR"/migrate_tile.metallib
    echo ""
    echo "Test with:"
    echo "  python -c \"from pstm.metal.python import is_available, get_device_info; print(get_device_info())\""
else
    echo -e "${RED}Build failed - module not created${NC}"
    exit 1
fi
