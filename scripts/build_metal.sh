#!/bin/bash
# Build PSTM Metal shaders into a compiled .metallib library
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
METAL_DIR="$PROJECT_ROOT/pstm/metal"
SHADER_DIR="$METAL_DIR/shaders"
BUILD_DIR="$METAL_DIR/build"
OUTPUT_LIB="$METAL_DIR/pstm_kernels.metallib"

echo "=============================================="
echo "Building PSTM Metal Shaders"
echo "=============================================="

mkdir -p "$BUILD_DIR"

# Only compile the main unified shader
METAL_FILE="$SHADER_DIR/pstm_migration.metal"

if [ ! -f "$METAL_FILE" ]; then
    echo "ERROR: $METAL_FILE not found"
    exit 1
fi

echo "Compiling pstm_migration.metal..."
xcrun -sdk macosx metal \
    -c "$METAL_FILE" \
    -o "$BUILD_DIR/pstm_migration.air" \
    -std=metal3.0 \
    -O3 \
    -ffast-math

echo "Linking to metallib..."
xcrun -sdk macosx metallib \
    "$BUILD_DIR/pstm_migration.air" \
    -o "$OUTPUT_LIB"

echo ""
echo "=============================================="
echo "Build successful!"
echo "Output: $OUTPUT_LIB"
echo "Size: $(ls -lh "$OUTPUT_LIB" | awk '{print $5}')"
echo "=============================================="
