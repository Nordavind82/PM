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

# List of shader files to compile
SHADER_FILES=(
    "pstm_migration.metal"
    "pstm_curved_ray.metal"
    "pstm_anisotropic_vti.metal"
    "pstm_trace_centric.metal"
)

AIR_FILES=()

# Compile each shader to .air
for SHADER in "${SHADER_FILES[@]}"; do
    METAL_FILE="$SHADER_DIR/$SHADER"
    AIR_FILE="$BUILD_DIR/${SHADER%.metal}.air"

    if [ ! -f "$METAL_FILE" ]; then
        echo "WARNING: $METAL_FILE not found, skipping..."
        continue
    fi

    echo "Compiling $SHADER..."
    xcrun -sdk macosx metal \
        -c "$METAL_FILE" \
        -o "$AIR_FILE" \
        -std=metal3.0 \
        -O3 \
        -ffast-math

    AIR_FILES+=("$AIR_FILE")
done

if [ ${#AIR_FILES[@]} -eq 0 ]; then
    echo "ERROR: No shader files compiled"
    exit 1
fi

echo ""
echo "Linking ${#AIR_FILES[@]} shader(s) to metallib..."
xcrun -sdk macosx metallib \
    "${AIR_FILES[@]}" \
    -o "$OUTPUT_LIB"

echo ""
echo "=============================================="
echo "Build successful!"
echo "Output: $OUTPUT_LIB"
echo "Size: $(ls -lh "$OUTPUT_LIB" | awk '{print $5}')"
echo "Shaders included:"
for SHADER in "${SHADER_FILES[@]}"; do
    if [ -f "$SHADER_DIR/$SHADER" ]; then
        echo "  - $SHADER"
    fi
done
echo "=============================================="
