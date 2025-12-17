#pragma once

#include "metal_types.h"
#include <memory>
#include <string>

namespace pstm {

// Forward declaration for Objective-C implementation
class MetalKernelImpl;

/**
 * Metal GPU kernel for PSTM migration.
 *
 * Uses Apple Metal for GPU acceleration on Apple Silicon.
 * Provides zero-copy buffer sharing with numpy arrays.
 */
class MetalKernel {
public:
    MetalKernel();
    ~MetalKernel();

    // Non-copyable
    MetalKernel(const MetalKernel&) = delete;
    MetalKernel& operator=(const MetalKernel&) = delete;

    // Move semantics
    MetalKernel(MetalKernel&&) noexcept;
    MetalKernel& operator=(MetalKernel&&) noexcept;

    /**
     * Check if Metal is available on this system.
     */
    static bool is_available();

    /**
     * Get device information.
     */
    static std::string get_device_name();
    static size_t get_device_memory();

    /**
     * Initialize the kernel with compiled shader.
     * @param shader_path Path to .metallib file (optional, uses embedded path if empty)
     * @return true if initialization successful
     */
    bool initialize(const std::string& shader_path = "");

    /**
     * Execute migration kernel.
     *
     * All arrays must be contiguous in memory.
     *
     * @param amplitudes Trace amplitudes [n_traces, n_samples], float32
     * @param source_x Source X coordinates [n_traces], float64
     * @param source_y Source Y coordinates [n_traces], float64
     * @param receiver_x Receiver X coordinates [n_traces], float64
     * @param receiver_y Receiver Y coordinates [n_traces], float64
     * @param midpoint_x Midpoint X coordinates [n_traces], float64
     * @param midpoint_y Midpoint Y coordinates [n_traces], float64
     * @param image Output image [nx, ny, nt], float64 (modified in-place)
     * @param fold Output fold [nx, ny], int32 (modified in-place)
     * @param x_coords Output X axis [nx], float64
     * @param y_coords Output Y axis [ny], float64
     * @param t_coords_ms Output time axis [nt], float64
     * @param vrms RMS velocity [nt], float64
     * @param params Migration parameters
     * @return Kernel execution metrics
     */
    KernelMetrics migrate_tile(
        const float* amplitudes,
        const double* source_x,
        const double* source_y,
        const double* receiver_x,
        const double* receiver_y,
        const double* midpoint_x,
        const double* midpoint_y,
        double* image,
        int32_t* fold,
        const double* x_coords,
        const double* y_coords,
        const double* t_coords_ms,
        const double* vrms,
        const MigrationParams& params
    );

    /**
     * Wait for all GPU operations to complete.
     */
    void synchronize();

    /**
     * Release GPU resources.
     */
    void cleanup();

private:
    std::unique_ptr<MetalKernelImpl> impl_;
};

} // namespace pstm
