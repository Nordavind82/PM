#pragma once

#include <cstdint>

namespace pstm {

// Migration parameters passed to GPU kernel
struct MigrationParams {
    float max_dip_deg;
    float min_aperture;
    float max_aperture;
    float taper_fraction;
    float dt_ms;
    float t_start_ms;
    int32_t apply_spreading;
    int32_t apply_obliquity;
    int32_t n_traces;
    int32_t n_samples;
    int32_t nx;
    int32_t ny;
    int32_t nt;
};

// Timing metrics returned from kernel execution
struct KernelMetrics {
    double kernel_time_ms;
    double total_time_ms;
    int64_t traces_processed;
    int64_t samples_written;
};

} // namespace pstm
