#include <metal_stdlib>
using namespace metal;

// =============================================================================
// PSTM Trace-Centric Migration Kernel
//
// KEY INSIGHT: Instead of processing each output point and looping over traces,
// we process each trace ONCE and scatter contributions to output points.
//
// This eliminates redundant trace processing when traces overlap multiple tiles.
// Each trace is loaded and processed exactly once for the entire output grid.
//
// Trade-off: Requires atomic operations for accumulation, but avoids O(tiles × traces)
// complexity, reducing to O(traces) total operations.
// =============================================================================

// Migration parameters for trace-centric kernel
struct TraceCentricParams {
    // Grid parameters
    float dx;                    // Output grid spacing X (meters)
    float dy;                    // Output grid spacing Y (meters)
    float dt_ms;                 // Time sampling (ms)
    float t_start_ms;            // Start time (ms)

    // Output grid bounds (world coordinates)
    float x_min;                 // Minimum X coordinate
    float y_min;                 // Minimum Y coordinate
    float t_min_ms;              // Minimum time (ms)

    // Aperture parameters
    float max_aperture;          // Maximum aperture (meters)
    float min_aperture;          // Minimum aperture (meters)
    float taper_fraction;        // Taper fraction (0-1)
    float max_dip_deg;           // Maximum dip angle (degrees)

    // Amplitude correction flags
    int apply_spreading;         // Apply geometrical spreading
    int apply_obliquity;         // Apply obliquity factor
    int apply_aa;                // Apply anti-aliasing
    float aa_dominant_freq;      // Dominant frequency for AA (Hz)

    // Dimensions
    int n_traces;
    int n_samples;               // Input samples per trace
    int nx;                      // Output grid X dimension
    int ny;                      // Output grid Y dimension
    int nt;                      // Output grid time dimension
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Atomic add for float using compare-and-swap.
 * Metal doesn't have native atomic float add, so we implement it.
 */
inline void atomic_add_float(device atomic_float* addr, float value) {
    float expected = atomic_load_explicit(addr, memory_order_relaxed);
    float desired;
    do {
        desired = expected + value;
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, desired,
        memory_order_relaxed, memory_order_relaxed
    ));
}

/**
 * Compute DSR (Double Square Root) travel time.
 */
inline float compute_dsr_traveltime(
    float ds2,           // Distance squared to source
    float dr2,           // Distance squared to receiver
    float t0_half_sq,    // (t0/2)^2 in seconds^2
    float inv_v_sq       // 1/v^2
) {
    return sqrt(t0_half_sq + ds2 * inv_v_sq) + sqrt(t0_half_sq + dr2 * inv_v_sq);
}

/**
 * Linear interpolation of trace amplitude.
 */
inline float linear_interp(
    device const float* trace,
    int n_samples,
    float sample_idx
) {
    if (sample_idx < 0.0f || sample_idx >= float(n_samples - 1)) {
        return 0.0f;
    }
    int idx0 = int(sample_idx);
    float frac = sample_idx - float(idx0);
    return trace[idx0] * (1.0f - frac) + trace[idx0 + 1] * frac;
}

/**
 * Compute cosine taper weight.
 */
inline float compute_taper(float distance, float aperture, float taper_fraction) {
    float taper_start = aperture * (1.0f - taper_fraction);
    if (distance <= taper_start) return 1.0f;
    if (distance >= aperture) return 0.0f;
    float t = (distance - taper_start) / (aperture - taper_start);
    return 0.5f * (1.0f + cos(t * M_PI_F));
}

// =============================================================================
// Trace-Centric Migration Kernel
// Each thread processes ONE trace and scatters to all affected output points
// =============================================================================
kernel void pstm_migrate_trace_centric(
    // Input trace data
    device const float* amplitudes [[buffer(0)]],      // [n_traces, n_samples]
    device const float* source_x [[buffer(1)]],        // [n_traces]
    device const float* source_y [[buffer(2)]],
    device const float* receiver_x [[buffer(3)]],
    device const float* receiver_y [[buffer(4)]],
    device const float* midpoint_x [[buffer(5)]],
    device const float* midpoint_y [[buffer(6)]],

    // Output arrays (atomic for concurrent writes)
    device atomic_float* image [[buffer(7)]],          // [nx, ny, nt]
    device atomic_int* fold [[buffer(8)]],             // [nx, ny]

    // Pre-computed time-dependent values
    device const float* t0_half_sq [[buffer(9)]],      // [nt] - (t0/2)^2
    device const float* inv_v_sq [[buffer(10)]],       // [nt] - 1/v^2
    device const float* t0_s [[buffer(11)]],           // [nt] - t0 in seconds
    device const float* apertures [[buffer(12)]],      // [nt] - aperture at each time

    // Parameters
    constant TraceCentricParams& params [[buffer(13)]],

    // Thread position - one thread per trace
    uint trace_id [[thread_position_in_grid]]
) {
    // Bounds check
    if (trace_id >= uint(params.n_traces)) {
        return;
    }

    // Get trace geometry
    float sx = source_x[trace_id];
    float sy = source_y[trace_id];
    float rx = receiver_x[trace_id];
    float ry = receiver_y[trace_id];
    float mx = midpoint_x[trace_id];
    float my = midpoint_y[trace_id];

    // Get trace amplitudes pointer
    device const float* trace = amplitudes + trace_id * params.n_samples;

    // Compute which output bins this trace can contribute to
    // Based on maximum aperture, find the rectangular region of output points
    float max_ap = params.max_aperture;

    // Output grid bounds for this trace (based on midpoint + aperture)
    int ix_min = max(0, int((mx - max_ap - params.x_min) / params.dx));
    int ix_max = min(params.nx - 1, int((mx + max_ap - params.x_min) / params.dx));
    int iy_min = max(0, int((my - max_ap - params.y_min) / params.dy));
    int iy_max = min(params.ny - 1, int((my + max_ap - params.y_min) / params.dy));

    // Skip if trace is completely outside output grid
    if (ix_max < 0 || ix_min >= params.nx || iy_max < 0 || iy_min >= params.ny) {
        return;
    }

    // Track if we contributed to any output point (for fold counting)
    bool contributed = false;
    int contribution_ix = 0;
    int contribution_iy = 0;

    // Loop over output points in the aperture region
    for (int ix = ix_min; ix <= ix_max; ix++) {
        float ox = params.x_min + (float(ix) + 0.5f) * params.dx;

        for (int iy = iy_min; iy <= iy_max; iy++) {
            float oy = params.y_min + (float(iy) + 0.5f) * params.dy;

            // Distance from output point to midpoint
            float dm = sqrt((ox - mx) * (ox - mx) + (oy - my) * (oy - my));

            // Distance squared to source and receiver
            float ds2 = (ox - sx) * (ox - sx) + (oy - sy) * (oy - sy);
            float dr2 = (ox - rx) * (ox - rx) + (oy - ry) * (oy - ry);

            // Loop over output times
            for (int it = 0; it < params.nt; it++) {
                float aperture = apertures[it];

                // Skip if outside aperture for this time
                if (dm > aperture) {
                    continue;
                }

                float t0_half_sq_val = t0_half_sq[it];
                float inv_v_sq_val = inv_v_sq[it];
                float t0_val = t0_s[it];
                float velocity = 1.0f / sqrt(inv_v_sq_val);

                // Compute DSR traveltime
                float t_travel = compute_dsr_traveltime(ds2, dr2, t0_half_sq_val, inv_v_sq_val);

                // Convert to sample index in input trace
                float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;

                // Bounds check on input trace
                if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) {
                    continue;
                }

                // Get interpolated amplitude
                float amp = linear_interp(trace, params.n_samples, sample_idx);

                // Apply aperture taper
                amp *= compute_taper(dm, aperture, params.taper_fraction);

                // Apply geometrical spreading correction: 1 / (v * t)
                if (params.apply_spreading && t_travel > 0.001f) {
                    amp *= 1.0f / (velocity * t_travel);
                }

                // Apply obliquity correction: t0 / t_travel
                if (params.apply_obliquity && t_travel > 0.001f) {
                    amp *= t0_val / t_travel;
                }

                // Atomic add to output image
                int img_idx = ix * params.ny * params.nt + iy * params.nt + it;
                atomic_add_float(&image[img_idx], amp);

                contributed = true;
                contribution_ix = ix;
                contribution_iy = iy;
            }
        }
    }

    // Update fold count (once per trace that contributed)
    if (contributed) {
        int fold_idx = contribution_ix * params.ny + contribution_iy;
        atomic_fetch_add_explicit(&fold[fold_idx], 1, memory_order_relaxed);
    }
}

// =============================================================================
// Optimized Trace-Centric Kernel with Time-First Processing
// This variant processes time samples first for better memory access patterns
// =============================================================================
kernel void pstm_migrate_trace_centric_v2(
    // Input trace data
    device const float* amplitudes [[buffer(0)]],
    device const float* source_x [[buffer(1)]],
    device const float* source_y [[buffer(2)]],
    device const float* receiver_x [[buffer(3)]],
    device const float* receiver_y [[buffer(4)]],
    device const float* midpoint_x [[buffer(5)]],
    device const float* midpoint_y [[buffer(6)]],

    // Output arrays (atomic)
    device atomic_float* image [[buffer(7)]],
    device atomic_int* fold [[buffer(8)]],

    // Pre-computed values
    device const float* t0_half_sq [[buffer(9)]],
    device const float* inv_v_sq [[buffer(10)]],
    device const float* t0_s [[buffer(11)]],
    device const float* apertures [[buffer(12)]],

    // Parameters
    constant TraceCentricParams& params [[buffer(13)]],

    // Thread position - 2D: (trace_id, time_chunk)
    uint2 gid [[thread_position_in_grid]]
) {
    uint trace_id = gid.x;
    uint time_chunk = gid.y;

    // Time chunk processing (8 time samples per chunk for better occupancy)
    const int CHUNK_SIZE = 8;
    int it_start = time_chunk * CHUNK_SIZE;
    int it_end = min(it_start + CHUNK_SIZE, params.nt);

    if (trace_id >= uint(params.n_traces) || it_start >= params.nt) {
        return;
    }

    // Get trace geometry
    float sx = source_x[trace_id];
    float sy = source_y[trace_id];
    float rx = receiver_x[trace_id];
    float ry = receiver_y[trace_id];
    float mx = midpoint_x[trace_id];
    float my = midpoint_y[trace_id];

    device const float* trace = amplitudes + trace_id * params.n_samples;

    // Get max aperture for this time chunk
    float max_ap = 0.0f;
    for (int it = it_start; it < it_end; it++) {
        max_ap = max(max_ap, apertures[it]);
    }

    // Output grid bounds for this trace
    int ix_min = max(0, int((mx - max_ap - params.x_min) / params.dx));
    int ix_max = min(params.nx - 1, int((mx + max_ap - params.x_min) / params.dx));
    int iy_min = max(0, int((my - max_ap - params.y_min) / params.dy));
    int iy_max = min(params.ny - 1, int((my + max_ap - params.y_min) / params.dy));

    if (ix_max < 0 || ix_min >= params.nx || iy_max < 0 || iy_min >= params.ny) {
        return;
    }

    // Process time samples in this chunk
    for (int it = it_start; it < it_end; it++) {
        float aperture = apertures[it];
        float t0_half_sq_val = t0_half_sq[it];
        float inv_v_sq_val = inv_v_sq[it];
        float t0_val = t0_s[it];
        float velocity = 1.0f / sqrt(inv_v_sq_val);

        // Loop over output points
        for (int ix = ix_min; ix <= ix_max; ix++) {
            float ox = params.x_min + (float(ix) + 0.5f) * params.dx;

            for (int iy = iy_min; iy <= iy_max; iy++) {
                float oy = params.y_min + (float(iy) + 0.5f) * params.dy;

                float dm = sqrt((ox - mx) * (ox - mx) + (oy - my) * (oy - my));

                if (dm > aperture) continue;

                float ds2 = (ox - sx) * (ox - sx) + (oy - sy) * (oy - sy);
                float dr2 = (ox - rx) * (ox - rx) + (oy - ry) * (oy - ry);

                float t_travel = compute_dsr_traveltime(ds2, dr2, t0_half_sq_val, inv_v_sq_val);
                float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;

                if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) continue;

                float amp = linear_interp(trace, params.n_samples, sample_idx);
                amp *= compute_taper(dm, aperture, params.taper_fraction);

                if (params.apply_spreading && t_travel > 0.001f) {
                    amp *= 1.0f / (velocity * t_travel);
                }
                if (params.apply_obliquity && t_travel > 0.001f) {
                    amp *= t0_val / t_travel;
                }

                int img_idx = ix * params.ny * params.nt + iy * params.nt + it;
                atomic_add_float(&image[img_idx], amp);
            }
        }
    }
}
