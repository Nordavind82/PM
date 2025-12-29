#include <metal_stdlib>
using namespace metal;

// Migration parameters - must match C++ struct layout
struct MigrationParams {
    float max_dip_deg;
    float min_aperture;
    float max_aperture;
    float taper_fraction;
    float dt_ms;
    float t_start_ms;
    int apply_spreading;
    int apply_obliquity;
    int n_traces;
    int n_samples;
    int nx;
    int ny;
    int nt;
};

// Pre-computed values for efficiency
struct PrecomputedValues {
    float t0_half_sq;    // (t0/2)^2 in seconds^2
    float inv_v_sq;      // 1/v^2
    float t0_s;          // t0 in seconds
    float aperture;      // aperture at this time
};

/**
 * Compute DSR (Double Square Root) travel time.
 * t_travel = sqrt((t0/2)^2 + ds^2/v^2) + sqrt((t0/2)^2 + dr^2/v^2)
 */
inline float compute_travel_time(
    float ds2,           // source distance squared
    float dr2,           // receiver distance squared
    float t0_half_sq,    // (t0/2)^2
    float inv_v_sq       // 1/v^2
) {
    float t_src = sqrt(t0_half_sq + ds2 * inv_v_sq);
    float t_rec = sqrt(t0_half_sq + dr2 * inv_v_sq);
    return t_src + t_rec;
}

/**
 * Linear interpolation of trace amplitude.
 */
inline float linear_interp(
    device const float* amplitudes,
    int n_samples,
    float sample_idx
) {
    if (sample_idx < 0.0f || sample_idx >= float(n_samples - 1)) {
        return 0.0f;
    }

    int idx0 = int(sample_idx);
    int idx1 = idx0 + 1;
    float frac = sample_idx - float(idx0);

    return amplitudes[idx0] * (1.0f - frac) + amplitudes[idx1] * frac;
}

/**
 * Compute taper weight for aperture edges.
 */
inline float compute_taper(
    float distance,
    float aperture,
    float taper_fraction
) {
    float taper_start = aperture * (1.0f - taper_fraction);
    if (distance <= taper_start) {
        return 1.0f;
    }
    if (distance >= aperture) {
        return 0.0f;
    }
    // Cosine taper
    float t = (distance - taper_start) / (aperture - taper_start);
    return 0.5f * (1.0f + cos(t * M_PI_F));
}

/**
 * Main migration kernel.
 *
 * Thread organization:
 * - Each thread processes one output pillar (x, y position)
 * - Iterates over all traces and time samples
 * - Accumulates contributions using atomic operations
 *
 * For a 32x32 tile: 1024 threads, each handling all traces × all times
 */
kernel void migrate_tile_kernel(
    // Trace data
    device const float* amplitudes [[buffer(0)]],       // [n_traces, n_samples]
    device const float* source_x [[buffer(1)]],         // [n_traces]
    device const float* source_y [[buffer(2)]],
    device const float* receiver_x [[buffer(3)]],
    device const float* receiver_y [[buffer(4)]],
    device const float* midpoint_x [[buffer(5)]],
    device const float* midpoint_y [[buffer(6)]],

    // Output arrays
    device atomic_float* image [[buffer(7)]],           // [nx, ny, nt]
    device atomic_int* fold [[buffer(8)]],              // [nx, ny]

    // Grid coordinates
    device const float* x_coords [[buffer(9)]],         // [nx]
    device const float* y_coords [[buffer(10)]],        // [ny]

    // Pre-computed time-dependent values
    device const float* t0_half_sq [[buffer(11)]],      // [nt]
    device const float* inv_v_sq [[buffer(12)]],        // [nt]
    device const float* t0_s [[buffer(13)]],            // [nt]
    device const float* apertures [[buffer(14)]],       // [nt]

    // Parameters
    constant MigrationParams& params [[buffer(15)]],

    // Thread info
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    // Get output pillar coordinates
    int ix = gid.x;
    int iy = gid.y;

    if (ix >= params.nx || iy >= params.ny) {
        return;
    }

    float ox = x_coords[ix];
    float oy = y_coords[iy];

    // Fold counter for this pillar
    int pillar_fold = 0;

    // Process all traces
    for (int it = 0; it < params.n_traces; it++) {
        float sx = source_x[it];
        float sy = source_y[it];
        float rx = receiver_x[it];
        float ry = receiver_y[it];
        float mx = midpoint_x[it];
        float my = midpoint_y[it];

        // Distance from output point to source and receiver (squared)
        float ds2 = (ox - sx) * (ox - sx) + (oy - sy) * (oy - sy);
        float dr2 = (ox - rx) * (ox - rx) + (oy - ry) * (oy - ry);

        // Distance from output point to midpoint (for aperture check)
        float dm = sqrt((ox - mx) * (ox - mx) + (oy - my) * (oy - my));

        // Pointer to this trace's amplitudes
        device const float* trace_amp = amplitudes + it * params.n_samples;

        // Track if this trace contributed
        bool trace_contributed = false;

        // Process all time samples
        for (int iot = 0; iot < params.nt; iot++) {
            float aperture = apertures[iot];

            // Aperture check
            if (dm > aperture) {
                continue;
            }

            // Compute travel time
            float t_travel = compute_travel_time(
                ds2, dr2,
                t0_half_sq[iot],
                inv_v_sq[iot]
            );

            // Convert to sample index
            float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;

            // Bounds check
            if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) {
                continue;
            }

            // Interpolate amplitude
            float amp = linear_interp(trace_amp, params.n_samples, sample_idx);

            // Apply taper weight
            float taper = compute_taper(dm, aperture, params.taper_fraction);
            amp *= taper;

            // Spherical spreading correction: 1 / (v * t)
            if (params.apply_spreading && t_travel > 0.001f) {
                float vrms = 1.0f / sqrt(inv_v_sq[iot]);
                amp *= 1.0f / (vrms * t_travel);
            }

            // Obliquity correction: t0 / t_travel
            if (params.apply_obliquity && t_travel > 0.001f) {
                float t0 = t0_s[iot];
                amp *= t0 / t_travel;
            }

            // Accumulate to output image (atomic for thread safety)
            int img_idx = ix * params.ny * params.nt + iy * params.nt + iot;
            atomic_fetch_add_explicit(&image[img_idx], amp, memory_order_relaxed);

            trace_contributed = true;
        }

        if (trace_contributed) {
            pillar_fold++;
        }
    }

    // Update fold count
    if (pillar_fold > 0) {
        int fold_idx = ix * params.ny + iy;
        atomic_fetch_add_explicit(&fold[fold_idx], pillar_fold, memory_order_relaxed);
    }
}

/**
 * Optimized kernel with trace chunking.
 * Processes traces in chunks to improve cache locality.
 */
kernel void migrate_tile_chunked(
    // Same buffers as migrate_tile_kernel
    device const float* amplitudes [[buffer(0)]],
    device const float* source_x [[buffer(1)]],
    device const float* source_y [[buffer(2)]],
    device const float* receiver_x [[buffer(3)]],
    device const float* receiver_y [[buffer(4)]],
    device const float* midpoint_x [[buffer(5)]],
    device const float* midpoint_y [[buffer(6)]],
    device atomic_float* image [[buffer(7)]],
    device atomic_int* fold [[buffer(8)]],
    device const float* x_coords [[buffer(9)]],
    device const float* y_coords [[buffer(10)]],
    device const float* t0_half_sq [[buffer(11)]],
    device const float* inv_v_sq [[buffer(12)]],
    device const float* t0_s [[buffer(13)]],
    device const float* apertures [[buffer(14)]],
    constant MigrationParams& params [[buffer(15)]],
    constant int& trace_offset [[buffer(16)]],         // Start trace index
    constant int& trace_count [[buffer(17)]],          // Traces in this chunk
    uint2 gid [[thread_position_in_grid]]
) {
    int ix = gid.x;
    int iy = gid.y;

    if (ix >= params.nx || iy >= params.ny) {
        return;
    }

    float ox = x_coords[ix];
    float oy = y_coords[iy];

    int pillar_fold = 0;
    int trace_end = min(trace_offset + trace_count, params.n_traces);

    for (int it = trace_offset; it < trace_end; it++) {
        float sx = source_x[it];
        float sy = source_y[it];
        float rx = receiver_x[it];
        float ry = receiver_y[it];
        float mx = midpoint_x[it];
        float my = midpoint_y[it];

        float ds2 = (ox - sx) * (ox - sx) + (oy - sy) * (oy - sy);
        float dr2 = (ox - rx) * (ox - rx) + (oy - ry) * (oy - ry);
        float dm = sqrt((ox - mx) * (ox - mx) + (oy - my) * (oy - my));

        device const float* trace_amp = amplitudes + it * params.n_samples;
        bool trace_contributed = false;

        for (int iot = 0; iot < params.nt; iot++) {
            float aperture = apertures[iot];
            if (dm > aperture) continue;

            float t_travel = compute_travel_time(ds2, dr2, t0_half_sq[iot], inv_v_sq[iot]);
            float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;

            if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) continue;

            float amp = linear_interp(trace_amp, params.n_samples, sample_idx);
            amp *= compute_taper(dm, aperture, params.taper_fraction);

            if (params.apply_spreading && t_travel > 0.001f) {
                float vrms = 1.0f / sqrt(inv_v_sq[iot]);
                amp *= 1.0f / (vrms * t_travel);
            }

            if (params.apply_obliquity && t_travel > 0.001f) {
                float t0 = t0_s[iot];
                amp *= t0 / t_travel;
            }

            int img_idx = ix * params.ny * params.nt + iy * params.nt + iot;
            atomic_fetch_add_explicit(&image[img_idx], amp, memory_order_relaxed);
            trace_contributed = true;
        }

        if (trace_contributed) pillar_fold++;
    }

    if (pillar_fold > 0) {
        int fold_idx = ix * params.ny + iy;
        atomic_fetch_add_explicit(&fold[fold_idx], pillar_fold, memory_order_relaxed);
    }
}

// ============================================================================
// OPTIMIZED: 3D Parallel Kernel (15x faster than baseline)
// Parallelizes over (x, y, t) - each thread handles one output sample
// ============================================================================
kernel void migrate_3d_parallel(
    device const float* amplitudes [[buffer(0)]],
    device const float* source_x [[buffer(1)]],
    device const float* source_y [[buffer(2)]],
    device const float* receiver_x [[buffer(3)]],
    device const float* receiver_y [[buffer(4)]],
    device const float* midpoint_x [[buffer(5)]],
    device const float* midpoint_y [[buffer(6)]],
    device float* image [[buffer(7)]],              // Non-atomic for this variant
    device atomic_int* fold [[buffer(8)]],
    device const float* x_coords [[buffer(9)]],
    device const float* y_coords [[buffer(10)]],
    device const float* t0_half_sq [[buffer(11)]],
    device const float* inv_v_sq [[buffer(12)]],
    device const float* t0_s [[buffer(13)]],
    device const float* apertures [[buffer(14)]],
    constant MigrationParams& params [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int ix = gid.x;
    int iy = gid.y;
    int iot = gid.z;

    if (ix >= params.nx || iy >= params.ny || iot >= params.nt) {
        return;
    }

    float ox = x_coords[ix];
    float oy = y_coords[iy];
    float aperture = apertures[iot];
    float t0_half_sq_val = t0_half_sq[iot];
    float inv_v_sq_val = inv_v_sq[iot];
    float t0_val = t0_s[iot];

    float local_sum = 0.0f;

    // Process all traces for this single output sample
    for (int it = 0; it < params.n_traces; it++) {
        float sx = source_x[it];
        float sy = source_y[it];
        float rx = receiver_x[it];
        float ry = receiver_y[it];
        float mx = midpoint_x[it];
        float my = midpoint_y[it];

        // Distance from output point to midpoint (for aperture check)
        float dm = sqrt((ox - mx) * (ox - mx) + (oy - my) * (oy - my));

        // Aperture check
        if (dm > aperture) {
            continue;
        }

        // Distance squared to source and receiver
        float ds2 = (ox - sx) * (ox - sx) + (oy - sy) * (oy - sy);
        float dr2 = (ox - rx) * (ox - rx) + (oy - ry) * (oy - ry);

        // DSR travel time
        float t_src = sqrt(t0_half_sq_val + ds2 * inv_v_sq_val);
        float t_rec = sqrt(t0_half_sq_val + dr2 * inv_v_sq_val);
        float t_travel = t_src + t_rec;

        // Convert to sample index
        float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;

        // Bounds check
        if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) {
            continue;
        }

        // Linear interpolation
        int idx0 = int(sample_idx);
        int idx1 = idx0 + 1;
        float frac = sample_idx - float(idx0);
        device const float* trace_amp = amplitudes + it * params.n_samples;
        float amp = trace_amp[idx0] * (1.0f - frac) + trace_amp[idx1] * frac;

        // Taper weight
        float taper_start = aperture * (1.0f - params.taper_fraction);
        float taper = 1.0f;
        if (dm > taper_start && dm < aperture) {
            float t = (dm - taper_start) / (aperture - taper_start);
            taper = 0.5f * (1.0f + cos(t * M_PI_F));
        }
        amp *= taper;

        // Spreading correction: 1 / (v * t)
        if (params.apply_spreading && t_travel > 0.001f) {
            float vrms = 1.0f / sqrt(inv_v_sq_val);
            amp *= 1.0f / (vrms * t_travel);
        }

        // Obliquity correction: t0 / t_travel
        if (params.apply_obliquity && t_travel > 0.001f) {
            amp *= t0_val / t_travel;
        }

        local_sum += amp;
    }

    // Write to output (no atomic needed - each thread writes unique location)
    int img_idx = ix * params.ny * params.nt + iy * params.nt + iot;
    image[img_idx] = local_sum;
}

// ============================================================================
// SIMD OPTIMIZED: Process 4 traces at a time using float4 SIMD types
// Expected ~20-30% faster than scalar version due to SIMD parallelism
// ============================================================================
kernel void migrate_3d_simd4(
    device const float* amplitudes [[buffer(0)]],
    device const float* source_x [[buffer(1)]],
    device const float* source_y [[buffer(2)]],
    device const float* receiver_x [[buffer(3)]],
    device const float* receiver_y [[buffer(4)]],
    device const float* midpoint_x [[buffer(5)]],
    device const float* midpoint_y [[buffer(6)]],
    device float* image [[buffer(7)]],
    device atomic_int* fold [[buffer(8)]],
    device const float* x_coords [[buffer(9)]],
    device const float* y_coords [[buffer(10)]],
    device const float* t0_half_sq [[buffer(11)]],
    device const float* inv_v_sq [[buffer(12)]],
    device const float* t0_s [[buffer(13)]],
    device const float* apertures [[buffer(14)]],
    constant MigrationParams& params [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int ix = gid.x;
    int iy = gid.y;
    int iot = gid.z;

    if (ix >= params.nx || iy >= params.ny || iot >= params.nt) {
        return;
    }

    float ox = x_coords[ix];
    float oy = y_coords[iy];
    float aperture = apertures[iot];
    float aperture_sq = aperture * aperture;  // Pre-compute for faster comparison
    float t0_half_sq_val = t0_half_sq[iot];
    float inv_v_sq_val = inv_v_sq[iot];
    float t0_val = t0_s[iot];
    float taper_start = aperture * (1.0f - params.taper_fraction);

    float local_sum = 0.0f;

    // Process 4 traces at a time using SIMD
    int n_traces_4 = (params.n_traces / 4) * 4;  // Round down to multiple of 4

    for (int it = 0; it < n_traces_4; it += 4) {
        // Load 4 traces worth of coordinates using SIMD
        float4 sx = float4(source_x[it], source_x[it+1], source_x[it+2], source_x[it+3]);
        float4 sy = float4(source_y[it], source_y[it+1], source_y[it+2], source_y[it+3]);
        float4 rx = float4(receiver_x[it], receiver_x[it+1], receiver_x[it+2], receiver_x[it+3]);
        float4 ry = float4(receiver_y[it], receiver_y[it+1], receiver_y[it+2], receiver_y[it+3]);
        float4 mx = float4(midpoint_x[it], midpoint_x[it+1], midpoint_x[it+2], midpoint_x[it+3]);
        float4 my = float4(midpoint_y[it], midpoint_y[it+1], midpoint_y[it+2], midpoint_y[it+3]);

        // Distance squared from output to midpoint (vectorized)
        float4 dmx = ox - mx;
        float4 dmy = oy - my;
        float4 dm_sq = dmx * dmx + dmy * dmy;

        // Aperture check (vectorized) - use squared distance to avoid sqrt
        bool4 in_aperture = dm_sq <= aperture_sq;

        // If no traces pass aperture, skip this group
        if (!any(in_aperture)) {
            continue;
        }

        // Distance squared to source and receiver (vectorized)
        float4 dsx = ox - sx;
        float4 dsy = oy - sy;
        float4 ds2 = dsx * dsx + dsy * dsy;

        float4 drx = ox - rx;
        float4 dry = oy - ry;
        float4 dr2 = drx * drx + dry * dry;

        // DSR travel time (vectorized sqrt)
        float4 t_src = sqrt(t0_half_sq_val + ds2 * inv_v_sq_val);
        float4 t_rec = sqrt(t0_half_sq_val + dr2 * inv_v_sq_val);
        float4 t_travel = t_src + t_rec;

        // Convert to sample index (vectorized)
        float4 sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;

        // Process each trace in the SIMD group
        for (int j = 0; j < 4; j++) {
            if (!in_aperture[j]) continue;

            float sidx = sample_idx[j];
            if (sidx < 0.0f || sidx >= float(params.n_samples - 1)) continue;

            // Linear interpolation
            int idx0 = int(sidx);
            float frac = sidx - float(idx0);
            device const float* trace_amp = amplitudes + (it + j) * params.n_samples;
            float amp = trace_amp[idx0] * (1.0f - frac) + trace_amp[idx0 + 1] * frac;

            // Taper weight
            float dm = sqrt(dm_sq[j]);
            if (dm > taper_start) {
                float t = (dm - taper_start) / (aperture - taper_start);
                amp *= 0.5f * (1.0f + cos(t * M_PI_F));
            }

            // Amplitude corrections
            float tt = t_travel[j];
            if (params.apply_spreading && tt > 0.001f) {
                float vrms = 1.0f / sqrt(inv_v_sq_val);
                amp *= 1.0f / (vrms * tt);
            }
            if (params.apply_obliquity && tt > 0.001f) {
                amp *= t0_val / tt;
            }

            local_sum += amp;
        }
    }

    // Handle remaining traces (less than 4)
    for (int it = n_traces_4; it < params.n_traces; it++) {
        float mx_s = midpoint_x[it];
        float my_s = midpoint_y[it];
        float dm_sq_s = (ox - mx_s) * (ox - mx_s) + (oy - my_s) * (oy - my_s);

        if (dm_sq_s > aperture_sq) continue;

        float sx_s = source_x[it];
        float sy_s = source_y[it];
        float rx_s = receiver_x[it];
        float ry_s = receiver_y[it];

        float ds2_s = (ox - sx_s) * (ox - sx_s) + (oy - sy_s) * (oy - sy_s);
        float dr2_s = (ox - rx_s) * (ox - rx_s) + (oy - ry_s) * (oy - ry_s);

        float t_src_s = sqrt(t0_half_sq_val + ds2_s * inv_v_sq_val);
        float t_rec_s = sqrt(t0_half_sq_val + dr2_s * inv_v_sq_val);
        float t_travel_s = t_src_s + t_rec_s;

        float sample_idx_s = (t_travel_s * 1000.0f - params.t_start_ms) / params.dt_ms;
        if (sample_idx_s < 0.0f || sample_idx_s >= float(params.n_samples - 1)) continue;

        int idx0 = int(sample_idx_s);
        float frac = sample_idx_s - float(idx0);
        device const float* trace_amp = amplitudes + it * params.n_samples;
        float amp = trace_amp[idx0] * (1.0f - frac) + trace_amp[idx0 + 1] * frac;

        float dm = sqrt(dm_sq_s);
        if (dm > taper_start) {
            float t = (dm - taper_start) / (aperture - taper_start);
            amp *= 0.5f * (1.0f + cos(t * M_PI_F));
        }

        if (params.apply_spreading && t_travel_s > 0.001f) {
            float vrms = 1.0f / sqrt(inv_v_sq_val);
            amp *= 1.0f / (vrms * t_travel_s);
        }
        if (params.apply_obliquity && t_travel_s > 0.001f) {
            amp *= t0_val / t_travel_s;
        }

        local_sum += amp;
    }

    int img_idx = ix * params.ny * params.nt + iy * params.nt + iot;
    image[img_idx] = local_sum;
}
