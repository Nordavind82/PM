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

// ============================================================================
// OPTIMIZATION 1: Parallelize over (x, y, t) - 3D grid
// Each thread handles one output sample, processes all traces for that sample
// Threads: nx * ny * nt (e.g., 32*32*500 = 512,000)
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
    int local_fold = 0;

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

        // Spreading correction
        if (params.apply_spreading && t_travel > 0.001f) {
            float vrms = 1.0f / sqrt(inv_v_sq_val);
            amp *= 1.0f / (vrms * t_travel);
        }

        // Obliquity correction
        if (params.apply_obliquity && t_travel > 0.001f) {
            amp *= t0_val / t_travel;
        }

        local_sum += amp;
        local_fold = 1;  // At least one trace contributed
    }

    // Write to output (no atomic needed - each thread writes unique location)
    int img_idx = ix * params.ny * params.nt + iy * params.nt + iot;
    image[img_idx] = local_sum;

    // Fold update (still needs atomic since multiple time samples share same fold)
    if (local_fold > 0 && iot == 0) {  // Only count once per pillar
        // Actually, we need to track unique traces per pillar, not per sample
        // For simplicity, skip fold in this variant
    }
}

// ============================================================================
// OPTIMIZATION 2: Threadgroup shared memory for velocity data
// Cache velocity arrays in fast threadgroup memory
// ============================================================================
kernel void migrate_shared_memory(
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
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]
) {
    // Shared memory for time-dependent values (limited to 32KB per threadgroup)
    // Cache first 256 time samples (4 arrays × 256 × 4 bytes = 4KB)
    threadgroup float shared_t0_half_sq[256];
    threadgroup float shared_inv_v_sq[256];
    threadgroup float shared_t0_s[256];
    threadgroup float shared_apertures[256];

    // Cooperative load into shared memory
    uint local_idx = tid.y * tg_size.x + tid.x;
    uint tg_total = tg_size.x * tg_size.y;

    uint max_shared = min(uint(256), uint(params.nt));
    for (uint i = local_idx; i < max_shared; i += tg_total) {
        shared_t0_half_sq[i] = t0_half_sq[i];
        shared_inv_v_sq[i] = inv_v_sq[i];
        shared_t0_s[i] = t0_s[i];
        shared_apertures[i] = apertures[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int ix = gid.x;
    int iy = gid.y;

    if (ix >= params.nx || iy >= params.ny) {
        return;
    }

    float ox = x_coords[ix];
    float oy = y_coords[iy];
    int pillar_fold = 0;

    for (int it = 0; it < params.n_traces; it++) {
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
            // Use shared memory for cached values, device memory for overflow
            float aperture = (iot < 256) ? shared_apertures[iot] : apertures[iot];
            if (dm > aperture) continue;

            float t0_half_sq_val = (iot < 256) ? shared_t0_half_sq[iot] : t0_half_sq[iot];
            float inv_v_sq_val = (iot < 256) ? shared_inv_v_sq[iot] : inv_v_sq[iot];

            float t_src = sqrt(t0_half_sq_val + ds2 * inv_v_sq_val);
            float t_rec = sqrt(t0_half_sq_val + dr2 * inv_v_sq_val);
            float t_travel = t_src + t_rec;

            float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;
            if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) continue;

            int idx0 = int(sample_idx);
            float frac = sample_idx - float(idx0);
            float amp = trace_amp[idx0] * (1.0f - frac) + trace_amp[idx0 + 1] * frac;

            float taper_start = aperture * (1.0f - params.taper_fraction);
            if (dm > taper_start && dm < aperture) {
                float t = (dm - taper_start) / (aperture - taper_start);
                amp *= 0.5f * (1.0f + cos(t * M_PI_F));
            }

            if (params.apply_spreading && t_travel > 0.001f) {
                float vrms = 1.0f / sqrt(inv_v_sq_val);
                amp *= 1.0f / (vrms * t_travel);
            }

            float t0_val = (iot < 256) ? shared_t0_s[iot] : t0_s[iot];
            if (params.apply_obliquity && t_travel > 0.001f) {
                amp *= t0_val / t_travel;
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
// OPTIMIZATION 3: Process traces in batches with parallel reduction
// Each threadgroup processes a batch of traces, accumulates locally
// ============================================================================
kernel void migrate_trace_batches(
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
    constant int& trace_batch_start [[buffer(16)]],
    constant int& trace_batch_size [[buffer(17)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ix = gid.x;
    int iy = gid.y;

    if (ix >= params.nx || iy >= params.ny) {
        return;
    }

    float ox = x_coords[ix];
    float oy = y_coords[iy];

    int trace_end = (trace_batch_start + trace_batch_size < params.n_traces) ?
                    (trace_batch_start + trace_batch_size) : params.n_traces;
    int pillar_fold = 0;

    // Process only this batch of traces
    for (int it = trace_batch_start; it < trace_end; it++) {
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

            float t0_half_sq_val = t0_half_sq[iot];
            float inv_v_sq_val = inv_v_sq[iot];

            float t_src = sqrt(t0_half_sq_val + ds2 * inv_v_sq_val);
            float t_rec = sqrt(t0_half_sq_val + dr2 * inv_v_sq_val);
            float t_travel = t_src + t_rec;

            float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;
            if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) continue;

            int idx0 = int(sample_idx);
            float frac = sample_idx - float(idx0);
            float amp = trace_amp[idx0] * (1.0f - frac) + trace_amp[idx0 + 1] * frac;

            float taper_start = aperture * (1.0f - params.taper_fraction);
            if (dm > taper_start && dm < aperture) {
                float t = (dm - taper_start) / (aperture - taper_start);
                amp *= 0.5f * (1.0f + cos(t * M_PI_F));
            }

            if (params.apply_spreading && t_travel > 0.001f) {
                float vrms = 1.0f / sqrt(inv_v_sq_val);
                amp *= 1.0f / (vrms * t_travel);
            }

            if (params.apply_obliquity && t_travel > 0.001f) {
                amp *= t0_s[iot] / t_travel;
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
// OPTIMIZATION 4: SIMD-friendly vectorized inner loop
// Process 4 time samples at once using SIMD
// ============================================================================
kernel void migrate_simd_vectorized(
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
    device const float4* t0_half_sq_vec [[buffer(11)]],  // Packed as float4
    device const float4* inv_v_sq_vec [[buffer(12)]],
    device const float4* t0_s_vec [[buffer(13)]],
    device const float4* apertures_vec [[buffer(14)]],
    constant MigrationParams& params [[buffer(15)]],
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
    int nt_vec = (params.nt + 3) / 4;  // Number of float4 vectors

    for (int it = 0; it < params.n_traces; it++) {
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

        // Process 4 time samples at a time
        for (int iv = 0; iv < nt_vec; iv++) {
            float4 aperture = apertures_vec[iv];

            // Check if any of the 4 samples are within aperture
            bool4 in_aperture = dm <= aperture;
            if (!any(in_aperture)) continue;

            float4 t0_half_sq_v = t0_half_sq_vec[iv];
            float4 inv_v_sq_v = inv_v_sq_vec[iv];

            // Vectorized DSR
            float4 t_src = sqrt(t0_half_sq_v + ds2 * inv_v_sq_v);
            float4 t_rec = sqrt(t0_half_sq_v + dr2 * inv_v_sq_v);
            float4 t_travel = t_src + t_rec;

            float4 sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;

            // Process each of the 4 samples
            for (int k = 0; k < 4; k++) {
                int iot = iv * 4 + k;
                if (iot >= params.nt) break;
                if (!in_aperture[k]) continue;

                float sidx = sample_idx[k];
                if (sidx < 0.0f || sidx >= float(params.n_samples - 1)) continue;

                int idx0 = int(sidx);
                float frac = sidx - float(idx0);
                float amp = trace_amp[idx0] * (1.0f - frac) + trace_amp[idx0 + 1] * frac;

                float ap = aperture[k];
                float taper_start = ap * (1.0f - params.taper_fraction);
                if (dm > taper_start && dm < ap) {
                    float t = (dm - taper_start) / (ap - taper_start);
                    amp *= 0.5f * (1.0f + cos(t * M_PI_F));
                }

                float tt = t_travel[k];
                if (params.apply_spreading && tt > 0.001f) {
                    float vrms = 1.0f / sqrt(inv_v_sq_v[k]);
                    amp *= 1.0f / (vrms * tt);
                }

                if (params.apply_obliquity && tt > 0.001f) {
                    amp *= t0_s_vec[iv][k] / tt;
                }

                int img_idx = ix * params.ny * params.nt + iy * params.nt + iot;
                atomic_fetch_add_explicit(&image[img_idx], amp, memory_order_relaxed);
                trace_contributed = true;
            }
        }

        if (trace_contributed) pillar_fold++;
    }

    if (pillar_fold > 0) {
        int fold_idx = ix * params.ny + iy;
        atomic_fetch_add_explicit(&fold[fold_idx], pillar_fold, memory_order_relaxed);
    }
}
