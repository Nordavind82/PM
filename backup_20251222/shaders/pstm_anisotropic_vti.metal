#include <metal_stdlib>
using namespace metal;

// =============================================================================
// PSTM Anisotropic VTI Migration Kernel
// Implements Alkhalifah-Tsvankin (1995) formulation using eta parameter
// =============================================================================

// Migration parameters for VTI anisotropic kernel
struct VTIParams {
    // Grid parameters
    float dx;                    // Output grid spacing X (meters)
    float dy;                    // Output grid spacing Y (meters)
    float dt_ms;                 // Time sampling (ms)
    float t_start_ms;            // Start time (ms)

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

    // VTI anisotropy parameters
    float eta_constant;          // Constant eta (used if eta_is_1d = 0)
    int eta_is_1d;               // Whether eta varies with time

    // Dimensions
    int n_traces;
    int n_samples;
    int nx;
    int ny;
    int nt;
};

// =============================================================================
// VTI Traveltime Functions
// =============================================================================

/**
 * Compute VTI non-hyperbolic moveout correction term.
 *
 * From Alkhalifah (1998):
 * t²(x) = t₀² + x²/V² - 2η·x⁴ / [V²·(t₀²·V² + (1+2η)·x²)]
 *
 * This function returns the correction term that is subtracted from t²_hyperbolic.
 */
inline float vti_correction_term(
    float t0_s,          // Zero-offset two-way time (s)
    float x,             // Offset (m)
    float v_nmo,         // NMO velocity (m/s)
    float eta            // Anisotropy parameter
) {
    if (v_nmo <= 0.0f || t0_s <= 0.0f) return 0.0f;

    float x2 = x * x;
    float v2 = v_nmo * v_nmo;
    float t02 = t0_s * t0_s;

    float denom = v2 * (t02 * v2 + (1.0f + 2.0f * eta) * x2);

    if (abs(denom) < 1e-20f) return 0.0f;

    return (2.0f * eta * x2 * x2) / denom;
}

/**
 * Compute VTI non-hyperbolic moveout traveltime.
 *
 * t²(x) = t₀² + x²/V² - correction
 */
inline float vti_traveltime(
    float t0_s,          // Zero-offset two-way time (s)
    float x,             // Offset (m)
    float v_nmo,         // NMO velocity (m/s)
    float eta            // Anisotropy parameter
) {
    if (v_nmo <= 0.0f || t0_s <= 0.0f) return t0_s;

    float x2 = x * x;
    float v2 = v_nmo * v_nmo;
    float t02 = t0_s * t0_s;

    // Hyperbolic term
    float t2_hyper = t02 + x2 / v2;

    // Non-hyperbolic correction
    float correction = vti_correction_term(t0_s, x, v_nmo, eta);

    float t2 = t2_hyper - correction;

    return sqrt(max(t2, 0.0f));
}

/**
 * Compute one-way VTI traveltime for a single ray leg.
 */
inline float vti_one_way_time(
    float t0_one_way,    // One-way zero-offset time (s)
    float h,             // Horizontal distance (m)
    float v_nmo,         // NMO velocity (m/s)
    float eta            // Anisotropy parameter
) {
    if (v_nmo <= 0.0f || t0_one_way <= 0.0f) return t0_one_way;

    float h2 = h * h;
    float v2 = v_nmo * v_nmo;
    float t02 = t0_one_way * t0_one_way;

    // Hyperbolic term
    float t2_hyper = t02 + h2 / v2;

    // Non-hyperbolic correction (scaled for one-way)
    float denom = v2 * (t02 * v2 + (1.0f + 2.0f * eta) * h2);

    float correction = 0.0f;
    if (abs(denom) > 1e-20f) {
        correction = (2.0f * eta * h2 * h2) / denom;
    }

    float t2 = t2_hyper - correction;

    return sqrt(max(t2, 0.0f));
}

/**
 * Compute DSR traveltime with VTI corrections.
 * Applies non-hyperbolic correction to both source and receiver legs.
 */
inline float vti_dsr_traveltime(
    float t0_s,          // Zero-offset two-way time (s)
    float h_s,           // Source horizontal distance from image point (m)
    float h_r,           // Receiver horizontal distance from image point (m)
    float v_nmo,         // NMO velocity (m/s)
    float eta            // Anisotropy parameter
) {
    float t0_half = t0_s / 2.0f;

    float t_s = vti_one_way_time(t0_half, h_s, v_nmo, eta);
    float t_r = vti_one_way_time(t0_half, h_r, v_nmo, eta);

    return t_s + t_r;
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
 * Compute cosine taper weight for aperture edges.
 */
inline float compute_taper(float distance, float aperture, float taper_fraction) {
    float taper_start = aperture * (1.0f - taper_fraction);
    if (distance <= taper_start) return 1.0f;
    if (distance >= aperture) return 0.0f;
    float t = (distance - taper_start) / (aperture - taper_start);
    return 0.5f * (1.0f + cos(t * M_PI_F));
}

/**
 * Compute anti-aliasing weight based on local dip.
 */
inline float compute_aa_weight(
    float ox, float oy,
    float mx, float my,
    float velocity,
    float t_travel,
    float dx, float dy,
    float dominant_freq
) {
    if (t_travel < 0.001f) return 1.0f;

    float denom = velocity * velocity * t_travel * 0.5f;
    float dip_x = (mx - ox) / denom;
    float dip_y = (my - oy) / denom;

    float sin_theta_x = min(abs(dip_x) * velocity * 0.5f, 1.0f);
    float sin_theta_y = min(abs(dip_y) * velocity * 0.5f, 1.0f);
    float sin_theta = max(sin_theta_x, sin_theta_y);

    if (sin_theta < 0.01f) return 1.0f;

    float grid_spacing = max(dx, dy);
    float f_max = velocity / (4.0f * grid_spacing * sin_theta);
    f_max = min(f_max, 500.0f);

    return max(0.0f, 1.0f - dominant_freq / f_max);
}

// =============================================================================
// VTI Anisotropic Migration Kernel - 3D Parallel
// =============================================================================

kernel void pstm_migrate_vti(
    // Input trace data
    device const float* amplitudes [[buffer(0)]],
    device const float* source_x [[buffer(1)]],
    device const float* source_y [[buffer(2)]],
    device const float* receiver_x [[buffer(3)]],
    device const float* receiver_y [[buffer(4)]],
    device const float* midpoint_x [[buffer(5)]],
    device const float* midpoint_y [[buffer(6)]],

    // Output arrays
    device float* image [[buffer(7)]],
    device atomic_int* fold [[buffer(8)]],

    // Grid coordinates
    device const float* x_coords [[buffer(9)]],
    device const float* y_coords [[buffer(10)]],

    // Pre-computed time-dependent values
    device const float* t0_s [[buffer(11)]],          // [nt] - t0 in seconds
    device const float* inv_v_sq [[buffer(12)]],      // [nt] - 1/v^2
    device const float* apertures [[buffer(13)]],     // [nt] - aperture at each time

    // Eta array (1D or constant)
    device const float* eta_array [[buffer(14)]],     // [nt] if eta_is_1d, else unused

    // Parameters
    constant VTIParams& params [[buffer(15)]],

    // Thread position
    uint3 gid [[thread_position_in_grid]]
) {
    int ix = gid.x;
    int iy = gid.y;
    int it = gid.z;

    // Bounds check
    if (ix >= params.nx || iy >= params.ny || it >= params.nt) {
        return;
    }

    // Get output point coordinates
    float ox = x_coords[ix];
    float oy = y_coords[iy];

    // Get time-dependent values for this output time
    float aperture = apertures[it];
    float inv_v_sq_val = inv_v_sq[it];
    float t0_val = t0_s[it];
    float velocity = 1.0f / sqrt(inv_v_sq_val);

    // Get eta for this time
    float eta = params.eta_is_1d ? eta_array[it] : params.eta_constant;

    // Accumulator
    float local_sum = 0.0f;
    int trace_count = 0;

    // Process all input traces
    for (int tr = 0; tr < params.n_traces; tr++) {
        // Get trace geometry
        float sx = source_x[tr];
        float sy = source_y[tr];
        float rx = receiver_x[tr];
        float ry = receiver_y[tr];
        float mx = midpoint_x[tr];
        float my = midpoint_y[tr];

        // Distance from output point to midpoint (for aperture check)
        float dm = sqrt((ox - mx) * (ox - mx) + (oy - my) * (oy - my));

        // Aperture check
        if (dm > aperture) {
            continue;
        }

        // Horizontal distances to source and receiver
        float h_s = sqrt((ox - sx) * (ox - sx) + (oy - sy) * (oy - sy));
        float h_r = sqrt((ox - rx) * (ox - rx) + (oy - ry) * (oy - ry));

        // Compute VTI DSR traveltime
        float t_travel = vti_dsr_traveltime(t0_val, h_s, h_r, velocity, eta);

        // Convert to sample index
        float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;

        // Bounds check on input trace
        if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) {
            continue;
        }

        // Get interpolated amplitude
        device const float* trace = amplitudes + tr * params.n_samples;
        float amp = linear_interp(trace, params.n_samples, sample_idx);

        // Apply aperture taper
        amp *= compute_taper(dm, aperture, params.taper_fraction);

        // Apply anti-aliasing weight
        if (params.apply_aa) {
            float aa_weight = compute_aa_weight(
                ox, oy, mx, my,
                velocity, t_travel,
                params.dx, params.dy,
                params.aa_dominant_freq
            );
            amp *= aa_weight;
        }

        // Apply geometrical spreading correction
        if (params.apply_spreading && t_travel > 0.001f) {
            amp *= 1.0f / (velocity * t_travel);
        }

        // Apply obliquity correction: t0 / t_travel
        if (params.apply_obliquity && t_travel > 0.001f) {
            amp *= t0_val / t_travel;
        }

        local_sum += amp;
        trace_count++;
    }

    // Write to output image
    int img_idx = ix * params.ny * params.nt + iy * params.nt + it;
    image[img_idx] = local_sum;

    // Update fold (only for first time sample)
    if (it == 0 && trace_count > 0) {
        int fold_idx = ix * params.ny + iy;
        atomic_fetch_add_explicit(&fold[fold_idx], trace_count, memory_order_relaxed);
    }
}

// =============================================================================
// SIMD Optimized VTI Kernel
// =============================================================================

kernel void pstm_migrate_vti_simd(
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
    device const float* t0_s [[buffer(11)]],
    device const float* inv_v_sq [[buffer(12)]],
    device const float* apertures [[buffer(13)]],
    device const float* eta_array [[buffer(14)]],
    constant VTIParams& params [[buffer(15)]],
    uint3 gid [[thread_position_in_grid]]
) {
    int ix = gid.x;
    int iy = gid.y;
    int it = gid.z;

    if (ix >= params.nx || iy >= params.ny || it >= params.nt) {
        return;
    }

    float ox = x_coords[ix];
    float oy = y_coords[iy];
    float aperture = apertures[it];
    float aperture_sq = aperture * aperture;
    float inv_v_sq_val = inv_v_sq[it];
    float t0_val = t0_s[it];
    float velocity = 1.0f / sqrt(inv_v_sq_val);
    float eta = params.eta_is_1d ? eta_array[it] : params.eta_constant;
    float taper_start = aperture * (1.0f - params.taper_fraction);

    float local_sum = 0.0f;
    int trace_count = 0;

    // Process 4 traces at a time using SIMD
    int n_traces_4 = (params.n_traces / 4) * 4;

    for (int tr = 0; tr < n_traces_4; tr += 4) {
        // Load 4 traces using SIMD types
        float4 sx = float4(source_x[tr], source_x[tr+1], source_x[tr+2], source_x[tr+3]);
        float4 sy = float4(source_y[tr], source_y[tr+1], source_y[tr+2], source_y[tr+3]);
        float4 rx = float4(receiver_x[tr], receiver_x[tr+1], receiver_x[tr+2], receiver_x[tr+3]);
        float4 ry = float4(receiver_y[tr], receiver_y[tr+1], receiver_y[tr+2], receiver_y[tr+3]);
        float4 mx = float4(midpoint_x[tr], midpoint_x[tr+1], midpoint_x[tr+2], midpoint_x[tr+3]);
        float4 my = float4(midpoint_y[tr], midpoint_y[tr+1], midpoint_y[tr+2], midpoint_y[tr+3]);

        // Vectorized distance calculation
        float4 dmx = ox - mx;
        float4 dmy = oy - my;
        float4 dm_sq = dmx * dmx + dmy * dmy;

        // Aperture check (vectorized)
        bool4 in_aperture = dm_sq <= aperture_sq;
        if (!any(in_aperture)) continue;

        // Process each trace in SIMD group
        for (int j = 0; j < 4; j++) {
            if (!in_aperture[j]) continue;

            // Horizontal distances
            float h_s = sqrt((ox - sx[j]) * (ox - sx[j]) + (oy - sy[j]) * (oy - sy[j]));
            float h_r = sqrt((ox - rx[j]) * (ox - rx[j]) + (oy - ry[j]) * (oy - ry[j]));

            // VTI traveltime
            float t_travel = vti_dsr_traveltime(t0_val, h_s, h_r, velocity, eta);

            float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;
            if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) continue;

            // Interpolate amplitude
            int idx0 = int(sample_idx);
            float frac = sample_idx - float(idx0);
            device const float* trace = amplitudes + (tr + j) * params.n_samples;
            float amp = trace[idx0] * (1.0f - frac) + trace[idx0 + 1] * frac;

            // Taper
            float dm = sqrt(dm_sq[j]);
            if (dm > taper_start) {
                float t = (dm - taper_start) / (aperture - taper_start);
                amp *= 0.5f * (1.0f + cos(t * M_PI_F));
            }

            // Anti-aliasing
            if (params.apply_aa) {
                float aa_weight = compute_aa_weight(
                    ox, oy, mx[j], my[j],
                    velocity, t_travel,
                    params.dx, params.dy,
                    params.aa_dominant_freq
                );
                amp *= aa_weight;
            }

            // Amplitude corrections
            if (params.apply_spreading && t_travel > 0.001f) {
                amp *= 1.0f / (velocity * t_travel);
            }
            if (params.apply_obliquity && t_travel > 0.001f) {
                amp *= t0_val / t_travel;
            }

            local_sum += amp;
            trace_count++;
        }
    }

    // Handle remaining traces
    for (int tr = n_traces_4; tr < params.n_traces; tr++) {
        float mx_s = midpoint_x[tr];
        float my_s = midpoint_y[tr];
        float dm_sq_s = (ox - mx_s) * (ox - mx_s) + (oy - my_s) * (oy - my_s);

        if (dm_sq_s > aperture_sq) continue;

        float h_s = sqrt((ox - source_x[tr]) * (ox - source_x[tr]) +
                         (oy - source_y[tr]) * (oy - source_y[tr]));
        float h_r = sqrt((ox - receiver_x[tr]) * (ox - receiver_x[tr]) +
                         (oy - receiver_y[tr]) * (oy - receiver_y[tr]));

        float t_travel = vti_dsr_traveltime(t0_val, h_s, h_r, velocity, eta);

        float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;
        if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) continue;

        int idx0 = int(sample_idx);
        float frac = sample_idx - float(idx0);
        device const float* trace = amplitudes + tr * params.n_samples;
        float amp = trace[idx0] * (1.0f - frac) + trace[idx0 + 1] * frac;

        float dm = sqrt(dm_sq_s);
        if (dm > taper_start) {
            float t = (dm - taper_start) / (aperture - taper_start);
            amp *= 0.5f * (1.0f + cos(t * M_PI_F));
        }

        if (params.apply_aa) {
            float aa_weight = compute_aa_weight(
                ox, oy, mx_s, my_s,
                velocity, t_travel,
                params.dx, params.dy,
                params.aa_dominant_freq
            );
            amp *= aa_weight;
        }

        if (params.apply_spreading && t_travel > 0.001f) {
            amp *= 1.0f / (velocity * t_travel);
        }
        if (params.apply_obliquity && t_travel > 0.001f) {
            amp *= t0_val / t_travel;
        }

        local_sum += amp;
        trace_count++;
    }

    int img_idx = ix * params.ny * params.nt + iy * params.nt + it;
    image[img_idx] = local_sum;

    if (it == 0 && trace_count > 0) {
        int fold_idx = ix * params.ny + iy;
        atomic_fetch_add_explicit(&fold[fold_idx], trace_count, memory_order_relaxed);
    }
}
