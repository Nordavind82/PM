#include <metal_stdlib>
using namespace metal;

// =============================================================================
// PSTM Curved Ray Migration Kernel
// Accounts for velocity gradient causing ray bending in V(z) = V0 + k*z media
// =============================================================================

// Migration parameters for curved ray kernel
struct CurvedRayParams {
    // Grid parameters
    float dx;                    // Output grid spacing X (meters)
    float dy;                    // Output grid spacing Y (meters)
    float dt_ms;                 // Time sampling (ms)
    float t_start_ms;            // Start time (ms)

    // Velocity gradient parameters
    float v0;                    // Surface velocity (m/s)
    float k;                     // Velocity gradient (1/s): V(z) = v0 + k*z

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
    int n_samples;
    int nx;
    int ny;
    int nt;
};

// =============================================================================
// Curved Ray Traveltime Functions
// =============================================================================

/**
 * Convert one-way vertical time to depth for V(z) = V0 + k*z.
 * For vertical rays: t = (1/k) * ln((V0 + k*z) / V0)
 * Solving for z: z = (V0 / k) * (exp(k*t) - 1)
 */
inline float time_to_depth(float t_s, float v0, float k) {
    if (abs(k) < 1e-10f) {
        // Constant velocity
        return v0 * t_s;
    }
    return (v0 / k) * (exp(k * t_s) - 1.0f);
}

/**
 * Compute one-way traveltime for curved ray in V(z) = V0 + k*z medium.
 *
 * In a medium with linear velocity gradient, rays follow circular arcs.
 * t = (1/k) * ln[(V_z + sqrt(V_z^2 + k^2*x^2)) / (V_0 + sqrt(V_0^2 + k^2*x^2))]
 */
inline float curved_ray_traveltime(float x, float z, float v0, float k) {
    if (abs(k) < 1e-10f) {
        // Near-zero gradient: use straight ray
        float r = sqrt(x * x + z * z);
        return r > 0.0f ? r / v0 : 0.0f;
    }

    float v_z = v0 + k * z;
    float kx = k * x;

    float term1 = v_z + sqrt(v_z * v_z + kx * kx);
    float term2 = v0 + sqrt(v0 * v0 + kx * kx);

    if (term2 <= 0.0f) return 0.0f;

    return (1.0f / k) * log(term1 / term2);
}

/**
 * Compute DSR traveltime with curved rays.
 * Uses curved ray formula for both source and receiver legs.
 */
inline float curved_ray_dsr_traveltime(
    float h_s,           // Source horizontal distance from image point
    float h_r,           // Receiver horizontal distance from image point
    float z,             // Image point depth (meters)
    float v0,            // Surface velocity
    float k              // Velocity gradient
) {
    float t_s = curved_ray_traveltime(h_s, z, v0, k);
    float t_r = curved_ray_traveltime(h_r, z, v0, k);
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
// Curved Ray Migration Kernel - 3D Parallel
// =============================================================================

kernel void pstm_migrate_curved_ray(
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

    // Time axis (for depth conversion)
    device const float* t0_s [[buffer(11)]],          // [nt] - t0 in seconds
    device const float* apertures [[buffer(12)]],     // [nt] - aperture at each time

    // Parameters
    constant CurvedRayParams& params [[buffer(13)]],

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

    // Get time for this output sample
    float t0_val = t0_s[it];
    float aperture = apertures[it];

    // Convert t0 to depth using curved ray model
    float depth = time_to_depth(t0_val / 2.0f, params.v0, params.k);  // One-way time

    // Velocity at image point depth
    float v_at_depth = params.v0 + params.k * depth;

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

        // Compute curved ray DSR traveltime
        float t_travel = curved_ray_dsr_traveltime(h_s, h_r, depth, params.v0, params.k);

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
                v_at_depth, t_travel,
                params.dx, params.dy,
                params.aa_dominant_freq
            );
            amp *= aa_weight;
        }

        // Apply geometrical spreading correction
        // For curved rays, use depth-dependent velocity
        if (params.apply_spreading && t_travel > 0.001f) {
            amp *= 1.0f / (v_at_depth * t_travel);
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
// SIMD Optimized Curved Ray Kernel
// =============================================================================

kernel void pstm_migrate_curved_ray_simd(
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
    device const float* apertures [[buffer(12)]],
    constant CurvedRayParams& params [[buffer(13)]],
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
    float t0_val = t0_s[it];
    float aperture = apertures[it];
    float aperture_sq = aperture * aperture;

    // Convert t0 to depth
    float depth = time_to_depth(t0_val / 2.0f, params.v0, params.k);
    float v_at_depth = params.v0 + params.k * depth;
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

            // Curved ray traveltime
            float t_travel = curved_ray_dsr_traveltime(h_s, h_r, depth, params.v0, params.k);

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
                    v_at_depth, t_travel,
                    params.dx, params.dy,
                    params.aa_dominant_freq
                );
                amp *= aa_weight;
            }

            // Amplitude corrections
            if (params.apply_spreading && t_travel > 0.001f) {
                amp *= 1.0f / (v_at_depth * t_travel);
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

        float t_travel = curved_ray_dsr_traveltime(h_s, h_r, depth, params.v0, params.k);

        float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;
        if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) continue;

        int idx0 = int(sample_idx);
        float frac = sample_idx - float(idx0);
        device const float* trace = amplitudes + tr * params.n_samples;
        float amp = trace[idx0] * (1.0f - frac) + trace[idx0 + 1] * frac;

        float dm = sqrt(dm_sq_s);
        if (dm > aperture * (1.0f - params.taper_fraction)) {
            float t = (dm - aperture * (1.0f - params.taper_fraction)) /
                      (aperture * params.taper_fraction);
            amp *= 0.5f * (1.0f + cos(t * M_PI_F));
        }

        if (params.apply_aa) {
            float aa_weight = compute_aa_weight(
                ox, oy, mx_s, my_s,
                v_at_depth, t_travel,
                params.dx, params.dy,
                params.aa_dominant_freq
            );
            amp *= aa_weight;
        }

        if (params.apply_spreading && t_travel > 0.001f) {
            amp *= 1.0f / (v_at_depth * t_travel);
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
