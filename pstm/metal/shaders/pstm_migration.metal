#include <metal_stdlib>
using namespace metal;

// =============================================================================
// PSTM Migration Kernel with Anti-Aliasing Support
// Compiled Metal shader for maximum GPU performance
// =============================================================================

// Migration parameters - must match Python struct layout exactly
struct MigrationParams {
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

    // Anti-aliasing parameters
    int apply_aa;                // Apply anti-aliasing
    float aa_dominant_freq;      // Dominant frequency for AA (Hz)

    // Dimensions
    int n_traces;
    int n_samples;
    int nx;
    int ny;
    int nt;

    // 3D velocity flag (added for lateral velocity variation support)
    int use_3d_velocity;         // If 1, use 3D velocity cube [nx, ny, nt]
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Compute DSR (Double Square Root) travel time.
 * t_travel = sqrt((t0/2)^2 + ds^2/v^2) + sqrt((t0/2)^2 + dr^2/v^2)
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
 * 
 * The maximum unaliased frequency is: f_max = v / (4 * dx * sin(theta))
 * where theta is the local dip angle.
 * 
 * Local dip is computed from: dt/dx = (x_m - x_out) / (v^2 * t / 2)
 * sin(theta) = |dt/dx| * v / 2
 */
inline float compute_aa_weight(
    float ox, float oy,          // Output point position
    float mx, float my,          // Midpoint position
    float velocity,              // Local velocity (m/s)
    float t_travel,              // Total traveltime (s)
    float dx, float dy,          // Grid spacing (m)
    float dominant_freq          // Dominant frequency (Hz)
) {
    if (t_travel < 0.001f) return 1.0f;
    
    // Compute local dip in x and y directions
    // dt/dx = (mx - ox) / (v^2 * t / 2)
    float denom = velocity * velocity * t_travel * 0.5f;
    float dip_x = (mx - ox) / denom;
    float dip_y = (my - oy) / denom;
    
    // sin(theta) = |dt/dx| * v / 2, clamped to [0, 1]
    float sin_theta_x = min(abs(dip_x) * velocity * 0.5f, 1.0f);
    float sin_theta_y = min(abs(dip_y) * velocity * 0.5f, 1.0f);
    
    // Combined sin(theta) using max of both directions
    float sin_theta = max(sin_theta_x, sin_theta_y);
    
    // Avoid division by zero for near-zero dip
    if (sin_theta < 0.01f) return 1.0f;
    
    // Maximum unaliased frequency
    float grid_spacing = max(dx, dy);
    float f_max = velocity / (4.0f * grid_spacing * sin_theta);
    
    // Clamp to Nyquist (assume 500 Hz Nyquist for 1ms sampling)
    f_max = min(f_max, 500.0f);
    
    // Triangle filter response: W(f) = max(0, 1 - f/f_max)
    float aa_weight = max(0.0f, 1.0f - dominant_freq / f_max);
    
    return aa_weight;
}

// =============================================================================
// Main Migration Kernel - 3D Parallel (each thread = one output sample)
// Supports both 1D velocity (time-only) and 3D velocity (x, y, t)
// =============================================================================
kernel void pstm_migrate_3d(
    // Input trace data
    device const float* amplitudes [[buffer(0)]],      // [n_traces, n_samples]
    device const float* source_x [[buffer(1)]],        // [n_traces]
    device const float* source_y [[buffer(2)]],
    device const float* receiver_x [[buffer(3)]],
    device const float* receiver_y [[buffer(4)]],
    device const float* midpoint_x [[buffer(5)]],
    device const float* midpoint_y [[buffer(6)]],

    // Output arrays
    device float* image [[buffer(7)]],                 // [nx, ny, nt] - non-atomic, unique per thread
    device atomic_int* fold [[buffer(8)]],             // [nx, ny]

    // Grid coordinates (flattened 2D for rotated grid support)
    device const float* x_coords [[buffer(9)]],        // [nx*ny] flattened X coordinates
    device const float* y_coords [[buffer(10)]],       // [nx*ny] flattened Y coordinates

    // Pre-computed time-dependent values
    device const float* t0_half_sq [[buffer(11)]],     // [nt] - (t0/2)^2
    device const float* inv_v_sq [[buffer(12)]],       // [nt] for 1D, [nx*ny*nt] for 3D velocity
    device const float* t0_s [[buffer(13)]],           // [nt] - t0 in seconds
    device const float* apertures [[buffer(14)]],      // [nt] - aperture at each time

    // Parameters
    constant MigrationParams& params [[buffer(15)]],

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

    // Get output point coordinates (2D flattened indexing for rotated grids)
    int coord_idx = ix * params.ny + iy;
    float ox = x_coords[coord_idx];
    float oy = y_coords[coord_idx];

    // Get time-dependent values for this output time
    float aperture = apertures[it];
    float t0_half_sq_val = t0_half_sq[it];
    float t0_val = t0_s[it];

    // Get velocity - support both 1D and 3D velocity models
    float inv_v_sq_val;
    if (params.use_3d_velocity) {
        // 3D velocity: index by (ix, iy, it) - same layout as output image
        int vel_idx = ix * params.ny * params.nt + iy * params.nt + it;
        inv_v_sq_val = inv_v_sq[vel_idx];
    } else {
        // 1D velocity: index by time only (center pillar for all x,y)
        inv_v_sq_val = inv_v_sq[it];
    }
    float velocity = 1.0f / sqrt(inv_v_sq_val);  // Recover velocity
    
    // Accumulator for this output sample
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
        
        // Distance squared to source and receiver
        float ds2 = (ox - sx) * (ox - sx) + (oy - sy) * (oy - sy);
        float dr2 = (ox - rx) * (ox - rx) + (oy - ry) * (oy - ry);
        
        // Compute DSR traveltime
        float t_travel = compute_dsr_traveltime(ds2, dr2, t0_half_sq_val, inv_v_sq_val);
        
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
        
        // Apply geometrical spreading correction: 1 / (v * t)
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
    
    // Write to output image (no atomic needed - each thread writes unique location)
    int img_idx = ix * params.ny * params.nt + iy * params.nt + it;
    image[img_idx] = local_sum;
    
    // Update fold (only for first time sample to avoid overcounting)
    if (it == 0 && trace_count > 0) {
        int fold_idx = ix * params.ny + iy;
        atomic_fetch_add_explicit(&fold[fold_idx], trace_count, memory_order_relaxed);
    }
}

// =============================================================================
// SIMD Optimized Kernel - Process 4 traces at a time
// =============================================================================
kernel void pstm_migrate_3d_simd(
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
    int it = gid.z;
    
    if (ix >= params.nx || iy >= params.ny || it >= params.nt) {
        return;
    }

    // Get output point coordinates (2D flattened indexing for rotated grids)
    int coord_idx = ix * params.ny + iy;
    float ox = x_coords[coord_idx];
    float oy = y_coords[coord_idx];
    float aperture = apertures[it];
    float aperture_sq = aperture * aperture;
    float t0_half_sq_val = t0_half_sq[it];
    float inv_v_sq_val = inv_v_sq[it];
    float t0_val = t0_s[it];
    float velocity = 1.0f / sqrt(inv_v_sq_val);
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
        
        // DSR computation (vectorized)
        float4 dsx = ox - sx;
        float4 dsy = oy - sy;
        float4 ds2 = dsx * dsx + dsy * dsy;
        
        float4 drx = ox - rx;
        float4 dry = oy - ry;
        float4 dr2 = drx * drx + dry * dry;
        
        float4 t_src = sqrt(t0_half_sq_val + ds2 * inv_v_sq_val);
        float4 t_rec = sqrt(t0_half_sq_val + dr2 * inv_v_sq_val);
        float4 t_travel = t_src + t_rec;
        
        float4 sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_ms;
        
        // Process each trace in SIMD group
        for (int j = 0; j < 4; j++) {
            if (!in_aperture[j]) continue;
            
            float sidx = sample_idx[j];
            if (sidx < 0.0f || sidx >= float(params.n_samples - 1)) continue;
            
            // Interpolate amplitude
            int idx0 = int(sidx);
            float frac = sidx - float(idx0);
            device const float* trace = amplitudes + (tr + j) * params.n_samples;
            float amp = trace[idx0] * (1.0f - frac) + trace[idx0 + 1] * frac;
            
            // Taper
            float dm = sqrt(dm_sq[j]);
            if (dm > taper_start) {
                float t = (dm - taper_start) / (aperture - taper_start);
                amp *= 0.5f * (1.0f + cos(t * M_PI_F));
            }
            
            float tt = t_travel[j];
            
            // Anti-aliasing
            if (params.apply_aa) {
                float aa_weight = compute_aa_weight(
                    ox, oy, mx[j], my[j],
                    velocity, tt,
                    params.dx, params.dy,
                    params.aa_dominant_freq
                );
                amp *= aa_weight;
            }
            
            // Amplitude corrections
            if (params.apply_spreading && tt > 0.001f) {
                amp *= 1.0f / (velocity * tt);
            }
            if (params.apply_obliquity && tt > 0.001f) {
                amp *= t0_val / tt;
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
        
        float sx_s = source_x[tr];
        float sy_s = source_y[tr];
        float rx_s = receiver_x[tr];
        float ry_s = receiver_y[tr];
        
        float ds2_s = (ox - sx_s) * (ox - sx_s) + (oy - sy_s) * (oy - sy_s);
        float dr2_s = (ox - rx_s) * (ox - rx_s) + (oy - ry_s) * (oy - ry_s);
        
        float t_travel_s = compute_dsr_traveltime(ds2_s, dr2_s, t0_half_sq_val, inv_v_sq_val);
        
        float sample_idx_s = (t_travel_s * 1000.0f - params.t_start_ms) / params.dt_ms;
        if (sample_idx_s < 0.0f || sample_idx_s >= float(params.n_samples - 1)) continue;
        
        int idx0 = int(sample_idx_s);
        float frac = sample_idx_s - float(idx0);
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
                velocity, t_travel_s,
                params.dx, params.dy,
                params.aa_dominant_freq
            );
            amp *= aa_weight;
        }
        
        if (params.apply_spreading && t_travel_s > 0.001f) {
            amp *= 1.0f / (velocity * t_travel_s);
        }
        if (params.apply_obliquity && t_travel_s > 0.001f) {
            amp *= t0_val / t_travel_s;
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

// =============================================================================
// Time-Variant Sampling Support
// =============================================================================

// Maximum number of time windows
constant int MAX_TIME_WINDOWS = 8;

// Time window for time-variant sampling
struct TimeWindow {
    float t_start_ms;        // Window start time (ms)
    float t_end_ms;          // Window end time (ms)
    float dt_effective_ms;   // Effective sample rate in this window (ms)
    int downsample_factor;   // Downsample factor (1, 2, 4, 8)
    int sample_offset;       // Start index in output array
    int n_samples;           // Number of samples in this window
};

// Parameters for time-variant migration
struct TimeVariantParams {
    // Grid parameters (same as MigrationParams)
    float dx;
    float dy;
    float dt_base_ms;        // Base sample rate (ms)
    float t_start_ms;

    // Aperture parameters
    float max_aperture;
    float min_aperture;
    float taper_fraction;
    float max_dip_deg;

    // Amplitude correction flags
    int apply_spreading;
    int apply_obliquity;
    int apply_aa;
    float aa_dominant_freq;

    // Dimensions
    int n_traces;
    int n_samples;           // Input trace samples
    int nx;
    int ny;
    int n_windows;           // Number of time windows
    int total_output_samples; // Total output samples across all windows

    // 3D velocity support
    int use_3d_velocity;     // If 1, vrms is [nx*ny*nt_base] instead of [nt_base]
    int nt_base;             // Number of time samples in velocity model
};

/**
 * Time-variant migration kernel.
 *
 * Processes time windows with varying sample rates. Each thread handles
 * one (x, y, window_sample) combination. Coarser sampling at deeper times
 * provides significant speedup while maintaining quality.
 */
kernel void pstm_migrate_time_variant(
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

    // Grid coordinates (flattened 2D for rotated grid support)
    device const float* x_coords [[buffer(9)]],    // [nx*ny] flattened X coordinates
    device const float* y_coords [[buffer(10)]],   // [nx*ny] flattened Y coordinates

    // Velocity model (1D for now)
    device const float* vrms [[buffer(11)]],       // [nt_base] velocity at base sampling

    // Time windows
    constant TimeWindow* windows [[buffer(12)]],

    // Parameters
    constant TimeVariantParams& params [[buffer(13)]],

    // Thread position: (ix, iy, flat_sample_idx)
    uint3 gid [[thread_position_in_grid]]
) {
    int ix = gid.x;
    int iy = gid.y;
    int flat_idx = gid.z;  // Flattened index across all windows

    // Bounds check
    if (ix >= params.nx || iy >= params.ny || flat_idx >= params.total_output_samples) {
        return;
    }

    // Find which window this sample belongs to
    int window_idx = 0;
    int local_sample = flat_idx;
    for (int w = 0; w < params.n_windows; w++) {
        if (flat_idx < windows[w].sample_offset + windows[w].n_samples) {
            window_idx = w;
            local_sample = flat_idx - windows[w].sample_offset;
            break;
        }
    }

    TimeWindow win = windows[window_idx];

    // Compute output time for this sample
    float t_out_ms = win.t_start_ms + local_sample * win.dt_effective_ms;
    float t_out_s = t_out_ms / 1000.0f;

    // Get output point coordinates (2D flattened indexing for rotated grids)
    int coord_idx = ix * params.ny + iy;

    // Get velocity at this time - support both 1D and 3D velocity models
    int velo_t_idx = int(t_out_ms / params.dt_base_ms);
    velo_t_idx = min(velo_t_idx, params.nt_base - 1);

    float velocity;
    if (params.use_3d_velocity) {
        // 3D velocity: index by (ix, iy, t) - same layout as output image
        int vel_idx = ix * params.ny * params.nt_base + iy * params.nt_base + velo_t_idx;
        velocity = vrms[vel_idx];
    } else {
        // 1D velocity: index by time only
        velocity = vrms[velo_t_idx];
    }

    // Pre-compute values for DSR
    float t0_half = t_out_s / 2.0f;
    float t0_half_sq = t0_half * t0_half;
    float inv_v_sq = 1.0f / (velocity * velocity);

    // Get output point coordinates (using coord_idx computed above)
    float ox = x_coords[coord_idx];
    float oy = y_coords[coord_idx];

    // Time-dependent aperture (simple linear model)
    float aperture = min(params.max_aperture, params.min_aperture + velocity * t_out_s * 0.5f);
    float aperture_sq = aperture * aperture;
    float taper_start = aperture * (1.0f - params.taper_fraction);

    // Accumulator
    float local_sum = 0.0f;
    int trace_count = 0;

    // Process all traces
    for (int tr = 0; tr < params.n_traces; tr++) {
        float mx = midpoint_x[tr];
        float my = midpoint_y[tr];

        // Aperture check
        float dm_sq = (ox - mx) * (ox - mx) + (oy - my) * (oy - my);
        if (dm_sq > aperture_sq) continue;

        float sx = source_x[tr];
        float sy = source_y[tr];
        float rx = receiver_x[tr];
        float ry = receiver_y[tr];

        // DSR traveltime
        float ds2 = (ox - sx) * (ox - sx) + (oy - sy) * (oy - sy);
        float dr2 = (ox - rx) * (ox - rx) + (oy - ry) * (oy - ry);
        float t_travel = sqrt(t0_half_sq + ds2 * inv_v_sq) + sqrt(t0_half_sq + dr2 * inv_v_sq);

        // Convert to sample index in input trace
        float sample_idx = (t_travel * 1000.0f - params.t_start_ms) / params.dt_base_ms;
        if (sample_idx < 0.0f || sample_idx >= float(params.n_samples - 1)) continue;

        // Linear interpolation
        int idx0 = int(sample_idx);
        float frac = sample_idx - float(idx0);
        device const float* trace = amplitudes + tr * params.n_samples;
        float amp = trace[idx0] * (1.0f - frac) + trace[idx0 + 1] * frac;

        // Aperture taper
        float dm = sqrt(dm_sq);
        if (dm > taper_start) {
            float t = (dm - taper_start) / (aperture - taper_start);
            amp *= 0.5f * (1.0f + cos(t * M_PI_F));
        }

        // Anti-aliasing
        if (params.apply_aa) {
            float aa_weight = compute_aa_weight(
                ox, oy, mx, my,
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
            amp *= t_out_s / t_travel;
        }

        local_sum += amp;
        trace_count++;
    }

    // Write to output (flat index for time-variant output)
    int img_idx = ix * params.ny * params.total_output_samples + iy * params.total_output_samples + flat_idx;
    image[img_idx] = local_sum;

    // Update fold for first sample only
    if (flat_idx == 0 && trace_count > 0) {
        int fold_idx = ix * params.ny + iy;
        atomic_fetch_add_explicit(&fold[fold_idx], trace_count, memory_order_relaxed);
    }
}
