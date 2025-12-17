// Kernel benchmark utilities for testing optimization variants
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>

namespace py = pybind11;

// Forward declarations
namespace pstm {
namespace metal {
    bool is_available();
    bool initialize_device();
    id<MTLDevice> get_device();
    id<MTLCommandQueue> get_command_queue();
}
}

// Parameters struct matching shader
struct BenchmarkParams {
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

// Pre-compute time terms
static void precompute_time_terms(
    const double* t_coords_ms,
    const double* vrms,
    int nt,
    float max_dip_deg,
    float min_aperture,
    float max_aperture,
    std::vector<float>& t0_half_sq,
    std::vector<float>& inv_v_sq,
    std::vector<float>& t0_s,
    std::vector<float>& apertures
) {
    t0_half_sq.resize(nt);
    inv_v_sq.resize(nt);
    t0_s.resize(nt);
    apertures.resize(nt);

    float tan_dip = std::tan(max_dip_deg * M_PI / 180.0f);

    for (int i = 0; i < nt; i++) {
        float t0 = static_cast<float>(t_coords_ms[i]) / 1000.0f;
        float v = static_cast<float>(vrms[i]);

        t0_s[i] = t0;
        t0_half_sq[i] = (t0 / 2.0f) * (t0 / 2.0f);
        inv_v_sq[i] = 1.0f / (v * v);

        float ap = v * t0 * tan_dip / 2.0f;
        apertures[i] = std::max(min_aperture, std::min(max_aperture, ap));
    }
}

// Benchmark a specific kernel variant
py::dict benchmark_kernel_variant(
    const std::string& variant_name,
    py::array_t<float> amplitudes,
    py::array_t<double> source_x,
    py::array_t<double> source_y,
    py::array_t<double> receiver_x,
    py::array_t<double> receiver_y,
    py::array_t<double> midpoint_x,
    py::array_t<double> midpoint_y,
    py::array_t<double> x_coords,
    py::array_t<double> y_coords,
    py::array_t<double> t_coords_ms,
    py::array_t<double> vrms,
    py::dict config,
    int n_runs
) {
    @autoreleasepool {
        if (!pstm::metal::is_available()) {
            throw std::runtime_error("Metal not available");
        }

        id<MTLDevice> device = pstm::metal::get_device();
        id<MTLCommandQueue> commandQueue = pstm::metal::get_command_queue();

        // Get array info
        auto amp_info = amplitudes.request();
        int n_traces = static_cast<int>(amp_info.shape[0]);
        int n_samples = static_cast<int>(amp_info.shape[1]);
        int nx = static_cast<int>(x_coords.size());
        int ny = static_cast<int>(y_coords.size());
        int nt = static_cast<int>(t_coords_ms.size());

        // Build params
        BenchmarkParams params;
        params.max_dip_deg = config.contains("max_dip_deg") ? config["max_dip_deg"].cast<float>() : 45.0f;
        params.min_aperture = config.contains("min_aperture") ? config["min_aperture"].cast<float>() : 100.0f;
        params.max_aperture = config.contains("max_aperture") ? config["max_aperture"].cast<float>() : 2500.0f;
        params.taper_fraction = config.contains("taper_fraction") ? config["taper_fraction"].cast<float>() : 0.1f;
        params.dt_ms = config.contains("dt_ms") ? config["dt_ms"].cast<float>() : 4.0f;
        params.t_start_ms = config.contains("t_start_ms") ? config["t_start_ms"].cast<float>() : 0.0f;
        params.apply_spreading = config.contains("apply_spreading") ? (config["apply_spreading"].cast<bool>() ? 1 : 0) : 1;
        params.apply_obliquity = config.contains("apply_obliquity") ? (config["apply_obliquity"].cast<bool>() ? 1 : 0) : 1;
        params.n_traces = n_traces;
        params.n_samples = n_samples;
        params.nx = nx;
        params.ny = ny;
        params.nt = nt;

        // Pre-compute time terms
        std::vector<float> t0_half_sq, inv_v_sq, t0_s, apertures;
        precompute_time_terms(
            t_coords_ms.data(), vrms.data(), nt,
            params.max_dip_deg, params.min_aperture, params.max_aperture,
            t0_half_sq, inv_v_sq, t0_s, apertures
        );

        // Convert coordinates to float
        std::vector<float> f_source_x(n_traces), f_source_y(n_traces);
        std::vector<float> f_receiver_x(n_traces), f_receiver_y(n_traces);
        std::vector<float> f_midpoint_x(n_traces), f_midpoint_y(n_traces);
        std::vector<float> f_x_coords(nx), f_y_coords(ny);

        for (int i = 0; i < n_traces; i++) {
            f_source_x[i] = static_cast<float>(source_x.at(i));
            f_source_y[i] = static_cast<float>(source_y.at(i));
            f_receiver_x[i] = static_cast<float>(receiver_x.at(i));
            f_receiver_y[i] = static_cast<float>(receiver_y.at(i));
            f_midpoint_x[i] = static_cast<float>(midpoint_x.at(i));
            f_midpoint_y[i] = static_cast<float>(midpoint_y.at(i));
        }
        for (int i = 0; i < nx; i++) f_x_coords[i] = static_cast<float>(x_coords.at(i));
        for (int i = 0; i < ny; i++) f_y_coords[i] = static_cast<float>(y_coords.at(i));

        // Load shader library - baseline, 3d_parallel, simd4 are in migrate_tile.metallib
        // Other optimized variants are in migrate_optimized.metallib
        bool use_main_lib = (variant_name == "baseline" || variant_name == "3d_parallel" || variant_name == "simd4");

        NSString* shader_path = nil;
        if (use_main_lib) {
            #ifdef SHADER_PATH
            shader_path = [NSString stringWithUTF8String:SHADER_PATH];
            #else
            shader_path = @"migrate_tile.metallib";
            #endif
        } else {
            #ifdef SHADER_OPT_PATH
            shader_path = [NSString stringWithUTF8String:SHADER_OPT_PATH];
            #else
            shader_path = @"migrate_optimized.metallib";
            #endif
        }

        NSError* error = nil;
        NSURL* url = [NSURL fileURLWithPath:shader_path];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];

        if (!library) {
            throw std::runtime_error("Failed to load shader library: " + std::string([shader_path UTF8String]));
        }

        // Get kernel function
        NSString* kernelName = nil;
        if (variant_name == "baseline") {
            kernelName = @"migrate_tile_kernel";
        } else if (variant_name == "3d_parallel") {
            kernelName = @"migrate_3d_parallel";
        } else if (variant_name == "simd4") {
            // New SIMD4 kernel - uses 3D dispatch like 3d_parallel
            kernelName = @"migrate_3d_simd4";
        } else if (variant_name == "shared_memory") {
            kernelName = @"migrate_shared_memory";
        } else if (variant_name == "trace_batches") {
            kernelName = @"migrate_trace_batches";
        } else if (variant_name == "simd_vectorized") {
            kernelName = @"migrate_simd_vectorized";
        } else {
            throw std::runtime_error("Unknown variant: " + variant_name);
        }

        id<MTLFunction> function = [library newFunctionWithName:kernelName];
        if (!function) {
            throw std::runtime_error("Kernel function not found: " + std::string([kernelName UTF8String]));
        }

        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipelineState) {
            throw std::runtime_error("Failed to create pipeline state");
        }

        // Create buffers
        size_t image_size = nx * ny * nt;
        std::vector<float> f_image(image_size, 0.0f);
        std::vector<int32_t> f_fold(nx * ny, 0);

        id<MTLBuffer> b_amplitudes = [device newBufferWithBytes:amp_info.ptr length:n_traces * n_samples * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_source_x = [device newBufferWithBytes:f_source_x.data() length:n_traces * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_source_y = [device newBufferWithBytes:f_source_y.data() length:n_traces * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_receiver_x = [device newBufferWithBytes:f_receiver_x.data() length:n_traces * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_receiver_y = [device newBufferWithBytes:f_receiver_y.data() length:n_traces * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_midpoint_x = [device newBufferWithBytes:f_midpoint_x.data() length:n_traces * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_midpoint_y = [device newBufferWithBytes:f_midpoint_y.data() length:n_traces * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_image = [device newBufferWithLength:image_size * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_fold = [device newBufferWithLength:nx * ny * sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_x_coords = [device newBufferWithBytes:f_x_coords.data() length:nx * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_y_coords = [device newBufferWithBytes:f_y_coords.data() length:ny * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_t0_half_sq = [device newBufferWithBytes:t0_half_sq.data() length:nt * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_inv_v_sq = [device newBufferWithBytes:inv_v_sq.data() length:nt * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_t0_s = [device newBufferWithBytes:t0_s.data() length:nt * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_apertures = [device newBufferWithBytes:apertures.data() length:nt * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_params = [device newBufferWithBytes:&params length:sizeof(BenchmarkParams) options:MTLResourceStorageModeShared];

        // Warmup
        {
            memset([b_image contents], 0, image_size * sizeof(float));
            memset([b_fold contents], 0, nx * ny * sizeof(int32_t));

            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipelineState];

            [encoder setBuffer:b_amplitudes offset:0 atIndex:0];
            [encoder setBuffer:b_source_x offset:0 atIndex:1];
            [encoder setBuffer:b_source_y offset:0 atIndex:2];
            [encoder setBuffer:b_receiver_x offset:0 atIndex:3];
            [encoder setBuffer:b_receiver_y offset:0 atIndex:4];
            [encoder setBuffer:b_midpoint_x offset:0 atIndex:5];
            [encoder setBuffer:b_midpoint_y offset:0 atIndex:6];
            [encoder setBuffer:b_image offset:0 atIndex:7];
            [encoder setBuffer:b_fold offset:0 atIndex:8];
            [encoder setBuffer:b_x_coords offset:0 atIndex:9];
            [encoder setBuffer:b_y_coords offset:0 atIndex:10];
            [encoder setBuffer:b_t0_half_sq offset:0 atIndex:11];
            [encoder setBuffer:b_inv_v_sq offset:0 atIndex:12];
            [encoder setBuffer:b_t0_s offset:0 atIndex:13];
            [encoder setBuffer:b_apertures offset:0 atIndex:14];
            [encoder setBuffer:b_params offset:0 atIndex:15];

            MTLSize gridSize, threadgroupSize;
            bool is_3d = (variant_name == "3d_parallel" || variant_name == "simd4");
            if (is_3d) {
                // 3D grid: (nx, ny, nt)
                gridSize = MTLSizeMake(nx, ny, nt);
                NSUInteger w = pipelineState.threadExecutionWidth;
                threadgroupSize = MTLSizeMake(w, 1, 1);
            } else {
                // 2D grid: (nx, ny)
                gridSize = MTLSizeMake(nx, ny, 1);
                NSUInteger w = pipelineState.threadExecutionWidth;
                NSUInteger h = pipelineState.maxTotalThreadsPerThreadgroup / w;
                threadgroupSize = MTLSizeMake(w, h, 1);
            }

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }

        // Benchmark runs
        std::vector<double> times;
        bool is_3d_dispatch = (variant_name == "3d_parallel" || variant_name == "simd4");
        for (int run = 0; run < n_runs; run++) {
            memset([b_image contents], 0, image_size * sizeof(float));
            memset([b_fold contents], 0, nx * ny * sizeof(int32_t));

            auto start = std::chrono::high_resolution_clock::now();

            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipelineState];

            [encoder setBuffer:b_amplitudes offset:0 atIndex:0];
            [encoder setBuffer:b_source_x offset:0 atIndex:1];
            [encoder setBuffer:b_source_y offset:0 atIndex:2];
            [encoder setBuffer:b_receiver_x offset:0 atIndex:3];
            [encoder setBuffer:b_receiver_y offset:0 atIndex:4];
            [encoder setBuffer:b_midpoint_x offset:0 atIndex:5];
            [encoder setBuffer:b_midpoint_y offset:0 atIndex:6];
            [encoder setBuffer:b_image offset:0 atIndex:7];
            [encoder setBuffer:b_fold offset:0 atIndex:8];
            [encoder setBuffer:b_x_coords offset:0 atIndex:9];
            [encoder setBuffer:b_y_coords offset:0 atIndex:10];
            [encoder setBuffer:b_t0_half_sq offset:0 atIndex:11];
            [encoder setBuffer:b_inv_v_sq offset:0 atIndex:12];
            [encoder setBuffer:b_t0_s offset:0 atIndex:13];
            [encoder setBuffer:b_apertures offset:0 atIndex:14];
            [encoder setBuffer:b_params offset:0 atIndex:15];

            MTLSize gridSize, threadgroupSize;
            if (is_3d_dispatch) {
                gridSize = MTLSizeMake(nx, ny, nt);
                NSUInteger w = pipelineState.threadExecutionWidth;
                threadgroupSize = MTLSizeMake(w, 1, 1);
            } else {
                gridSize = MTLSizeMake(nx, ny, 1);
                NSUInteger w = pipelineState.threadExecutionWidth;
                NSUInteger h = pipelineState.maxTotalThreadsPerThreadgroup / w;
                threadgroupSize = MTLSizeMake(w, h, 1);
            }

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(elapsed_ms);
        }

        // Calculate statistics
        double mean = 0;
        for (double t : times) mean += t;
        mean /= times.size();

        double variance = 0;
        for (double t : times) variance += (t - mean) * (t - mean);
        variance /= times.size();
        double std_dev = std::sqrt(variance);

        // Get result for validation
        float* gpu_image = (float*)[b_image contents];
        double image_sum = 0;
        for (size_t i = 0; i < image_size; i++) {
            image_sum += std::abs(gpu_image[i]);
        }

        py::dict result;
        result["variant"] = variant_name;
        result["mean_time_ms"] = mean;
        result["std_time_ms"] = std_dev;
        result["traces_per_second"] = n_traces / (mean / 1000.0);
        result["image_sum"] = image_sum;
        result["n_runs"] = n_runs;

        return result;
    }
}

// Export benchmark function
void register_benchmark_functions(py::module& m) {
    m.def("benchmark_kernel_variant", &benchmark_kernel_variant,
        py::arg("variant_name"),
        py::arg("amplitudes"),
        py::arg("source_x"),
        py::arg("source_y"),
        py::arg("receiver_x"),
        py::arg("receiver_y"),
        py::arg("midpoint_x"),
        py::arg("midpoint_y"),
        py::arg("x_coords"),
        py::arg("y_coords"),
        py::arg("t_coords_ms"),
        py::arg("vrms"),
        py::arg("config"),
        py::arg("n_runs") = 3,
        R"doc(
Benchmark a specific kernel variant.

Parameters
----------
variant_name : str
    One of: "baseline", "3d_parallel", "shared_memory", "trace_batches", "simd_vectorized"
... (other params same as migrate_tile)
n_runs : int
    Number of benchmark runs (default: 3)

Returns
-------
dict
    Benchmark results including mean_time_ms, std_time_ms, traces_per_second
)doc");
}
