#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_kernel.h"
#include <stdexcept>
#include <chrono>
#include <cmath>

// Forward declarations from metal_device.mm
namespace pstm {
namespace metal {
    bool is_available();
    bool initialize_device();
    id<MTLDevice> get_device();
    id<MTLCommandQueue> get_command_queue();
    std::string get_device_name();
    size_t get_device_memory();
}
}

namespace pstm {

// Implementation class holding Objective-C objects
class MetalKernelImpl {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pipelineState;      // 3D parallel kernel (fast)
    id<MTLComputePipelineState> pipelineStateBaseline;  // Original 2D kernel
    id<MTLComputePipelineState> pipelineStateChunked;
    bool initialized;

    MetalKernelImpl()
        : device(nil)
        , commandQueue(nil)
        , library(nil)
        , pipelineState(nil)
        , pipelineStateBaseline(nil)
        , pipelineStateChunked(nil)
        , initialized(false)
    {}

    ~MetalKernelImpl() {
        cleanup();
    }

    void cleanup() {
        @autoreleasepool {
            pipelineStateChunked = nil;
            pipelineStateBaseline = nil;
            pipelineState = nil;
            library = nil;
            // Don't release device/commandQueue as they're global singletons
            initialized = false;
        }
    }
};

// Static methods
bool MetalKernel::is_available() {
    return metal::is_available();
}

std::string MetalKernel::get_device_name() {
    return metal::get_device_name();
}

size_t MetalKernel::get_device_memory() {
    return metal::get_device_memory();
}

// Constructor/Destructor
MetalKernel::MetalKernel()
    : impl_(std::make_unique<MetalKernelImpl>())
{}

MetalKernel::~MetalKernel() = default;

MetalKernel::MetalKernel(MetalKernel&&) noexcept = default;
MetalKernel& MetalKernel::operator=(MetalKernel&&) noexcept = default;

bool MetalKernel::initialize(const std::string& shader_path) {
    if (impl_->initialized) {
        return true;
    }

    @autoreleasepool {
        // Initialize Metal device
        if (!metal::initialize_device()) {
            return false;
        }

        impl_->device = metal::get_device();
        impl_->commandQueue = metal::get_command_queue();

        // Determine shader path
        std::string path = shader_path;
        if (path.empty()) {
            // Use compiled-in path
            #ifdef SHADER_PATH
            path = SHADER_PATH;
            #else
            // Try to find it relative to the module
            path = "migrate_tile.metallib";
            #endif
        }

        NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:nsPath];

        NSError* error = nil;
        impl_->library = [impl_->device newLibraryWithURL:url error:&error];

        if (!impl_->library) {
            if (error) {
                NSLog(@"Failed to load Metal library: %@", error);
            }
            return false;
        }

        // Create pipeline states - use optimized SIMD4 kernel (fastest: 35% faster than 3D parallel)
        id<MTLFunction> functionSIMD4 = [impl_->library newFunctionWithName:@"migrate_3d_simd4"];
        if (!functionSIMD4) {
            // Fallback to 3D parallel if SIMD4 not available
            NSLog(@"SIMD4 kernel not found, falling back to 3D parallel");
            functionSIMD4 = [impl_->library newFunctionWithName:@"migrate_3d_parallel"];
        }
        if (!functionSIMD4) {
            NSLog(@"Failed to find kernel function");
            return false;
        }

        impl_->pipelineState = [impl_->device newComputePipelineStateWithFunction:functionSIMD4 error:&error];
        if (!impl_->pipelineState) {
            if (error) {
                NSLog(@"Failed to create SIMD4 pipeline state: %@", error);
            }
            return false;
        }

        // Also load baseline kernel (for debugging/comparison)
        id<MTLFunction> functionBaseline = [impl_->library newFunctionWithName:@"migrate_tile_kernel"];
        if (functionBaseline) {
            impl_->pipelineStateBaseline = [impl_->device newComputePipelineStateWithFunction:functionBaseline error:&error];
        }

        // Create chunked pipeline
        id<MTLFunction> functionChunked = [impl_->library newFunctionWithName:@"migrate_tile_chunked"];
        if (functionChunked) {
            impl_->pipelineStateChunked = [impl_->device newComputePipelineStateWithFunction:functionChunked error:&error];
        }

        impl_->initialized = true;
        return true;
    }
}

// Helper to create a buffer from host memory
static id<MTLBuffer> create_buffer(id<MTLDevice> device, const void* data, size_t size, bool readonly = true) {
    MTLResourceOptions options = readonly
        ? MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache
        : MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;

    id<MTLBuffer> buffer = [device newBufferWithBytes:data length:size options:options];
    return buffer;
}

// Helper to create a shared buffer for output (zero-copy when possible)
static id<MTLBuffer> create_shared_buffer(id<MTLDevice> device, void* data, size_t size) {
    // Use MTLResourceStorageModeShared for unified memory access
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:data
                                                     length:size
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];

    // If noCopy fails (e.g., alignment issues), fall back to regular buffer
    if (!buffer) {
        buffer = [device newBufferWithLength:size
                                     options:MTLResourceStorageModeShared];
        if (buffer) {
            memcpy([buffer contents], data, size);
        }
    }
    return buffer;
}

// Pre-compute time-dependent values
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
        float t0 = static_cast<float>(t_coords_ms[i]) / 1000.0f;  // to seconds
        float v = static_cast<float>(vrms[i]);

        t0_s[i] = t0;
        t0_half_sq[i] = (t0 / 2.0f) * (t0 / 2.0f);
        inv_v_sq[i] = 1.0f / (v * v);

        // Aperture = v * t0 * tan(dip) / 2, clamped to [min, max]
        float ap = v * t0 * tan_dip / 2.0f;
        apertures[i] = std::max(min_aperture, std::min(max_aperture, ap));
    }
}

KernelMetrics MetalKernel::migrate_tile(
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
) {
    KernelMetrics metrics = {0.0, 0.0, 0, 0};

    if (!impl_->initialized) {
        throw std::runtime_error("Metal kernel not initialized");
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    @autoreleasepool {
        // Pre-compute time-dependent values
        std::vector<float> t0_half_sq, inv_v_sq, t0_s, apertures;
        precompute_time_terms(
            t_coords_ms, vrms, params.nt,
            params.max_dip_deg, params.min_aperture, params.max_aperture,
            t0_half_sq, inv_v_sq, t0_s, apertures
        );

        // Convert double coordinates to float for GPU
        std::vector<float> f_source_x(params.n_traces);
        std::vector<float> f_source_y(params.n_traces);
        std::vector<float> f_receiver_x(params.n_traces);
        std::vector<float> f_receiver_y(params.n_traces);
        std::vector<float> f_midpoint_x(params.n_traces);
        std::vector<float> f_midpoint_y(params.n_traces);
        std::vector<float> f_x_coords(params.nx);
        std::vector<float> f_y_coords(params.ny);

        for (int i = 0; i < params.n_traces; i++) {
            f_source_x[i] = static_cast<float>(source_x[i]);
            f_source_y[i] = static_cast<float>(source_y[i]);
            f_receiver_x[i] = static_cast<float>(receiver_x[i]);
            f_receiver_y[i] = static_cast<float>(receiver_y[i]);
            f_midpoint_x[i] = static_cast<float>(midpoint_x[i]);
            f_midpoint_y[i] = static_cast<float>(midpoint_y[i]);
        }
        for (int i = 0; i < params.nx; i++) {
            f_x_coords[i] = static_cast<float>(x_coords[i]);
        }
        for (int i = 0; i < params.ny; i++) {
            f_y_coords[i] = static_cast<float>(y_coords[i]);
        }

        // Create GPU output buffer (float for atomic operations, will convert back)
        size_t image_size = params.nx * params.ny * params.nt;
        std::vector<float> f_image(image_size, 0.0f);

        // Create input buffers
        id<MTLBuffer> b_amplitudes = create_buffer(impl_->device, amplitudes,
            params.n_traces * params.n_samples * sizeof(float));
        id<MTLBuffer> b_source_x = create_buffer(impl_->device, f_source_x.data(),
            params.n_traces * sizeof(float));
        id<MTLBuffer> b_source_y = create_buffer(impl_->device, f_source_y.data(),
            params.n_traces * sizeof(float));
        id<MTLBuffer> b_receiver_x = create_buffer(impl_->device, f_receiver_x.data(),
            params.n_traces * sizeof(float));
        id<MTLBuffer> b_receiver_y = create_buffer(impl_->device, f_receiver_y.data(),
            params.n_traces * sizeof(float));
        id<MTLBuffer> b_midpoint_x = create_buffer(impl_->device, f_midpoint_x.data(),
            params.n_traces * sizeof(float));
        id<MTLBuffer> b_midpoint_y = create_buffer(impl_->device, f_midpoint_y.data(),
            params.n_traces * sizeof(float));

        // Output buffers
        id<MTLBuffer> b_image = create_buffer(impl_->device, f_image.data(),
            image_size * sizeof(float), false);
        id<MTLBuffer> b_fold = create_buffer(impl_->device, fold,
            params.nx * params.ny * sizeof(int32_t), false);

        // Grid coordinate buffers
        id<MTLBuffer> b_x_coords = create_buffer(impl_->device, f_x_coords.data(),
            params.nx * sizeof(float));
        id<MTLBuffer> b_y_coords = create_buffer(impl_->device, f_y_coords.data(),
            params.ny * sizeof(float));

        // Pre-computed buffers
        id<MTLBuffer> b_t0_half_sq = create_buffer(impl_->device, t0_half_sq.data(),
            params.nt * sizeof(float));
        id<MTLBuffer> b_inv_v_sq = create_buffer(impl_->device, inv_v_sq.data(),
            params.nt * sizeof(float));
        id<MTLBuffer> b_t0_s = create_buffer(impl_->device, t0_s.data(),
            params.nt * sizeof(float));
        id<MTLBuffer> b_apertures = create_buffer(impl_->device, apertures.data(),
            params.nt * sizeof(float));

        // Parameters buffer
        id<MTLBuffer> b_params = create_buffer(impl_->device, &params, sizeof(MigrationParams));

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:impl_->pipelineState];

        // Set buffers
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

        // Calculate 3D thread groups for migrate_3d_parallel kernel
        // Each thread handles one output sample (x, y, t)
        // Total threads = nx * ny * nt (e.g., 32 * 32 * 500 = 512,000)
        NSUInteger maxThreadsPerGroup = impl_->pipelineState.maxTotalThreadsPerThreadgroup;  // Usually 1024

        // Use 8x8x16 = 1024 threads per group (good balance for 3D)
        NSUInteger groupX = 8;
        NSUInteger groupY = 8;
        NSUInteger groupZ = maxThreadsPerGroup / (groupX * groupY);  // 16 with 1024 max
        if (groupZ > (NSUInteger)params.nt) groupZ = (NSUInteger)params.nt;

        MTLSize threadsPerGroup = MTLSizeMake(groupX, groupY, groupZ);
        MTLSize gridSize = MTLSizeMake(params.nx, params.ny, params.nt);

        auto kernel_start = std::chrono::high_resolution_clock::now();

        // Dispatch using dispatchThreads for exact grid coverage
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];

        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        auto kernel_end = std::chrono::high_resolution_clock::now();

        // Check for errors
        if (commandBuffer.status == MTLCommandBufferStatusError) {
            NSLog(@"Metal kernel error: %@", commandBuffer.error);
            throw std::runtime_error("Metal kernel execution failed");
        }

        // Copy results back
        float* gpu_image = (float*)[b_image contents];
        int32_t* gpu_fold = (int32_t*)[b_fold contents];

        // Convert float image back to double and accumulate
        for (size_t i = 0; i < image_size; i++) {
            image[i] += static_cast<double>(gpu_image[i]);
        }
        // Copy fold (already int32)
        memcpy(fold, gpu_fold, params.nx * params.ny * sizeof(int32_t));

        auto total_end = std::chrono::high_resolution_clock::now();

        // Calculate metrics
        metrics.kernel_time_ms = std::chrono::duration<double, std::milli>(
            kernel_end - kernel_start).count();
        metrics.total_time_ms = std::chrono::duration<double, std::milli>(
            total_end - total_start).count();
        metrics.traces_processed = params.n_traces;
        metrics.samples_written = image_size;
    }

    return metrics;
}

void MetalKernel::synchronize() {
    // GPU commands are synchronous in current implementation
    // (waitUntilCompleted in migrate_tile)
}

void MetalKernel::cleanup() {
    if (impl_) {
        impl_->cleanup();
    }
}

} // namespace pstm
