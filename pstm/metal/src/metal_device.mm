#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string>
#include <stdexcept>

namespace pstm {
namespace metal {

// Global Metal device (singleton pattern)
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_commandQueue = nil;
static bool g_initialized = false;

bool initialize_device() {
    if (g_initialized) {
        return g_device != nil;
    }

    @autoreleasepool {
        // Get default GPU device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            g_initialized = true;
            return false;
        }

        // Create command queue
        g_commandQueue = [g_device newCommandQueue];
        if (!g_commandQueue) {
            g_device = nil;
            g_initialized = true;
            return false;
        }

        g_initialized = true;
        return true;
    }
}

bool is_available() {
    if (!g_initialized) {
        initialize_device();
    }
    return g_device != nil;
}

id<MTLDevice> get_device() {
    if (!g_initialized) {
        initialize_device();
    }
    return g_device;
}

id<MTLCommandQueue> get_command_queue() {
    if (!g_initialized) {
        initialize_device();
    }
    return g_commandQueue;
}

std::string get_device_name() {
    if (!is_available()) {
        return "No Metal device";
    }
    @autoreleasepool {
        return std::string([[g_device name] UTF8String]);
    }
}

size_t get_device_memory() {
    if (!is_available()) {
        return 0;
    }
    // recommendedMaxWorkingSetSize gives the optimal memory usage
    return [g_device recommendedMaxWorkingSetSize];
}

size_t get_max_buffer_length() {
    if (!is_available()) {
        return 0;
    }
    return [g_device maxBufferLength];
}

} // namespace metal
} // namespace pstm
