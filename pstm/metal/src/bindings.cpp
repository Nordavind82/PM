#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "metal_kernel.h"
#include <stdexcept>

namespace py = pybind11;

// Global kernel instance for reuse
static std::unique_ptr<pstm::MetalKernel> g_kernel;

// Ensure kernel is initialized
static pstm::MetalKernel& get_kernel(const std::string& shader_path = "") {
    if (!g_kernel) {
        g_kernel = std::make_unique<pstm::MetalKernel>();
    }
    if (!g_kernel->initialize(shader_path)) {
        throw std::runtime_error("Failed to initialize Metal kernel. Check that Metal is available.");
    }
    return *g_kernel;
}

// Check if Metal is available
bool is_available() {
    return pstm::MetalKernel::is_available();
}

// Get device information
py::dict get_device_info() {
    py::dict info;
    info["available"] = is_available();
    info["device_name"] = pstm::MetalKernel::get_device_name();
    info["device_memory_gb"] = pstm::MetalKernel::get_device_memory() / (1024.0 * 1024.0 * 1024.0);
    return info;
}

// Initialize kernel (optional, will auto-init on first use)
bool initialize(const std::string& shader_path = "") {
    try {
        get_kernel(shader_path);
        return true;
    } catch (...) {
        return false;
    }
}

// Main migration function
py::dict migrate_tile(
    py::array_t<float, py::array::c_style | py::array::forcecast> amplitudes,
    py::array_t<double, py::array::c_style | py::array::forcecast> source_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> source_y,
    py::array_t<double, py::array::c_style | py::array::forcecast> receiver_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> receiver_y,
    py::array_t<double, py::array::c_style | py::array::forcecast> midpoint_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> midpoint_y,
    py::array_t<double, py::array::c_style> image,  // Output, must be writable
    py::array_t<int32_t, py::array::c_style> fold,   // Output, must be writable
    py::array_t<double, py::array::c_style | py::array::forcecast> x_coords,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_coords,
    py::array_t<double, py::array::c_style | py::array::forcecast> t_coords_ms,
    py::array_t<double, py::array::c_style | py::array::forcecast> vrms,
    py::dict config
) {
    // Validate array shapes
    auto amp_info = amplitudes.request();
    if (amp_info.ndim != 2) {
        throw std::runtime_error("amplitudes must be 2D array [n_traces, n_samples]");
    }
    int n_traces = static_cast<int>(amp_info.shape[0]);
    int n_samples = static_cast<int>(amp_info.shape[1]);

    auto img_info = image.request();
    if (img_info.ndim != 3) {
        throw std::runtime_error("image must be 3D array [nx, ny, nt]");
    }
    int nx = static_cast<int>(img_info.shape[0]);
    int ny = static_cast<int>(img_info.shape[1]);
    int nt = static_cast<int>(img_info.shape[2]);

    // Validate coordinate array sizes
    if (source_x.size() != n_traces || source_y.size() != n_traces ||
        receiver_x.size() != n_traces || receiver_y.size() != n_traces ||
        midpoint_x.size() != n_traces || midpoint_y.size() != n_traces) {
        throw std::runtime_error("Coordinate arrays must have n_traces elements");
    }

    if (x_coords.size() != nx) {
        throw std::runtime_error("x_coords must have nx elements");
    }
    if (y_coords.size() != ny) {
        throw std::runtime_error("y_coords must have ny elements");
    }
    if (t_coords_ms.size() != nt || vrms.size() != nt) {
        throw std::runtime_error("t_coords_ms and vrms must have nt elements");
    }

    auto fold_info = fold.request();
    if (fold_info.ndim != 2 || fold_info.shape[0] != nx || fold_info.shape[1] != ny) {
        throw std::runtime_error("fold must be 2D array [nx, ny]");
    }

    // Build parameters struct
    pstm::MigrationParams params;
    params.max_dip_deg = config.contains("max_dip_deg") ?
        config["max_dip_deg"].cast<float>() : 45.0f;
    params.min_aperture = config.contains("min_aperture") ?
        config["min_aperture"].cast<float>() : 100.0f;
    params.max_aperture = config.contains("max_aperture") ?
        config["max_aperture"].cast<float>() : 2500.0f;
    params.taper_fraction = config.contains("taper_fraction") ?
        config["taper_fraction"].cast<float>() : 0.1f;
    params.dt_ms = config.contains("dt_ms") ?
        config["dt_ms"].cast<float>() : 4.0f;
    params.t_start_ms = config.contains("t_start_ms") ?
        config["t_start_ms"].cast<float>() : 0.0f;
    params.apply_spreading = config.contains("apply_spreading") ?
        (config["apply_spreading"].cast<bool>() ? 1 : 0) : 1;
    params.apply_obliquity = config.contains("apply_obliquity") ?
        (config["apply_obliquity"].cast<bool>() ? 1 : 0) : 1;
    params.n_traces = n_traces;
    params.n_samples = n_samples;
    params.nx = nx;
    params.ny = ny;
    params.nt = nt;

    // Get shader path from config if provided
    std::string shader_path = config.contains("shader_path") ?
        config["shader_path"].cast<std::string>() : "";

    // Get kernel and execute
    auto& kernel = get_kernel(shader_path);

    auto metrics = kernel.migrate_tile(
        static_cast<const float*>(amp_info.ptr),
        source_x.data(),
        source_y.data(),
        receiver_x.data(),
        receiver_y.data(),
        midpoint_x.data(),
        midpoint_y.data(),
        static_cast<double*>(img_info.ptr),
        static_cast<int32_t*>(fold_info.ptr),
        x_coords.data(),
        y_coords.data(),
        t_coords_ms.data(),
        vrms.data(),
        params
    );

    // Return metrics as dict
    py::dict result;
    result["kernel_time_ms"] = metrics.kernel_time_ms;
    result["total_time_ms"] = metrics.total_time_ms;
    result["traces_processed"] = metrics.traces_processed;
    result["samples_written"] = metrics.samples_written;
    result["traces_per_second"] = metrics.traces_processed / (metrics.total_time_ms / 1000.0);

    return result;
}

// Cleanup resources
void cleanup() {
    if (g_kernel) {
        g_kernel->cleanup();
        g_kernel.reset();
    }
}

// Forward declaration from kernel_benchmark.mm
void register_benchmark_functions(py::module& m);

PYBIND11_MODULE(pstm_metal, m) {
    m.doc() = "PSTM Metal GPU kernel for seismic migration";

    m.def("is_available", &is_available,
        "Check if Metal GPU is available on this system");

    m.def("get_device_info", &get_device_info,
        "Get information about the Metal GPU device");

    m.def("initialize", &initialize,
        py::arg("shader_path") = "",
        "Initialize the Metal kernel (optional, auto-initializes on first use)");

    m.def("migrate_tile", &migrate_tile,
        py::arg("amplitudes"),
        py::arg("source_x"),
        py::arg("source_y"),
        py::arg("receiver_x"),
        py::arg("receiver_y"),
        py::arg("midpoint_x"),
        py::arg("midpoint_y"),
        py::arg("image"),
        py::arg("fold"),
        py::arg("x_coords"),
        py::arg("y_coords"),
        py::arg("t_coords_ms"),
        py::arg("vrms"),
        py::arg("config"),
        R"doc(
Execute PSTM migration on GPU.

Parameters
----------
amplitudes : ndarray[float32]
    Trace amplitudes, shape (n_traces, n_samples)
source_x, source_y : ndarray[float64]
    Source coordinates, shape (n_traces,)
receiver_x, receiver_y : ndarray[float64]
    Receiver coordinates, shape (n_traces,)
midpoint_x, midpoint_y : ndarray[float64]
    Midpoint coordinates, shape (n_traces,)
image : ndarray[float64]
    Output image array, shape (nx, ny, nt), modified in-place
fold : ndarray[int32]
    Output fold array, shape (nx, ny), modified in-place
x_coords, y_coords : ndarray[float64]
    Output grid coordinates
t_coords_ms : ndarray[float64]
    Output time axis in milliseconds
vrms : ndarray[float64]
    RMS velocity at each time sample
config : dict
    Migration parameters:
    - max_dip_deg: Maximum dip angle (default: 45.0)
    - min_aperture: Minimum aperture in meters (default: 100.0)
    - max_aperture: Maximum aperture in meters (default: 2500.0)
    - taper_fraction: Edge taper fraction (default: 0.1)
    - dt_ms: Sample interval in ms (default: 4.0)
    - t_start_ms: Start time in ms (default: 0.0)
    - apply_spreading: Apply spherical spreading correction (default: True)
    - apply_obliquity: Apply obliquity correction (default: True)
    - shader_path: Path to compiled shader (optional)

Returns
-------
dict
    Execution metrics including kernel_time_ms, total_time_ms, traces_per_second
)doc");

    m.def("cleanup", &cleanup,
        "Release Metal GPU resources");

    // Register benchmark functions
    register_benchmark_functions(m);
}
