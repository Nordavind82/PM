cmake_minimum_required(VERSION 3.18)
project(pstm_metal LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    # Try to find pybind11 via pip
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_RESULT
    )
    if(PYBIND11_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

# macOS specific settings
if(APPLE)
    # Find Metal framework
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)

    message(STATUS "Metal framework: ${METAL_FRAMEWORK}")
    message(STATUS "Foundation framework: ${FOUNDATION_FRAMEWORK}")
endif()

# Compile Metal shaders to metallib
set(SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/migrate_tile.metal)
set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/migrate_tile.metallib)

set(SHADER_OPT_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/migrate_optimized.metal)
set(SHADER_OPT_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/migrate_optimized.metallib)

# Metal shader compiler flags for fast math optimizations
# -ffast-math: Enable fast math (less precise but faster sqrt, rsqrt, etc.)
# -O3: Maximum optimization level
set(METAL_COMPILER_FLAGS -ffast-math -O3)

add_custom_command(
    OUTPUT ${SHADER_OUTPUT}
    COMMAND xcrun -sdk macosx metal ${METAL_COMPILER_FLAGS} -c ${SHADER_SOURCE} -o ${CMAKE_CURRENT_BINARY_DIR}/migrate_tile.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/migrate_tile.air -o ${SHADER_OUTPUT}
    DEPENDS ${SHADER_SOURCE}
    COMMENT "Compiling Metal shader (with fast-math)"
)

add_custom_command(
    OUTPUT ${SHADER_OPT_OUTPUT}
    COMMAND xcrun -sdk macosx metal ${METAL_COMPILER_FLAGS} -c ${SHADER_OPT_SOURCE} -o ${CMAKE_CURRENT_BINARY_DIR}/migrate_optimized.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/migrate_optimized.air -o ${SHADER_OPT_OUTPUT}
    DEPENDS ${SHADER_OPT_SOURCE}
    COMMENT "Compiling Metal shader (optimized with fast-math)"
)

add_custom_target(metal_shaders DEPENDS ${SHADER_OUTPUT} ${SHADER_OPT_OUTPUT})

# Source files
set(SOURCES
    src/metal_device.mm
    src/migration_kernel.mm
    src/kernel_benchmark.mm
    src/bindings.cpp
)

# Create Python module
pybind11_add_module(pstm_metal ${SOURCES})

# Dependencies
add_dependencies(pstm_metal metal_shaders)

target_include_directories(pstm_metal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pstm_metal PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${METALKIT_FRAMEWORK}
)

# Embed shader paths
target_compile_definitions(pstm_metal PRIVATE
    SHADER_PATH="${SHADER_OUTPUT}"
    SHADER_OPT_PATH="${SHADER_OPT_OUTPUT}"
)

# Enable fast math and optimization
target_compile_options(pstm_metal PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffast-math>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-O3 -ffast-math -fobjc-arc>
)

# Copy metallib to output directory alongside the module
add_custom_command(TARGET pstm_metal POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_OUTPUT} $<TARGET_FILE_DIR:pstm_metal>/migrate_tile.metallib
    COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_OPT_OUTPUT} $<TARGET_FILE_DIR:pstm_metal>/migrate_optimized.metallib
    COMMENT "Copying metallib to output directory"
)

# Install target
install(TARGETS pstm_metal DESTINATION .)
install(FILES ${SHADER_OUTPUT} DESTINATION .)
