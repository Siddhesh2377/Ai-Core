# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

project("ai_core")

# ============================================================================
# OpenCL Configuration - Multi-ABI Support
# ============================================================================

# Force OpenCL detection for llama.cpp
set(GGML_OPENCL ON CACHE BOOL "Enable OpenCL" FORCE)
set(GGML_USE_OPENCL ON CACHE BOOL "Enable OpenCL backend" FORCE)
set(OpenCL_FOUND TRUE CACHE BOOL "OpenCL found" FORCE)

# Add OpenCL compile definitions globally
add_compile_definitions(GGML_USE_OPENCL)
add_compile_definitions(GGML_OPENCL)

# Get include path from Gradle (or use default)
if(DEFINED OpenCL_INCLUDE_DIR)
    message(STATUS "Using OpenCL headers from: ${OpenCL_INCLUDE_DIR}")
    set(OpenCL_INCLUDE_DIRS "${OpenCL_INCLUDE_DIR}" CACHE PATH "OpenCL include directory" FORCE)
    include_directories(SYSTEM "${OpenCL_INCLUDE_DIR}")
    add_compile_options("-I${OpenCL_INCLUDE_DIR}")
else()
    # Fallback to local path
    set(OpenCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencl/include")
    set(OpenCL_INCLUDE_DIRS "${OpenCL_INCLUDE_DIR}" CACHE PATH "OpenCL include directory" FORCE)
    include_directories(SYSTEM "${OpenCL_INCLUDE_DIR}")
    add_compile_options("-I${OpenCL_INCLUDE_DIR}")
    message(STATUS "Using default OpenCL headers: ${OpenCL_INCLUDE_DIR}")
endif()

# Construct ABI-specific library path
if(DEFINED OPENCL_LIB_BASE AND ANDROID_ABI)
    set(OPENCL_LIBRARY_PATH "${OPENCL_LIB_BASE}/${ANDROID_ABI}/libOpenCL.so")
    message(STATUS "Using OpenCL library for ${ANDROID_ABI}: ${OPENCL_LIBRARY_PATH}")

    # Create imported library target
    add_library(OpenCL SHARED IMPORTED)
    set_target_properties(OpenCL PROPERTIES
            IMPORTED_LOCATION "${OPENCL_LIBRARY_PATH}"
    )

    set(OpenCL_LIBRARIES "${OPENCL_LIBRARY_PATH}" CACHE STRING "OpenCL library" FORCE)
    set(OpenCL_LIBRARY "${OPENCL_LIBRARY_PATH}" CACHE STRING "OpenCL library" FORCE)
else()
    # Fallback: use system OpenCL (runtime linking)
    message(WARNING "OPENCL_LIB_BASE not set, using system OpenCL")
    set(OpenCL_LIBRARIES "OpenCL" CACHE STRING "OpenCL library" FORCE)
    set(OpenCL_LIBRARY "OpenCL" CACHE STRING "OpenCL library" FORCE)
endif()

message(STATUS "=== OpenCL Configuration Summary ===")
message(STATUS "  GGML_OPENCL:        ${GGML_OPENCL}")
message(STATUS "  ANDROID_ABI:        ${ANDROID_ABI}")
message(STATUS "  Include Directory:  ${OpenCL_INCLUDE_DIR}")
message(STATUS "  Library Path:       ${OpenCL_LIBRARY}")
message(STATUS "=====================================")

# ============================================================================
# Disable Vulkan completely
# ============================================================================
set(GGML_VULKAN OFF CACHE BOOL "Disable Vulkan" FORCE)

# ============================================================================
# Add llama.cpp subdirectory
# ============================================================================
include_directories(/home/home/dev/AI/llama.cpp/src)
add_subdirectory(/home/home/dev/AI/llama.cpp build-llama)

# ============================================================================
# Create main library
# ============================================================================
add_library(${CMAKE_PROJECT_NAME} SHARED
        ai_core.cpp
        cpu_helper.cpp
        embed_search.cpp
)

# ============================================================================
# cpufeatures helper
# ============================================================================
set(NDK_CPUFEATURES_DIR ${ANDROID_NDK}/sources/android/cpufeatures)

add_library(cpufeatures STATIC
        ${NDK_CPUFEATURES_DIR}/cpu-features.c
)

target_include_directories(cpufeatures PUBLIC
        ${NDK_CPUFEATURES_DIR}
)

# ============================================================================
# Link everything (NO VULKAN!)
# ============================================================================
target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        common
        cpufeatures
        android
        log
        OpenCL  # OpenCL library (ABI-specific or system)
)

message(STATUS "=== Build Configuration Complete for ${ANDROID_ABI} ===")