cmake_minimum_required(VERSION 3.22.1)
project(ai_core LANGUAGES C CXX)

# ------------------------------------------------------------------
# 1) Enable Vulkan, disable OpenCL
# ------------------------------------------------------------------
set(GGML_OPENCL   OFF CACHE BOOL "" FORCE)
set(GGML_USE_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN   OFF  CACHE BOOL "" FORCE)
add_compile_definitions(GGML_VULKAN=0)

# ------------------------------------------------------------------
# 2) llama.cpp sub-project
# ------------------------------------------------------------------
include_directories(/home/home/dev/AI/llama.cpp/src)
add_subdirectory(/home/home/dev/AI/llama.cpp build-llama)

# ------------------------------------------------------------------
# 3) Vulkan C++ headers (vulkan-hpp interface)
# ------------------------------------------------------------------
add_library(vulkanhpp_iface INTERFACE)
target_include_directories(vulkanhpp_iface INTERFACE
        /home/home/dev/AI/includes/gpu/Vulkan-Headers/include
)
target_link_libraries(llama PRIVATE vulkanhpp_iface)

# ------------------------------------------------------------------
# 4) Your JNI library
# ------------------------------------------------------------------
add_library(${CMAKE_PROJECT_NAME} SHARED
        ai_core.cpp
        cpu_helper.cpp
)

# Link Vulkan headers interface here (AFTER it's created)
target_link_libraries(${PROJECT_NAME} PRIVATE vulkanhpp_iface)

# ------------------------------------------------------------------
# 5) CPU features helper
# ------------------------------------------------------------------
set(NDK_CPUFEATURES_DIR ${ANDROID_NDK}/sources/android/cpufeatures)
add_library(cpufeatures STATIC ${NDK_CPUFEATURES_DIR}/cpu-features.c)
target_include_directories(cpufeatures PUBLIC ${NDK_CPUFEATURES_DIR})

# ------------------------------------------------------------------
# 6) Find Vulkan system lib
# ------------------------------------------------------------------
find_library(VULKAN_LIB vulkan)

# ------------------------------------------------------------------
# 7) Link everything together
# ------------------------------------------------------------------
target_link_libraries(${CMAKE_PROJECT_NAME}
        PRIVATE llama
        PRIVATE common
        PRIVATE cpufeatures
        PRIVATE android
        PRIVATE log
        PRIVATE ${VULKAN_LIB}
)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
message(STATUS "=== Build type: ${CMAKE_BUILD_TYPE} ===")
