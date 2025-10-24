# ------------------------------------------------------------------
#   ai_core - Android / C++ JNI library
# ------------------------------------------------------------------
cmake_minimum_required(VERSION 3.22.1)
project(ai_core LANGUAGES C CXX)

# 1️⃣  C++ standard & Android ABI
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2️⃣  Where the llama.cpp source lives (adjust if you clone elsewhere)
set(LLAMACPP_DIR /home/home/dev/AI/llama.cpp)          # <-- change this if needed
if(NOT EXISTS "${LLAMACPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Could not find llama.cpp in ${LLAMACPP_DIR}")
endif()

# 3️⃣  Add llama.cpp as a subdirectory and build it
add_subdirectory(${LLAMACPP_DIR} llama-build)

# 4️⃣  Gather all *source files* that belong to the JNI library
set(SRC_FILES
        src/ai_core.cpp
        src/ai_core_embed.cpp
        src/ai_core_mtmd.cpp
        src/state/global_state.cpp
        src/state/model_state.cpp
        src/utils/jni_utils.cpp
        src/utils/utf8_utils.cpp
        src/chat/chat_template.cpp
        src/tool_calling/tool_call_state.cpp
        src/cpu/cpu_helper.cpp
)

# 5️⃣  Optional: the single header file `llama.h` from the sub‑project
#      (the sub‑project already installs it in <build-llama>/include)
#      We just add the build directory so the compiler can find it.
include_directories(${LLAMACPP_DIR})           # llama-api headers
include_directories(${LLAMACPP_DIR}/include)    # if <build-llama>/include is used

# 6️⃣  The JNI shared library
add_library(ai_core SHARED ${SRC_FILES})

# 7️⃣  Android‑specific CPU‑features library (optional but
#      typically required for the NDK)
set(NDK_CPUFEATURES_DIR ${ANDROID_NDK}/sources/android/cpufeatures)
add_library(cpufeatures STATIC ${NDK_CPUFEATURES_DIR}/cpu-features.c)
target_include_directories(cpufeatures PUBLIC ${NDK_CPUFEATURES_DIR})

# 8️⃣  Define the build output directory where pre-built libraries are located
set(BUILD_OUTPUT_DIR /home/home/AndroidStudioProjects/Ai-Core/scripts/build-output)

# 9️⃣  Use the current ABI to determine the correct library path
set(ABI ${ANDROID_ABI})

# Define path for pre-built mtmd library
set(LIB_MTMD ${BUILD_OUTPUT_DIR}/${ABI}/bin/libmtmd.so)

# Check if pre-built libraries exist
if(NOT EXISTS ${LIB_MTMD})
    message(FATAL_ERROR "Pre-built libmtmd.so not found at ${LIB_MTMD}")
endif()

# Add pre-built mtmd as an imported library
add_library(mtmd SHARED IMPORTED)
set_target_properties(mtmd PROPERTIES IMPORTED_LOCATION ${LIB_MTMD})

# 🔟  Link everything together
target_link_libraries(ai_core
        PRIVATE llama
        PRIVATE ggml
        PRIVATE ggml-cpu
        PRIVATE ggml-base
        PRIVATE mtmd          # Add mtmd library to link
        PRIVATE cpufeatures
        PRIVATE android
        PRIVATE log
)

# ========== Optional Settings ==========

# 🔟  Optional – set the build type (Release is default for NDK builds)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# 🥷  Show what we are building
message(STATUS "=== ai_core build type: ${CMAKE_BUILD_TYPE} ===")
message(STATUS "=== Building for ABI: ${ANDROID_ABI} ===")