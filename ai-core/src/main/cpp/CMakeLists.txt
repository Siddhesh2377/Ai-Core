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
add_subdirectory(${LLAMACPP_DIR} llama-build)

# 3️⃣  Gather all *source files* that belong to the JNI library
set(SRC_FILES
        src/ai_core.cpp
        src/ai_core_embed.cpp          # ← embedding JNI
        src/model_state.cpp
        src/utils/jni_utils.cpp
        src/utils/utf8_utils.cpp
        src/utils/chat_template.cpp
        src/utils/tool_call_state.cpp
        src/cpu/cpu_helper.cpp
)
# 4️⃣  Optional: the single header file `llama.h` from the sub‑project
#      (the sub‑project already installs it in <build-llama>/include)
#      We just add the build directory so the compiler can find it.
include_directories(${LLAMACPP_DIR})           # llama-api headers
include_directories(${LLAMACPP_DIR}/include)    # if <build-llama>/include is used

# 5️⃣  The JNI shared library
add_library(ai_core SHARED ${SRC_FILES})

# 6️⃣  Android‑specific CPU‑features library (optional but
#      typically required for the NDK)
set(NDK_CPUFEATURES_DIR ${ANDROID_NDK}/sources/android/cpufeatures)
add_library(cpufeatures STATIC ${NDK_CPUFEATURES_DIR}/cpu-features.c)
target_include_directories(cpufeatures PUBLIC ${NDK_CPUFEATURES_DIR})

# 7️⃣  Link everything together
target_link_libraries(ai_core
        PRIVATE llama          # built from the llama‑cpp sub‑project
        PRIVATE cpufeatures
        PRIVATE android        # Android NDK system library
        PRIVATE log           # Android log
)

# 8️⃣  Optional – set the build type (Release is default for NDk builds)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# 9️⃣  Show what we are building
message(STATUS "=== ai_core build type: ${CMAKE_BUILD_TYPE} ===")