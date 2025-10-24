#!/bin/bash

# Check if the llama.cpp directory path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/llama.cpp"
    exit 1
fi

# Define variables
LLAMACPP_DIR=$1
ANDROID_NDK=$ANDROID_NDK  # Ensure the ANDROID_NDK environment variable is set
BUILD_OUTPUT_DIR=$(pwd)/build-output

# Check if llama.cpp directory exists
if [ ! -f "${LLAMACPP_DIR}/CMakeLists.txt" ]; then
    echo "Error: Could not find llama.cpp in ${LLAMACPP_DIR}"
    exit 1
fi

# Function to build for a given architecture
build_architecture() {
    ARCH=$1
    ARCH_BUILD_DIR=$(pwd)/build-${ARCH}

    # Configure CMake for the architecture
    cmake -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
          -DANDROID_ABI=${ARCH} \
          -DANDROID_PLATFORM=android-30 \
          -DCMAKE_BUILD_TYPE=Release \
          -DLLAMA_CURL=OFF \
          -DLLAMA_BUILD_TOOLS=ON \
          -DBUILD_SHARED_LIBS=ON \
          -DLLAMA_BUILD_GGML=ON \
          -DLLAMA_BUILD_GGML_CPU=ON \
          -DLLAMA_BUILD_GGML_BASE=ON \
          -DLLAMA_BUILD_LLAMA=ON \
          -B ${ARCH_BUILD_DIR} \
          -S ${LLAMACPP_DIR}

    if [ $? -ne 0 ]; then
        echo "Error: Configuration failed for ${ARCH}"
        exit 1
    fi

    # Build the project for the architecture
    cmake --build ${ARCH_BUILD_DIR} --config Release
    if [ $? -ne 0 ]; then
        echo "Error: Build failed for ${ARCH}"
        exit 1
    fi

    # Create output directory for the architecture
    mkdir -p ${BUILD_OUTPUT_DIR}/${ARCH}/bin

    # Copy the built libraries to the output directory
    cp -v ${ARCH_BUILD_DIR}/bin/libllama.so \
          ${ARCH_BUILD_DIR}/bin/libggml.so \
          ${ARCH_BUILD_DIR}/bin/libggml-cpu.so \
          ${ARCH_BUILD_DIR}/bin/libggml-base.so \
          ${ARCH_BUILD_DIR}/bin/libmtmd.so \
          ${BUILD_OUTPUT_DIR}/${ARCH}/bin/
}

# Build for arm64-v8a and x86_64
build_architecture arm64-v8a
build_architecture x86_64

echo "Build process completed successfully."