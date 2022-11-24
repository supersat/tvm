#!/bin/bash

exit

cmake -S apps/hexagon_api -B apps/hexagon_api/build \
  -DUSE_ANDROID_TOOLCHAIN=$HEXAGON_SDK_ROOT/android-ndk-r19c/build/cmake/android.toolchain.cmake \
  -DANDROID_PLATFORM=android-28 -DANDROID_ABI=arm64-v8a -DUSE_HEXAGON_ARCH=v69 \
  -DUSE_HEXAGON_SDK=$HEXAGON_SDK_ROOT -DUSE_HEXAGON_TOOLCHAIN=$HEXAGON_TOOLCHAIN \
  -DUSE_OUTPUT_BINARY_DIR=$TVM_HOME/build/hexagon_api_output && \
cmake --build apps/hexagon_api/build -j$(nproc) && \
cmake -B build -DUSE_LLVM=/opt/llvm/bin/llvm-config -DUSE_CPP_RPC=ON \
  -DUSE_HEXAGON_SDK=$HEXAGON_SDK_ROOT -DUSE_HEXAGON=ON && \
cmake --build build -j$(nproc)

