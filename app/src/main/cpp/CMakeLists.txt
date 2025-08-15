# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
# Copyright (c) 2022 Magic Leap, Inc. All Rights Reserved.
# Use of this file is governed by the Software License Agreement,
# located here: https://www.magicleap.com/software-license-agreement-ml2
# Terms and conditions applicable to third-party materials accompanying
# this distribution may also be found in the top-level NOTICE file
# appearing herein.
# %COPYRIGHT_END%
# ---------------------------------------------------------------------
# %BANNER_END%

cmake_minimum_required(VERSION 3.22.1)

project(camera_mixed_reality)


# Set the ONNX Runtime prebuilt shared library path
set(ORT_LIB "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so")

file(TO_CMAKE_PATH "$ENV{MLSDK}" MLSDK)
list(APPEND CMAKE_MODULE_PATH "${MLSDK}/cmake" "${ANDROID_NDK}/../../mlsdk/cmake")

find_package(MagicLeap REQUIRED)
find_package(MagicLeapAppFramework REQUIRED)

# Import the ONNX Runtime shared library
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_LIB}"
        # Point to the parent of 'onnxruntime/' folder in include
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/onnxruntime/include"
)

add_library(camera_mixed_reality SHARED main.cpp)

include(DeprecatedApiUsage)
target_include_directories(camera_mixed_reality PRIVATE
        "${CMAKE_SOURCE_DIR}/onnxruntime/include"  # ONNX Runtime headers
)

use_deprecated_api(camera_mixed_reality)

target_link_libraries(camera_mixed_reality
        ML::app_framework
        ML::camera
        ML::media_format
        ML::media_recorder
        ML::media_error
        onnxruntime    # ONNX Runtime shared library
        dl             # For dlopen, dlsym
)

# Optional: set standard C++ version and flags
set_target_properties(camera_mixed_reality PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)
if (COMMAND copy_artifacts)
    copy_artifacts(camera_mixed_reality)
endif()











