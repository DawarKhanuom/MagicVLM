cmake_minimum_required(VERSION 3.22.1)
project(camera_mixed_reality)

# ---- Magic Leap SDK ----
file(TO_CMAKE_PATH "$ENV{MLSDK}" MLSDK)
list(APPEND CMAKE_MODULE_PATH "${MLSDK}/cmake" "${ANDROID_NDK}/../../mlsdk/cmake")
find_package(MagicLeap REQUIRED)
find_package(MagicLeapAppFramework REQUIRED)

# ---- ONNX Runtime (.so + headers) ----
# Use the relative path to jniLibs
set(ORT_LIB "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so")
if (NOT EXISTS "${ORT_LIB}")
    message(FATAL_ERROR "libonnxruntime.so not found at: ${ORT_LIB}")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/onnxruntime/include"
)

# ---- App sources ----
add_library(camera_mixed_reality SHARED
        main.cpp
)

target_include_directories(camera_mixed_reality PRIVATE
        "${CMAKE_SOURCE_DIR}"                       # e.g., stb_image.h in same folder
        "${CMAKE_SOURCE_DIR}/onnxruntime/include"   # ORT headers
)

include(DeprecatedApiUsage)
use_deprecated_api(camera_mixed_reality)

target_link_libraries(camera_mixed_reality
        ML::app_framework
        ML::camera
        ML::media_format
        ML::media_recorder
        ML::media_error
        onnxruntime
)

set_target_properties(camera_mixed_reality PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

if (COMMAND copy_artifacts)
    copy_artifacts(camera_mixed_reality)
endif()
