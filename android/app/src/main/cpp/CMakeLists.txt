cmake_minimum_required(VERSION 3.22.1)
project("yoloe_ncnn")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable RTTI for custom NCNN layers
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti -fexceptions")

# NCNN library path (CPU-only version)
set(NCNN_DIR ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan/${ANDROID_ABI})

# Include NCNN headers
include_directories(${NCNN_DIR}/include/ncnn)
include_directories(${NCNN_DIR}/include)

# Import NCNN as static library
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${NCNN_DIR}/lib/libncnn.a)

# Find required Android libraries
find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# OpenMP support (required by NCNN)
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# 16KB ELF page alignment for Android 15+ compatibility
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

# Create YOLOE shared library
add_library(
    yoloe_ncnn
    SHARED
    yoloe_ncnn.cpp
)

# Link YOLOE libraries
target_link_libraries(
    yoloe_ncnn
    ncnn
    OpenMP::OpenMP_CXX
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}
)

# Set YOLOE NCNN include directory
target_include_directories(
    yoloe_ncnn
    PRIVATE
    ${NCNN_DIR}/include
)

# Create SHARP shared library
add_library(
    sharp_ncnn
    SHARED
    sharp_ncnn.cpp
)

# Link SHARP libraries
target_link_libraries(
    sharp_ncnn
    ncnn
    OpenMP::OpenMP_CXX
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}
)

# Set SHARP NCNN include directory
target_include_directories(
    sharp_ncnn
    PRIVATE
    ${NCNN_DIR}/include
)
