cmake_minimum_required(VERSION 3.22.1)
project("yoloe_ncnn")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable RTTI to match NCNN build (NCNN built with -fno-rtti)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fexceptions")

# NCNN library path (with Vulkan support) - using latest release 20260113
set(NCNN_DIR ${CMAKE_SOURCE_DIR}/ncnn-20260113-android-vulkan/${ANDROID_ABI})

# Include NCNN headers
include_directories(${NCNN_DIR}/include/ncnn)
include_directories(${NCNN_DIR}/include)

# Import NCNN as static library
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${NCNN_DIR}/lib/libncnn.a)

# Import glslang libraries (required for NCNN Vulkan build, even if not using Vulkan)
add_library(glslang STATIC IMPORTED)
set_target_properties(glslang PROPERTIES IMPORTED_LOCATION ${NCNN_DIR}/lib/libglslang.a)

add_library(SPIRV STATIC IMPORTED)
set_target_properties(SPIRV PROPERTIES IMPORTED_LOCATION ${NCNN_DIR}/lib/libSPIRV.a)

add_library(glslang-default-resource-limits STATIC IMPORTED)
set_target_properties(glslang-default-resource-limits PROPERTIES IMPORTED_LOCATION ${NCNN_DIR}/lib/libglslang-default-resource-limits.a)

add_library(GenericCodeGen STATIC IMPORTED)
set_target_properties(GenericCodeGen PROPERTIES IMPORTED_LOCATION ${NCNN_DIR}/lib/libGenericCodeGen.a)

add_library(MachineIndependent STATIC IMPORTED)
set_target_properties(MachineIndependent PROPERTIES IMPORTED_LOCATION ${NCNN_DIR}/lib/libMachineIndependent.a)

add_library(OSDependent STATIC IMPORTED)
set_target_properties(OSDependent PROPERTIES IMPORTED_LOCATION ${NCNN_DIR}/lib/libOSDependent.a)

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
    glslang
    SPIRV
    glslang-default-resource-limits
    GenericCodeGen
    MachineIndependent
    OSDependent
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
    glslang
    SPIRV
    glslang-default-resource-limits
    GenericCodeGen
    MachineIndependent
    OSDependent
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
