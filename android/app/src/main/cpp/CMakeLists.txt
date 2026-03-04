cmake_minimum_required(VERSION 3.22.1)
project("furnit_native")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 16KB ELF page alignment for Android 15+ compatibility
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

option(ENABLE_NCNN "Build NCNN libraries (yoloe_ncnn, sharp_ncnn)" OFF)

# ============================================================================
# ExecuTorch C++ tile inference (always built)
# Links against the pre-built libexecutorch.so extracted from executorch.aar.
# Headers are vendored in third_party/executorch/ (relative to android/ root).
# Include path must be parent of executorch so <executorch/extension/...> resolves.
# ============================================================================
set(EXECUTORCH_ROOT ${CMAKE_SOURCE_DIR}/../../../../third_party/executorch)
set(EXECUTORCH_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../../../../third_party)
set(EXECUTORCH_LIB_DIR ${CMAKE_SOURCE_DIR}/executorch_lib)

add_library(executorch_prebuilt SHARED IMPORTED)
set_target_properties(executorch_prebuilt PROPERTIES IMPORTED_LOCATION ${EXECUTORCH_LIB_DIR}/libexecutorch.so)

add_library(
    sharp_executorch_tiles
    SHARED
    sharp_executorch_tiles.cpp
)

# executorch headers use <c10/util/irange.h> from portable_type/c10
target_include_directories(
    sharp_executorch_tiles
    PRIVATE
    ${EXECUTORCH_INCLUDE_DIR}
    ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

# Skip ExecuTorch's generated cmake_macros.h (not present in vendored headers)
target_compile_definitions(sharp_executorch_tiles PRIVATE C10_USING_CUSTOM_GENERATED_MACROS)
set_target_properties(sharp_executorch_tiles PROPERTIES
    COMPILE_FLAGS "-frtti -fexceptions"
)

target_link_libraries(
    sharp_executorch_tiles
    executorch_prebuilt
    ${log-lib}
)

# ============================================================================
# NCNN libraries (opt-in: -DENABLE_NCNN=ON)
# ============================================================================
if(ENABLE_NCNN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fexceptions")

    set(NCNN_DIR ${CMAKE_SOURCE_DIR}/ncnn-20260113-android-vulkan/${ANDROID_ABI})

    include_directories(${NCNN_DIR}/include/ncnn)
    include_directories(${NCNN_DIR}/include)

    add_library(ncnn STATIC IMPORTED)
    set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${NCNN_DIR}/lib/libncnn.a)

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

    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

    add_library(yoloe_ncnn SHARED yoloe_ncnn.cpp)
    target_link_libraries(yoloe_ncnn ncnn glslang SPIRV glslang-default-resource-limits GenericCodeGen MachineIndependent OSDependent OpenMP::OpenMP_CXX ${log-lib} ${android-lib} ${jnigraphics-lib})
    target_include_directories(yoloe_ncnn PRIVATE ${NCNN_DIR}/include)

    add_library(sharp_ncnn SHARED sharp_ncnn.cpp)
    target_link_libraries(sharp_ncnn ncnn glslang SPIRV glslang-default-resource-limits GenericCodeGen MachineIndependent OSDependent OpenMP::OpenMP_CXX ${log-lib} ${android-lib} ${jnigraphics-lib})
    target_include_directories(sharp_ncnn PRIVATE ${NCNN_DIR}/include)
endif()
