cmake_minimum_required(VERSION 3.22.1)
project("furnit_native")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 16KB ELF page alignment for Android 15+ compatibility
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)
find_library(vulkan-lib vulkan)

# ============================================================================
# ExecuTorch C++ tile inference (SHARP 16-tile Part4b)
# Links against pre-built libexecutorch.so in executorch_lib/ (extract from AAR or build).
# Headers from android/third_party/executorch (include path = parent of executorch/).
# ============================================================================
# From app/src/main/cpp, ../../../../ = android/
set(EXECUTORCH_BASE ${CMAKE_SOURCE_DIR}/../../../../third_party)
set(EXECUTORCH_ROOT ${EXECUTORCH_BASE}/executorch)
set(EXECUTORCH_INCLUDE_DIR ${EXECUTORCH_BASE})
# Gradle passes -DEXECUTORCH_LIB_DIR per product flavor (executorch_lib_etVulkan vs executorch_lib_etCpu).
if(NOT EXECUTORCH_LIB_DIR)
    set(EXECUTORCH_LIB_DIR "${CMAKE_SOURCE_DIR}/executorch_lib")
endif()
# ETDump in-app profiling: set ON only when linking a local ExecuTorch build with DEVTOOLS (AAR has no ETDumpGen).
if(NOT DEFINED EXECUTORCH_PROFILING_ETDUMP)
    set(EXECUTORCH_PROFILING_ETDUMP OFF)
endif()

# When using a split ExecuTorch build (libexecutorch.so + libexecutorch_core.so), both must be
# linked so runtime/extension symbols (e.g. _get_log_timestamp, make_tensor_ptr) resolve.
if(EXISTS "${EXECUTORCH_LIB_DIR}/libexecutorch.so" AND EXISTS "${EXECUTORCH_ROOT}/extension/module/module.h")
    add_library(executorch_prebuilt SHARED IMPORTED)
    set_target_properties(executorch_prebuilt PROPERTIES IMPORTED_LOCATION ${EXECUTORCH_LIB_DIR}/libexecutorch.so)

    set(EXECUTORCH_CORE_SO "${EXECUTORCH_LIB_DIR}/libexecutorch_core.so")
    if(EXISTS "${EXECUTORCH_CORE_SO}")
      add_library(executorch_core_prebuilt SHARED IMPORTED)
      set_target_properties(executorch_core_prebuilt PROPERTIES IMPORTED_LOCATION ${EXECUTORCH_CORE_SO})
      set(EXECUTORCH_LIBS executorch_prebuilt executorch_core_prebuilt)
    else()
      set(EXECUTORCH_LIBS executorch_prebuilt)
    endif()

    # Extension .so files (Module, make_tensor_ptr) when using ExecuTorch shared build with extension.
    set(EXECUTORCH_EXTENSION_LIBS "")
    foreach(ext_lib extension_data_loader extension_flat_tensor extension_named_data_map extension_tensor extension_module_shared)
      set(ext_so "${EXECUTORCH_LIB_DIR}/lib${ext_lib}.so")
      if(EXISTS "${ext_so}")
        add_library(${ext_lib}_prebuilt SHARED IMPORTED)
        set_target_properties(${ext_lib}_prebuilt PROPERTIES IMPORTED_LOCATION ${ext_so})
        list(APPEND EXECUTORCH_EXTENSION_LIBS ${ext_lib}_prebuilt)
      endif()
    endforeach()
    if(EXECUTORCH_EXTENSION_LIBS)
      list(APPEND EXECUTORCH_LIBS ${EXECUTORCH_EXTENSION_LIBS})
    endif()

    add_library(sharp_executorch_tiles SHARED sharp_executorch_tiles.cpp)

    target_include_directories(sharp_executorch_tiles PRIVATE
        ${EXECUTORCH_INCLUDE_DIR}
        ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
    )
    target_compile_definitions(sharp_executorch_tiles PRIVATE C10_USING_CUSTOM_GENERATED_MACROS)
    set_target_properties(sharp_executorch_tiles PROPERTIES COMPILE_FLAGS "-frtti -fexceptions")
    target_link_libraries(sharp_executorch_tiles ${EXECUTORCH_LIBS} ${log-lib})

    # Full INT8 pipeline in C++ (Part1–4b single, no tiles)
    # -O3 -ffast-math: aggressive auto-vectorization; NEON intrinsics used explicitly for downsample
    add_library(sharp_executorch_full SHARED
        sharp_executorch_full_common.cpp
        sharp_executorch_full.cpp
        sharp_executorch_full_vulkan.cpp)
    target_include_directories(sharp_executorch_full PRIVATE
        ${EXECUTORCH_INCLUDE_DIR}
        ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
    )
    # ETDump profiling: opt-in only. The AAR's libexecutorch.so is built without devtools, so
    # ETDumpGen symbols are undefined if we compile the ETDump path. Enable only when linking
    # against a local ExecuTorch build with EXECUTORCH_BUILD_DEVTOOLS=ON (see EXECUTORCH_VULKAN_PROFILING.md).
    if(EXISTS "${EXECUTORCH_ROOT}/devtools/etdump/etdump_flatcc.h" AND EXECUTORCH_PROFILING_ETDUMP)
      target_compile_definitions(sharp_executorch_full PRIVATE EXECUTORCH_HAS_ETDUMP=1)
    endif()
    target_compile_definitions(sharp_executorch_full PRIVATE C10_USING_CUSTOM_GENERATED_MACROS)
    set_target_properties(sharp_executorch_full PROPERTIES COMPILE_FLAGS "-frtti -fexceptions -O3 -ffast-math")
    target_link_libraries(sharp_executorch_full ${EXECUTORCH_LIBS} ${log-lib})
endif()

# ============================================================================
# NCNN libraries (opt-in: -DENABLE_NCNN=ON when furnit.enableNative=true)
# - yoloe_ncnn: segmentation (always when ENABLE_NCNN=ON)
# - sharp_ncnn: legacy SHARP via NCNN (opt-in ENABLE_NCNN_SHARP=ON, furnit.enableNcnnSharp=true)
# ============================================================================
option(ENABLE_NCNN "Build NCNN libraries (yoloe_ncnn; sharp_ncnn if ENABLE_NCNN_SHARP)" OFF)
option(ENABLE_NCNN_SHARP "Build libsharp_ncnn (SHARP NCNN path; ExecuTorch is separate)" OFF)

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

    if(ENABLE_NCNN_SHARP)
        add_library(sharp_ncnn SHARED sharp_ncnn.cpp)
        target_link_libraries(sharp_ncnn ncnn glslang SPIRV glslang-default-resource-limits GenericCodeGen MachineIndependent OSDependent OpenMP::OpenMP_CXX ${log-lib} ${android-lib} ${jnigraphics-lib})
        target_include_directories(sharp_ncnn PRIVATE ${NCNN_DIR}/include)
    endif()
endif()

# ============================================================================
# Vulkan 1536x1536 support test (no ExecuTorch dependency)
# ============================================================================
if(vulkan-lib)
    add_library(vulkan_1536_test SHARED vulkan_1536_test.cpp)
    target_link_libraries(vulkan_1536_test ${vulkan-lib} ${log-lib})
endif()
