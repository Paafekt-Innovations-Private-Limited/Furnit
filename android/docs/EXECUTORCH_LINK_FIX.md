# ExecuTorch split build: linking libexecutorch_core.so

When using a **split** ExecuTorch build (BUILD_SHARED_LIBS=ON), the build produces two shared libraries. The app must link **both** so that all ExecuTorch symbols resolve. This doc records the steps applied and how to repeat them.

---

## Why this is needed

A split build produces:

- **libexecutorch.so** – depends on the core runtime
- **libexecutorch_core.so** – runtime (logging, abort, evalue, etc.)

If the app links only libexecutorch.so, the linker reports undefined symbols such as:

- executorch::runtime::internal::_get_log_timestamp
- executorch::runtime::runtime_abort
- executorch::runtime::internal::vlogf
- executorch::runtime::BoxedEvalueList<...>::get() const
- (and possibly executorch::extension::make_tensor_ptr if extensions are in the core build)

Linking **both** libexecutorch.so and libexecutorch_core.so resolves these.

---

## Steps applied

### 1. Copy libexecutorch_core.so into the app

From the ExecuTorch build directory, copy the core library into Furnit executorch_lib:

  EXECUTORCH_BUILD=/Users/al/Documents/tries01/executorch/cmake-out-android
  FURNIT_LIB=/Users/al/Documents/tries01/Furnit/android/app/src/main/cpp/executorch_lib

  cp -f "$EXECUTORCH_BUILD/libexecutorch_core.so" "$FURNIT_LIB/libexecutorch_core.so"

**Result:** android/app/src/main/cpp/executorch_lib/ contains both libexecutorch.so and libexecutorch_core.so.

### 2. Update Furnit CMake to link both libraries

**File:** android/app/src/main/cpp/CMakeLists.txt

- After defining the executorch_prebuilt IMPORTED target and its IMPORTED_LOCATION, add:
  - Set EXECUTORCH_CORE_SO to "${EXECUTORCH_LIB_DIR}/libexecutorch_core.so".
  - If EXISTS "${EXECUTORCH_CORE_SO}":
    - Add IMPORTED target executorch_core_prebuilt with IMPORTED_LOCATION ${EXECUTORCH_CORE_SO}.
    - Set EXECUTORCH_LIBS to executorch_prebuilt executorch_core_prebuilt.
  - Else:
    - Set EXECUTORCH_LIBS to executorch_prebuilt.
- For both sharp_executorch_tiles and sharp_executorch_full, change:
  - target_link_libraries(... executorch_prebuilt ${log-lib})
  - to:
  - target_link_libraries(... ${EXECUTORCH_LIBS} ${log-lib}).

So when libexecutorch_core.so is present, both .so files are linked; when it is missing, only libexecutorch.so is linked (backward compatible).

**Exact block added** (after set_target_properties(executorch_prebuilt ...)):

    set(EXECUTORCH_CORE_SO "${EXECUTORCH_LIB_DIR}/libexecutorch_core.so")
    if(EXISTS "${EXECUTORCH_CORE_SO}")
      add_library(executorch_core_prebuilt SHARED IMPORTED)
      set_target_properties(executorch_core_prebuilt PROPERTIES IMPORTED_LOCATION ${EXECUTORCH_CORE_SO})
      set(EXECUTORCH_LIBS executorch_prebuilt executorch_core_prebuilt)
    else()
      set(EXECUTORCH_LIBS executorch_prebuilt)
    endif()

And both target_link_libraries lines use ${EXECUTORCH_LIBS} instead of executorch_prebuilt.

### 3. Rebuild the app

  cd android
  ./gradlew clean
  ./gradlew assembleDebug

---

## Summary

| Step | Action |
|------|--------|
| 1 | Copy libexecutorch_core.so from ExecuTorch build dir to android/app/src/main/cpp/executorch_lib/. |
| 2 | In android/app/src/main/cpp/CMakeLists.txt, add optional executorch_core_prebuilt and set EXECUTORCH_LIBS; link ${EXECUTORCH_LIBS} for both native libs. |
| 3 | Run ./gradlew clean assembleDebug from android/. |

If undefined symbols remain (e.g. from extensions), the ExecuTorch build may need to be a single monolithic .so or include additional extension libraries; see BUILD_EXECUTORCH_SO.md if present.

---
## Extension undefined symbols (Module::Module, make_tensor_ptr)

If the linker reports:
- undefined symbol: executorch::extension::module::Module::Module(...)
- undefined symbol: executorch::extension::make_tensor_ptr(...)

the app must also link the ExecuTorch extension shared libs.

### Fix summary
1. **ExecuTorch**: In root CMakeLists.txt, when BUILD_SHARED_LIBS is ON, link extension_module_static and extension_tensor into the executorch target. In extension/module/CMakeLists.txt, when BUILD_SHARED_LIBS and Android, add shared target extension_module_shared.
2. **Rebuild ExecuTorch**: cmake --build . -j8 --target executorch extension_module_shared
3. **Furnit**: Copy libexecutorch.so, libexecutorch_core.so, libextension_*.so into executorch_lib/. In CMakeLists.txt add IMPORTED targets for each extension .so and link them (foreach ext_lib ... extension_module_shared). Put all .so in jniLibs/arm64-v8a for runtime.
4. **Build**: cd android && ./gradlew assembleDebug
