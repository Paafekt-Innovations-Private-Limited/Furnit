// NativeBLAS.cpp
// JNI wrapper for native BLAS operations using Eigen
#include <jni.h>
#include <android/log.h>
#include <Eigen/Dense>
#include <vector>

#define LOG_TAG "NativeBLAS"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

/**
 * Matrix-vector multiplication: y = A * x
 * A is (m x n) in row-major layout
 * x is (n x 1)
 * y is (m x 1)
 */
JNIEXPORT void JNICALL
Java_com_example_smartypants_NativeBLAS_sgemv(
    JNIEnv* env,
    jobject /* this */,
    jint m,
    jint n,
    jfloatArray A,
    jfloatArray x,
    jfloatArray y
) {
    // Get array pointers
    jfloat* A_ptr = env->GetFloatArrayElements(A, nullptr);
    jfloat* x_ptr = env->GetFloatArrayElements(x, nullptr);
    jfloat* y_ptr = env->GetFloatArrayElements(y, nullptr);
    
    if (!A_ptr || !x_ptr || !y_ptr) {
        LOGE("Failed to get array pointers");
        return;
    }
    
    // Map to Eigen matrices (row-major)
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        mat_A(A_ptr, m, n);
    Eigen::Map<Eigen::VectorXf> vec_x(x_ptr, n);
    Eigen::Map<Eigen::VectorXf> vec_y(y_ptr, m);
    
    // Perform matrix-vector multiplication
    vec_y = mat_A * vec_x;
    
    // Release arrays
    env->ReleaseFloatArrayElements(A, A_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(x, x_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(y, y_ptr, 0);  // Copy back
}

/**
 * Matrix-matrix multiplication: C = A * B
 * A is (m x k) in row-major
 * B is (k x n) in row-major
 * C is (m x n) in row-major
 */
JNIEXPORT void JNICALL
Java_com_example_smartypants_NativeBLAS_sgemm(
    JNIEnv* env,
    jobject /* this */,
    jint m,
    jint n,
    jint k,
    jfloatArray A,
    jfloatArray B,
    jfloatArray C
) {
    jfloat* A_ptr = env->GetFloatArrayElements(A, nullptr);
    jfloat* B_ptr = env->GetFloatArrayElements(B, nullptr);
    jfloat* C_ptr = env->GetFloatArrayElements(C, nullptr);
    
    if (!A_ptr || !B_ptr || !C_ptr) {
        LOGE("Failed to get array pointers");
        return;
    }
    
    // Map to Eigen matrices
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        mat_A(A_ptr, m, k);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        mat_B(B_ptr, k, n);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        mat_C(C_ptr, m, n);
    
    // Perform matrix-matrix multiplication
    mat_C = mat_A * mat_B;
    
    // Release arrays
    env->ReleaseFloatArrayElements(A, A_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(B, B_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(C, C_ptr, 0);  // Copy back
}

/**
 * Batch matrix-vector multiplication for mask computation
 * Computes max logits across multiple detections
 * 
 * prototypes: (32 x planeSize) in column-major (each plane is contiguous)
 * coeffs: (detCount x 32) in row-major
 * output: (planeSize) max logits
 */
JNIEXPORT void JNICALL
Java_com_example_smartypants_NativeBLAS_computeMaskLogits(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray prototypes,
    jfloatArray coeffs,
    jfloatArray output,
    jint planeSize,
    jint detCount
) {
    jfloat* proto_ptr = env->GetFloatArrayElements(prototypes, nullptr);
    jfloat* coeff_ptr = env->GetFloatArrayElements(coeffs, nullptr);
    jfloat* out_ptr = env->GetFloatArrayElements(output, nullptr);
    
    if (!proto_ptr || !coeff_ptr || !out_ptr) {
        LOGE("Failed to get array pointers");
        return;
    }
    
    // Reorganize prototypes to row-major: (planeSize x 32)
    // Each row represents one pixel's 32 prototype values
    Eigen::Matrix<float, Eigen::Dynamic, 32, Eigen::RowMajor> A(planeSize, 32);
    
    for (int pixel = 0; pixel < planeSize; pixel++) {
        for (int k = 0; k < 32; k++) {
            A(pixel, k) = proto_ptr[k * planeSize + pixel];
        }
    }
    
    // Initialize output to -infinity
    Eigen::Map<Eigen::VectorXf> max_logits(out_ptr, planeSize);
    max_logits.setConstant(-std::numeric_limits<float>::infinity());
    
    // For each detection
    for (int det = 0; det < detCount; det++) {
        // Get coefficients for this detection
        Eigen::Map<Eigen::Matrix<float, 32, 1>> 
            coeffs_vec(coeff_ptr + det * 32, 32);
        
        // Compute logits: A * coeffs (planeSize x 32) * (32 x 1) = (planeSize x 1)
        Eigen::VectorXf logits = A * coeffs_vec;
        
        // Update max
        max_logits = max_logits.cwiseMax(logits);
    }
    
    // Release arrays
    env->ReleaseFloatArrayElements(prototypes, proto_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(coeffs, coeff_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(output, out_ptr, 0);  // Copy back
    
    LOGD("Computed mask logits: %d pixels, %d detections", planeSize, detCount);
}

/**
 * Element-wise max for updating max logits
 */
JNIEXPORT void JNICALL
Java_com_example_smartypants_NativeBLAS_vmax(
    JNIEnv* env,
    jobject /* this */,
    jfloatArray a,
    jfloatArray b,
    jfloatArray result,
    jint size
) {
    jfloat* a_ptr = env->GetFloatArrayElements(a, nullptr);
    jfloat* b_ptr = env->GetFloatArrayElements(b, nullptr);
    jfloat* res_ptr = env->GetFloatArrayElements(result, nullptr);
    
    if (!a_ptr || !b_ptr || !res_ptr) {
        LOGE("Failed to get array pointers");
        return;
    }
    
    Eigen::Map<Eigen::VectorXf> vec_a(a_ptr, size);
    Eigen::Map<Eigen::VectorXf> vec_b(b_ptr, size);
    Eigen::Map<Eigen::VectorXf> vec_res(res_ptr, size);
    
    vec_res = vec_a.cwiseMax(vec_b);
    
    env->ReleaseFloatArrayElements(a, a_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(b, b_ptr, JNI_ABORT);
    env->ReleaseFloatArrayElements(result, res_ptr, 0);
}

} // extern "C"
