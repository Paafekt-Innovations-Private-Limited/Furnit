// LinearAlgebra.swift
// Pure math helpers for ``RoomGeometryEngine`` (covariance PCA / Jacobi).

import Foundation
import simd

// MARK: - 3×3 symmetric matrix (column-major simd_float3x3)

/// Column-major: index `matrix[col][row]`.
func matrixGet(_ matrix: simd_float3x3, row: Int, col: Int) -> Float {
    matrix[col][row]
}

func matrixSet(_ matrix: inout simd_float3x3, row: Int, col: Int, value: Float) {
    matrix[col][row] = value
}

// MARK: - Jacobi eigendecomposition

/// Unit eigenvector for the **smallest** eigenvalue of a real symmetric 3×3 matrix, or `nil` if degenerate.
///
/// Ten sweeps; each sweep applies Jacobi rotations for pairs (0,1), (0,2), (1,2).
func smallestEigenvector(ofSymmetric3x3 matrix: simd_float3x3) -> SIMD3<Float>? {
    var a = matrix
    var v = matrix_identity_float3x3

    for _ in 0 ..< 10 {
        jacobiRotate(matrix: &a, eigenvectors: &v, p: 0, q: 1)
        jacobiRotate(matrix: &a, eigenvectors: &v, p: 0, q: 2)
        jacobiRotate(matrix: &a, eigenvectors: &v, p: 1, q: 2)
    }

    let eigenvalues = SIMD3<Float>(
        matrixGet(a, row: 0, col: 0),
        matrixGet(a, row: 1, col: 1),
        matrixGet(a, row: 2, col: 2)
    )

    let minIndex: Int
    if eigenvalues.x <= eigenvalues.y && eigenvalues.x <= eigenvalues.z {
        minIndex = 0
    } else if eigenvalues.y <= eigenvalues.z {
        minIndex = 1
    } else {
        minIndex = 2
    }

    let eigenvector = v[minIndex]
    let lengthSq = simd_length_squared(eigenvector)
    guard lengthSq > 1e-8 else { return nil }
    return eigenvector / sqrt(lengthSq)
}

func jacobiRotate(
    matrix a: inout simd_float3x3,
    eigenvectors v: inout simd_float3x3,
    p: Int,
    q: Int
) {
    let apq = matrixGet(a, row: p, col: q)
    guard abs(apq) > 1e-8 else { return }

    let app = matrixGet(a, row: p, col: p)
    let aqq = matrixGet(a, row: q, col: q)

    let tau = (aqq - app) / (2.0 * apq)
    let t: Float
    if tau >= 0 {
        t = 1.0 / (tau + sqrt(1.0 + tau * tau))
    } else {
        t = 1.0 / (tau - sqrt(1.0 + tau * tau))
    }

    let c = 1.0 / sqrt(1.0 + t * t)
    let s = t * c

    matrixSet(&a, row: p, col: p, value: app - t * apq)
    matrixSet(&a, row: q, col: q, value: aqq + t * apq)
    matrixSet(&a, row: p, col: q, value: 0)
    matrixSet(&a, row: q, col: p, value: 0)

    for r in 0 ..< 3 where r != p && r != q {
        let arp = matrixGet(a, row: r, col: p)
        let arq = matrixGet(a, row: r, col: q)
        matrixSet(&a, row: r, col: p, value: c * arp - s * arq)
        matrixSet(&a, row: p, col: r, value: c * arp - s * arq)
        matrixSet(&a, row: r, col: q, value: s * arp + c * arq)
        matrixSet(&a, row: q, col: r, value: s * arp + c * arq)
    }

    for r in 0 ..< 3 {
        let vrp = matrixGet(v, row: r, col: p)
        let vrq = matrixGet(v, row: r, col: q)
        matrixSet(&v, row: r, col: p, value: c * vrp - s * vrq)
        matrixSet(&v, row: r, col: q, value: s * vrp + c * vrq)
    }
}

// MARK: - Vector

func normalizeOrZero(_ vector: SIMD3<Float>) -> SIMD3<Float> {
    let lengthSquared = simd_length_squared(vector)
    guard lengthSquared > 1e-8 else { return .zero }
    return vector / sqrt(lengthSquared)
}
