// FurnitureFitBlas.swift
// Thin wrapper around BlasScopy (BLASWrapper.m) to avoid Swift deprecation warnings from legacy BLAS.

import Foundation

enum FurnitureFitBlas {
    typealias Dim = Int32

    @inline(__always)
    static func scopy(
        _ n: Dim,
        _ x: UnsafePointer<Float>,
        _ incx: Dim,
        _ y: UnsafeMutablePointer<Float>,
        _ incy: Dim
    ) {
        BlasScopy(n, x, incx, y, incy)
    }
}
