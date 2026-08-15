import simd

struct Point3 {
    var x: Float
    var y: Float
    var z: Float
}

struct Plane {
    var normal: SIMD3<Float>
    var d: Float
}

struct RoomExtentResult {
    var width: Float
    var depth: Float
    var height: Float
    var confidence: Float
    var approximate: Bool
    var debug: String
}

struct DepthMaskResult {
    var valid: [Bool]
    var debug: String
}

enum RoomExtent {
    static func buildInvalidDepthMask(
        depth: [Float],
        width: Int,
        height: Int,
        detections: [(cls: Int, box: (x0: Int, y0: Int, x1: Int, y1: Int), conf: Float)],
        focalPx: Float,
        cx: Float,
        cy: Float,
        depthP98: Float? = nil
    ) -> DepthMaskResult {
        let pixelCount = max(0, width * height)
        var valid = [Bool](repeating: true, count: pixelCount)
        guard width > 1, height > 1, depth.count == pixelCount else {
            return DepthMaskResult(valid: valid, debug: "invalid input valid=0/\(pixelCount)")
        }

        func index(_ x: Int, _ y: Int) -> Int { y * width + x }

        var central: [Float] = []
        central.reserveCapacity((width / 2) * (height / 2))
        for y in (height / 4)..<(3 * height / 4) {
            for x in (width / 4)..<(3 * width / 4) {
                let sample = depth[index(x, y)]
                if sample.isFinite, sample > 0 {
                    central.append(sample)
                }
            }
        }
        central.sort()
        let wallRef = central.isEmpty ? 0 : central[central.count / 2]
        let beyondCut = wallRef > 0 ? wallRef * 1.5 : Float.greatestFiniteMagnitude

        var beyond = 0
        for i in 0..<pixelCount where depth[i].isFinite && depth[i] > beyondCut {
            valid[i] = false
            beyond += 1
        }

        var window = 0
        let p98: Float?
        if let depthP98 {
            p98 = depthP98
        } else {
            let sorted = depth.filter { $0.isFinite && $0 > 0 }.sorted()
            if sorted.isEmpty {
                p98 = nil
            } else {
                let p98Index = min(sorted.count - 1, max(0, Int(Double(sorted.count - 1) * 0.98)))
                p98 = sorted[p98Index]
            }
        }
        if let p98 {
            for i in 0..<pixelCount where depth[i].isFinite && depth[i] >= p98 {
                if valid[i] { window += 1 }
                valid[i] = false
            }
        }

        var mirror = 0
        for detection in detections {
            let x0 = max(0, min(width - 1, detection.box.x0))
            let y0 = max(0, min(height - 1, detection.box.y0))
            let x1 = max(0, min(width - 1, detection.box.x1))
            let y1 = max(0, min(height - 1, detection.box.y1))
            guard x0 < x1, y0 < y1 else { continue }

            var objectDepths: [Float] = []
            objectDepths.reserveCapacity((x1 - x0 + 1) * (y1 - y0 + 1))
            for y in y0...y1 {
                for x in x0...x1 {
                    let sample = depth[index(x, y)]
                    if sample.isFinite, sample > 0 {
                        objectDepths.append(sample)
                    }
                }
            }
            objectDepths.sort()
            guard !objectDepths.isEmpty else { continue }
            let p20 = objectDepths[min(objectDepths.count - 1, objectDepths.count / 5)]
            if wallRef > 0, p20 > wallRef * 1.3 {
                let padX = max(1, (x1 - x0) / 2)
                let padY = max(1, (y1 - y0) / 2)
                let mx0 = max(0, x0 - padX)
                let mx1 = min(width - 1, x1 + padX)
                let my0 = max(0, y0 - padY)
                let my1 = min(height - 1, y1 + padY)
                for y in my0...my1 {
                    for x in mx0...mx1 {
                        let i = index(x, y)
                        if valid[i] { mirror += 1 }
                        valid[i] = false
                    }
                }
            }
        }

        let validCount = valid.lazy.filter { $0 }.count
        return DepthMaskResult(
            valid: valid,
            debug: String(
                format: "wallRef=%.3f beyondCut=%.3f beyond=%d window=%d mirror≈%dpx valid=%d/%d focal=%.1f c=(%.1f,%.1f)",
                wallRef,
                beyondCut,
                beyond,
                window,
                mirror,
                validCount,
                pixelCount,
                focalPx,
                cx,
                cy
            )
        )
    }

    static func unprojectLeveled(
        depth: [Float],
        width: Int,
        height: Int,
        valid: [Bool],
        focalPx: Float,
        cx: Float,
        cy: Float,
        gravity rotation: simd_float3x3,
        stride step: Int = 2
    ) -> [Point3] {
        guard width > 1, height > 1, depth.count == width * height, valid.count == depth.count else {
            return []
        }
        let sampleStep = max(1, step)
        var points: [Point3] = []
        points.reserveCapacity((width / sampleStep) * (height / sampleStep))

        for y in Swift.stride(from: 0, to: height, by: sampleStep) {
            for x in Swift.stride(from: 0, to: width, by: sampleStep) {
                let i = y * width + x
                let d = depth[i]
                if !valid[i] || !d.isFinite || d <= 0 { continue }
                let cameraX = (Float(x) - cx) * d / focalPx
                let cameraY = (Float(y) - cy) * d / focalPx
                let leveled = rotation * SIMD3<Float>(cameraX, cameraY, d)
                // Existing room-measurement leveling uses positive Y below camera.
                // RoomExtent uses Y-up world coordinates, so flip the leveled Y.
                points.append(Point3(x: leveled.x, y: -leveled.y, z: leveled.z))
            }
        }
        return points
    }

    static func roomExtentFromWalls(
        points: [Point3],
        scale: Float,
        cameraHeight: Float = 1.65
    ) -> RoomExtentResult {
        guard points.count > 500 else {
            return RoomExtentResult(
                width: 0,
                depth: 0,
                height: 0,
                confidence: 0.2,
                approximate: true,
                debug: "too few points \(points.count)"
            )
        }

        let ys = points.map { $0.y * scale }.sorted()
        let floorY = percentile(ys, fraction: 0.05)
        let ceilingY = percentile(ys, fraction: 0.95)
        var ceilingClearance = ceilingY - floorY - cameraHeight
        ceilingClearance = min(1.8, max(0.4, ceilingClearance))
        let height = cameraHeight + ceilingClearance

        let lowBand = floorY + 0.4
        let highBand = ceilingY - 0.4
        var slabX: [Float] = []
        var slabZ: [Float] = []
        slabX.reserveCapacity(points.count / 2)
        slabZ.reserveCapacity(points.count / 2)
        for point in points {
            let metricY = point.y * scale
            if metricY > lowBand, metricY < highBand {
                slabX.append(point.x * scale)
                slabZ.append(point.z * scale)
            }
        }
        let slabCount = slabX.count
        let sourceX: [Float]
        let sourceZ: [Float]
        if slabCount > 300 {
            sourceX = slabX
            sourceZ = slabZ
        } else {
            sourceX = points.map { $0.x * scale }
            sourceZ = points.map { $0.z * scale }
        }

        let xSpan = robustSpan(sourceX)
        let zSpan = robustSpan(sourceZ)
        var width = xSpan.high - xSpan.low
        var depth = zSpan.high - zSpan.low

        var approximate = false
        var confidence: Float = 0.7
        func flag(_ condition: Bool) {
            if condition {
                approximate = true
                confidence = 0.3
            }
        }
        flag(!(2.0...3.6).contains(height))
        flag(!(1.2...12).contains(width))
        flag(!(1.2...15).contains(depth))

        width = min(12, max(1.0, width))
        depth = min(15, max(1.0, depth))

        return RoomExtentResult(
            width: width,
            depth: depth,
            height: height,
            confidence: confidence,
            approximate: approximate,
            debug: String(
                format: "floorY=%.3f ceilY=%.3f clr=%.3f slab=%d pts=%d W=%.3f D=%.3f H=%.3f",
                floorY,
                ceilingY,
                ceilingClearance,
                slabCount,
                points.count,
                width,
                depth,
                height
            )
        )
    }

    private static func robustSpan(_ values: [Float]) -> (low: Float, high: Float) {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return (0, 0) }
        return (percentile(sorted, fraction: 0.025), percentile(sorted, fraction: 0.975))
    }

    private static func percentile(_ sorted: [Float], fraction: Double) -> Float {
        guard !sorted.isEmpty else { return 0 }
        let index = min(sorted.count - 1, max(0, Int((Double(sorted.count - 1) * fraction).rounded())))
        return sorted[index]
    }
}
