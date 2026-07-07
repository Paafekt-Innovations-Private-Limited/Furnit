import UIKit
import simd

enum VanishingPointGravity {
    struct Result: Sendable {
        let levelingRotation: simd_float3x3
        let confidence: Double
        let debug: String
    }

    static func refine(
        levelingRotation: simd_float3x3,
        image: UIImage,
        focalPx: Double
    ) -> Result {
        // Scaffold only. A production version should run line-segment detection,
        // cluster orthogonal vanishing points, and replace GeoCalib gravity only
        // when VP inliers are strong and disagreement is large.
        logDebug(String(
            format: "[VP] vps=0 pitch=nan roll=nan focal=%.1f inliers=0 source=geocalib_fused reason=unimplemented",
            focalPx
        ))
        return Result(
            levelingRotation: levelingRotation,
            confidence: 0,
            debug: "vp_refiner_unimplemented_using_input_gravity"
        )
    }
}
