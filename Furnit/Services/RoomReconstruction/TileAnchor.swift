import Foundation

struct TileAnchor: ScaleAnchor {
    func candidates(ctx: SceneContext) -> [ScaleCandidate] {
        // Scaffold only. Real implementation should:
        // 1. select uncluttered floor/wall patches,
        // 2. rectify with gravity + focal,
        // 3. find dominant tile periods via FFT/autocorrelation,
        // 4. snap to standard tile sizes and emit real/raw scale candidates.
        []
    }
}
