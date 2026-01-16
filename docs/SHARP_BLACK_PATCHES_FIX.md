# SHARP 3D Room Black Patches Fix

## Problem
When generating 3D rooms from photos using the SHARP (Single-image House-scale Accurate Reconstruction with Priors) model, black patches appeared on walls, ceilings, and corners of the rendered room.

![Black patches example](../assets/sharp_black_patches_example.png)

## Root Cause
The `SHARPService.swift` had aggressive gaussian filtering that was designed to remove "fog" and "weak" pixels from the 3D splat output. However, the filters were too strict and were removing **valid wall/ceiling/corner gaussians** that had lower opacity values.

### Original Filter Thresholds (Too Aggressive)
```swift
// These were removing valid scene content:
private static let minAlpha: Float = 0.30           // Removed splats with < 30% opacity
private static let fogAlphaThreshold: Float = 0.50  // Fog filter: opacity < 50%
private static let fogScaleThreshold: Float = 0.020 // Combined with large scale
private static let edgeMarginX/Y/Z: Float = 0.12    // 12% edge trimming
```

### What Was Being Filtered Out
- Semi-transparent wall surfaces (20-30% opacity)
- Ceiling areas with softer rendering
- Corner regions where multiple surfaces meet
- Edge areas of the room

## Solution
**Disabled all gaussian filtering entirely.** The SHARP model output is now used directly without any post-processing filters.

### Code Change (SHARPService.swift)
```swift
private func filterGaussians(_ params: [Float]) -> [Float] {
    let inputCount = params.count / Self.paramsPerGaussian
    logDebug("SHARP: Processing \(inputCount) Gaussians (no filtering)")

    // Return all gaussians without filtering - filtering was causing black patches
    // by removing valid wall/ceiling splats that had lower opacity
    return params

    // ... rest of filtering code is now unreachable (kept for reference)
}
```

## Additional Improvements Made

### 1. Background Color (SharpRoomView.swift)
Changed from harsh black to neutral gray so any remaining sparse areas blend better:
```javascript
scene.background = new THREE.Color(0x808080);  // Was 0x1a1a1a (near-black)
```

### 2. SparkJS Renderer Settings
Adjusted to spread gaussians more for better coverage:
```javascript
const spark = new SparkRenderer({
    maxStdDev: 3.0,        // Larger gaussians to fill gaps
    preBlurAmount: 0.5,    // Soft blur to blend edges
    blurAmount: 0.3,       // Post-blur
    falloff: 0.8,          // Gentler opacity falloff
    focalAdjustment: 1.5   // Less aggressive focal adjustment
});
```

### 3. Battery Optimization
Added render-on-demand to prevent continuous 60fps rendering:
- Auto-orbit now runs at 30fps (not 60fps)
- Completely stops rendering when user isn't interacting and auto-orbit is disabled

## Result
After disabling filtering:
- ✅ No more black patches on walls/ceilings
- ✅ Complete room coverage from the SHARP model
- ✅ Gray background for any truly missing areas (outside camera view)

## Files Modified
- `Furnit/Services/OnDevice/SHARPService.swift` - Disabled filtering
- `Furnit/Views/SharpRoomView.swift` - Gray background, SparkJS settings, battery optimization

## Date
January 2026
