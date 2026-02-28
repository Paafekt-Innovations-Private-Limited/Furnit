# Paafekt standards

Standards used across Furnit/Paafekt (iOS and Android) for consistent UX and behavior.

---

## Landscape room opening position (3D viewer)

**Standard:** When a room created from a **landscape** photo is opened, the 3D viewer must open with the camera in the **same “good” position** so the room is framed correctly and the user sees the expected view without adjusting.

### Definition (Android / WebGL SHARP viewer)

- **Box3 axis mapping (landscape):** `roomWidth = size.x`, `roomHeight = size.y`, `roomDepth = size.z`. Depth axis: Z (back wall = min Z, front = max Z in local space; camera looks along +Z toward target).
- **Camera position** (relative to room box center):
  - `L_CAM_X = 0` (centered in X)
  - `L_CAM_Y = 0.00207` (slightly above floor, in room height units)
  - `L_CAM_Z = -0.130` (camera 13% of room depth in from the back wall)
  - Formula: `camera.position.set(center.x + L_CAM_X * W, center.y + L_CAM_Y * H, center.z + L_CAM_Z * D)`
- **Look-at target:**
  - `L_TGT_Z = -0.444` (target 44.4% of depth from back; looking toward front of room)
  - Formula: `controls.target.set(center.x + L_CAM_X * W, center.y, center.z + L_TGT_Z * D)`

**Constants (Paafekt standard):**

```text
L_CAM_X = 0
L_CAM_Y = 0.00207
L_CAM_Z = -0.130
L_TGT_Z = -0.444
```

**Where implemented:**

- **Android:** `SharpRoomActivity` WebView JS (autoFrameRoom): when `isPortrait === false`, use the L_* constants above for camera and target. Same values used for both Box3-derived bounds and fallback dimensions.
- **iOS:** Match the same viewing angle and framing when opening a landscape PLY/RealityKit room (back-center camera, look at front); exact constants may map to RealityKit’s coordinate system but the user-visible “good position” should match.

### Why this is the standard

This position was validated as the preferred landscape opening view; adopting it everywhere keeps behavior consistent across Android, iOS, and any future clients.
