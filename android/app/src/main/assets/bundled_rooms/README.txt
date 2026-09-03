Bundled sample GLB rooms

Assets used by ModelManager:
  scandinavian_minimal.glb
  industrial_loft.glb

The GLBs are versioned with Git LFS. After a fresh clone, run `git lfs pull`
before building if either file is represented by an LFS pointer.

Shared geometry/material source and reproducible build:
  scripts/bundled_room_assets_v3/

The generator writes Android GLB files here and stages matching iOS USDZ files in
`/tmp/furnit-room-assets-v3/`. The checked-in iOS counterparts live in:
  Furnit/Assets.xcassets/scandinavian_minimal.dataset/
  Furnit/Assets.xcassets/industrial_loft.dataset/

Camera/axis convention (both rooms):
  Front/window wall: -Z
  Back/solid wall: +Z
  Viewer starts near the +Z back center looking toward -Z
  X = width, Y = up, Z = depth
  Floor at Y = 0

Do not use KTX2/BasisU, WebP, or Draco in these Android GLBs. The v3 build
validates plain glTF 2.0 with embedded PNG/JPEG textures for the bundled Three.js
viewer and the SceneView-based Furniture Fit path.

Authored dimensions and opening orientation:
  scandinavian_minimal: 5.8 x 2.8 x 4.6 m; portrait
  industrial_loft:      7.2 x 3.2 x 5.4 m; landscape

Both designs keep structural details on the perimeter and contain no interior
column, divider, bar, or center ceiling beam. Automated validation includes GLB
structure, Android packaging/build, ARKit USDZ compliance, iPhoneOS build, and
local Blender renders. Final Android/iOS device appearance and orientation remain
unconfirmed as of 2026-09-03.
