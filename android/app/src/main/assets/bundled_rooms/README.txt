Bundled demo GLB rooms

Place asset paths used by ModelManager here:
  scandinavian_minimal.glb
  industrial_loft.glb

These *.glb files are gitignored (copy locally or use scripts/bundled_room_assets_v2/).

Source of truth on iOS:
  scandinavian_minimal.usdz
  industrial_loft.usdz

Handoff notes, USDA sources, and texture PNGs live in scripts/bundled_room_assets_v2/.

Camera/axis convention (both rooms):
  Front/window wall: -Z
  Back/solid wall: +Z
  Camera outside +Z looking toward -Z
  X = width, Y = up, Z = depth
  Floor at Y = 0

Do NOT use KTX2/BasisU or WebP in Android GLBs — SceneView/Filament crashes on KHR_texture_basisu.
Embedded PNG textures only (see bundled_room_assets_v2 VALIDATION.txt).

Expected bounds:
  scandinavian_minimal: 5.0 x 2.55 x 4.0 m
  industrial_loft:       8.0 x 3.05 x 6.0 m

Android camera uses bbox after centering at origin; Z is the default depth axis.
