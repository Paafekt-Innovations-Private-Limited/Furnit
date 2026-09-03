# Bundled room assets v3

`build_rooms.py` generates the same two lightweight room shells for both apps:

- Android GLB output: `app/src/main/assets/bundled_rooms/`
- iOS USDZ staging output: `/tmp/furnit-room-assets-v3/`

Run from `android/`:

```bash
python3 scripts/bundled_room_assets_v3/build_rooms.py
```

The public identifiers remain `scandinavian_minimal` and `industrial_loft`, so
neither app needs data migration. Both designs keep all structural detailing at the
perimeter; neither includes an interior column, divider, bar, or center ceiling beam.

The app-owned procedural textures in `textures/` are the reproducible source inputs
for both platform outputs. Local preview renders and intermediate generated files stay
under the ignored `generated/` directory.
