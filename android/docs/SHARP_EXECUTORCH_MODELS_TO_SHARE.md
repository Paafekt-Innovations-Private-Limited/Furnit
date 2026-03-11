# SHARP ExecuTorch models – list for sharing (Android)

Use this list to upload from your **external drive backup** to Google Drive and share with your friend. They pull latest from git and push these models to the device.

---

## Where the models are on your external drive

After running `./backup_models_v3_mar10th2026.sh`, the backup is at:

**`/Volumes/LaCie/mar10th2026/v3/`**

It contains two folders. Upload both folders to Google Drive (keep the same folder names):

- `executorch_int8_models/`
- `executorch_models/`

---

## Files in each folder (checklist for Google Drive)

### Folder: `executorch_int8_models/`

| File | Required? | Notes |
|------|-----------|--------|
| `sharp_split_part1_int8.pte` | **Yes** | Encoder part 1 |
| `sharp_split_part2_int8.pte` | **Yes** | Encoder part 2 |
| `sharp_split_part3_int8.pte` | **Yes** | Encoder part 3 |
| `sharp_split_part1_b4_int8.pte` | Optional | Batch-4 Part1 (fewer launches) |
| `sharp_split_part2_b4_int8.pte` | Optional | Batch-4 Part2 |
| `sharp_split_part4b_int8.pte` | Optional | INT8 Part4b; C++ pipeline prefers this when present |

Tiles (optional; for Part4b tiled path in Settings):

| File | Required? |
|------|-----------|
| `sharp_split_part4b_tile_full.pte` | Optional |
| `sharp_split_part4b_tile_00.pte` … `sharp_split_part4b_tile_15.pte` | Optional (16 files) |

---

### Folder: `executorch_models/`

| File | Required? | Notes |
|------|-----------|--------|
| `sharp_split_part4a_chunk_512.pte` | **Yes** | Decoder Part4a (first 512 tokens) |
| `sharp_split_part4a_chunk_65.pte` | **Yes** | Decoder Part4a (remaining 65 tokens) |
| `sharp_split_part4b.pte` | **Yes** | Decoder Part4b (FP32 fallback; always needed) |
| `sharp_split_part4b_int8.pte` | Optional | Same as in int8 folder; can live in either folder |
| `sharp_split_part4b_tile_full.pte` | Optional | Same as in int8 folder |
| `sharp_split_part4b_tile_00.pte` … `sharp_split_part4b_tile_15.pte` | Optional | Same as in int8 folder |

---

## Minimum set (Android works with only these 6 files)

- **executorch_int8_models:**  
  `sharp_split_part1_int8.pte`, `sharp_split_part2_int8.pte`, `sharp_split_part3_int8.pte`
- **executorch_models:**  
  `sharp_split_part4a_chunk_512.pte`, `sharp_split_part4a_chunk_65.pte`, `sharp_split_part4b.pte`

Everything else is optional (faster or INT8 Part4b / tiled path).

---

## Instructions for your friend

1. **Pull latest from git** (e.g. `git pull origin segmentios` or `main`).
2. **Download** the two folders from your Google Drive:  
   `executorch_int8_models` and `executorch_models`.
3. **Place them** in the repo under `android/`:
   - `android/executorch_int8_models/`
   - `android/executorch_models/`
4. **Connect device**, install and run the app once (so the models path exists).
5. **Push models to device:**
   ```bash
   cd android
   ./push_sharp_executorch_int8_models.sh
   ```
6. **Build/run** the app and test.

If they build with models in assets (APK-packaged), the 6 required .pte files must be present in those two folders before building; see `android/app/build.gradle` (task that copies into `assets/models/`).

---

## ADB commands to push models (e.g. Blackview Shark)

**Prerequisites:** USB debugging enabled, device connected, Furnit app installed and opened at least once.

Set the destination and your local folders (adjust paths if needed):

```bash
DEST="/sdcard/Android/data/com.furnit.android/files/models"
INT8_DIR="$HOME/Downloads/executorch_int8_models"   # or path where you extracted the zip
CHUNKED_DIR="$HOME/Downloads/executorch_models"
```

If multiple devices are connected, use the Blackview serial (get it with `adb devices`):

```bash
ADB="adb -s YOUR_DEVICE_SERIAL"
# e.g. ADB="adb -s 1234567890ABCD"
```

Create the models directory on the device:

```bash
$ADB shell "mkdir -p $DEST"
```

Push the 6 required files:

```bash
$ADB push "$INT8_DIR/sharp_split_part1_int8.pte"   "$DEST/"
$ADB push "$INT8_DIR/sharp_split_part2_int8.pte"   "$DEST/"
$ADB push "$INT8_DIR/sharp_split_part3_int8.pte"   "$DEST/"
$ADB push "$CHUNKED_DIR/sharp_split_part4a_chunk_512.pte" "$DEST/"
$ADB push "$CHUNKED_DIR/sharp_split_part4a_chunk_65.pte"  "$DEST/"
$ADB push "$CHUNKED_DIR/sharp_split_part4b.pte"    "$DEST/"
```

Optional (INT8 Part4b, batch-4, tiles — only if you have the files):

```bash
# INT8 Part4b (recommended when available)
[ -f "$CHUNKED_DIR/sharp_split_part4b_int8.pte" ] && $ADB push "$CHUNKED_DIR/sharp_split_part4b_int8.pte" "$DEST/"
# or if it's in int8 folder:
[ -f "$INT8_DIR/sharp_split_part4b_int8.pte" ]    && $ADB push "$INT8_DIR/sharp_split_part4b_int8.pte" "$DEST/"

# Batch-4 Part1/Part2 (optional)
[ -f "$INT8_DIR/sharp_split_part1_b4_int8.pte" ]  && $ADB push "$INT8_DIR/sharp_split_part1_b4_int8.pte" "$DEST/"
[ -f "$INT8_DIR/sharp_split_part2_b4_int8.pte" ]  && $ADB push "$INT8_DIR/sharp_split_part2_b4_int8.pte" "$DEST/"

# Part4b tiles (17 files; optional)
for f in sharp_split_part4b_tile_full.pte sharp_split_part4b_tile_{00..15}.pte; do
  [ -f "$INT8_DIR/$f" ]  && $ADB push "$INT8_DIR/$f"  "$DEST/" && continue
  [ -f "$CHUNKED_DIR/$f" ] && $ADB push "$CHUNKED_DIR/$f" "$DEST/"
done
```

One-liner from repo (if folders are in `android/`):

```bash
cd /path/to/Furnit/android
./push_sharp_executorch_int8_models.sh
```

To target a specific device (e.g. Blackview Shark when several are connected):

```bash
adb devices
# copy the serial for the Shark, then:
INT8_DIR="$(pwd)/executorch_int8_models" CHUNKED_DIR="$(pwd)/executorch_models" adb -s SERIAL shell "mkdir -p /sdcard/Android/data/com.furnit.android/files/models"
adb -s SERIAL push executorch_int8_models/sharp_split_part1_int8.pte /sdcard/Android/data/com.furnit.android/files/models/
adb -s SERIAL push executorch_int8_models/sharp_split_part2_int8.pte /sdcard/Android/data/com.furnit.android/files/models/
adb -s SERIAL push executorch_int8_models/sharp_split_part3_int8.pte /sdcard/Android/data/com.furnit.android/files/models/
adb -s SERIAL push executorch_models/sharp_split_part4a_chunk_512.pte /sdcard/Android/data/com.furnit.android/files/models/
adb -s SERIAL push executorch_models/sharp_split_part4a_chunk_65.pte  /sdcard/Android/data/com.furnit.android/files/models/
adb -s SERIAL push executorch_models/sharp_split_part4b.pte /sdcard/Android/data/com.furnit.android/files/models/
```
Replace `SERIAL` with the device id from `adb devices`.
