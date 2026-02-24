# Install on Phone and Test ExecuTorch (SHARP)

Steps to install the Furnit Android app on a **USB-linked phone** and test **ExecuTorch** (Vulkan) on the SHARP model.

## 1. Prerequisites

- **Phone:** USB debugging enabled (Settings → Developer options → USB debugging).
- **PC:** ADB in PATH. Check: `adb devices` (device should show as "device", not "unauthorized").
- **SHARP weights:** A `.pt` checkpoint (e.g. from team; often `sharp_2572gikvuh.pt` or similar).
- **Python env:** For exporting `.pte` models: `torch`, ExecuTorch, and SHARP source (see `android/third_party/ml-sharp` or `export_sharp_executorch_split4.py`).

## 2. Install the app on the phone

From the project root:

```bash
cd android
./gradlew installDebug
```

Or build and install in one step:

```bash
cd android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

If Gradle fails with "Could not determine usable wildcard IP", run with full permissions or from Android Studio: **Run → Run 'app'** with the device selected.

## 3. Get ExecuTorch SHARP models (.pte)

You can use either:

- **Single full Vulkan** — one `sharp_full_vulkan.pte` (~1.2GB). Select **"ExecuTorch Vulkan (single model)"** in Settings (last option). Needs 6GB+ free RAM.
- **Split** — 4-part `.pte` files. Select **"ExecuTorch"** or **"New classes"** in Settings.

### Option A: You have SHARP weights and export env

**Single full Vulkan (one .pte, one forward):**

```bash
cd android
python export_sharp_executorch_full_vulkan.py --weights /path/to/sharp.pt --output-dir executorch_models
./push_sharp_executorch_models.sh executorch_models
```

Then in the app: Settings → Developer → **ExecuTorch Vulkan (single model)**. Needs 6GB+ free RAM; otherwise use split below.

**Split Vulkan (GPU, ~30s room with enough RAM):**

```bash
cd android
./export_sharp_executorch_vulkan_full.sh
```

This exports to `android/executorch_models/` with:
- Vulkan backend
- Part1+Part2 combined (one .pte)
- Chunked Part 4 (part4a_chunk_512, part4a_chunk_65, part4b)
- Part 3

If the script fails (missing `WEIGHTS` or `SHARP_SRC`), set them and run the Python export manually:

```bash
cd android
# Set paths to your SHARP checkpoint and ml-sharp source
export WEIGHTS=/path/to/sharp.pt
export SHARP_SRC=/path/to/ml-sharp/src   # or android/third_party/ml-sharp/src

python3 export_sharp_executorch_split4.py \
  --backend vulkan \
  --combined-part1-part2 \
  --chunked-part4 \
  --sharp-src "$SHARP_SRC" \
  --weights "$WEIGHTS" \
  --output-dir executorch_models
```

**Minimal split (no chunked Part 4):**

```bash
python3 export_sharp_executorch_split4.py \
  --backend vulkan \
  --weights /path/to/sharp.pt \
  --output-dir executorch_models
```

This produces: `sharp_split_part1.pte`, `sharp_split_part2.pte`, `sharp_split_part3.pte`, `sharp_split_part4.pte` (~2.5GB total).

### Option B: You already have .pte files

If a colleague or Drive has an `executorch_models/` (or similar) folder with the split `.pte` files, use that folder in step 4.

## 4. Push models to the phone

**One-time:** Run the app once on the phone so the app data directory exists, then push:

```bash
cd android
./push_sharp_executorch_models.sh executorch_models
```

`executorch_models` is the folder containing the `.pte` files (from export or from team). The script pushes to:

- `/sdcard/Android/data/com.furnit.android/files/models/`
- `/data/local/tmp/furnit/` (fallback)

To push a different folder:

```bash
./push_sharp_executorch_models.sh /path/to/folder/with/pte/files
```

**Optional:** Add a `backend.txt` in that folder containing the line `vulkan` so the app logs which backend was intended.

## 5. Select ExecuTorch in the app

1. Open **Furnit** on the phone.
2. Go to **Settings** (gear) → **Developer**.
3. Under **Inference Backend**, choose one:
   - **ExecuTorch Vulkan (single model)** — if you pushed `sharp_full_vulkan.pte`; one forward, needs 6GB+ free RAM.
   - **New classes** or **ExecuTorch** — for 4-part split models (Part1–Part4 .pte).
4. Go back to the home screen.

## 6. Test SHARP (ExecuTorch)

1. Tap **Create Room from Photo** (or the photo/camera entry point).
2. Choose a photo (gallery or take one).
3. Select **AI Room** (AI-powered 3D construction).
4. Wait for generation:
   - **Vulkan + enough free RAM (~5GB+):** often ~30s–1 min.
   - **Low RAM or CPU fallback:** 10–20+ min or LMK kill.

Watch **logcat** to confirm backend and progress:

```bash
adb logcat -s ExecutorchSharp:* Progress0:*
```

You should see lines like:
- `ExecuTorch models backend: vulkan` (or xnnpack)
- `Part 1: Patch 0/35...`, then Part 2, Part 3, Part 4
- `Backend proof: check logcat for ExecuTorch/Vulkan init messages...`

## 7. Verify models on device

```bash
adb shell ls -la /sdcard/Android/data/com.furnit.android/files/models/
```

You should see the `.pte` files (and optionally `backend.txt`). The app copies them to internal storage on first use for faster mmap.

## Troubleshooting

| Issue | What to do |
|-------|------------|
| **"No device connected"** | Enable USB debugging, accept the RSA prompt on phone, run `adb devices`. |
| **"Model not available" / ExecuTorch not used** | Push models (step 4), select "New classes" (step 5), restart app. |
| **Very slow or LMK kill** | Close other apps; use chunked Part 4 + Part1+2 combined; device needs Vulkan 1.1+ and enough RAM. |
| **Export fails (missing sharp.pt / SHARP_SRC)** | Set `WEIGHTS` and `SHARP_SRC`; ensure Python env has `torch` and ExecuTorch. |

## Summary

1. **Install:** `cd android && ./gradlew installDebug` (phone connected).
2. **Export (if you have weights):**
   - Single full: `python export_sharp_executorch_full_vulkan.py --weights /path/to/sharp.pt --output-dir executorch_models`
   - Split: `./export_sharp_executorch_vulkan_full.sh` or `export_sharp_executorch_split4.py` (see above).
3. **Push:** `./push_sharp_executorch_models.sh executorch_models`.
4. **App:** Settings → Developer → Inference Backend = **ExecuTorch Vulkan (single model)** (full) or **New classes** / **ExecuTorch** (split).
5. **Test:** Create Room from Photo → pick image → **AI Room**.
