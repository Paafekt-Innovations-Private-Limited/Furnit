# Furnit tester package (no code checkout)

You only need **adb** (Android Platform Tools), a USB cable, and this folder. **macOS:** open **Terminal** (or iTerm), `cd` into the unzipped folder — the `.sh` scripts run there like any shell script.

## 1. Install Platform Tools

- **macOS / Linux / Windows:** [Android Platform Tools](https://developer.android.com/tools/releases/platform-tools) — unzip and add the folder to your `PATH`, or run `adb` with a full path.

## 2. Phone

- Settings → Developer options → **USB debugging** ON  
- Connect the phone; when prompted, **allow USB debugging** for this computer.

**Android emulator open?** These scripts **use the physical phone**, not the emulator. Close the emulator or run `adb emu kill` so only the real device appears — or if you must keep both attached, pick the phone explicitly:

```bash
adb devices
export ANDROID_SERIAL=PASTE_YOUR_PHONE_SERIAL_HERE
./install_furnit_apk.sh
./push_furnit_sharp_models.sh
```

Check:

```bash
adb devices
```

You should see a line ending in `device` (not `unauthorized`).

## 3. Install the app

Put the `.apk` you received **in this same folder** (next to these scripts), then:

```bash
chmod +x install_furnit_apk.sh push_furnit_sharp_models.sh
./install_furnit_apk.sh
```

The APK can be **any name** (e.g. `furnit.apk` or the long `friend-nomodels-…-debug.apk`).

Or install manually (still use one device: `adb -s SERIAL install …` if needed):

```bash
adb install -r /path/to/furnit.apk
```

## 4. Push SHARP models (if the APK was the “lightweight” build)

Unzip the **`models_cpuvulkan_hybrid`** folder **into this folder** so you have:

```
this_folder/
  adb_common.sh
  install_furnit_apk.sh
  push_furnit_sharp_models.sh
  models_cpuvulkan_hybrid/
    sharp_split_part*.pte
    ...
```

Then:

```bash
./push_furnit_sharp_models.sh
```

If the models live somewhere else:

```bash
./push_furnit_sharp_models.sh /path/to/models_cpuvulkan_hybrid
```

**If `permission denied` when running `./…sh`:** run `chmod +x *.sh` or use `bash push_furnit_sharp_models.sh`.

## 5. Open the app

Launch **Furnit** on the phone. AI room / SHARP features need the pushed models; other parts work with the APK alone.

---

**What you were given:** small APK + `models_cpuvulkan_hybrid` from Google Drive + these scripts (`adb_common.sh` is required — do not delete). **You do not need** Android Studio or the Furnit source tree.
