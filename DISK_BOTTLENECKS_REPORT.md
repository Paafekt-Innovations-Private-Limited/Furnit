# Laptop disk bottlenecks report

**Current status:** ~18–22 GB free on Data volume. **Disk:** 460 GB total (APFS), **~421 GB used**, 96% full.  
**Recommendation:** Aim to keep 40–60 GB free. On a 460 GB disk, iOS + Android dev tools alone can use **120+ GB** — that’s why it fills up.

---

## 0. THE BIG PICTURE — why 460 GB isn’t enough

The **single largest consumer** is **iOS Simulator**. Your Mac has **four iOS Simulator runtimes** mounted as separate APFS volumes on the same disk. Each one is **16–19 GB**:

| What | Size | Where |
|------|------|--------|
| **iOS Simulator runtime** (iOS_22D8075) | **18 GB** | `/Library/Developer/CoreSimulator/Volumes/` |
| **iOS Simulator runtime** (iOS_22E238) | **19 GB** | same |
| **iOS Simulator runtime** (iOS_23C54) | **16 GB** | same |
| **iOS Simulator runtime** (iOS_23A343) | **16 GB** | same |
| **SimRuntime Cryptex bundles** (×2) | **~17 GB** | `CoreSimulator/Cryptex/Images/bundle/` |
| **Total iOS Simulator** | **~86 GB** | |

Then add:

- **Android SDK:** ~20 GB  
- **Xcode** (app, Archives, DeviceSupport, DerivedData, DVTDownloads): **~30+ GB**  
- **Your user data** (Cursor, Chrome, Documents, Downloads, Library): **~50+ GB**  
- **System, /Applications, /Library:** **~50+ GB**  
- **Rest:** caches, logs, Containers, etc.

So **iOS + Android dev alone is ~120+ GB**. That’s why a 512 GB (or 460 GB usable) disk hits “no space” — not because of one 8 GB app, but because **simulator runtimes + SDK + Xcode + Android** add up to a very large chunk.

**Main levers:**

1. **Delete old iOS Simulator runtimes** (Xcode → Settings → Platforms). Keep only the one or two iOS versions you actually use. **Saves ~16–50+ GB.**
2. **Delete Xcode Archives** if you don’t need old builds. **Saves ~7 GB.**
3. **Trim Android SDK** (unused API levels / system images). **Saves ~5–15 GB.**
4. **Optionally** prune Cursor backup, Chrome ML, Gradle, Downloads. **Saves ~15–20 GB more.**

---

## 1. Largest space users (by area)

| Area | Size | Notes |
|------|------|--------|
| **Cursor** | **8.8 GB** | See §2 below |
| **Xcode iOS DeviceSupport** | **16 GB** | 3 device/iOS versions (~5.3–5.5 GB each) |
| **Xcode Archives** | **7 GB** | Single date folder (2026-03-09) |
| **Chrome** | **6.2 GB** | ~4 GB is on-device ML models |
| **Android SDK** | **20 GB** | Trim unused SDKs/images in Android Studio |
| **Gradle** | **2.5 GB** | Can clear caches again if needed |
| **Downloads** | **10 GB** | Manual cleanup |
| **Xcode DVTDownloads** | **2.1 GB** | Xcode downloaded components |
| **Library/Containers** | **3.3 GB** | App sandbox data |
| **VS Code (Code)** | **1.3 GB** | |
| **Xcode UserData** | **1 GB** | Snippets, breakpoints, etc. |
| **Logs** | **205 MB** | |

---

## 2. Cursor (~8.8 GB) — big win

Almost all of it is in **globalStorage**:

- `state.vscdb` — **4.3 GB** (main state DB: chat history, indexing, etc.)
- `state.vscdb.backup` — **4.0 GB** (backup of same)

**Options:**

1. **Remove backup only (safe, ~4 GB):**  
   Quit Cursor, then in Terminal:
   ```bash
   rm "/Users/al/Library/Application Support/Cursor/User/globalStorage/state.vscdb.backup"
   ```
   Cursor will keep working; only the backup is gone.

2. **Reset Cursor state (reclaim most of 8 GB, loses in-app history):**  
   Quit Cursor, then:
   ```bash
   rm "/Users/al/Library/Application Support/Cursor/User/globalStorage/state.vscdb"
   rm "/Users/al/Library/Application Support/Cursor/User/globalStorage/state.vscdb.backup"
   ```
   Cursor will create a new DB; old chat/conversation history in Cursor is lost. Your project files are untouched.

---

## 3. iOS Simulator runtimes (~86 GB) — biggest win

You have **four** iOS simulator runtimes (22.x and 23.x). Each is **16–19 GB**. You only need **one** (or two if you test on two iOS versions).

**How to remove (safe):**

1. Open **Xcode**.
2. **Xcode → Settings** (or Preferences) → **Platforms** (or **Components** in older Xcode).
3. You’ll see **iOS 22.x**, **iOS 23.x**, etc. Each has a version and a size (~16–18 GB).
4. Select an iOS version you **don’t** need (e.g. older 22.x if you only use 23.x).
5. Click the **minus (−)** or **Delete** / **Remove** button.
6. Repeat for other runtimes you don’t need.

**Keep:** The one (or two) iOS version(s) you actually use for development.  
**Remove:** The rest. **Each removed runtime frees ~16–19 GB.**

Example: keep only **iOS 23.x** (one runtime), remove both 22.x runtimes → **frees ~35–37 GB.**

---

## 4. Xcode Archives (7 GB)

One folder: `~/Library/Developer/Xcode/Archives/2026-03-09` (~7 GB).

- **If you don’t need to resubmit or debug those builds:** delete the whole Archives folder:
  ```bash
  rm -rf ~/Library/Developer/Xcode/Archives/*
  ```
- **If you might need some:** open Archives in Finder, delete only the date folders you don’t need.

**Frees:** up to ~7 GB.

---

## 5. Xcode iOS DeviceSupport (16 GB)

Three versions (each ~5.3–5.5 GB):

- iPhone13,2 26.2.1 (23C71)
- iPhone18,1 26.3 (23D127)
- iPhone18,1 26.3.1 (23D8133)

**Option A – In Xcode (safest):**  
Xcode → Settings → Platforms → select an old iOS version → minus (−) to delete.  
Keep the version that matches the device you actually debug on.

**Option B – Manually:**  
If you only use one device (e.g. iPhone 18 with 26.3.1), you can delete the other two folders, e.g.:
```bash
# Example: keep only 26.3.1, remove the other two
rm -rf ~/Library/Developer/Xcode/iOS\ DeviceSupport/iPhone13,2\ 26.2.1\ \(23C71\)
rm -rf ~/Library/Developer/Xcode/iOS\ DeviceSupport/iPhone18,1\ 26.3\ \(23D127\)
```
**Frees:** ~5–11 GB depending on how many you remove.

---

## 6. Chrome (6.2 GB, ~4 GB ML models)

- **OptGuideOnDeviceModel** — ~4 GB (Chrome’s on-device optimization/ML models).

**Options:**

1. In Chrome: Settings → Privacy and security → Clear browsing data → Advanced → pick a time range; or use “Clear data” for site data/cache.  
2. Delete the ML model folder (Chrome will re-download if it needs it):
   ```bash
   rm -rf "/Users/al/Library/Application Support/Google/Chrome/OptGuideOnDeviceModel"
   ```
   **Frees:** ~4 GB.

---

## 7. Android SDK (20 GB)

- Open **Android Studio** → **Settings/Preferences** → **Languages & Frameworks** → **Android SDK**.
- **SDK Platforms:** Uncheck old Android versions you don’t build for.
- **SDK Tools** tab: Uncheck unused build-tools, old NDK versions, or system images you don’t use.
- Apply → let it remove the selected packages.

**Frees:** typically 2–8 GB depending on what you remove.

---

## 8. Gradle (2.5 GB)

Safe to clear caches (next Android build will re-download):

```bash
rm -rf ~/.gradle/caches
```

**Frees:** ~2.4 GB.

---

## 9. Downloads (10 GB)

- Open **Finder → Downloads**.
- Sort by size; move or delete large files you don’t need.
- Empty **Trash** after deleting.

---

## 10. Quick summary: order of impact

| Action | Approx. space freed |
|--------|----------------------|
| **Remove 1–2 iOS Simulator runtimes** (Xcode → Settings → Platforms) | **~16–50 GB** |
| Delete Xcode Archives | ~7 GB |
| Remove 1–2 old iOS DeviceSupport versions | ~5–11 GB |
| Delete Cursor both state DBs (reset state) | ~8 GB |
| Trim Android SDK (unused API levels/images) | ~5–15 GB |
| Remove Cursor `state.vscdb.backup` | ~4 GB |
| Delete Chrome OptGuideOnDeviceModel | ~4 GB |
| Clear Gradle caches | ~2.4 GB |
| Clean Downloads + Trash | variable |

**Why 460 GB fills up:** iOS Simulator (~86 GB) + Android SDK (~20 GB) + Xcode (~30+ GB) + your data (~50+ GB) + system (~50+ GB) = **240+ GB** before you add much else. The **first** thing to do is remove extra **iOS Simulator runtimes** (keep only the one iOS version you use) — that alone can free **35–50+ GB**.

---

## 11. One-shot “safe” cleanup (no Cursor reset, no DeviceSupport)

Run with Cursor and Chrome closed:

```bash
# Cursor backup only
rm -f "/Users/al/Library/Application Support/Cursor/User/globalStorage/state.vscdb.backup"

# Xcode Archives
rm -rf ~/Library/Developer/Xcode/Archives/*

# Chrome on-device ML
rm -rf "/Users/al/Library/Application Support/Google/Chrome/OptGuideOnDeviceModel"

# Gradle
rm -rf ~/.gradle/caches

echo "Done. Check: df -h /System/Volumes/Data"
```

Then run `df -h /System/Volumes/Data` to confirm free space.
