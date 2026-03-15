# Checking Memory of App Variations (iOS)

Ways to measure and compare memory usage across different flows, builds, and devices.

---

## 1. In-app logs (resident size)

The app already logs **resident memory (RSS)** in MB when **Debug mode** is on.

**Enable:** Settings → turn on **Debug mode** (or whatever toggle exposes debug logs).

**Where it’s logged:**
- **FurnitureFitView**: `logMemory(_ tag:)` prints `🧠 [tag] Memory: X.X MB` at:
  - FRAME START, AFTER INFERENCE, AFTER STAGE 5b/5c, AFTER BUILD MASK, FRAME END
- **SHARPService**: `SHARP: Device RAM: X MB` when loading the model (total device RAM, not process).

**How to compare variations:**
1. Reproduce the same flow (e.g. open room → tap FurnitureFit → wait one frame).
2. In Xcode console (or device console), note the **last** `🧠 [FRAME END] Memory: X.X MB` for that flow.
3. Repeat for another variation (e.g. different screen, before/after loading SHARP) and compare numbers.

**Limitation:** Only runs in code paths that call `logMemory` (mainly FurnitureFit pipeline). For other screens you’d need to add similar logging or use Instruments.

---

## 2. Xcode Debug Navigator (Memory gauge)

**Use:** While running from Xcode, open **Debug Navigator** (⌘7) → **Memory**.

- Shows **live process memory** (footprint) in MB.
- Good for: “home” vs “FurnitureFit open” vs “SHARP room loaded” on the same run.

**How to compare variations:**
1. Run app, wait for idle → note Memory value (e.g. “baseline”).
2. Navigate to a heavy flow (e.g. open a room, enable FurnitureFit, or run SHARP) → note Memory again.
3. Go back / dismiss → see if memory drops (no leak) or stays high.
4. Repeat with a different build (e.g. Release) or different device to compare variations.

---

## 3. Instruments (detailed breakdown)

**Use:** Xcode → **Product → Profile** (⌘I) or run from Instruments.

**Useful templates:**
- **Allocations**: Live allocations; see what’s on the heap and compare before/after a flow.
- **Leaks**: Find leaks; run the same flow several times and check for growth.
- **VM Tracker** (in Allocations): Resident / dirty size over time; good for “app variation” comparisons.

**How to compare variations:**
1. Start recording.
2. Do **variation A** (e.g. open home only) → mark a flag / note time.
3. Do **variation B** (e.g. open room + FurnitureFit) → mark again.
4. Stop; in VM Tracker or Allocations summary, compare memory at those points.
5. Repeat with another build (Debug vs Release) or device to compare “app variations”.

---

## 4. Device Console (TestFlight / release builds)

If you need memory insight on **TestFlight** (or release) builds:

1. Connect the device to the Mac.
2. **Window → Devices and Simulators** → select device → **Open Console**.
3. Select your app process; filter by your app name or “Memory” / “🧠”.
4. Reproduce the flow; you’ll see any `logMemory` output that’s printed in that build (if debug logging is enabled in release).
5. For full footprint without code changes, use **Instruments** with a **Release** scheme and “Profile” on the device.

---

## 5. Quick comparison table

| Goal | Method | Notes |
|------|--------|--------|
| Compare “home” vs “FurnitureFit” vs “SHARP” | Xcode Memory gauge | Easiest; same run, different screens. |
| Compare Debug vs Release | Xcode Memory gauge or Instruments | Run each build, same flow, note memory. |
| Compare devices / OS versions | Same flow on each; note gauge or logs | Baseline and after heavy flow. |
| See RSS in code paths (FurnitureFit) | Debug mode + console `🧠 [tag]` | Use existing `logMemory` points. |
| Deep dive (leaks, what’s allocated) | Instruments → Allocations / Leaks | For “why is this variation higher?”. |

---

## 6. Adding memory logs to other flows

To log resident memory in another view (e.g. SharpRoomView, ModelViewerView):

```swift
private func logMemory(_ tag: String) {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    if result == KERN_SUCCESS {
        let usedMB = Double(info.resident_size) / 1024.0 / 1024.0
        print("🧠 [\(tag)] Memory: \(String(format: "%.1f", usedMB)) MB")
    }
}
```

Call it on appear and after heavy work (e.g. after SHARP load, after first FurnitureFit frame) so you can compare “app variations” from logs.
