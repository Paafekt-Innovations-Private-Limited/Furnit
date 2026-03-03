# Ultralytics code review: Android duplicate SharpRoomActivity / viewer-open fix

**Copy the sections below into Ultralytics (or your reviewer) to get this code reviewed.**

---

## 1. Problem

After SHARP inference completes on Android, we start a single Activity (`SharpRoomActivity`) to show the 3D Gaussian splat in a WebView. Logcat showed the Activity opening **twice** in quick succession: two `onCreate` runs, two copies of the ~293 MB PLY file, two WebView inits. We only call `startActivity(SharpRoomActivity)` once from our completion callback; the second instance was still being created (e.g. system creating a second activity before the first was “on top” with `singleTop`). Goal: ensure only one viewer instance and one PLY load per “open viewer” flow.

---

## 2. Approach

- **singleTask** for `SharpRoomActivity` so at most one instance exists in the task; second intent goes to `onNewIntent`.
- **App-wide debounce** in the caller: static path + timestamp; skip second `startActivity` for same path within 3 s.
- **Post viewer open** from completion callback so any pending touch is processed first, then we open once.
- **In-Activity guard**: companion `currentInstanceRef` (WeakReference) + `currentPlyPath`; if another instance is already showing the same path, `finish()` immediately in `onCreate` and skip PLY copy/WebView (set `isDuplicateInstance` and no heavy cleanup in `onDestroy`).
- **onNewIntent**: same-room check (path and folder canonical comparison); if same room, skip reload; else update state and reload PLY/WebView.

---

## 3. Code to review

**AndroidManifest.xml (SharpRoomActivity):**
```xml
<activity
    android:name=".SharpRoomActivity"
    android:exported="false"
    android:launchMode="singleTask"
    android:theme="@style/Theme.Furnit.FullScreen" />
```

**SinglePhotoRoomActivity.kt – companion (static debounce) and open:**
```kotlin
companion object {
    private const val VIEWER_OPEN_DEBOUNCE_MS = 3000L
    @Volatile
    private var lastOpenedPlyPath: String? = null
    @Volatile
    private var lastOpenedViewerTimeMs: Long = 0L
}

// In onComplete(result):
if (aiRoomOverlayRequested) {
    aiRoomOverlayRequested = false
    Handler(Looper.getMainLooper()).post { openSharpRoomWithResult(result) }
}

private fun openSharpRoomWithResult(result: SharpService.GenerationResult) {
    val plyPath = result.classicPlyFile.absolutePath
    val now = System.currentTimeMillis()
    val lastPath = lastOpenedPlyPath
    val lastTime = lastOpenedViewerTimeMs
    if (plyPath == lastPath && (now - lastTime) < VIEWER_OPEN_DEBOUNCE_MS) {
        // log and return (skip duplicate start)
        return
    }
    lastOpenedPlyPath = plyPath
    lastOpenedViewerTimeMs = now
    val intent = Intent(this, SharpRoomActivity::class.java).apply {
        addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT)
        putExtra(SharpRoomActivity.EXTRA_PLY_PATH, plyPath)
        // ... other extras
    }
    startActivity(intent)
}
```

**SharpRoomActivity.kt – companion, duplicate check in onCreate, onNewIntent, onDestroy:**
```kotlin
companion object {
    private var currentInstanceRef: java.lang.ref.WeakReference<SharpRoomActivity>? = null
    private var currentPlyPath: String? = null
    // ... EXTRA_* constants
}

private var isDuplicateInstance: Boolean = false

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    // ... window/insets setup ...
    plyPath = intent.getStringExtra(EXTRA_PLY_PATH)
    roomFolder = intent.getStringExtra(EXTRA_ROOM_FOLDER)
    if (plyPath == null && roomFolder != null) {
        val roomPly = File(roomFolder, "room.ply")
        if (roomPly.exists()) plyPath = roomPly.absolutePath
    }
    val pathToShow = plyPath
    if (pathToShow != null) {
        val existing = currentInstanceRef?.get()
        val samePath = currentPlyPath == pathToShow
        if (existing != null && existing != this && samePath) {
            isDuplicateInstance = true
            finish()
            return
        }
        currentInstanceRef = java.lang.ref.WeakReference(this)
        currentPlyPath = pathToShow
    }
    // ... rest of onCreate (copy PLY, build WebView, setContentView, loadWebGLViewer())
}

override fun onNewIntent(intent: Intent) {
    super.onNewIntent(intent)
    setIntent(intent)
    val newPath = intent.getStringExtra(EXTRA_PLY_PATH)
        ?: intent.getStringExtra(EXTRA_ROOM_FOLDER)?.let { File(it, "room.ply").takeIf { f -> f.exists() }?.absolutePath }
    val newFolder = intent.getStringExtra(EXTRA_ROOM_FOLDER)
    val currentPath = plyPath
    val currentFolder = roomFolder
    val sameRoom = when {
        newPath == null && newFolder == null -> false
        currentPath == null && currentFolder == null -> false
        newFolder != null && currentFolder != null ->
            runCatching { File(newFolder).canonicalPath == File(currentFolder).canonicalPath }.getOrDefault(newFolder == currentFolder)
        newPath != null && currentPath != null -> when {
            newPath == currentPath -> true
            else -> runCatching { File(newPath).canonicalPath == File(currentPath).canonicalPath }.getOrDefault(false)
        }
        else -> false
    }
    if (sameRoom) return  // skip re-load
    // else: update plyPath/roomFolder from intent, copy PLY, loadWebGLViewer()
}

override fun onDestroy() {
    if (currentInstanceRef?.get() == this) {
        currentInstanceRef = null
        currentPlyPath = null
    }
    if (isDuplicateInstance) {
        super.onDestroy()
        return
    }
    stopBrainDetection()
    cameraExecutor.shutdown()
    furnitureFitManager?.close()
    webView.destroy()
    super.onDestroy()
}
```

---

## 4. Questions for reviewer

1. **Correctness** – Any race or lifecycle case where we could clear `currentInstanceRef` too early (e.g. first activity destroyed before second’s `onCreate` runs) and still get two full loads, or where `singleTask` + `onNewIntent` could leave the UI in a bad state?
2. **WeakReference** – Is holding the “current” Activity in a `WeakReference` and clearing it in `onDestroy` when `get() == this` the right pattern here, or would you use a different way to detect “another instance already showing this path”?
3. **singleTask** – Are there downsides (e.g. back stack, multiple tasks) we should be aware of for a full-screen WebGL viewer that can also be opened from the room list with different rooms?
4. **Simpler alternative** – Would you prefer to rely only on `singleTask` + `onNewIntent` (and remove the static debounce and in-Activity duplicate check), or is the layered defense reasonable for Android’s sometimes-duplicate creation behavior?
5. **Edge cases** – Configuration change, process death, or user rapidly opening different rooms from the list: anything we should handle explicitly?

---

## 5. Context

- **Stack:** Android (Kotlin), AppCompatActivity, single task for the app.
- **Flow:** User picks photo → SHARP inference (~3 min) → onComplete runs on main thread → we open SharpRoomActivity with the result’s PLY path. We observed a second SharpRoomActivity instance created shortly after the first (same path); both copied the same large PLY and built a WebView.

---

## 6. Ultralytics review response (summary)

- **Lifecycle:** `super.onCreate()` is already called at the start of `onCreate` before the duplicate check; added an explicit comment that the early `finish()` return keeps lifecycle intact.
- **WeakReference / guard:** Clearing `currentInstanceRef` only when `currentInstanceRef?.get() == this` in `onDestroy` is correct—we only clear when the **finishing** instance is the stored one, so a valid new instance is never cleared during rapid switch.
- **singleTask:** Back stack is affected (activities above SharpRoomActivity are cleared). If preserving list/filter state on back is important, consider `singleTop` + `FLAG_ACTIVITY_REORDER_TO_FRONT` instead; we keep `singleTask` for strongest duplicate prevention with heavy (293 MB) asset load.
- **Simplify:** Relying on `singleTask` + `onNewIntent` is usually enough; the static debounce is good “belt-and-suspenders” for the initial double-trigger.
- **Process death:** Static `lastOpenedPlyPath` is cleared on process death; on restore, `onCreate` runs without debounce, which is desired so the restored activity can load.

---

## 7. References (Ultralytics)

- **Platform (workflow, deploy, export):** [Ultralytics Platform](https://docs.ultralytics.com/hub/) — Upload → Annotate → Train → Export → Deploy; 17 export formats (including ExecuTorch); deployment and monitoring.
- **Community (threading, Android, edge cases):** [Ultralytics Community Forum](https://community.ultralytics.com) — for complex threading and mobile deployment questions.
