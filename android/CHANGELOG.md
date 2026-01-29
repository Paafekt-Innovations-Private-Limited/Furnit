# Furnit Android - Recent Changes

## Room Viewer Enhancements

### Auto-Orbit Feature
- **Location**: `SettingsActivity.kt`, `SharpRoomActivity.kt`
- Added auto-orbit toggle in Settings under "Room Viewer" section
- Default is **OFF** (was previously ON)
- When enabled, camera slowly rotates around the room when idle
- Setting persisted in SharedPreferences (`auto_orbit_enabled`)

### Grey Screen Fix (Warm-up Rendering)
- **Location**: `SharpRoomActivity.kt` (WebGL JavaScript)
- Added 5-second warm-up period for continuous rendering
- Fixes issue where room appeared grey when auto-orbit was disabled
- SparkJS Gaussian splat needs time to fully load before static rendering works

```javascript
const WARMUP_DURATION = 5000;
const animationStartTime = performance.now();
// In animate loop:
const inWarmup = elapsed < WARMUP_DURATION;
if (inWarmup) { shouldRender = true; }
```

### Room Dimension Persistence
- **Location**: `Model.kt`, `ModelManager.kt`, `SharpRoomActivity.kt`, `ContentActivity.kt`

#### Model.kt
Added dimension fields to the Model data class:
```kotlin
val roomWidth: Float? = null
val roomHeight: Float? = null
val roomDepth: Float? = null
val photoOrientation: String = "portrait"
```

#### ModelManager.kt
Loads dimensions from metadata file:
```kotlin
lines.firstOrNull { it.startsWith("roomWidth=") }
    ?.substringAfter("roomWidth=")?.toFloatOrNull()?.let { roomWidth = it }
```

#### SharpRoomActivity.kt
- Saves dimensions to metadata when room is saved
- Reports dimensions from WebGL to Android via JavaScript interface
- Multiple dimension reports (500ms, 1500ms, 3000ms) to ensure delivery
- Prioritizes saved dimensions over JS-measured ones using `hasSavedDimensions` flag

#### ContentActivity.kt
- Passes saved dimensions when opening rooms from home screen
- Room cards now display actual dimensions (e.g., "3.5 × 2.8 m") instead of generic text

### Orientation Label
- **Location**: `SharpRoomActivity.kt`
- Shows correct orientation based on `photoOrientation` field
- Portrait: "held vertically"
- Landscape: "held horizontally"

### PLY File Handling
- **Location**: `ContentActivity.kt`
- Detects PLY files in room folders
- Opens `SharpRoomActivity` for PLY files (instead of `RoomViewerActivity`)
- Passes all saved metadata (dimensions, orientation) to SharpRoomActivity

### User-Created Room Icon
- **Location**: `ContentActivity.kt`, `res/drawable/ic_grid_3x3.xml`
- New vector drawable matching iOS `circle.grid.3x3.fill` SF Symbol
- 3x3 grid of filled purple circles
- Replaces old text-based grid icon

## File Changes Summary

| File | Changes |
|------|---------|
| `Model.kt` | Added `roomWidth`, `roomHeight`, `roomDepth`, `photoOrientation` fields |
| `ModelManager.kt` | Load dimensions from metadata file |
| `SettingsActivity.kt` | Added auto-orbit toggle (default OFF) |
| `SharpRoomActivity.kt` | Warm-up rendering, auto-orbit, dimension handling, orientation label |
| `ContentActivity.kt` | PLY detection, dimension passing, actual dimensions in cards, new icon |
| `ic_grid_3x3.xml` | New drawable for user-created room icon |

## Metadata Format

Room metadata is stored in `metadata.txt` within each room folder:

```
name=My Room
created=1706547200000
type=sharp
roomWidth=4.5
roomHeight=3.2
roomDepth=5.0
photoOrientation=portrait
```

## JavaScript Interface

`WebAppInterface` in SharpRoomActivity provides:

```kotlin
@JavascriptInterface
fun onLoaded()  // Called when WebGL viewer is ready

@JavascriptInterface
fun onDimensionsMeasured(width: Float, height: Float)  // Called with room dimensions
```
