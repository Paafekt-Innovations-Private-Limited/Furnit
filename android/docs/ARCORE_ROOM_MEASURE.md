# ARCore Room Measure

`ArMeasureActivity` implements a simple two-tap ARCore distance measurement flow. It returns the measured distance and anchor count to the caller.

Current constants:

```kotlin
ArMeasureActivity.EXTRA_ROOM_WIDTH_M
ArMeasureActivity.EXTRA_ROOM_HEIGHT_M
ArMeasureActivity.EXTRA_ROOM_DEPTH_M
ArMeasureActivity.RESULT_EXTRA_DISTANCE_M
ArMeasureActivity.RESULT_EXTRA_ANCHOR_COUNT
```

This activity is generic and is not tied to a specific room-generation backend. Furniture Fit also uses ARCore through `FurnitureFitArCameraController` for live overlay sizing where supported.
