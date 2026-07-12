package com.furnit.android.roomreconstruction

object RoomMeasurementConstants {
    const val WALL_MARGIN = 0.05f
    const val CAMERA_HEIGHT_PRIOR_METERS = 1.70f
    const val CAMERA_HEIGHT_RAW_VALID_MIN = 0.45f
    const val CAMERA_HEIGHT_RAW_VALID_MAX = 5.0f
    const val FLOOR_BAND_START_FRACTION = 0.78f
    const val FLOOR_CHAIR_EXCLUDE_U = 0.58f
    const val FLOOR_CHAIR_EXCLUDE_V = 0.55f
    const val PLAUSIBLE_SPAN_MIN = 1.2f
    const val PLAUSIBLE_SPAN_MAX = 8.0f
    const val MINIMUM_ROOM_WIDTH_METERS = 2.0f
    const val OBJECT_BBOX_CONFIDENCE_THRESHOLD = 0.30f
    const val GEO_EXIF_FOCAL_MATCH_RATIO_MIN = 0.85f
    const val GEO_EXIF_FOCAL_MATCH_RATIO_MAX = 1.15f
    const val DEPTH_METRIC_SCALE_MIN = 0.55f
    const val DEPTH_METRIC_SCALE_MAX = 1.45f
    const val MAX_PLAUSIBLE_ROLL_RADIANS = 0.6f
    const val MAX_PLAUSIBLE_PITCH_RADIANS = 0.9f
    const val CEILING_BAND_ROW_FRACTION = 0.18f
    const val MINIMUM_CEILING_CLEARANCE_METERS = 0.3f
    const val CEILING_ANCHORED_HEIGHT_MIN = 1.9f
    const val CEILING_ANCHORED_HEIGHT_MAX = 4.2f
    const val MAX_WORKING_IMAGE_DIMENSION = 1600

    val objectAnchorHeightMeters: Map<Int, Float> = mapOf(
        56 to 1.15f, // chair
    )
}
