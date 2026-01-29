package com.furnit.android.models

data class Model(
    val id: String,
    val name: String,
    val assetPath: String,
    val isUserCreated: Boolean = false,  // true for user-created rooms
    val thumbnailPath: String? = null,   // path to thumbnail image
    val createdAt: Long = 0L,            // creation timestamp for sorting
    val roomWidth: Float? = null,        // room width in meters
    val roomHeight: Float? = null,       // room height in meters
    val roomDepth: Float? = null,        // room depth in meters
    val photoOrientation: String = "portrait"  // "portrait" or "landscape"
)
