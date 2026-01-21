package com.furnit.android.models

data class Model(
    val id: String,
    val name: String,
    val assetPath: String,
    val isUserCreated: Boolean = false,  // true for user-created rooms
    val thumbnailPath: String? = null     // path to thumbnail image
)
