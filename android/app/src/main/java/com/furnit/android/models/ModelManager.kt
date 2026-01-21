package com.furnit.android.models

import android.content.Context

class ModelManager(private val context: Context) {
    private val models = mutableListOf<Model>()

    init {
        // Load bundled models from assets
        models.add(Model("vintage", "Vintage Living Room", "models/vintage.glb"))
        models.add(Model("cozy_room", "Cozy Living Room", "models/cozy_room.glb"))
    }

    fun listModels(): List<Model> = models

    fun getModel(id: String): Model? = models.find { it.id == id }
}
