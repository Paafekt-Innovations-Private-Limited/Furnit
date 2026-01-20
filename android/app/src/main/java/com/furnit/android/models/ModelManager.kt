package com.furnit.android.models

import android.content.Context

class ModelManager(private val context: Context) {
    private val models = mutableListOf<Model>()

    init {
        // Load bundled models from assets or list from a server
        models.add(Model("chair_01", "Chair", "models/chair_01.glb"))
    }

    fun listModels(): List<Model> = models

    fun getModel(id: String): Model? = models.find { it.id == id }
}
