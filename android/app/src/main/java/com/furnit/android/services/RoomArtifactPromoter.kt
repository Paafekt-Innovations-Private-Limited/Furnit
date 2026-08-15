package com.furnit.android.services

import java.io.File

/**
 * Promotes a generated preview folder without changing the GLB that the user inspected.
 * Metadata may be updated by the caller after this copy, but [room.glb] must remain byte-identical.
 */
internal object RoomArtifactPromoter {
    fun copyPreviewArtifact(
        previewRoomFolder: File,
        savedRoomFolder: File,
        glbFileName: String,
    ): File {
        require(previewRoomFolder.isDirectory) { "Preview room folder is missing" }
        require(!savedRoomFolder.exists()) { "Saved room folder already exists" }
        val previewGlb = File(previewRoomFolder, glbFileName)
        require(previewGlb.isFile) { "Preview GLB is missing" }

        try {
            check(previewRoomFolder.copyRecursively(savedRoomFolder, overwrite = false)) {
                "Could not copy preview room"
            }
            val savedGlb = File(savedRoomFolder, glbFileName)
            check(savedGlb.isFile && filesHaveIdenticalContent(previewGlb, savedGlb)) {
                "Saved GLB differs from preview GLB"
            }
            return savedGlb
        } catch (error: Exception) {
            savedRoomFolder.deleteRecursively()
            throw error
        }
    }

    private fun filesHaveIdenticalContent(first: File, second: File): Boolean {
        if (first.length() != second.length()) return false
        first.inputStream().buffered().use { firstInput ->
            second.inputStream().buffered().use { secondInput ->
                val firstBuffer = ByteArray(COMPARE_BUFFER_BYTES)
                val secondBuffer = ByteArray(COMPARE_BUFFER_BYTES)
                while (true) {
                    val firstCount = firstInput.read(firstBuffer)
                    val secondCount = secondInput.read(secondBuffer)
                    if (firstCount != secondCount) return false
                    if (firstCount < 0) return true
                    for (index in 0 until firstCount) {
                        if (firstBuffer[index] != secondBuffer[index]) return false
                    }
                }
            }
        }
    }

    private const val COMPARE_BUFFER_BYTES = 64 * 1024
}
