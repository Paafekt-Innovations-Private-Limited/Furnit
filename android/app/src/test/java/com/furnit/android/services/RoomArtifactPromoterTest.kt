package com.furnit.android.services

import java.nio.file.Files
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RoomArtifactPromoterTest {
    @Test
    fun promotedGlbIsByteIdenticalToPreview() {
        val testRoot = Files.createTempDirectory("room-artifact-parity").toFile()
        try {
            val previewFolder = testRoot.resolve("room_preview/room_123").apply { mkdirs() }
            val previewBytes = ByteArray(192 * 1024 + 17) { index ->
                ((index * 31 + 7) and 0xff).toByte()
            }
            previewFolder.resolve("room.glb").writeBytes(previewBytes)
            previewFolder.resolve("room_meta.json").writeText("{\"previewOnly\":true}")
            val savedFolder = testRoot.resolve("rooms/room_123")

            val savedGlb = RoomArtifactPromoter.copyPreviewArtifact(
                previewRoomFolder = previewFolder,
                savedRoomFolder = savedFolder,
                glbFileName = "room.glb",
            )

            assertTrue(savedGlb.isFile)
            assertArrayEquals(previewBytes, savedGlb.readBytes())
            assertTrue(savedFolder.resolve("room_meta.json").isFile)
            assertTrue(previewFolder.resolve("room.glb").isFile)
        } finally {
            testRoot.deleteRecursively()
        }
    }

    @Test
    fun existingDestinationIsRejectedWithoutMutation() {
        val testRoot = Files.createTempDirectory("room-artifact-existing-destination").toFile()
        try {
            val previewFolder = testRoot.resolve("room_preview/room_123").apply { mkdirs() }
            previewFolder.resolve("room.glb").writeBytes(byteArrayOf(1, 2, 3))
            val savedFolder = testRoot.resolve("rooms/room_123").apply { mkdirs() }

            val failed = runCatching {
                RoomArtifactPromoter.copyPreviewArtifact(
                    previewRoomFolder = previewFolder,
                    savedRoomFolder = savedFolder,
                    glbFileName = "room.glb",
                )
            }.isFailure

            assertTrue(failed)
            assertTrue(savedFolder.isDirectory)
            assertFalse(savedFolder.resolve("room.glb").exists())
        } finally {
            testRoot.deleteRecursively()
        }
    }
}
