package com.furnit.android.utils

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.nio.ByteBuffer
import java.nio.ByteOrder

class BundledRoomAssetStoreTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun validateGlb_acceptsVersionTwoWithMatchingLength() {
        val file = temporaryFolder.newFile("room.glb")
        file.writeBytes(minimalGlb(declaredLength = 24))

        assertNull(BundledRoomAssetStore.validateGlb(file))
    }

    @Test
    fun validateGlb_rejectsTruncatedOrMismatchedFiles() {
        val file = temporaryFolder.newFile("room.glb")
        file.writeBytes(minimalGlb(declaredLength = 28))

        assertEquals("GLB length mismatch", BundledRoomAssetStore.validateGlb(file))
    }

    private fun minimalGlb(declaredLength: Int): ByteArray =
        ByteBuffer.allocate(24)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putInt(0x46546C67)
            .putInt(2)
            .putInt(declaredLength)
            .putInt(4)
            .putInt(0x4E4F534A)
            .put("{}  ".toByteArray())
            .array()
}
