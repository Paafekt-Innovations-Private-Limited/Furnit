package com.furnit.android.services

import java.io.ByteArrayInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/**
 * Covers the on-device model cache used by every RTMDet entry point — live Furniture Fit, the
 * room-viewer inline brain, and Settings -> Image scan, which all reach it through
 * `FurnitureFitManager.initializeAuto()` -> `RTMDetLiteRtBackend.create()`.
 *
 * The regression these exist for: a truncated cached copy used to satisfy the old `length() > 0`
 * reuse check forever, so segmentation stayed dead across every relaunch.
 */
class RTMDetModelCacheTest {

    @get:Rule
    val folder = TemporaryFolder()

    private val installTime = 1_700_000_000_000L
    private val modelBytes = ByteArray(4096) { (it % 251).toByte() }

    private fun destination() = File(folder.root, "rtmdet-ins-m-raw-fp16.tflite")

    private fun temporary() = File(folder.root, "rtmdet-ins-m-raw-fp16.tflite.tmp")

    private fun extractModel(): Long =
        RTMDetModelCache.extract(destination(), temporary()) { ByteArrayInputStream(modelBytes) }

    @Test
    fun extractionWritesEveryByteAndReportsTheCount() {
        val written = extractModel()

        assertEquals(modelBytes.size.toLong(), written)
        assertArrayEquals(modelBytes, destination().readBytes())
    }

    @Test
    fun extractionLeavesNoTemporaryFileBehind() {
        extractModel()

        assertFalse("temporary file should be cleaned up", temporary().exists())
    }

    @Test
    fun completeCopyFromThisInstallIsReused() {
        val written = extractModel()
        val token = RTMDetModelCache.buildToken(installTime, written)

        assertTrue(RTMDetModelCache.isReusable(destination(), token, installTime))
    }

    /** The bug that disabled segmentation permanently. */
    @Test
    fun truncatedCopyIsRejectedInsteadOfReusedForever() {
        val written = extractModel()
        val token = RTMDetModelCache.buildToken(installTime, written)
        // Simulate an extraction cut short by process death or cacheDir reclamation.
        destination().writeBytes(modelBytes.copyOf(modelBytes.size / 2))

        assertTrue("precondition: a truncated file is still non-empty", destination().length() > 0L)
        assertFalse(RTMDetModelCache.isReusable(destination(), token, installTime))
    }

    @Test
    fun copyLongerThanRecordedIsRejected() {
        val written = extractModel()
        val token = RTMDetModelCache.buildToken(installTime, written)
        destination().writeBytes(modelBytes + modelBytes)

        assertFalse(RTMDetModelCache.isReusable(destination(), token, installTime))
    }

    @Test
    fun copyFromAPreviousInstallIsRejected() {
        val written = extractModel()
        val token = RTMDetModelCache.buildToken(installTime, written)

        assertFalse(RTMDetModelCache.isReusable(destination(), token, installTime + 1))
    }

    /** Pre-fix installs wrote a bare timestamp; those must re-extract once rather than be trusted. */
    @Test
    fun legacyTokenWithoutAByteCountIsRejected() {
        extractModel()

        assertFalse(RTMDetModelCache.isReusable(destination(), installTime.toString(), installTime))
    }

    @Test
    fun missingTokenIsRejected() {
        extractModel()

        assertFalse(RTMDetModelCache.isReusable(destination(), null, installTime))
    }

    @Test
    fun malformedTokenIsRejected() {
        val written = extractModel()

        assertFalse(RTMDetModelCache.isReusable(destination(), "not-a-token", installTime))
        assertFalse(RTMDetModelCache.isReusable(destination(), ":$written", installTime))
    }

    @Test
    fun missingCachedFileIsRejected() {
        val token = RTMDetModelCache.buildToken(installTime, modelBytes.size.toLong())

        assertFalse(RTMDetModelCache.isReusable(destination(), token, installTime))
    }

    @Test
    fun interruptedExtractionNeverPublishesAPartialDestination() {
        val failure = runCatching {
            RTMDetModelCache.extract(destination(), temporary()) { failingStream() }
        }

        assertTrue("extraction should surface the failure", failure.isFailure)
        assertFalse("destination must not exist after a failed copy", destination().exists())
        assertFalse("temporary file should be cleaned up", temporary().exists())
    }

    @Test
    fun interruptedExtractionDoesNotCorruptAnExistingGoodCopy() {
        val written = extractModel()
        val token = RTMDetModelCache.buildToken(installTime, written)

        runCatching { RTMDetModelCache.extract(destination(), temporary()) { failingStream() } }

        assertArrayEquals(modelBytes, destination().readBytes())
        assertTrue(RTMDetModelCache.isReusable(destination(), token, installTime))
    }

    @Test
    fun reExtractionRepairsATruncatedCopy() {
        destination().writeBytes(modelBytes.copyOf(16))

        val written = extractModel()
        val token = RTMDetModelCache.buildToken(installTime, written)

        assertArrayEquals(modelBytes, destination().readBytes())
        assertTrue(RTMDetModelCache.isReusable(destination(), token, installTime))
    }

    @Test
    fun emptyAssetIsRejectedRatherThanCached() {
        val failure = runCatching {
            RTMDetModelCache.extract(destination(), temporary()) { ByteArrayInputStream(ByteArray(0)) }
        }

        assertTrue(failure.isFailure)
        assertFalse(destination().exists())
    }

    @Test
    fun staleTemporaryFileFromAnEarlierFailureIsDiscarded() {
        temporary().writeBytes(ByteArray(64) { 7 })

        val written = extractModel()

        assertEquals(modelBytes.size.toLong(), written)
        assertArrayEquals(modelBytes, destination().readBytes())
    }

    /** Fails part-way through, the way a killed process or a full disk would. */
    private fun failingStream(): InputStream = object : InputStream() {
        private var delivered = 0

        override fun read(): Int {
            if (delivered >= 512) throw IOException("simulated interruption")
            delivered++
            return 0
        }

        override fun read(b: ByteArray, off: Int, len: Int): Int {
            if (delivered >= 512) throw IOException("simulated interruption")
            val count = minOf(len, 512 - delivered)
            java.util.Arrays.fill(b, off, off + count, 0)
            delivered += count
            return count
        }
    }
}
