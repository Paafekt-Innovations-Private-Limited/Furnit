package com.furnit.android

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Utility class for cleaning up test data.
 *
 * IMPORTANT: Only cleans TEST-created data, NOT user rooms.
 * Test rooms are identified by naming convention: room_test_*
 */
object TestCleanup {

    private const val TAG = "TestCleanup"

    // Test-only directories (safe to delete entirely)
    private val TEST_ONLY_DIRECTORIES = listOf(
        "test_camera",              // Camera test rooms
        "test_glb",                 // GLB generator test files
        "test_mask",                // Mask test files
        "test_room",                // Room processing tests
        "glb_test",                 // GlbGeneratorTest
        "glb_doublesided_test",     // GlbDoubleSidedTest
        "room_preview"              // Preview folder for RoomSaveFlowTest
    )

    // Pattern for test rooms INSIDE the rooms directory
    // Only deletes room_test_* folders, NOT user rooms
    private const val TEST_ROOM_PATTERN = "room_test_.*"

    // File patterns to clean (test output files only)
    private val TEST_FILE_PATTERNS = listOf(
        "TestRoom_*.glb",
        "TestRoom_*.png",
        "test_*.glb",
        "test_*.png",
        "test_*.jpg",
        "TestScreenshot_*.png",       // Screenshot tests
        "doublesided_test.glb",       // DoubleSided tests
        "mesh_count_test.glb",
        "material_names_test.glb",
        "doublesided_position_test.glb",
        "generator_string_test.glb",
        "version_test.glb"
    )

    /**
     * Clean all test data from app directories.
     * IMPORTANT: Only cleans test-created data, preserves user rooms.
     */
    fun cleanAll(context: Context) {
        Log.d(TAG, "=== Starting Test Cleanup (preserving user rooms) ===")

        cleanTestDirectories(context)
        cleanTestRoomsOnly(context)
        cleanTestFiles(context)
        cleanCacheDir(context)

        Log.d(TAG, "=== Test Cleanup Complete ===")
    }

    /**
     * Clean test-only directories (safe to delete entirely)
     */
    fun cleanTestDirectories(context: Context) {
        val filesDir = context.filesDir
        Log.d(TAG, "Cleaning test directories in: ${filesDir.absolutePath}")

        TEST_ONLY_DIRECTORIES.forEach { dirName ->
            val dir = File(filesDir, dirName)
            if (dir.exists()) {
                Log.d(TAG, "Deleting test directory: $dirName")
                deleteRecursively(dir)
            }
        }
    }

    /**
     * Clean only test-created rooms (room_test_*), NOT user rooms
     */
    fun cleanTestRoomsOnly(context: Context) {
        val roomsDir = File(context.filesDir, "rooms")
        if (roomsDir.exists()) {
            val testRoomPattern = TEST_ROOM_PATTERN.toRegex()
            roomsDir.listFiles()?.filter {
                it.isDirectory && it.name.matches(testRoomPattern)
            }?.forEach { testRoom ->
                Log.d(TAG, "Deleting test room: ${testRoom.name}")
                deleteRecursively(testRoom)
            }
        }
    }

    /**
     * Clean test files from external storage
     */
    fun cleanTestFiles(context: Context) {
        val externalDir = context.getExternalFilesDir(null) ?: return
        Log.d(TAG, "Cleaning test files in: ${externalDir.absolutePath}")

        TEST_FILE_PATTERNS.forEach { pattern ->
            val regex = pattern.replace("*", ".*").toRegex()
            externalDir.listFiles()?.filter { it.name.matches(regex) }?.forEach { file ->
                Log.d(TAG, "Deleting test file: ${file.name}")
                deleteRecursively(file)
            }
        }

        // Clean test directories in external storage
        TEST_ONLY_DIRECTORIES.forEach { dirName ->
            val dir = File(externalDir, dirName)
            if (dir.exists()) {
                deleteRecursively(dir)
            }
        }
    }

    /**
     * Clean cache directory (test files only)
     */
    fun cleanCacheDir(context: Context) {
        val cacheDir = context.cacheDir
        Log.d(TAG, "Cleaning cache: ${cacheDir.absolutePath}")

        TEST_ONLY_DIRECTORIES.forEach { dirName ->
            val dir = File(cacheDir, dirName)
            if (dir.exists()) {
                deleteRecursively(dir)
            }
        }

        // Clean test files directly in cache
        TEST_FILE_PATTERNS.forEach { pattern ->
            val regex = pattern.replace("*", ".*").toRegex()
            cacheDir.listFiles()?.filter { it.name.matches(regex) }?.forEach { file ->
                deleteRecursively(file)
            }
        }
    }

    /**
     * Recursively delete a file or directory
     */
    private fun deleteRecursively(file: File): Boolean {
        if (file.isDirectory) {
            file.listFiles()?.forEach { child ->
                deleteRecursively(child)
            }
        }
        val deleted = file.delete()
        if (deleted) {
            Log.d(TAG, "Deleted: ${file.absolutePath}")
        } else if (file.exists()) {
            Log.w(TAG, "Failed to delete: ${file.absolutePath}")
        }
        return deleted
    }

    /**
     * List all test data that would be cleaned (for debugging)
     */
    fun listTestData(context: Context): List<String> {
        val result = mutableListOf<String>()

        val filesDir = context.filesDir

        // Test-only directories
        TEST_ONLY_DIRECTORIES.forEach { dirName ->
            val dir = File(filesDir, dirName)
            if (dir.exists()) {
                result.add("Test dir: ${dir.absolutePath} (${countFiles(dir)} files)")
            }
        }

        // Test rooms inside rooms directory
        val roomsDir = File(filesDir, "rooms")
        if (roomsDir.exists()) {
            val testRoomPattern = TEST_ROOM_PATTERN.toRegex()
            roomsDir.listFiles()?.filter {
                it.isDirectory && it.name.matches(testRoomPattern)
            }?.forEach { testRoom ->
                result.add("Test room: ${testRoom.name}")
            }
        }

        // External storage test files
        val externalDir = context.getExternalFilesDir(null)
        if (externalDir != null) {
            TEST_FILE_PATTERNS.forEach { pattern ->
                val regex = pattern.replace("*", ".*").toRegex()
                externalDir.listFiles()?.filter { it.name.matches(regex) }?.forEach { file ->
                    result.add("Test file: ${file.absolutePath}")
                }
            }
        }

        // Cache test directories
        val cacheDir = context.cacheDir
        TEST_ONLY_DIRECTORIES.forEach { dirName ->
            val dir = File(cacheDir, dirName)
            if (dir.exists()) {
                result.add("Cache: ${dir.absolutePath} (${countFiles(dir)} files)")
            }
        }

        return result
    }

    private fun countFiles(dir: File): Int {
        if (!dir.isDirectory) return 1
        return dir.listFiles()?.sumOf { countFiles(it) } ?: 0
    }
}
