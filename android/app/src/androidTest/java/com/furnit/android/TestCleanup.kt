package com.furnit.android

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Utility class for cleaning up test data.
 *
 * This ensures that test-created rooms, files, and temporary data
 * are properly removed before and after tests run.
 */
object TestCleanup {

    private const val TAG = "TestCleanup"

    // Directories that may contain test data
    private val TEST_DIRECTORIES = listOf(
        "rooms",           // SinglePhotoRoomReconstructor rooms
        "sharp_rooms",     // SharpService rooms
        "test_camera",     // Camera test rooms
        "test_glb",        // GLB generator test files
        "test_mask",       // Mask test files
        "test_room",       // Room processing tests
        "TestRoom_*"       // Test room output files
    )

    // File patterns to clean
    private val TEST_FILE_PATTERNS = listOf(
        "TestRoom_*.glb",
        "TestRoom_*.png",
        "test_*.glb",
        "test_*.png",
        "test_*.jpg",
        "room_sharp_*.png",
        "sharp_room_*.txt"
    )

    /**
     * Clean all test data from app directories
     */
    fun cleanAll(context: Context) {
        Log.d(TAG, "=== Starting Test Cleanup ===")

        cleanInternalStorage(context)
        cleanExternalStorage(context)
        cleanCacheDir(context)

        Log.d(TAG, "=== Test Cleanup Complete ===")
    }

    /**
     * Clean internal storage (filesDir)
     */
    fun cleanInternalStorage(context: Context) {
        val filesDir = context.filesDir
        Log.d(TAG, "Cleaning internal storage: ${filesDir.absolutePath}")

        TEST_DIRECTORIES.forEach { dirName ->
            if (dirName.contains("*")) {
                // Pattern matching for directory names
                val pattern = dirName.replace("*", ".*").toRegex()
                filesDir.listFiles()?.filter { it.isDirectory && it.name.matches(pattern) }?.forEach { dir ->
                    deleteRecursively(dir)
                }
            } else {
                val dir = File(filesDir, dirName)
                if (dir.exists()) {
                    deleteRecursively(dir)
                }
            }
        }
    }

    /**
     * Clean external storage (getExternalFilesDir)
     */
    fun cleanExternalStorage(context: Context) {
        val externalDir = context.getExternalFilesDir(null) ?: return
        Log.d(TAG, "Cleaning external storage: ${externalDir.absolutePath}")

        // Clean test files by pattern
        TEST_FILE_PATTERNS.forEach { pattern ->
            val regex = pattern.replace("*", ".*").toRegex()
            externalDir.listFiles()?.filter { it.name.matches(regex) }?.forEach { file ->
                deleteRecursively(file)
            }
        }

        // Clean test directories
        TEST_DIRECTORIES.forEach { dirName ->
            val dir = File(externalDir, dirName)
            if (dir.exists()) {
                deleteRecursively(dir)
            }
        }
    }

    /**
     * Clean cache directory
     */
    fun cleanCacheDir(context: Context) {
        val cacheDir = context.cacheDir
        Log.d(TAG, "Cleaning cache: ${cacheDir.absolutePath}")

        TEST_DIRECTORIES.forEach { dirName ->
            val dir = File(cacheDir, dirName)
            if (dir.exists()) {
                deleteRecursively(dir)
            }
        }

        // Also clean test files directly in cache
        TEST_FILE_PATTERNS.forEach { pattern ->
            val regex = pattern.replace("*", ".*").toRegex()
            cacheDir.listFiles()?.filter { it.name.matches(regex) }?.forEach { file ->
                deleteRecursively(file)
            }
        }
    }

    /**
     * Clean only rooms directory (for quick cleanup between tests)
     */
    fun cleanRoomsOnly(context: Context) {
        val roomsDir = File(context.filesDir, "rooms")
        if (roomsDir.exists()) {
            Log.d(TAG, "Cleaning rooms directory")
            deleteRecursively(roomsDir)
        }

        val sharpRoomsDir = File(context.filesDir, "sharp_rooms")
        if (sharpRoomsDir.exists()) {
            Log.d(TAG, "Cleaning sharp_rooms directory")
            deleteRecursively(sharpRoomsDir)
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

        // Internal storage
        val filesDir = context.filesDir
        TEST_DIRECTORIES.forEach { dirName ->
            val dir = File(filesDir, dirName)
            if (dir.exists()) {
                result.add("Internal: ${dir.absolutePath} (${countFiles(dir)} files)")
            }
        }

        // External storage
        val externalDir = context.getExternalFilesDir(null)
        if (externalDir != null) {
            TEST_FILE_PATTERNS.forEach { pattern ->
                val regex = pattern.replace("*", ".*").toRegex()
                externalDir.listFiles()?.filter { it.name.matches(regex) }?.forEach { file ->
                    result.add("External: ${file.absolutePath}")
                }
            }
        }

        // Cache
        val cacheDir = context.cacheDir
        TEST_DIRECTORIES.forEach { dirName ->
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
