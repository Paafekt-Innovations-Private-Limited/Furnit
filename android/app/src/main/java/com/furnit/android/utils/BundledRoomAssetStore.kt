package com.furnit.android.utils

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/** Copies packaged sample rooms to a private file that WebViewAssetLoader can serve safely. */
object BundledRoomAssetStore {
    private const val GLB_MAGIC = 0x46546C67 // "glTF" as little-endian Int
    private const val GLB_VERSION = 2
    private const val GLB_JSON_CHUNK = 0x4E4F534A // "JSON" as little-endian Int
    private const val MAX_BUNDLED_GLB_BYTES = 32L * 1024L * 1024L

    fun stage(context: Context, roomId: String, assetPath: String): File {
        require(roomId.matches(Regex("[a-z0-9_]+"))) { "Invalid bundled room id" }
        require(assetPath.startsWith("bundled_rooms/") && assetPath.endsWith(".glb")) {
            "Invalid bundled room asset path"
        }

        val directory = File(context.cacheDir, "bundled_room_viewer").apply {
            check(isDirectory || mkdirs()) { "Could not create bundled room cache" }
        }
        val destination = File(directory, "$roomId.glb")
        val temporary = File(directory, ".$roomId.${System.nanoTime()}.tmp")

        try {
            context.assets.open(assetPath).use { input ->
                FileOutputStream(temporary).use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var total = 0L
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        total += count
                        check(total <= MAX_BUNDLED_GLB_BYTES) { "Bundled room exceeds safety limit" }
                        output.write(buffer, 0, count)
                    }
                    output.fd.sync()
                }
            }
            validateGlb(temporary)?.let { error -> throw IllegalArgumentException(error) }
            runCatching {
                Files.move(
                    temporary.toPath(),
                    destination.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            }.getOrElse {
                Files.move(
                    temporary.toPath(),
                    destination.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                )
            }
            return destination
        } finally {
            if (temporary.exists()) temporary.delete()
        }
    }

    fun validateGlb(file: File): String? {
        if (!file.isFile) return "file does not exist"
        if (file.length() !in 20..0xffffffffL) return "invalid file size ${file.length()}"
        return runCatching {
            RandomAccessFile(file, "r").use { input ->
                val header = ByteArray(12)
                input.readFully(header)
                val buffer = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
                val magic = buffer.int
                val version = buffer.int
                val declaredLength = buffer.int.toLong() and 0xffffffffL
                if (magic != GLB_MAGIC) return "bad GLB magic"
                if (version != GLB_VERSION) return "unsupported GLB version $version"
                if (declaredLength != file.length()) return "GLB length mismatch"

                var firstChunk = true
                var foundJson = false
                while (input.filePointer < declaredLength) {
                    if (declaredLength - input.filePointer < 8) return "truncated GLB chunk header"
                    val chunkHeader = ByteArray(8)
                    input.readFully(chunkHeader)
                    val chunkBuffer = ByteBuffer.wrap(chunkHeader).order(ByteOrder.LITTLE_ENDIAN)
                    val chunkLength = chunkBuffer.int.toLong() and 0xffffffffL
                    val chunkType = chunkBuffer.int
                    if (chunkLength % 4L != 0L) return "unaligned GLB chunk"
                    if (chunkLength > declaredLength - input.filePointer) return "truncated GLB chunk"
                    if (firstChunk && chunkType != GLB_JSON_CHUNK) return "first GLB chunk is not JSON"
                    if (chunkType == GLB_JSON_CHUNK) foundJson = true
                    input.seek(input.filePointer + chunkLength)
                    firstChunk = false
                }
                if (!foundJson) "GLB has no JSON chunk" else null
            }
        }.getOrElse { error -> error.message ?: error.javaClass.simpleName }
    }
}
