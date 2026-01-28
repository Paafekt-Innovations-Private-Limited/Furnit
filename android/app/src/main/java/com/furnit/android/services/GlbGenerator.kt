package com.furnit.android.services

import android.graphics.Bitmap
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

/**
 * GlbGenerator - Creates binary glTF 2.0 (GLB) files for room models
 *
 * Generates a 5-plane room structure with embedded textures:
 * - Floor, Ceiling, Front Wall, Left Wall, Right Wall
 */
class GlbGenerator {

    companion object {
        private const val TAG = "GlbGenerator"

        // GLB magic bytes and version
        private const val GLB_MAGIC = 0x46546C67 // "glTF" in little-endian
        private const val GLB_VERSION = 2
        private const val JSON_CHUNK_TYPE = 0x4E4F534A // "JSON" in little-endian
        private const val BIN_CHUNK_TYPE = 0x004E4942  // "BIN\0" in little-endian
    }

    data class PlaneGeometry(
        val positions: FloatArray,
        val normals: FloatArray,
        val uvs: FloatArray,
        val indices: ShortArray
    )

    data class RoomDimensions(
        val width: Float = 8.0f,
        val depth: Float = 9.0f,
        val height: Float = 5.6f
    )

    /**
     * Main entry point - generates a GLB file for the room
     */
    fun generateGlb(
        outputFile: File,
        dimensions: RoomDimensions,
        frontWallTexture: Bitmap,
        floorTexture: Bitmap,
        ceilingTexture: Bitmap,
        leftWallTexture: Bitmap,
        rightWallTexture: Bitmap
    ): Boolean {
        return try {
            Log.d(TAG, "Generating GLB: ${outputFile.absolutePath}")

            // Create plane geometries based on room dimensions
            val halfWidth = dimensions.width / 2f
            val halfDepth = dimensions.depth / 2f
            val height = dimensions.height
            val centerZ = halfDepth - dimensions.depth / 2f // Center point in z

            // Define 5 planes (positions calculated relative to camera at z=+depth/2)
            // Camera is at the back, looking at the front wall
            val floorPlane = createPlaneGeometry(
                centerX = 0f, centerY = 0f, centerZ = centerZ,
                width = dimensions.width, height = dimensions.depth,
                normalX = 0f, normalY = 1f, normalZ = 0f
            )

            val ceilingPlane = createPlaneGeometry(
                centerX = 0f, centerY = height, centerZ = centerZ,
                width = dimensions.width, height = dimensions.depth,
                normalX = 0f, normalY = -1f, normalZ = 0f
            )

            val frontWallPlane = createPlaneGeometry(
                centerX = 0f, centerY = height / 2f, centerZ = -halfDepth,
                width = dimensions.width, height = height,
                normalX = 0f, normalY = 0f, normalZ = 1f
            )

            val leftWallPlane = createPlaneGeometry(
                centerX = -halfWidth, centerY = height / 2f, centerZ = centerZ,
                width = dimensions.depth, height = height,
                normalX = 1f, normalY = 0f, normalZ = 0f
            )

            val rightWallPlane = createPlaneGeometry(
                centerX = halfWidth, centerY = height / 2f, centerZ = centerZ,
                width = dimensions.depth, height = height,
                normalX = -1f, normalY = 0f, normalZ = 0f
            )

            val planes = listOf(floorPlane, ceilingPlane, frontWallPlane, leftWallPlane, rightWallPlane)
            val textures = listOf(floorTexture, ceilingTexture, frontWallTexture, leftWallTexture, rightWallTexture)
            val textureNames = listOf("floor", "ceiling", "front_wall", "left_wall", "right_wall")

            // Convert textures to PNG bytes
            val textureBytes = textures.map { bitmapToPngBytes(it) }

            // Build binary buffer (geometry + textures)
            val binaryData = buildBinaryBuffer(planes, textureBytes)

            // Build JSON structure
            val json = buildGltfJson(planes, textureBytes, textureNames)

            // Assemble final GLB
            val glbData = assembleGlb(json, binaryData)

            // Write to file
            FileOutputStream(outputFile).use { fos ->
                fos.write(glbData)
            }

            Log.d(TAG, "GLB generated: ${outputFile.length()} bytes")
            true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate GLB", e)
            false
        }
    }

    /**
     * Creates a plane mesh with positions, normals, UVs, and indices
     */
    private fun createPlaneGeometry(
        centerX: Float, centerY: Float, centerZ: Float,
        width: Float, height: Float,
        normalX: Float, normalY: Float, normalZ: Float
    ): PlaneGeometry {
        val halfW = width / 2f
        val halfH = height / 2f

        // Generate 4 corners based on normal direction
        val positions: FloatArray
        val uvs: FloatArray

        when {
            normalY != 0f -> {
                // Horizontal plane (floor/ceiling)
                // For floor (normalY > 0): looking down from above
                // For ceiling (normalY < 0): looking up from below
                positions = floatArrayOf(
                    centerX - halfW, centerY, centerZ - halfH,  // bottom-left
                    centerX + halfW, centerY, centerZ - halfH,  // bottom-right
                    centerX + halfW, centerY, centerZ + halfH,  // top-right
                    centerX - halfW, centerY, centerZ + halfH   // top-left
                )
                // UV coordinates - flip based on normal direction
                uvs = if (normalY > 0) {
                    floatArrayOf(
                        0f, 0f,
                        1f, 0f,
                        1f, 1f,
                        0f, 1f
                    )
                } else {
                    floatArrayOf(
                        0f, 1f,
                        1f, 1f,
                        1f, 0f,
                        0f, 0f
                    )
                }
            }
            normalZ != 0f -> {
                // Front/back wall (perpendicular to Z)
                positions = floatArrayOf(
                    centerX - halfW, centerY - halfH, centerZ,  // bottom-left
                    centerX + halfW, centerY - halfH, centerZ,  // bottom-right
                    centerX + halfW, centerY + halfH, centerZ,  // top-right
                    centerX - halfW, centerY + halfH, centerZ   // top-left
                )
                uvs = floatArrayOf(
                    0f, 1f,
                    1f, 1f,
                    1f, 0f,
                    0f, 0f
                )
            }
            else -> {
                // Left/right wall (perpendicular to X)
                positions = if (normalX > 0) {
                    // Left wall - normal points right (+X)
                    floatArrayOf(
                        centerX, centerY - halfH, centerZ + halfH,  // bottom-left (back)
                        centerX, centerY - halfH, centerZ - halfH,  // bottom-right (front)
                        centerX, centerY + halfH, centerZ - halfH,  // top-right (front)
                        centerX, centerY + halfH, centerZ + halfH   // top-left (back)
                    )
                } else {
                    // Right wall - normal points left (-X)
                    floatArrayOf(
                        centerX, centerY - halfH, centerZ - halfH,  // bottom-left (front)
                        centerX, centerY - halfH, centerZ + halfH,  // bottom-right (back)
                        centerX, centerY + halfH, centerZ + halfH,  // top-right (back)
                        centerX, centerY + halfH, centerZ - halfH   // top-left (front)
                    )
                }
                uvs = floatArrayOf(
                    0f, 1f,
                    1f, 1f,
                    1f, 0f,
                    0f, 0f
                )
            }
        }

        // All 4 vertices share the same normal
        val normals = floatArrayOf(
            normalX, normalY, normalZ,
            normalX, normalY, normalZ,
            normalX, normalY, normalZ,
            normalX, normalY, normalZ
        )

        // Two triangles forming a quad
        val indices = shortArrayOf(0, 1, 2, 0, 2, 3)

        return PlaneGeometry(positions, normals, uvs, indices)
    }

    /**
     * Compresses bitmap to PNG bytes
     */
    private fun bitmapToPngBytes(bitmap: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 90, stream)
        return stream.toByteArray()
    }

    /**
     * Builds the binary buffer containing geometry and texture data
     */
    private fun buildBinaryBuffer(planes: List<PlaneGeometry>, textureBytes: List<ByteArray>): ByteArray {
        val stream = ByteArrayOutputStream()

        // Write geometry data for each plane
        for (plane in planes) {
            // Positions (12 bytes per vertex = 3 floats)
            writeFloatArray(stream, plane.positions)
            // Normals (12 bytes per vertex = 3 floats)
            writeFloatArray(stream, plane.normals)
            // UVs (8 bytes per vertex = 2 floats)
            writeFloatArray(stream, plane.uvs)
            // Indices (2 bytes per index)
            writeShortArray(stream, plane.indices)
        }

        // Write texture data (already PNG bytes)
        for (bytes in textureBytes) {
            stream.write(bytes)
        }

        return stream.toByteArray()
    }

    private fun writeFloatArray(stream: ByteArrayOutputStream, data: FloatArray) {
        val buffer = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (f in data) {
            buffer.putFloat(f)
        }
        stream.write(buffer.array())
    }

    private fun writeShortArray(stream: ByteArrayOutputStream, data: ShortArray) {
        val buffer = ByteBuffer.allocate(data.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (s in data) {
            buffer.putShort(s)
        }
        stream.write(buffer.array())
    }

    /**
     * Builds the glTF JSON structure
     */
    private fun buildGltfJson(
        planes: List<PlaneGeometry>,
        textureBytes: List<ByteArray>,
        textureNames: List<String>
    ): String {
        val sb = StringBuilder()
        sb.append("{")

        // Asset info
        sb.append("\"asset\":{\"version\":\"2.0\",\"generator\":\"Furnit Android\"},")

        // Scene
        sb.append("\"scene\":0,")
        sb.append("\"scenes\":[{\"nodes\":[0]}],")

        // Nodes - root node with children for each plane
        sb.append("\"nodes\":[")
        sb.append("{\"children\":[1,2,3,4,5]}")  // Root node
        for (i in planes.indices) {
            sb.append(",{\"mesh\":$i,\"name\":\"${textureNames[i]}\"}")
        }
        sb.append("],")

        // Meshes - one per plane
        sb.append("\"meshes\":[")
        for (i in planes.indices) {
            if (i > 0) sb.append(",")
            val posAccessor = i * 4
            val normalAccessor = i * 4 + 1
            val uvAccessor = i * 4 + 2
            val indexAccessor = i * 4 + 3
            sb.append("{\"primitives\":[{")
            sb.append("\"attributes\":{")
            sb.append("\"POSITION\":$posAccessor,")
            sb.append("\"NORMAL\":$normalAccessor,")
            sb.append("\"TEXCOORD_0\":$uvAccessor")
            sb.append("},")
            sb.append("\"indices\":$indexAccessor,")
            sb.append("\"material\":$i")
            sb.append("}]}")
        }
        sb.append("],")

        // Materials - one per plane with texture (doubleSided for visibility from both sides)
        sb.append("\"materials\":[")
        for (i in planes.indices) {
            if (i > 0) sb.append(",")
            sb.append("{\"pbrMetallicRoughness\":{")
            sb.append("\"baseColorTexture\":{\"index\":$i},")
            sb.append("\"metallicFactor\":0.0,")
            sb.append("\"roughnessFactor\":1.0")
            sb.append("},\"doubleSided\":true,\"name\":\"${textureNames[i]}_material\"}")
        }
        sb.append("],")

        // Textures - one per plane
        sb.append("\"textures\":[")
        for (i in planes.indices) {
            if (i > 0) sb.append(",")
            sb.append("{\"sampler\":0,\"source\":$i}")
        }
        sb.append("],")

        // Images - one per plane, referencing buffer views
        val imageBufferViewStart = planes.size * 4  // After geometry buffer views
        sb.append("\"images\":[")
        for (i in planes.indices) {
            if (i > 0) sb.append(",")
            sb.append("{\"bufferView\":${imageBufferViewStart + i},\"mimeType\":\"image/png\"}")
        }
        sb.append("],")

        // Samplers
        sb.append("\"samplers\":[{\"magFilter\":9729,\"minFilter\":9729,\"wrapS\":10497,\"wrapT\":10497}],")

        // Calculate buffer offsets for accessors and buffer views
        var offset = 0
        val bufferViews = mutableListOf<String>()
        val accessors = mutableListOf<String>()

        // Geometry buffer views and accessors (4 per plane: positions, normals, uvs, indices)
        for ((planeIdx, plane) in planes.withIndex()) {
            val vertexCount = plane.positions.size / 3
            val indexCount = plane.indices.size

            // Positions
            val posSize = plane.positions.size * 4
            bufferViews.add("{\"buffer\":0,\"byteOffset\":$offset,\"byteLength\":$posSize,\"target\":34962}")
            val minMax = calculateBounds(plane.positions)
            accessors.add("{\"bufferView\":${planeIdx * 4},\"componentType\":5126,\"count\":$vertexCount,\"type\":\"VEC3\",\"min\":[${minMax.first}],\"max\":[${minMax.second}]}")
            offset += posSize

            // Normals
            val normalSize = plane.normals.size * 4
            bufferViews.add("{\"buffer\":0,\"byteOffset\":$offset,\"byteLength\":$normalSize,\"target\":34962}")
            accessors.add("{\"bufferView\":${planeIdx * 4 + 1},\"componentType\":5126,\"count\":$vertexCount,\"type\":\"VEC3\"}")
            offset += normalSize

            // UVs
            val uvSize = plane.uvs.size * 4
            bufferViews.add("{\"buffer\":0,\"byteOffset\":$offset,\"byteLength\":$uvSize,\"target\":34962}")
            accessors.add("{\"bufferView\":${planeIdx * 4 + 2},\"componentType\":5126,\"count\":$vertexCount,\"type\":\"VEC2\"}")
            offset += uvSize

            // Indices
            val indexSize = plane.indices.size * 2
            bufferViews.add("{\"buffer\":0,\"byteOffset\":$offset,\"byteLength\":$indexSize,\"target\":34963}")
            accessors.add("{\"bufferView\":${planeIdx * 4 + 3},\"componentType\":5123,\"count\":$indexCount,\"type\":\"SCALAR\"}")
            offset += indexSize
        }

        // Image buffer views (no target for images)
        for (bytes in textureBytes) {
            bufferViews.add("{\"buffer\":0,\"byteOffset\":$offset,\"byteLength\":${bytes.size}}")
            offset += bytes.size
        }

        // Accessors
        sb.append("\"accessors\":[")
        sb.append(accessors.joinToString(","))
        sb.append("],")

        // Buffer views
        sb.append("\"bufferViews\":[")
        sb.append(bufferViews.joinToString(","))
        sb.append("],")

        // Buffer
        sb.append("\"buffers\":[{\"byteLength\":$offset}]")

        sb.append("}")
        return sb.toString()
    }

    private fun calculateBounds(positions: FloatArray): Pair<String, String> {
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var minZ = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var maxY = Float.MIN_VALUE
        var maxZ = Float.MIN_VALUE

        for (i in positions.indices step 3) {
            val x = positions[i]
            val y = positions[i + 1]
            val z = positions[i + 2]
            minX = minOf(minX, x)
            minY = minOf(minY, y)
            minZ = minOf(minZ, z)
            maxX = maxOf(maxX, x)
            maxY = maxOf(maxY, y)
            maxZ = maxOf(maxZ, z)
        }

        return Pair("$minX,$minY,$minZ", "$maxX,$maxY,$maxZ")
    }

    /**
     * Assembles the final GLB file from JSON and binary data
     */
    private fun assembleGlb(json: String, binaryData: ByteArray): ByteArray {
        // Pad JSON to 4-byte boundary with spaces
        val jsonBytes = json.toByteArray(Charsets.UTF_8)
        val jsonPadding = (4 - (jsonBytes.size % 4)) % 4
        val paddedJsonSize = jsonBytes.size + jsonPadding

        // Pad binary data to 4-byte boundary with zeros
        val binPadding = (4 - (binaryData.size % 4)) % 4
        val paddedBinSize = binaryData.size + binPadding

        // Calculate total file size
        val totalSize = 12 + // Header
                        8 + paddedJsonSize + // JSON chunk header + data
                        8 + paddedBinSize    // BIN chunk header + data

        val buffer = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN)

        // GLB Header (12 bytes)
        buffer.putInt(GLB_MAGIC)          // Magic "glTF"
        buffer.putInt(GLB_VERSION)        // Version 2
        buffer.putInt(totalSize)          // Total file size

        // JSON Chunk
        buffer.putInt(paddedJsonSize)     // Chunk length
        buffer.putInt(JSON_CHUNK_TYPE)    // Chunk type "JSON"
        buffer.put(jsonBytes)             // JSON data
        repeat(jsonPadding) { buffer.put(0x20.toByte()) } // Pad with spaces

        // Binary Chunk
        buffer.putInt(paddedBinSize)      // Chunk length
        buffer.putInt(BIN_CHUNK_TYPE)     // Chunk type "BIN\0"
        buffer.put(binaryData)            // Binary data
        repeat(binPadding) { buffer.put(0.toByte()) } // Pad with zeros

        return buffer.array()
    }
}
