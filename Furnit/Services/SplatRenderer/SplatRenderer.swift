import Foundation
import Metal
import MetalKit
import simd
import UIKit

// MARK: - Room Quad Vertex (for textured room planes)
struct RoomQuadVertex {
    var position: SIMD3<Float>
    var uv: SIMD2<Float>
}

// MARK: - Camera Uniforms (must match Metal CameraUniforms exactly)
struct CameraUniforms {
    var viewMatrix: simd_float4x4
    var projectionMatrix: simd_float4x4
    var debugScreenSpace: UInt32  // 0 = normal camera, 1 = screen-space bypass
    var padding: SIMD3<UInt32> = .zero  // Pad to 16-byte alignment
}

// MARK: - Cloud Bounds (for auto-framing camera)
struct CloudBounds {
    var min: SIMD3<Float>
    var max: SIMD3<Float>
    var averageCenter: SIMD3<Float>  // Center of mass (simple average of all splat positions)

    /// Bounding-box center (midpoint of min/max)
    var boxCenter: SIMD3<Float> {
        (min + max) * 0.5
    }

    var diagonal: SIMD3<Float> {
        max - min
    }

    var radius: Float {
        simd_length(diagonal) * 0.5
    }
}

// MARK: - Camera Mode
enum SplatCameraMode {
    case room       // Normal room view
    case sharpDebug // Auto-frame the SHARP cloud
}

// MARK: - Splat Renderer
/// Metal-based renderer for Gaussian splats from SHARP.
class SplatRenderer: NSObject {

    // MARK: - Metal Objects
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var pipelineState: MTLRenderPipelineState?
    private var debugPipelineState: MTLRenderPipelineState?
    private var depthState: MTLDepthStencilState?

    // MARK: - Debug Mode
    /// When true, uses direct 2D mapping to verify data packing (bypass camera)
    var debugMode: Bool = false

    /// When true, forces camera to frame the splat cloud perfectly
    var debugForceCamera: Bool = true

    // MARK: - Camera Mode & Cloud Bounds
    /// Camera mode: .room for normal view, .sharpDebug for auto-framing SHARP cloud
    var cameraMode: SplatCameraMode = .sharpDebug

    /// Bounds of the loaded splat cloud (set by loadSplats)
    var cloudBounds: CloudBounds?

    /// Flag to log camera info only once after loading new splats
    private var didLogCameraForCurrentCloud = false

    // MARK: - Buffers
    private var vertexBuffer: MTLBuffer?
    private var splatCount: Int = 0

    // MARK: - Room Wireframe
    private var roomWireframeBuffer: MTLBuffer?
    private var roomWireframeVertexCount: Int = 0
    private var linePipelineState: MTLRenderPipelineState?

    // MARK: - Room Quad Buffers (textured planes)
    private var floorVB: MTLBuffer?
    private var frontVB: MTLBuffer?
    private var leftVB: MTLBuffer?
    private var rightVB: MTLBuffer?
    private var ceilVB: MTLBuffer?

    private var floorVC: Int = 0
    private var frontVC: Int = 0
    private var leftVC: Int = 0
    private var rightVC: Int = 0
    private var ceilVC: Int = 0

    // MARK: - Room Textures
    private var floorTex: MTLTexture?
    private var frontTex: MTLTexture?
    private var leftTex: MTLTexture?
    private var rightTex: MTLTexture?
    private var ceilTex: MTLTexture?

    private var roomQuadPipeline: MTLRenderPipelineState?

    // MARK: - Room Dimensions (should match settings)
    let roomWidth: Float = 4.0
    let roomDepth: Float = 4.5
    let roomHeight: Float = 2.8

    // MARK: - Camera State
    var cameraPosition: SIMD3<Float> = SIMD3(0, 1.4, 4.0)  // In front of room, eye level
    var cameraTarget: SIMD3<Float> = SIMD3(0, 1.4, 0)      // Look at room center
    var cameraUp: SIMD3<Float> = SIMD3(0, 1, 0)

    var fieldOfView: Float = 60.0
    var nearPlane: Float = 0.05
    var farPlane: Float = 15.0

    // MARK: - Orbit Camera
    var orbitDistance: Float = 8.0  // Far enough to see the whole room
    var orbitAngleX: Float = 0.3    // Slight rotation to see room shape
    var orbitAngleY: Float = 0.3    // Looking slightly down at the room
    var cameraTargetOffset: SIMD3<Float> = .zero  // Pan offset for joystick movement

    // MARK: - Background Color
    var backgroundColor: MTLClearColor = MTLClearColor(red: 0.15, green: 0.15, blue: 0.2, alpha: 1.0)

    // MARK: - Initialization
    init?(device: MTLDevice) {
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            print("Failed to create command queue")
            return nil
        }
        self.commandQueue = queue

        super.init()

        setupPipeline()
        setupDebugPipeline()
        setupDepthState()
        setupLinePipeline()
        setupQuadPipeline()
        buildRoomWireframe()
        buildRoomQuads()
    }

    // MARK: - Vertex Descriptor
    private func makeVertexDescriptor() -> MTLVertexDescriptor {
        let vd = MTLVertexDescriptor()

        // attribute(0): positionRadius - float4 at offset 0
        vd.attributes[0].format = .float4
        vd.attributes[0].offset = 0
        vd.attributes[0].bufferIndex = 0

        // attribute(1): colorOpacity - float4 at offset 16
        vd.attributes[1].format = .float4
        vd.attributes[1].offset = MemoryLayout<SIMD4<Float>>.stride  // 16
        vd.attributes[1].bufferIndex = 0

        // Buffer layout - must match SplatVertex stride (32 bytes)
        vd.layouts[0].stride = MemoryLayout<SplatVertex>.stride  // 32
        vd.layouts[0].stepFunction = .perVertex
        vd.layouts[0].stepRate = 1

        return vd
    }

    // MARK: - Pipeline Setup
    private func setupPipeline() {
        guard let library = device.makeDefaultLibrary() else {
            print("Failed to load Metal library")
            return
        }

        guard let vertexFunction = library.makeFunction(name: "splat_vertex"),
              let fragmentFunction = library.makeFunction(name: "splat_fragment_gaussian") else {
            print("Failed to load splat shader functions")
            return
        }

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = makeVertexDescriptor()
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        // Alpha blending
        let colorAttachment = pipelineDescriptor.colorAttachments[0]!
        colorAttachment.isBlendingEnabled = true
        colorAttachment.sourceRGBBlendFactor = .sourceAlpha
        colorAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        colorAttachment.sourceAlphaBlendFactor = .one
        colorAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha

        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float

        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            print("Pipeline created successfully")
        } catch {
            print("Failed to create pipeline state: \(error)")
        }
    }

    private func setupDebugPipeline() {
        guard let library = device.makeDefaultLibrary() else { return }

        guard let vertexFunction = library.makeFunction(name: "splat_vertex_debug"),
              let fragmentFunction = library.makeFunction(name: "splat_fragment_gaussian") else {
            print("Failed to load debug shader functions")
            return
        }

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = makeVertexDescriptor()
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        // Alpha blending
        let colorAttachment = pipelineDescriptor.colorAttachments[0]!
        colorAttachment.isBlendingEnabled = true
        colorAttachment.sourceRGBBlendFactor = .sourceAlpha
        colorAttachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
        colorAttachment.sourceAlphaBlendFactor = .one
        colorAttachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha

        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float

        do {
            debugPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            print("Debug pipeline created successfully")
        } catch {
            print("Failed to create debug pipeline state: \(error)")
        }
    }

    private func setupDepthState() {
        let depthDescriptor = MTLDepthStencilDescriptor()
        depthDescriptor.depthCompareFunction = .lessEqual
        // DEBUG: Disable depth writes so splats overlay everything
        depthDescriptor.isDepthWriteEnabled = false
        depthState = device.makeDepthStencilState(descriptor: depthDescriptor)
        print("DEBUG: Depth state created with depthWrite=false")
    }

    // MARK: - Line Pipeline for Room Wireframe
    private func setupLinePipeline() {
        guard let library = device.makeDefaultLibrary() else {
            print("Failed to load Metal library for line pipeline")
            return
        }

        // Use simple shaders for lines (or create dedicated ones)
        guard let vertexFunction = library.makeFunction(name: "line_vertex"),
              let fragmentFunction = library.makeFunction(name: "line_fragment") else {
            print("Line shader functions not found - wireframe disabled")
            return
        }

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction

        // Simple vertex descriptor for lines: position (float3) + color (float3)
        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float3
        vd.attributes[0].offset = 0
        vd.attributes[0].bufferIndex = 0
        vd.attributes[1].format = .float3
        vd.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        vd.attributes[1].bufferIndex = 0
        vd.layouts[0].stride = MemoryLayout<SIMD3<Float>>.stride * 2
        vd.layouts[0].stepFunction = .perVertex
        pipelineDescriptor.vertexDescriptor = vd

        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float

        do {
            linePipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            print("Line pipeline created successfully")
        } catch {
            print("Failed to create line pipeline: \(error)")
        }
    }

    // MARK: - Room Wireframe
    private func buildRoomWireframe() {
        let hw = roomWidth * 0.5   // half width
        let hd = roomDepth * 0.5   // half depth
        let h = roomHeight

        // 8 corners of the room box
        let c0 = SIMD3<Float>(-hw, 0,  -hd) // floor front left
        let c1 = SIMD3<Float>( hw, 0,  -hd) // floor front right
        let c2 = SIMD3<Float>( hw, 0,   hd) // floor back right
        let c3 = SIMD3<Float>(-hw, 0,   hd) // floor back left
        let c4 = SIMD3<Float>(-hw, h,  -hd) // ceiling front left
        let c5 = SIMD3<Float>( hw, h,  -hd) // ceiling front right
        let c6 = SIMD3<Float>( hw, h,   hd) // ceiling back right
        let c7 = SIMD3<Float>(-hw, h,   hd) // ceiling back left

        // 12 edges as line segments (pairs of vertices)
        let edges: [(SIMD3<Float>, SIMD3<Float>)] = [
            // Floor rectangle
            (c0, c1), (c1, c2), (c2, c3), (c3, c0),
            // Ceiling rectangle
            (c4, c5), (c5, c6), (c6, c7), (c7, c4),
            // Vertical edges
            (c0, c4), (c1, c5), (c2, c6), (c3, c7)
        ]

        // Build vertex buffer: position + color for each vertex
        struct LineVertex {
            var position: SIMD3<Float>
            var color: SIMD3<Float>
        }

        var vertices: [LineVertex] = []
        let lineColor = SIMD3<Float>(0.7, 0.7, 0.7)  // Light grey

        for (a, b) in edges {
            vertices.append(LineVertex(position: a, color: lineColor))
            vertices.append(LineVertex(position: b, color: lineColor))
        }

        roomWireframeVertexCount = vertices.count
        let length = vertices.count * MemoryLayout<LineVertex>.stride
        roomWireframeBuffer = device.makeBuffer(bytes: vertices, length: length, options: [.storageModeShared])
        print("Room wireframe: \(vertices.count) vertices, \(roomWidth)×\(roomDepth)×\(roomHeight)m")
    }

    // MARK: - Room Quads (textured planes)
    private func buildRoomQuads() {
        let hw = roomWidth * 0.5   // half width  → X ∈ [-2, +2]
        let hd = roomDepth * 0.5   // half depth  → Z ∈ [-2.25, +2.25]
        let h = roomHeight         // height      → Y ∈ [0, 2.8]

        // Helper to build a quad from 4 named corners (proper UV orientation)
        // topLeft = (0,0), topRight = (1,0), bottomRight = (1,1), bottomLeft = (0,1)
        @inline(__always)
        func makeQuad(topLeft: SIMD3<Float>, topRight: SIMD3<Float>,
                      bottomRight: SIMD3<Float>, bottomLeft: SIMD3<Float>) -> [RoomQuadVertex] {
            return [
                // Triangle 1: topLeft → topRight → bottomRight
                RoomQuadVertex(position: topLeft, uv: SIMD2<Float>(0, 0)),
                RoomQuadVertex(position: topRight, uv: SIMD2<Float>(1, 0)),
                RoomQuadVertex(position: bottomRight, uv: SIMD2<Float>(1, 1)),
                // Triangle 2: topLeft → bottomRight → bottomLeft
                RoomQuadVertex(position: topLeft, uv: SIMD2<Float>(0, 0)),
                RoomQuadVertex(position: bottomRight, uv: SIMD2<Float>(1, 1)),
                RoomQuadVertex(position: bottomLeft, uv: SIMD2<Float>(0, 1)),
            ]
        }

        // 8 corners in room-space
        // Floor corners (y=0)
        let fl = SIMD3<Float>(-hw, 0, -hd)  // floor front left
        let fr = SIMD3<Float>( hw, 0, -hd)  // floor front right
        let br = SIMD3<Float>( hw, 0,  hd)  // floor back right
        let bl = SIMD3<Float>(-hw, 0,  hd)  // floor back left

        // Ceiling corners (y=h)
        let cfl = SIMD3<Float>(-hw, h, -hd) // ceiling front left
        let cfr = SIMD3<Float>( hw, h, -hd) // ceiling front right
        let cbr = SIMD3<Float>( hw, h,  hd) // ceiling back right
        let cbl = SIMD3<Float>(-hw, h,  hd) // ceiling back left

        // Front wall (XY plane at z=-hd, viewed from inside room at +Z)
        // topLeft=cfl, topRight=cfr, bottomRight=fr, bottomLeft=fl
        let frontVerts = makeQuad(topLeft: cfl, topRight: cfr, bottomRight: fr, bottomLeft: fl)
        frontVC = frontVerts.count
        frontVB = device.makeBuffer(bytes: frontVerts,
                                    length: frontVerts.count * MemoryLayout<RoomQuadVertex>.stride,
                                    options: .storageModeShared)

        // Floor (XZ plane at y=0, viewed from above)
        // "top" = back, "bottom" = front in UV space
        let floorVerts = makeQuad(topLeft: bl, topRight: br, bottomRight: fr, bottomLeft: fl)
        floorVC = floorVerts.count
        floorVB = device.makeBuffer(bytes: floorVerts,
                                    length: floorVerts.count * MemoryLayout<RoomQuadVertex>.stride,
                                    options: .storageModeShared)

        // Ceiling (XZ plane at y=h, viewed from below)
        let ceilVerts = makeQuad(topLeft: cfl, topRight: cfr, bottomRight: cbr, bottomLeft: cbl)
        ceilVC = ceilVerts.count
        ceilVB = device.makeBuffer(bytes: ceilVerts,
                                   length: ceilVerts.count * MemoryLayout<RoomQuadVertex>.stride,
                                   options: .storageModeShared)

        // Left wall (YZ plane at x=-hw, viewed from inside room at +X)
        // topLeft=cbl, topRight=cfl, bottomRight=fl, bottomLeft=bl
        let leftVerts = makeQuad(topLeft: cbl, topRight: cfl, bottomRight: fl, bottomLeft: bl)
        leftVC = leftVerts.count
        leftVB = device.makeBuffer(bytes: leftVerts,
                                   length: leftVerts.count * MemoryLayout<RoomQuadVertex>.stride,
                                   options: .storageModeShared)

        // Right wall (YZ plane at x=+hw, viewed from inside room at -X)
        // topLeft=cfr, topRight=cbr, bottomRight=br, bottomLeft=fr
        let rightVerts = makeQuad(topLeft: cfr, topRight: cbr, bottomRight: br, bottomLeft: fr)
        rightVC = rightVerts.count
        rightVB = device.makeBuffer(bytes: rightVerts,
                                    length: rightVerts.count * MemoryLayout<RoomQuadVertex>.stride,
                                    options: .storageModeShared)

        print("Room quads built: floor=\(floorVC), front=\(frontVC), left=\(leftVC), right=\(rightVC), ceil=\(ceilVC) vertices")
    }

    // MARK: - Quad Pipeline (textured)
    private func setupQuadPipeline() {
        guard let library = device.makeDefaultLibrary() else {
            print("Failed to load Metal library for quad pipeline")
            return
        }

        guard let vertexFunction = library.makeFunction(name: "quad_vertex"),
              let fragmentFunction = library.makeFunction(name: "quad_fragment") else {
            print("Quad shader functions not found - textured room disabled")
            return
        }

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction

        // Vertex descriptor for quads: position (float3) + uv (float2)
        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float3
        vd.attributes[0].offset = 0
        vd.attributes[0].bufferIndex = 0
        vd.attributes[1].format = .float2
        vd.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
        vd.attributes[1].bufferIndex = 0
        vd.layouts[0].stride = MemoryLayout<RoomQuadVertex>.stride
        vd.layouts[0].stepFunction = .perVertex
        pipelineDescriptor.vertexDescriptor = vd

        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float

        do {
            roomQuadPipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            print("Quad pipeline created successfully")
        } catch {
            print("Failed to create quad pipeline: \(error)")
        }
    }

    // MARK: - Update Room Textures
    /// Call this with the generated room textures from RoomBuilder
    func updateRoomTextures(floor: UIImage?, ceiling: UIImage?, front: UIImage?, left: UIImage?, right: UIImage?) {
        let loader = MTKTextureLoader(device: device)

        func tex(_ img: UIImage?) -> MTLTexture? {
            guard let img = img, let cg = img.cgImage else { return nil }
            return try? loader.newTexture(cgImage: cg, options: [.SRGB: false])
        }

        floorTex = tex(floor)
        ceilTex = tex(ceiling)
        frontTex = tex(front)
        leftTex = tex(left)
        rightTex = tex(right)

        print("Room textures updated: floor=\(floorTex != nil), ceil=\(ceilTex != nil), front=\(frontTex != nil), left=\(leftTex != nil), right=\(rightTex != nil)")
    }

    // MARK: - Load Splats
    func loadSplats(_ splats: [Splat]) {
        guard !splats.isEmpty else {
            vertexBuffer = nil
            splatCount = 0
            cloudBounds = nil
            return
        }

        // Create properly-aligned vertex buffer for Metal
        guard let result = splats.makeVertexBuffer(device: device) else {
            print("Failed to create vertex buffer")
            vertexBuffer = nil
            splatCount = 0
            cloudBounds = nil
            return
        }

        vertexBuffer = result.buffer
        splatCount = result.count
        cloudBounds = result.bounds
        didLogCameraForCurrentCloud = false  // Reset so we log camera once for new cloud

        print("Loaded \(splatCount) splats into GPU vertex buffer")
        print("Buffer length: \(result.buffer.length) bytes")
        print("Cloud bounds: averageCenter=\(result.bounds.averageCenter), radius=\(result.bounds.radius)")
    }

    /// Load splats from SHARP GaussianSplat array
    func loadFromSHARP(_ gaussianSplats: [SinglePhotoRoomReconstructor.GaussianSplat], roomSize: Float = 4.0) {
        // fromSHARP now includes semantic filtering (opacity, Y-band, depth) and importance capping
        let splats = [Splat].fromSHARP(gaussianSplats, targetSize: roomSize)
        loadSplats(splats)
    }

    // MARK: - Camera Matrix Helpers
    private func makeViewMatrix() -> simd_float4x4 {
        // Compute camera position from orbit angles
        // Camera orbits around the room center at eye height
        let eyeHeight = roomHeight * 0.5  // Eye level in the room

        let x = orbitDistance * cos(orbitAngleY) * sin(orbitAngleX)
        let y = orbitDistance * sin(orbitAngleY) + eyeHeight
        let z = orbitDistance * cos(orbitAngleY) * cos(orbitAngleX)

        cameraPosition = SIMD3(x, y, z)
        cameraTarget = SIMD3(0, eyeHeight, 0)  // Look at room center

        return lookAt(eye: cameraPosition, center: cameraTarget, up: cameraUp)
    }

    private func makeProjectionMatrix(aspectRatio: Float) -> simd_float4x4 {
        let fovRadians = fieldOfView * .pi / 180.0
        return perspective(fovY: fovRadians, aspect: aspectRatio, near: nearPlane, far: farPlane)
    }

    private func makeViewProjMatrix(aspectRatio: Float) -> simd_float4x4 {
        let view = makeViewMatrix()
        let proj = makeProjectionMatrix(aspectRatio: aspectRatio)
        return proj * view
    }

    // MARK: - Matrix Math
    private func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let z = simd_normalize(eye - center)
        let x = simd_normalize(simd_cross(up, z))
        let y = simd_cross(z, x)

        return simd_float4x4(columns: (
            SIMD4(x.x, y.x, z.x, 0),
            SIMD4(x.y, y.y, z.y, 0),
            SIMD4(x.z, y.z, z.z, 0),
            SIMD4(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1)
        ))
    }

    private func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let tanHalfFov = tan(fovY / 2)
        let m00 = 1 / (aspect * tanHalfFov)
        let m11 = 1 / tanHalfFov
        let m22 = -(far + near) / (far - near)
        let m23: Float = -1
        let m32 = -(2 * far * near) / (far - near)

        return simd_float4x4(columns: (
            SIMD4(m00, 0, 0, 0),
            SIMD4(0, m11, 0, 0),
            SIMD4(0, 0, m22, m23),
            SIMD4(0, 0, m32, 0)
        ))
    }

    // MARK: - Draw
    func draw(in view: MTKView) {
        // Select pipeline based on debug mode
        let activePipeline = debugMode ? debugPipelineState : pipelineState

        guard let pipeline = activePipeline,
              let depthState = depthState,
              let vertexBuffer = vertexBuffer,
              splatCount > 0 else {
            return
        }

        guard let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor else {
            return
        }

        // Set background color
        descriptor.colorAttachments[0].clearColor = backgroundColor

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else {
            return
        }

        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setCullMode(.none)  // DEBUG: no culling

        // Compute camera matrices based on camera mode
        let aspectRatio = Float(view.drawableSize.width / view.drawableSize.height)
        var uniforms: CameraUniforms

        switch cameraMode {
        case .sharpDebug:
            // ORBIT CAMERA: Use orbit angles to rotate around room center
            // Target is the room center (roughly at height/2) plus joystick pan offset
            let target = SIMD3<Float>(0, roomHeight * 0.5, 0) + cameraTargetOffset

            // Compute eye position from orbit angles and distance
            let cosY = cos(orbitAngleY)
            let sinY = sin(orbitAngleY)
            let cosX = cos(orbitAngleX)
            let sinX = sin(orbitAngleX)

            let eye = SIMD3<Float>(
                target.x + orbitDistance * cosY * sinX,
                target.y + orbitDistance * sinY,
                target.z + orbitDistance * cosY * cosX
            )
            let up = SIMD3<Float>(0, 1, 0)

            // FOV and near/far for room-scale viewing
            let fovY: Float = 60.0 * .pi / 180.0
            let nearZ: Float = 0.1
            let farZ: Float = 50.0

            // Log camera info only once per cloud load
            if !didLogCameraForCurrentCloud {
                print("ORBIT CAMERA: target=\(target), eye=\(eye), dist=\(orbitDistance), angleX=\(orbitAngleX), angleY=\(orbitAngleY)")
                didLogCameraForCurrentCloud = true
            }

            uniforms = CameraUniforms(
                viewMatrix: lookAt(eye: eye, center: target, up: up),
                projectionMatrix: perspective(fovY: fovY, aspect: aspectRatio, near: nearZ, far: farZ),
                debugScreenSpace: 0  // Use 3D camera, not screen-space bypass
            )

        case .room:
            // Normal room camera (orbit mode)
            uniforms = CameraUniforms(
                viewMatrix: makeViewMatrix(),
                projectionMatrix: makeProjectionMatrix(aspectRatio: aspectRatio),
                debugScreenSpace: 0
            )
        }

        // Set uniforms for all pipelines
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<CameraUniforms>.stride, index: 1)

        // 1) Draw textured room quads (if textures are loaded)
        if let quadPipeline = roomQuadPipeline {
            encoder.setRenderPipelineState(quadPipeline)

            func drawQuad(vb: MTLBuffer?, vc: Int, tex: MTLTexture?) {
                guard let vb = vb, vc > 0, let tex = tex else { return }
                encoder.setVertexBuffer(vb, offset: 0, index: 0)
                encoder.setFragmentTexture(tex, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: vc)
            }

            drawQuad(vb: floorVB, vc: floorVC, tex: floorTex)
            drawQuad(vb: ceilVB, vc: ceilVC, tex: ceilTex)
            drawQuad(vb: frontVB, vc: frontVC, tex: frontTex)
            drawQuad(vb: leftVB, vc: leftVC, tex: leftTex)
            drawQuad(vb: rightVB, vc: rightVC, tex: rightTex)
        }

        // 2) Draw splats as points (on top of room geometry)
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.drawPrimitives(
            type: .point,
            vertexStart: 0,
            vertexCount: splatCount
        )

        // Draw room wireframe (if available)
        if let linePipeline = linePipelineState,
           let wireframeBuffer = roomWireframeBuffer,
           roomWireframeVertexCount > 0 {
            encoder.setRenderPipelineState(linePipeline)
            encoder.setVertexBuffer(wireframeBuffer, offset: 0, index: 0)
            // Uniforms are already set at buffer index 1
            encoder.drawPrimitives(
                type: .line,
                vertexStart: 0,
                vertexCount: roomWireframeVertexCount
            )
        }

        encoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    // MARK: - Camera Controls
    func orbit(deltaX: Float, deltaY: Float) {
        orbitAngleX += deltaX * 0.01
        orbitAngleY = max(-Float.pi/2 + 0.1, min(Float.pi/2 - 0.1, orbitAngleY + deltaY * 0.01))
    }

    func zoom(delta: Float) {
        orbitDistance = max(1.0, min(20.0, orbitDistance - delta * 0.1))
    }

    /// Pan camera target in world XZ plane (joystick movement)
    /// deltaX: right/left, deltaZ: forward/back (relative to camera facing direction)
    func pan(deltaX: Float, deltaZ: Float) {
        // Get camera forward direction projected onto XZ plane
        let forwardX = sin(orbitAngleX)
        let forwardZ = cos(orbitAngleX)

        // Right vector (perpendicular to forward in XZ plane)
        let rightX = cos(orbitAngleX)
        let rightZ = -sin(orbitAngleX)

        // Move target based on joystick input
        let speed: Float = 0.05
        cameraTargetOffset.x += (rightX * deltaX + forwardX * deltaZ) * speed
        cameraTargetOffset.z += (rightZ * deltaX + forwardZ * deltaZ) * speed

        // Clamp to reasonable bounds (stay within room area)
        let maxOffset: Float = 5.0
        cameraTargetOffset.x = max(-maxOffset, min(maxOffset, cameraTargetOffset.x))
        cameraTargetOffset.z = max(-maxOffset, min(maxOffset, cameraTargetOffset.z))
    }

    /// Reset camera to default position
    func resetCamera() {
        orbitDistance = 8.0
        orbitAngleX = 0.3
        orbitAngleY = 0.3
        cameraTargetOffset = .zero
    }
}

// MARK: - MTKViewDelegate
extension SplatRenderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle resize if needed
    }

    // Note: draw(in:) is already implemented above
}
