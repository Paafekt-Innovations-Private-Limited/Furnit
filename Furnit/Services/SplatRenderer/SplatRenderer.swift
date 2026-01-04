import Foundation
import Metal
import MetalKit
import simd

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
        buildRoomWireframe()
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
            // Target is the room center (roughly at height/2)
            let target = SIMD3<Float>(0, roomHeight * 0.5, 0)

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

        // Bind vertex buffer and uniforms
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<CameraUniforms>.stride, index: 1)

        // Don't log every frame - splat count is logged once when loaded

        // Draw splats as points
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
}

// MARK: - MTKViewDelegate
extension SplatRenderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Handle resize if needed
    }

    // Note: draw(in:) is already implemented above
}
