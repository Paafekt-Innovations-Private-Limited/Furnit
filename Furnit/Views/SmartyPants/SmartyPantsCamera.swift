// SmartyPantsCamera.swift
// Camera capture management for SmartyPants detection

import Foundation
import AVFoundation
import UIKit

// MARK: - Camera Delegate Protocol

protocol SmartyPantsCameraDelegate: AnyObject {
    func camera(_ camera: SmartyPantsCamera, didOutput sampleBuffer: CMSampleBuffer)
}

// MARK: - Camera Manager

final class SmartyPantsCamera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    // MARK: Properties

    weak var delegate: SmartyPantsCameraDelegate?

    let captureSession = AVCaptureSession()
    let videoOutput = AVCaptureVideoDataOutput()
    let previewLayer = AVCaptureVideoPreviewLayer()

    private let sampleQueue = DispatchQueue(label: "com.smartypants.sampleQueue", qos: .userInteractive)

    var isRunning: Bool {
        return captureSession.isRunning
    }

    // MARK: - Initialization

    override init() {
        super.init()
        previewLayer.session = captureSession
        previewLayer.videoGravity = .resizeAspectFill
    }

    // MARK: - Setup

    func setup() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720
        captureSession.inputs.forEach { captureSession.removeInput($0) }
        captureSession.outputs.forEach { captureSession.removeOutput($0) }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            captureSession.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(input) {
                captureSession.addInput(input)
            }
            videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            videoOutput.alwaysDiscardsLateVideoFrames = true
            if captureSession.canAddOutput(videoOutput) {
                captureSession.addOutput(videoOutput)
            }
            if let conn = videoOutput.connection(with: .video) {
                conn.videoRotationAngle = 90
            }
            captureSession.commitConfiguration()
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession.startRunning()
            }
        } catch {
            captureSession.commitConfiguration()
        }
    }

    // MARK: - Start/Stop

    func start() {
        if !captureSession.isRunning {
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession.startRunning()
            }
        }
    }

    func stop() {
        DispatchQueue.global(qos: .userInitiated).async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
    }

    // MARK: - Permission

    func requestPermissionAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            start()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    self.start()
                }
            }
        default:
            break
        }
    }

    // MARK: - Delegate Control

    func pauseCapture() {
        videoOutput.setSampleBufferDelegate(nil, queue: nil)
    }

    func resumeCapture() {
        videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        delegate?.camera(self, didOutput: sampleBuffer)
    }
}
