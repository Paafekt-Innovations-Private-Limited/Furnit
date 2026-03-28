platform :ios, '15.0'

target 'Furnit' do
  use_frameworks!

  # ONNX Runtime for Objective-C (CPU/CoreML) – used only by the new blue icon path.
  # After adding this Podfile, run:
  #   cd Furnit
  #   pod install
  # Then open Furnit.xcworkspace instead of Furnit.xcodeproj.
  pod 'onnxruntime-objc'
end

post_install do |installer|
  installer.pods_project.targets.each do |target|
    target.build_configurations.each do |config|
      config.build_settings['CLANG_WARN_MISSING_SEARCH_PATHS'] = 'NO'
    end
  end
end

