import SwiftUI

struct RealityKitLoadingDebugView: View {
    let model: USDZModel
    @State private var loadingSteps: [String] = []
    @State private var currentStep = "Initializing..."
    
    var body: some View {
        VStack {
            Text("Loading Debug")
                .font(.title2)
                .fontWeight(.semibold)
            
            Text("Model: \(model.fileName)")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Divider()
            
            Text("Current Step:")
                .font(.headline)
            Text(currentStep)
                .font(.body)
                .foregroundColor(.blue)
                .padding(.bottom)
            
            Text("Completed Steps:")
                .font(.headline)
            
            ScrollView {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(loadingSteps, id: \.self) { step in
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                            Text(step)
                                .font(.caption)
                            Spacer()
                        }
                    }
                }
            }
            .frame(maxHeight: 200)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 4)
        .onAppear {
            simulateLoadingSteps()
        }
    }
    
    private func simulateLoadingSteps() {
        let steps = [
            "ARView initialized",
            "Camera configuration set",
            "World anchor created",
            "Lighting configured",
            "Model file located",
            "Model loading started",
            "Model entity created",
            "Boundary manager setup",
            "Camera positioning",
            "Scene ready"
        ]
        
        for (index, step) in steps.enumerated() {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(index) * 0.5) {
                withAnimation {
                    if index < steps.count - 1 {
                        loadingSteps.append(step)
                        currentStep = steps[index + 1]
                    } else {
                        loadingSteps.append(step)
                        currentStep = "Complete!"
                    }
                }
            }
        }
    }
}

#Preview {
    RealityKitLoadingDebugView(model: USDZModel(name: "Test", fileName: "test.usdz"))
}