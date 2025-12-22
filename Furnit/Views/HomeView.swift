import SwiftUI

struct HomeView: View {
    @StateObject private var modelManager = USDZModelManager()
    @Environment(\.appState) private var appState
    @State private var showingSettings = false
    @State private var debugInfo: String = "Initializing..."
    
    init() {
        logDebug("🔵 HomeView init called")
    }
    
    var body: some View {
        NavigationStack {
            contentView
                .navigationTitle("3D Room Models")
                .navigationBarTitleDisplayMode(.large)
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button {
                            showingSettings = true
                        } label: {
                            Image(systemName: "gearshape")
                                .font(.title3)
                        }
                        .accessibilityLabel("Settings")
                    }
                }
                .sheet(isPresented: $showingSettings) {
                    SettingsView()
                }
        }
        .onAppear {
            logDebug("🔵 HomeView body called")
            logDebug("🔵 modelManager.models.count: \(modelManager.models.count)")
            logDebug("🟢 ============ HomeView onAppear ============")
            checkEverything()
        }
        .onChange(of: modelManager.models) { _, newValue in
            logDebug("🟡 Models changed! New count: \(newValue.count)")
            debugInfo = "Models changed: \(newValue.count) models"
        }
    }
    
    private var contentView: some View {
        VStack {
            debugPanel
            
            if modelManager.models.isEmpty {
                emptyStateView
            } else {
                modelsListView
            }
        }
    }
    
    private var debugPanel: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Debug Info:")
                .font(.caption.bold())
            Text(debugInfo)
                .font(.caption)
                .foregroundColor(.red)
            Text("Models count: \(modelManager.models.count)")
                .font(.caption)
                .foregroundColor(.orange)
            Text("Model manager exists: YES")
                .font(.caption)
                .foregroundColor(.blue)
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
        .padding(.horizontal)
    }
    
    private var emptyStateView: some View {
        VStack {
            ContentUnavailableView(
                "No 3D Models Found",
                systemImage: "cube.transparent",
                description: Text("Check console for debug output")
            )
            
            VStack(spacing: 10) {
                Button("Test: Check Bundle Resources") {
                    checkBundleResources()
                }
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)
                
                Button("Test: Check Model Manager State") {
                    checkModelManagerState()
                }
                .padding()
                .background(Color.green)
                .foregroundColor(.white)
                .cornerRadius(8)
                
                Button("Test: Try Manual Model Creation") {
                    tryManualModelCreation()
                }
                .padding()
                .background(Color.purple)
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .padding()
        }
    }
    
    private var modelsListView: some View {
        VStack {
            Text("Models found! Count: \(modelManager.models.count)")
                .foregroundColor(.green)
                .padding()
            
            List {
                ForEach(Array(modelManager.models.enumerated()), id: \.offset) { index, model in
                    modelRow(for: model, at: index)
                }
            }
            .listStyle(PlainListStyle())
        }
    }
    
    private func modelRow(for model: USDZModel, at index: Int) -> some View {
        let _ = logDebug("🔵 Creating row for model \(index): \(model.displayName)")
        
        return Group {
            if let modelURL = getModelURL(for: model) {
                let _ = logDebug("✅ Found URL for model: \(model.displayName)")
                NavigationLink(destination: ModelViewerView(model: model)) {
                    ModelRowView(model: model)
                }
            } else {
                let _ = logDebug("❌ No URL found for model: \(model.displayName)")
                ModelRowView(model: model)
                    .opacity(0.5)
                    .overlay(
                        Text("URL not found")
                            .font(.caption)
                            .foregroundColor(.red)
                    )
            }
        }
    }
    
    private func checkEverything() {
        logDebug("🟢 Starting comprehensive check...")
        
        // 1. Check model manager
        logDebug("🟢 1. Model Manager Check:")
        logDebug("   - Manager exists: true")
        logDebug("   - Models array: \(modelManager.models)")
        logDebug("   - Models count: \(modelManager.models.count)")
        
        if !modelManager.models.isEmpty {
            logDebug("   - Models details:")
            for (index, model) in modelManager.models.enumerated() {
                logDebug("     [\(index)] displayName: \(model.displayName)")
            }
        }
        
        // 2. Check bundle for USDZ files
        logDebug("🟢 2. Bundle Resources Check:")
        checkBundleResources()
        
        // 3. Check specific paths
        logDebug("🟢 3. Specific Paths Check:")
        let testPaths = [
            "room.usdz",
            "Room.usdz",
            "model.usdz",
            "furniture.usdz",
            "chair.usdz",
            "sofa.usdz",
            "table.usdz",
            "test.usdz",
            "demo.usdz"
        ]
        
        for path in testPaths {
            if let url = Bundle.main.url(forResource: path.replacingOccurrences(of: ".usdz", with: ""), withExtension: "usdz") {
                logDebug("   ✅ Found: \(path) at \(url)")
            } else {
                logDebug("   ❌ Not found: \(path)")
            }
        }
        
        // 4. Check Documents directory
        logDebug("🟢 4. Documents Directory Check:")
        if let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            do {
                let items = try FileManager.default.contentsOfDirectory(at: documentsPath, includingPropertiesForKeys: nil)
                let usdzFiles = items.filter { $0.pathExtension == "usdz" }
                logDebug("   USDZ files in Documents: \(usdzFiles.count)")
                for file in usdzFiles {
                    logDebug("   - \(file.lastPathComponent)")
                }
            } catch {
                logDebug("   Error reading Documents: \(error)")
            }
        }
        
        debugInfo = "Check complete: \(modelManager.models.count) models"
    }
    
    private func checkBundleResources() {
        logDebug("🔍 Checking Bundle Resources...")
        
        if let resourcePath = Bundle.main.resourcePath {
            do {
                let items = try FileManager.default.contentsOfDirectory(atPath: resourcePath)
                let usdzFiles = items.filter { $0.hasSuffix(".usdz") || $0.hasSuffix(".USDZ") }
                logDebug("   Total items in bundle: \(items.count)")
                logDebug("   USDZ files found: \(usdzFiles.count)")
                for file in usdzFiles {
                    logDebug("   - \(file)")
                }
                
                debugInfo = "Bundle: \(usdzFiles.count) USDZ files"
            } catch {
                logDebug("   ❌ Error reading bundle: \(error)")
                debugInfo = "Bundle error: \(error.localizedDescription)"
            }
        } else {
            logDebug("   ❌ No resource path found")
            debugInfo = "No resource path"
        }
    }
    
    private func checkModelManagerState() {
        logDebug("🔍 Checking Model Manager State...")
        logDebug("   Manager: \(String(describing: modelManager))")
        logDebug("   Models: \(modelManager.models)")
        
        let count = modelManager.models.count
        logDebug("   Forced count check: \(count)")
        
        debugInfo = "Manager state: \(count) models"
    }
    
    private func tryManualModelCreation() {
        logDebug("🔍 Trying Manual Model Creation...")
        
        if let resourcePath = Bundle.main.resourcePath {
            do {
                let items = try FileManager.default.contentsOfDirectory(atPath: resourcePath)
                let usdzFiles = items.filter { $0.hasSuffix(".usdz") }
                
                if let firstUSDZ = usdzFiles.first {
                    logDebug("   Found USDZ: \(firstUSDZ)")
                    if let url = Bundle.main.url(forResource: firstUSDZ.replacingOccurrences(of: ".usdz", with: ""), withExtension: "usdz") {
                        logDebug("   ✅ Successfully created URL: \(url)")
                        debugInfo = "Manual: Found \(firstUSDZ)"
                    }
                } else {
                    logDebug("   ❌ No USDZ files to test")
                    debugInfo = "Manual: No USDZ files"
                }
            } catch {
                logDebug("   ❌ Error: \(error)")
                debugInfo = "Manual error: \(error.localizedDescription)"
            }
        }
    }
    
    private func getModelURL(for model: USDZModel) -> URL? {
        logDebug("🔵 getModelURL called for: \(model.displayName)")
        
        if let url = Bundle.main.url(forResource: model.displayName, withExtension: "usdz") {
            logDebug("   ✅ Found with .usdz extension")
            return url
        }
        
        if let url = Bundle.main.url(forResource: model.displayName, withExtension: nil) {
            logDebug("   ✅ Found without extension")
            return url
        }
        
        if let url = Bundle.main.url(forResource: model.displayName.lowercased(), withExtension: "usdz") {
            logDebug("   ✅ Found with lowercase")
            return url
        }
        
        logDebug("   ❌ No URL found for model: \(model.displayName)")
        return nil
    }
}

struct ModelRowView: View {
    let model: USDZModel
    
    init(model: USDZModel) {
        self.model = model
        logDebug("🔵 ModelRowView init for: \(model.displayName)")
    }
    
    var body: some View {
        HStack {
            Image(systemName: "cube.fill")
                .foregroundColor(.blue)
                .font(.title2)
                .frame(width: 40, height: 40)
                .background(Color.blue.opacity(0.1))
                .cornerRadius(8)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(model.displayName)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Text("3D Room Model")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Image(systemName: "chevron.right")
                .foregroundColor(.secondary)
                .font(.caption)
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    HomeView()
}
