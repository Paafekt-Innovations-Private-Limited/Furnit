import SwiftUI

struct HomeView: View {
    @StateObject private var modelManager = USDZModelManager()
    @Environment(\.appState) private var appState
    @State private var showingSettings = false
    @State private var debugInfo: String = "Initializing..."
    
    init() {
        print("🔵 HomeView init called")
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
            print("🔵 HomeView body called")
            print("🔵 modelManager.models.count: \(modelManager.models.count)")
            print("🟢 ============ HomeView onAppear ============")
            checkEverything()
        }
        .onChange(of: modelManager.models) { _, newValue in
            print("🟡 Models changed! New count: \(newValue.count)")
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
        let _ = print("🔵 Creating row for model \(index): \(model.displayName)")
        
        return Group {
            if let modelURL = getModelURL(for: model) {
                let _ = print("✅ Found URL for model: \(model.displayName)")
                NavigationLink(destination: ModelViewerView(model: model)) {
                    ModelRowView(model: model)
                }
            } else {
                let _ = print("❌ No URL found for model: \(model.displayName)")
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
        print("🟢 Starting comprehensive check...")
        
        // 1. Check model manager
        print("🟢 1. Model Manager Check:")
        print("   - Manager exists: true")
        print("   - Models array: \(modelManager.models)")
        print("   - Models count: \(modelManager.models.count)")
        
        if !modelManager.models.isEmpty {
            print("   - Models details:")
            for (index, model) in modelManager.models.enumerated() {
                print("     [\(index)] displayName: \(model.displayName)")
            }
        }
        
        // 2. Check bundle for USDZ files
        print("🟢 2. Bundle Resources Check:")
        checkBundleResources()
        
        // 3. Check specific paths
        print("🟢 3. Specific Paths Check:")
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
                print("   ✅ Found: \(path) at \(url)")
            } else {
                print("   ❌ Not found: \(path)")
            }
        }
        
        // 4. Check Documents directory
        print("🟢 4. Documents Directory Check:")
        if let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            do {
                let items = try FileManager.default.contentsOfDirectory(at: documentsPath, includingPropertiesForKeys: nil)
                let usdzFiles = items.filter { $0.pathExtension == "usdz" }
                print("   USDZ files in Documents: \(usdzFiles.count)")
                for file in usdzFiles {
                    print("   - \(file.lastPathComponent)")
                }
            } catch {
                print("   Error reading Documents: \(error)")
            }
        }
        
        debugInfo = "Check complete: \(modelManager.models.count) models"
    }
    
    private func checkBundleResources() {
        print("🔍 Checking Bundle Resources...")
        
        if let resourcePath = Bundle.main.resourcePath {
            do {
                let items = try FileManager.default.contentsOfDirectory(atPath: resourcePath)
                let usdzFiles = items.filter { $0.hasSuffix(".usdz") || $0.hasSuffix(".USDZ") }
                print("   Total items in bundle: \(items.count)")
                print("   USDZ files found: \(usdzFiles.count)")
                for file in usdzFiles {
                    print("   - \(file)")
                }
                
                debugInfo = "Bundle: \(usdzFiles.count) USDZ files"
            } catch {
                print("   ❌ Error reading bundle: \(error)")
                debugInfo = "Bundle error: \(error.localizedDescription)"
            }
        } else {
            print("   ❌ No resource path found")
            debugInfo = "No resource path"
        }
    }
    
    private func checkModelManagerState() {
        print("🔍 Checking Model Manager State...")
        print("   Manager: \(String(describing: modelManager))")
        print("   Models: \(modelManager.models)")
        
        let count = modelManager.models.count
        print("   Forced count check: \(count)")
        
        debugInfo = "Manager state: \(count) models"
    }
    
    private func tryManualModelCreation() {
        print("🔍 Trying Manual Model Creation...")
        
        if let resourcePath = Bundle.main.resourcePath {
            do {
                let items = try FileManager.default.contentsOfDirectory(atPath: resourcePath)
                let usdzFiles = items.filter { $0.hasSuffix(".usdz") }
                
                if let firstUSDZ = usdzFiles.first {
                    print("   Found USDZ: \(firstUSDZ)")
                    if let url = Bundle.main.url(forResource: firstUSDZ.replacingOccurrences(of: ".usdz", with: ""), withExtension: "usdz") {
                        print("   ✅ Successfully created URL: \(url)")
                        debugInfo = "Manual: Found \(firstUSDZ)"
                    }
                } else {
                    print("   ❌ No USDZ files to test")
                    debugInfo = "Manual: No USDZ files"
                }
            } catch {
                print("   ❌ Error: \(error)")
                debugInfo = "Manual error: \(error.localizedDescription)"
            }
        }
    }
    
    private func getModelURL(for model: USDZModel) -> URL? {
        print("🔵 getModelURL called for: \(model.displayName)")
        
        if let url = Bundle.main.url(forResource: model.displayName, withExtension: "usdz") {
            print("   ✅ Found with .usdz extension")
            return url
        }
        
        if let url = Bundle.main.url(forResource: model.displayName, withExtension: nil) {
            print("   ✅ Found without extension")
            return url
        }
        
        if let url = Bundle.main.url(forResource: model.displayName.lowercased(), withExtension: "usdz") {
            print("   ✅ Found with lowercase")
            return url
        }
        
        print("   ❌ No URL found for model: \(model.displayName)")
        return nil
    }
}

struct ModelRowView: View {
    let model: USDZModel
    
    init(model: USDZModel) {
        self.model = model
        print("🔵 ModelRowView init for: \(model.displayName)")
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
