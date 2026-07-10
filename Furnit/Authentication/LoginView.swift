import SwiftUI

// MARK: - Country Model
struct Country: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let code: String
    let flag: String
    let dialCode: String

    static let allCountries: [Country] = [
        Country(name: "India", code: "IN", flag: "🇮🇳", dialCode: "+91"),
        Country(name: "United States", code: "US", flag: "🇺🇸", dialCode: "+1"),
        Country(name: "United Kingdom", code: "GB", flag: "🇬🇧", dialCode: "+44"),
        Country(name: "Canada", code: "CA", flag: "🇨🇦", dialCode: "+1"),
        Country(name: "Australia", code: "AU", flag: "🇦🇺", dialCode: "+61"),
        Country(name: "Germany", code: "DE", flag: "🇩🇪", dialCode: "+49"),
        Country(name: "France", code: "FR", flag: "🇫🇷", dialCode: "+33"),
        Country(name: "Italy", code: "IT", flag: "🇮🇹", dialCode: "+39"),
        Country(name: "Spain", code: "ES", flag: "🇪🇸", dialCode: "+34"),
        Country(name: "Brazil", code: "BR", flag: "🇧🇷", dialCode: "+55"),
        Country(name: "Mexico", code: "MX", flag: "🇲🇽", dialCode: "+52"),
        Country(name: "Japan", code: "JP", flag: "🇯🇵", dialCode: "+81"),
        Country(name: "South Korea", code: "KR", flag: "🇰🇷", dialCode: "+82"),
        Country(name: "China", code: "CN", flag: "🇨🇳", dialCode: "+86"),
        Country(name: "Singapore", code: "SG", flag: "🇸🇬", dialCode: "+65"),
        Country(name: "Malaysia", code: "MY", flag: "🇲🇾", dialCode: "+60"),
        Country(name: "Indonesia", code: "ID", flag: "🇮🇩", dialCode: "+62"),
        Country(name: "Thailand", code: "TH", flag: "🇹🇭", dialCode: "+66"),
        Country(name: "Vietnam", code: "VN", flag: "🇻🇳", dialCode: "+84"),
        Country(name: "Philippines", code: "PH", flag: "🇵🇭", dialCode: "+63"),
        Country(name: "Pakistan", code: "PK", flag: "🇵🇰", dialCode: "+92"),
        Country(name: "Bangladesh", code: "BD", flag: "🇧🇩", dialCode: "+880"),
        Country(name: "Sri Lanka", code: "LK", flag: "🇱🇰", dialCode: "+94"),
        Country(name: "Nepal", code: "NP", flag: "🇳🇵", dialCode: "+977"),
        Country(name: "United Arab Emirates", code: "AE", flag: "🇦🇪", dialCode: "+971"),
        Country(name: "Saudi Arabia", code: "SA", flag: "🇸🇦", dialCode: "+966"),
        Country(name: "Qatar", code: "QA", flag: "🇶🇦", dialCode: "+974"),
        Country(name: "Kuwait", code: "KW", flag: "🇰🇼", dialCode: "+965"),
        Country(name: "Oman", code: "OM", flag: "🇴🇲", dialCode: "+968"),
        Country(name: "Bahrain", code: "BH", flag: "🇧🇭", dialCode: "+973"),
        Country(name: "South Africa", code: "ZA", flag: "🇿🇦", dialCode: "+27"),
        Country(name: "Nigeria", code: "NG", flag: "🇳🇬", dialCode: "+234"),
        Country(name: "Kenya", code: "KE", flag: "🇰🇪", dialCode: "+254"),
        Country(name: "Egypt", code: "EG", flag: "🇪🇬", dialCode: "+20"),
        Country(name: "Russia", code: "RU", flag: "🇷🇺", dialCode: "+7"),
        Country(name: "Netherlands", code: "NL", flag: "🇳🇱", dialCode: "+31"),
        Country(name: "Belgium", code: "BE", flag: "🇧🇪", dialCode: "+32"),
        Country(name: "Switzerland", code: "CH", flag: "🇨🇭", dialCode: "+41"),
        Country(name: "Austria", code: "AT", flag: "🇦🇹", dialCode: "+43"),
        Country(name: "Sweden", code: "SE", flag: "🇸🇪", dialCode: "+46"),
        Country(name: "Norway", code: "NO", flag: "🇳🇴", dialCode: "+47"),
        Country(name: "Denmark", code: "DK", flag: "🇩🇰", dialCode: "+45"),
        Country(name: "Finland", code: "FI", flag: "🇫🇮", dialCode: "+358"),
        Country(name: "Ireland", code: "IE", flag: "🇮🇪", dialCode: "+353"),
        Country(name: "Portugal", code: "PT", flag: "🇵🇹", dialCode: "+351"),
        Country(name: "Greece", code: "GR", flag: "🇬🇷", dialCode: "+30"),
        Country(name: "Turkey", code: "TR", flag: "🇹🇷", dialCode: "+90"),
        Country(name: "Poland", code: "PL", flag: "🇵🇱", dialCode: "+48"),
        Country(name: "New Zealand", code: "NZ", flag: "🇳🇿", dialCode: "+64"),
        Country(name: "Argentina", code: "AR", flag: "🇦🇷", dialCode: "+54"),
        Country(name: "Chile", code: "CL", flag: "🇨🇱", dialCode: "+56"),
        Country(name: "Colombia", code: "CO", flag: "🇨🇴", dialCode: "+57"),
        Country(name: "Peru", code: "PE", flag: "🇵🇪", dialCode: "+51"),
        Country(name: "Israel", code: "IL", flag: "🇮🇱", dialCode: "+972"),
    ].sorted { $0.name < $1.name }

    static let defaultCountry = allCountries.first { $0.code == "IN" } ?? allCountries[0]
    
    // ✅ Detect country based on device locale
    static func detectCountryFromLocale() -> Country {
        // Get the device's region code
        if let regionCode = Locale.current.region?.identifier {
            logDebug("🌍 [Country] Detected region code: \(regionCode)")
            
            // Find matching country in our list
            if let country = allCountries.first(where: { $0.code == regionCode }) {
                logDebug("✅ [Country] Auto-selected: \(country.name) (\(country.dialCode))")
                return country
            } else {
                logDebug("⚠️ [Country] Region '\(regionCode)' not in country list, using default")
            }
        } else {
            logDebug("⚠️ [Country] Could not detect region, using default")
        }
        
        return defaultCountry
    }
}

// MARK: - Country Picker View
struct CountryPickerView: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var selectedCountry: Country
    @State private var searchText = ""

    var filteredCountries: [Country] {
        if searchText.isEmpty {
            return Country.allCountries
        }
        return Country.allCountries.filter {
            $0.name.localizedCaseInsensitiveContains(searchText) ||
            $0.dialCode.contains(searchText) ||
            $0.code.localizedCaseInsensitiveContains(searchText)
        }
    }

    var body: some View {
        NavigationStack {
            List {
                ForEach(filteredCountries) { country in
                    Button(action: {
                        selectedCountry = country
                        dismiss()
                    }) {
                        HStack(spacing: 12) {
                            Text(country.flag)
                                .font(.title2)

                            Text(country.name)
                                .foregroundColor(.primary)

                            Spacer()

                            Text(country.dialCode)
                                .foregroundColor(.secondary)

                            if country.id == selectedCountry.id {
                                Image(systemName: "checkmark")
                                    .foregroundStyle(Theme.Palette.accent)
                            }
                        }
                    }
                }
            }
            .searchable(text: $searchText, prompt: L10n.Country.searchPlaceholder)
            .navigationTitle(L10n.Country.selectTitle)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(L10n.Common.cancel) {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Login View
struct LoginView: View {
    @EnvironmentObject var authManager: AuthenticationManager
    @State private var name = ""
    @State private var phoneNumber = ""
    @State private var selectedCountry = Country.detectCountryFromLocale()  // ✅ Auto-detect country
    @State private var showCountryPicker = false
    @State private var showOTPView = false
    @State private var isLoading = false
    @State private var showError = false
    @State private var errorMessage = ""
    @FocusState private var nameFieldFocused: Bool
    @FocusState private var phoneFieldFocused: Bool

    var body: some View {
        NavigationStack {
            ZStack {
                Theme.Palette.background
                    .ignoresSafeArea()
                    .onTapGesture {
                        nameFieldFocused = false
                        phoneFieldFocused = false
                    }

                VStack(spacing: 0) {
                    Spacer(minLength: Theme.Space.xxl)

                    // Premium Paafekt mark — centered above the sign-in card
                    VStack(spacing: Theme.Space.md) {
                        Image("PaafektMark")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 96, height: 96)
                            .accessibilityLabel(L10n.App.name)

                        Text(L10n.App.name)
                            .font(Theme.Typo.title())
                            .foregroundStyle(Theme.Palette.textPrimary)

                        Text(L10n.App.tagline)
                            .font(Theme.Typo.body())
                            .foregroundStyle(Theme.Palette.textSecondary)
                    }
                    .multilineTextAlignment(.center)
                    .padding(.bottom, Theme.Space.xl)

                    // Login Form
                    VStack(spacing: 20) {
                        // Name Field
                        VStack(alignment: .leading, spacing: 8) {
                            Label(L10n.Login.yourName, systemImage: "person.fill")
                                .font(Theme.Typo.caption())
                                .foregroundStyle(Theme.Palette.textSecondary)

                            TextField(L10n.Login.enterName, text: $name)
                                .textFieldStyle(CustomTextFieldStyle())
                                .focused($nameFieldFocused)
                                .submitLabel(.next)
                                .onSubmit {
                                    phoneFieldFocused = true
                                }
                                .disabled(isLoading)
                        }

                        // Phone Field with Country Code
                        VStack(alignment: .leading, spacing: 8) {
                            Label(L10n.Login.phoneNumber, systemImage: "phone.fill")
                                .font(Theme.Typo.caption())
                                .foregroundStyle(Theme.Palette.textSecondary)

                            HStack(spacing: 8) {
                                // Country Code Button
                                Button(action: {
                                    showCountryPicker = true
                                }) {
                                    HStack(spacing: 4) {
                                        Text(selectedCountry.flag)
                                            .font(.title3)
                                        Text(selectedCountry.dialCode)
                                            .font(.subheadline)
                                            .fontWeight(.medium)
                                        Image(systemName: "chevron.down")
                                            .font(.caption2)
                                    }
                                    .foregroundStyle(Theme.Palette.textPrimary)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 14)
                                    .background(Theme.Palette.surfaceHi)
                                    .cornerRadius(Theme.Radius.control)
                                }
                                .disabled(isLoading)

                                // Phone Number Field
                                TextField(L10n.Login.phonePlaceholder, text: $phoneNumber)
                                    .keyboardType(.phonePad)
                                    .focused($phoneFieldFocused)
                                    .padding()
                                    .background(Theme.Palette.surfaceHi)
                                    .cornerRadius(Theme.Radius.control)
                                    .foregroundStyle(Theme.Palette.textPrimary)
                                    .onChange(of: phoneNumber) { _, newValue in
                                        phoneNumber = formatPhoneInput(newValue)
                                    }
                                    .disabled(isLoading)
                            }
                        }

                        // Send OTP Button
                        Button(action: sendOTP) {
                            HStack {
                                if isLoading {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: Theme.Palette.accentText))
                                        .scaleEffect(0.8)
                                } else {
                                    Image(systemName: "paperplane.fill")
                                }
                                Text(isLoading ? L10n.Login.sending : L10n.Login.sendOTP)
                            }
                        }
                        .buttonStyle(PrimaryButtonStyle())
                        .disabled(!isValidInput || isLoading)
                        .opacity(isValidInput ? 1.0 : 0.7)
                        .animation(.easeInOut(duration: 0.2), value: isValidInput)

                        // Hint
                        Text(L10n.Login.otpHint)
                            .font(Theme.Typo.caption())
                            .foregroundStyle(Theme.Palette.textSecondary)
                    }
                    .padding(Theme.Space.xl)
                    .paafektCardSurface()
                    .padding(.horizontal, Theme.Space.lg)

                    Spacer(minLength: Theme.Space.xxl)
                }
            }
            .navigationDestination(isPresented: $showOTPView) {
                OTPVerificationView(
                    name: name,
                    phoneNumber: fullPhoneNumber,
                    authManager: authManager
                )
            }
            .sheet(isPresented: $showCountryPicker) {
                CountryPickerView(selectedCountry: $selectedCountry)
            }
            .alert(L10n.Common.error, isPresented: $showError) {
                Button(L10n.Common.ok, role: .cancel) { }
            } message: {
                Text(errorMessage)
            }
        }
    }

    private var fullPhoneNumber: String {
        let cleanNumber = phoneNumber.filter { $0.isNumber }
        return "\(selectedCountry.dialCode)\(cleanNumber)"
    }

    private var isValidInput: Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        phoneNumber.filter { $0.isNumber }.count >= 10
    }

    private func sendOTP() {
        guard isValidInput else {
            errorMessage = L10n.Login.validationError
            showError = true
            return
        }

        isLoading = true

        // Send OTP via Firebase with full phone number
        authManager.sendOTP(to: fullPhoneNumber) { success, error in
            isLoading = false

            if success {
                showOTPView = true
            } else {
                errorMessage = error ?? L10n.Login.sendFailed
                showError = true
            }
        }
    }

    private func formatPhoneInput(_ input: String) -> String {
        let cleaned = input.filter { $0.isNumber }
        return String(cleaned.prefix(10))
    }
}

// Custom TextField Style
struct CustomTextFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .padding()
            .background(Theme.Palette.surfaceHi)
            .cornerRadius(Theme.Radius.control)
            .foregroundStyle(Theme.Palette.textPrimary)
            .autocapitalization(.none)
            .disableAutocorrection(true)
    }
}

#Preview {
    LoginView()
}
