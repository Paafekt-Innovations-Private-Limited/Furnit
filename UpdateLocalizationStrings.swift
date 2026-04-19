import Foundation

// MARK: - Localization Update Script
// This script automatically adds missing localization strings to all .lproj folders

struct LocalizationEntry {
    let key: String
    let value: String
    let comment: String
}

let newStrings: [String: [LocalizationEntry]] = [
    "en": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "On: choose the box with the highest score among those above the slider.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "Off: choose the largest box among those above the slider.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "Both modes only consider detections at or above the Primary detection confidence slider. Parser/NMS behavior is unchanged.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "Single Image Scan",
            comment: "Settings - Single Image Scan"
        )
    ],
    "ar": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "تشغيل: اختر المربع ذو أعلى نقاط من بين تلك الموجودة فوق المنزلق.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "إيقاف: اختر أكبر مربع من بين تلك الموجودة فوق المنزلق.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "يأخذ كلا الوضعين في الاعتبار فقط الاكتشافات عند أو أعلى من منزلق ثقة الكشف الأساسي. سلوك المحلل/NMS لم يتغير.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "مسح صورة واحدة",
            comment: "Settings - Single Image Scan"
        )
    ],
    "bn": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "চালু: স্লাইডারের উপরে থাকা বক্সগুলির মধ্যে সর্বোচ্চ স্কোর সহ বক্সটি বেছে নিন।",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "বন্ধ: স্লাইডারের উপরে থাকা বক্সগুলির মধ্যে সবচেয়ে বড় বক্সটি বেছে নিন।",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "উভয় মোড শুধুমাত্র প্রাথমিক সনাক্তকরণ আত্মবিশ্বাস স্লাইডারে বা তার উপরে থাকা সনাক্তকরণ বিবেচনা করে। পার্সার/NMS আচরণ অপরিবর্তিত।",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "একক ছবি স্ক্যান",
            comment: "Settings - Single Image Scan"
        )
    ],
    "de": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "Ein: Wähle die Box mit der höchsten Punktzahl unter denen über dem Schieberegler.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "Aus: Wähle die größte Box unter denen über dem Schieberegler.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "Beide Modi berücksichtigen nur Erkennungen auf oder über dem Primären Erkennungs-Konfidenzschieberegler. Parser/NMS-Verhalten bleibt unverändert.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "Einzelbild-Scan",
            comment: "Settings - Single Image Scan"
        )
    ],
    "es": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "Activado: elige la caja con la puntuación más alta entre las que están sobre el control deslizante.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "Desactivado: elige la caja más grande entre las que están sobre el control deslizante.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "Ambos modos solo consideran detecciones en o por encima del control deslizante de confianza de detección primaria. El comportamiento del Parser/NMS no cambia.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "Escaneo de imagen única",
            comment: "Settings - Single Image Scan"
        )
    ],
    "es-MX": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "Activado: elige la caja con la puntuación más alta entre las que están arriba del control deslizante.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "Desactivado: elige la caja más grande entre las que están arriba del control deslizante.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "Ambos modos solo consideran detecciones en o por encima del control deslizante de confianza de detección primaria. El comportamiento del Parser/NMS no cambia.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "Escaneo de imagen única",
            comment: "Settings - Single Image Scan"
        )
    ],
    "fr": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "Activé : choisir la boîte avec le score le plus élevé parmi celles au-dessus du curseur.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "Désactivé : choisir la plus grande boîte parmi celles au-dessus du curseur.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "Les deux modes ne considèrent que les détections au niveau ou au-dessus du curseur de confiance de détection primaire. Le comportement du Parser/NMS reste inchangé.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "Analyse d'image unique",
            comment: "Settings - Single Image Scan"
        )
    ],
    "hi": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "ऑन: स्लाइडर के ऊपर वाले बॉक्सों में से सबसे अधिक स्कोर वाले बॉक्स को चुनें।",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "ऑफ: स्लाइडर के ऊपर वाले बॉक्सों में से सबसे बड़े बॉक्स को चुनें।",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "दोनों मोड केवल प्राथमिक पहचान विश्वास स्लाइडर पर या उससे ऊपर की पहचान पर विचार करते हैं। पार्सर/NMS व्यवहार अपरिवर्तित रहता है।",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "एकल छवि स्कैन",
            comment: "Settings - Single Image Scan"
        )
    ],
    "kn": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "ಆನ್: ಸ್ಲೈಡರ್‌ಗಿಂತ ಮೇಲಿರುವವುಗಳಲ್ಲಿ ಅತ್ಯಧಿಕ ಅಂಕಗಳನ್ನು ಹೊಂದಿರುವ ಪೆಟ್ಟಿಗೆಯನ್ನು ಆರಿಸಿ.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "ಆಫ್: ಸ್ಲೈಡರ್‌ಗಿಂತ ಮೇಲಿರುವವುಗಳಲ್ಲಿ ದೊಡ್ಡ ಪೆಟ್ಟಿಗೆಯನ್ನು ಆರಿಸಿ.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "ಎರಡೂ ಮೋಡ್‌ಗಳು ಪ್ರಾಥಮಿಕ ಪತ್ತೆ ವಿಶ್ವಾಸ ಸ್ಲೈಡರ್‌ನಲ್ಲಿ ಅಥವಾ ಅದಕ್ಕಿಂತ ಹೆಚ್ಚಿನ ಪತ್ತೆಗಳನ್ನು ಮಾತ್ರ ಪರಿಗಣಿಸುತ್ತವೆ. ಪಾರ್ಸರ್/NMS ನಡವಳಿಕೆ ಬದಲಾಗುವುದಿಲ್ಲ.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "ಏಕ ಚಿತ್ರ ಸ್ಕ್ಯಾನ್",
            comment: "Settings - Single Image Scan"
        )
    ],
    "ml": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "ഓൺ: സ്ലൈഡറിന് മുകളിലുള്ളവയിൽ ഏറ്റവും ഉയർന്ന സ്കോർ ഉള്ള ബോക്‌സ് തിരഞ്ഞെടുക്കുക.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "ഓഫ്: സ്ലൈഡറിന് മുകളിലുള്ളവയിൽ വലിയ ബോക്‌സ് തിരഞ്ഞെടുക്കുക.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "രണ്ട് മോഡുകളും പ്രാഥമിക കണ്ടെത്തൽ വിശ്വാസ സ്ലൈഡറിലോ അതിനു മുകളിലോ ഉള്ള കണ്ടെത്തലുകൾ മാത്രം പരിഗണിക്കുന്നു. പാഴ്‌സർ/NMS പെരുമാറ്റം മാറ്റമില്ലാതെ തുടരുന്നു.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "ഒറ്റ ചിത്ര സ്കാൻ",
            comment: "Settings - Single Image Scan"
        )
    ],
    "ta": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "ஆன்: ஸ்லைடருக்கு மேலே உள்ளவற்றில் அதிக மதிப்பெண் கொண்ட பெட்டியைத் தேர்ந்தெடுக்கவும்.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "ஆஃப்: ஸ்லைடருக்கு மேலே உள்ளவற்றில் பெரிய பெட்டியைத் தேர்ந்தெடுக்கவும்.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "இரண்டு முறைகளும் முதன்மை கண்டறிதல் நம்பிக்கை ஸ்லைடரில் அல்லது அதற்கு மேல் உள்ள கண்டறிதல்களை மட்டுமே கருத்தில் கொள்கின்றன. பார்சர்/NMS நடத்தை மாறாமல் உள்ளது.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "ஒற்றைப் படச் சோதனை",
            comment: "Settings - Single Image Scan"
        )
    ],
    "te": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "ఆన్: స్లయిడర్ పైన ఉన్న వాటిలో అత్యధిక స్కోర్ ఉన్న బాక్స్‌ను ఎంచుకోండి.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "ఆఫ్: స్లయిడర్ పైన ఉన్న వాటిలో పెద్ద బాక్స్‌ను ఎంచుకోండి.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "రెండు మోడ్‌లు ప్రాథమిక గుర్తింపు విశ్వాసం స్లయిడర్ వద్ద లేదా అంతకంటే ఎక్కువగా ఉన్న గుర్తింపులను మాత్రమే పరిగణిస్తాయి. పార్సర్/NMS ప్రవర్తన మారదు.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "ఒకే చిత్ర స్కాన్",
            comment: "Settings - Single Image Scan"
        )
    ],
    "zh-Hans": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "开启：在滑块上方的框中选择得分最高的框。",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "关闭：在滑块上方的框中选择最大的框。",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "两种模式仅考虑位于或高于主检测置信度滑块的检测。解析器/NMS 行为保持不变。",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "单图像扫描",
            comment: "Settings - Single Image Scan"
        )
    ],
    "zh-Hant": [
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOn",
            value: "開啟：在滑桿上方的框中選擇得分最高的框。",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionByHighestConfidenceOff",
            value: "關閉：在滑桿上方的框中選擇最大的框。",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.primarySelectionBehaviorNote",
            value: "兩種模式僅考慮位於或高於主檢測信心滑桿的檢測。解析器/NMS 行為保持不變.",
            comment: "Settings - Furniture Segmentation"
        ),
        LocalizationEntry(
            key: "settings.singleImageScanSection",
            value: "單圖像掃描",
            comment: "Settings - Single Image Scan"
        )
    ]
]

func resolveLocalizationRoot(fileManager: FileManager) -> URL? {
    let currentDirectoryURL = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
    let scriptDirectoryURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    let candidateRoots: [URL] = [
        currentDirectoryURL.appendingPathComponent("Furnit", isDirectory: true),
        currentDirectoryURL,
        scriptDirectoryURL.appendingPathComponent("Furnit", isDirectory: true),
        scriptDirectoryURL
    ]

    for candidateRoot in candidateRoots {
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: candidateRoot.path, isDirectory: &isDirectory), isDirectory.boolValue {
            let englishStringsFileURL = candidateRoot
                .appendingPathComponent("en.lproj", isDirectory: true)
                .appendingPathComponent("Localizable.strings", isDirectory: false)
            if fileManager.fileExists(atPath: englishStringsFileURL.path) {
                return candidateRoot
            }
        }
    }

    return nil
}

func escapedStringsValue(_ value: String) -> String {
    value
        .replacingOccurrences(of: "\\", with: "\\\\")
        .replacingOccurrences(of: "\"", with: "\\\"")
}

// MARK: - Main Execution Function
func updateLocalizations() {
    print("🚀 Starting Localization Update Script...")
    print("📝 This will add 4 new strings to all 14 language files\n")

    let fileManager = FileManager.default
    let currentPath = fileManager.currentDirectoryPath
    print("📂 Current directory: \(currentPath)\n")

    guard let localizationRootURL = resolveLocalizationRoot(fileManager: fileManager) else {
        print("❌ Could not find the localization root containing `en.lproj/Localizable.strings`.")
        print("   Tried relative to both the current directory and the script location.")
        return
    }

    print("📁 Localization root: \(localizationRootURL.path)\n")

    var successCount = 0
    var failCount = 0

    for (languageCode, entries) in newStrings {
        let lprojFolder = "\(languageCode).lproj"
        let lprojFolderURL = localizationRootURL.appendingPathComponent(lprojFolder, isDirectory: true)
        let stringsFileURL = lprojFolderURL.appendingPathComponent("Localizable.strings", isDirectory: false)
        
        print("🌍 Processing: \(languageCode) (\(lprojFolder))...")
        
        // Check if .lproj folder exists
        var isDirectory: ObjCBool = false
        if !fileManager.fileExists(atPath: lprojFolderURL.path, isDirectory: &isDirectory) || !isDirectory.boolValue {
            print("   ⚠️  Warning: \(lprojFolder) folder not found - skipping")
            failCount += 1
            continue
        }

        do {
            let existingContent: String
            if fileManager.fileExists(atPath: stringsFileURL.path) {
                existingContent = try String(contentsOf: stringsFileURL, encoding: .utf8)
            } else {
                existingContent = ""
            }

            let missingEntries = entries.filter { entry in
                !existingContent.contains("\"\(entry.key)\" = ")
            }

            if missingEntries.isEmpty {
                print("   ℹ️  Already up to date")
                successCount += 1
                continue
            }

            var stringsToAdd = "\n// MARK: - Auto-generated additions (Settings - Furniture Segmentation & Single Image Scan)\n"
            for entry in missingEntries {
                stringsToAdd += "/* \(entry.comment) */\n"
                stringsToAdd += "\"\(entry.key)\" = \"\(escapedStringsValue(entry.value))\";\n"
                stringsToAdd += "\n"
            }

            if fileManager.fileExists(atPath: stringsFileURL.path) {
                let fileHandle = try FileHandle(forWritingTo: stringsFileURL)
                defer { try? fileHandle.close() }
                try fileHandle.seekToEnd()
                if let data = stringsToAdd.data(using: .utf8) {
                    try fileHandle.write(contentsOf: data)
                }
                print("   ✅ Updated: \(stringsFileURL.lastPathComponent) (+\(missingEntries.count) strings)")
            } else {
                try stringsToAdd.write(to: stringsFileURL, atomically: true, encoding: .utf8)
                print("   ✅ Created new file: \(stringsFileURL.path)")
            }

            successCount += 1
        } catch {
            print("   ❌ Error: \(error.localizedDescription)")
            failCount += 1
        }
    }

    print("\n" + String(repeating: "=", count: 50))
    print("📊 Summary:")
    print("   ✅ Successfully updated: \(successCount) files")
    print("   ❌ Failed: \(failCount) files")
    print(String(repeating: "=", count: 50))
    print("\n🎉 Done! Please verify the changes in Xcode.")
}

// Run the script
updateLocalizations()
