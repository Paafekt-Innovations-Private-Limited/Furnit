"""FAQ string overrides per locale (base keys loaded from en.lproj)."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EN_PATH = REPO / "Furnit" / "en.lproj" / "Localizable.strings"

KEYS = [
    "faq.howToCreateAnswer",
    "faq.howToTakePhotoAnswer",
    "faq.howToSaveRoomAnswer",
    "faq.whatIsBrainIcon",
    "faq.whatIsBrainIconAnswer",
    "faq.whatIsViewfinderButton",
    "faq.whatIsViewfinderButtonAnswer",
    "faq.howToScreenshotAnswer",
    "faq.whatIsSegmentationAnswer",
    "faq.howToSegmentAnswer",
    "faq.howToViewAnswer",
    "faq.howToNavigateAnswer",
    "faq.whatDoArrowsDo",
    "faq.whatDoArrowsDoAnswer",
    "faq.adjustDimensionsAnswer",
    "faq.arAssistedSizingAnswer",
    "faq.resetOverlayScaleAnswer",
    "faq.multiplePiecesAnswer",
    "faq.whatIsPlacementIntelligenceAnswer",
]

_EN_CACHE: dict[str, str] | None = None


def _load_en() -> dict[str, str]:
    global _EN_CACHE
    if _EN_CACHE is not None:
        return dict(_EN_CACHE)
    text = EN_PATH.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for key in KEYS:
        m = re.search(rf'^"{re.escape(key)}" = "(.*)";$', text, re.MULTILINE)
        if not m:
            raise KeyError(f"Missing {key} in en.lproj")
        found[key] = m.group(1).replace("\\n", "\n").replace('\\"', '"')
    _EN_CACHE = found
    return dict(found)


def _bundle(**overrides: str) -> dict[str, str]:
    base = _load_en()
    base.update(overrides)
    return base


KN = {
    "faq.howToCreateAnswer": "ಹೋಮ್ ಸ್ಕ್ರೀನ್ ಟೂಲ್‌ಬಾರ್‌ನಲ್ಲಿ Photo → 3D (ಅಥವಾ ಖಾಲಿ ಸ್ಥಿತಿಯಲ್ಲಿ Create Room from Photo) ಟ್ಯಾಪ್ ಮಾಡಿ, ನಂತರ ನಿಮ್ಮ ಕೊಠಡಿಯ ಫೋಟೋ ತೆಗೆದುಕೊಳ್ಳಿ ಅಥವಾ ಆಯ್ಕೆಮಾಡಿ. 3D ಕೊಠಡಿ ರಚಿಸಲು ಎರಡು ಆಯ್ಕೆಗಳನ್ನು ನೋಡುತ್ತೀರಿ.",
    "faq.howToTakePhotoAnswer": "ಕೊಠಡಿ ರಚಿಸುವಾಗ Take a Photo ಆಯ್ಕೆಮಾಡಿ. Standard (1x) ಅಥವಾ Wide Angle (0.5×) ಆಯ್ಕೆಮಾಡಬಹುದು. ಸಾಧನವನ್ನು ಸ್ಥಿರವಾಗಿ ಹಿಡಿದು ಕ್ಯಾಪ್ಚರ್ ಮಾಡಿ. ಫೋಟೋ ನಿಮ್ಮ 3D ಕೊಠಡಿ ನಿರ್ಮಾಣಕ್ಕೆ ಬಳಸಲಾಗುತ್ತದೆ.",
    "faq.howToSaveRoomAnswer": "3D ಕೊಠಡಿ ರಚಿಸಿದ ನಂತರ, ಕೊಠಡಿಯನ್ನು ಟ್ಯಾಪ್ ಮಾಡಿ ನಿಯಂತ್ರಣಗಳನ್ನು ತೋರಿಸಿ, ನಂತರ ಉಳಿಸು ಐಕಾನ್ (ಡೌನ್‌ಲೋಡ್ ಬಾಣ) ಟ್ಯಾಪ್ ಮಾಡಿ ಹೆಸರು ನಮೂದಿಸಿ. ಉಳಿಸದೆ ಪ್ರಿವ್ಯೂ ಬಿಟ್ಟರೆ ಆ ಕೊಠಡಿ ತ್ಯಜಿಸಲಾಗುತ್ತದೆ.",
    "faq.whatIsBrainIcon": "Fit ಬಟನ್ ಏನು ಮಾಡುತ್ತದೆ?",
    "faq.whatIsBrainIconAnswer": "Room viewer ನಲ್ಲಿ ಕೆಳಗೆ-ಬಲದ ಚಿನ್ನದ Fit ಬಟನ್ ಟ್ಯಾಪ್ ಮಾಡಿ SmartyPants ಪೀಠೋಪಕರಣ segmentation ಪ್ರಾರಂಭಿಸಿ. ಡಿಫಾಲ್ಟ್ ಆಗಿ ಅತ್ಯಧಿಕ-confidence ವಸ್ತುವನ್ನು 3D ಕೊಠಡಿಯ ಮೇಲೆ ಪಾರದರ್ಶಕ cutout ಆಗಿ auto-segment ಮಾಡುತ್ತದೆ. ನಿರ್ಗಮಿಸಲು ಮೇಲೆ-ಬಲ Done ಟ್ಯಾಪ್ ಮಾಡಿ. ನಿರ್ದಿಷ್ಟ pieces ಆಯ್ಕೆಗಾಗಿ ನಿಯಂತ್ರಣಗಳನ್ನು summon ಮಾಡಿ ಮೊದಲು full-video mode ಸಕ್ರಿಯಗೊಳಿಸಿ.",
    "faq.whatIsViewfinderButton": "Room viewer ನಲ್ಲಿ full-video mode ಏನು ಮಾಡುತ್ತದೆ?",
    "faq.whatIsViewfinderButtonAnswer": "Fit ಸಕ್ರಿಯವಾಗಿರುವಾಗ ನಿಯಂತ್ರಣಗಳನ್ನು summon ಮಾಡಿ viewfinder ಐಕಾನ್ ಟ್ಯಾಪ್ ಮಾಡಿ full-video mode ಟಾಗಲ್ ಮಾಡಿ. live camera + detection boxes—ಒಂದು ಅಥವಾ ಹೆಚ್ಚು items ಟ್ಯಾಪ್ ಮಾಡಿ, ನಂತರ Segment. camera ಮರೆಮಾಡಿ cutouts 3D room ಮೇಲೆ composite. segment mode ಬಿಡಲು ಅಥವಾ Fit ಬಿಡಲು ಮೇಲೆ-ಬಲ Done ಟ್ಯಾಪ್ ಮಾಡಿ.",
    "faq.howToScreenshotAnswer": "ಕೊಠಡಿಯನ್ನು ಟ್ಯಾಪ್ ಮಾಡಿ ನಿಯಂತ್ರಣಗಳನ್ನು summon ಮಾಡಿ, ನಂತರ Capture ಟ್ಯಾಪ್ ಮಾಡಿ. Fit ಆನ್ ಇದ್ದಾಗ furniture overlays ಸೇರಿ ಪ್ರಸ್ತುತ ನೋಟವನ್ನು Photos ಲೈಬ್ರರಿಗೆ ಉಳಿಸುತ್ತದೆ.",
    "faq.whatIsSegmentationAnswer": "Room viewer ನಲ್ಲಿ Fit ಬಟನ್ ಮೂಲಕ furniture segmentation ಚಲಿಸುತ್ತದೆ. ಡಿಫಾಲ್ಟ್ ಆಗಿ ಒಂದು primary item auto-segment. full-video mode ನಲ್ಲಿ ಮೊದಲು items ಆಯ್ಕೆ; Segment ಆಯ್ಕೆ ಮಾಡಿದ pieces ಗಾಗಿ ಮಾತ್ರ cutouts.",
    "faq.howToSegmentAnswer": "Room viewer ನಲ್ಲಿ Fit ಟ್ಯಾಪ್ ಮಾಡಿ. quick check ಗಾಗಿ main item auto-segment. ನಿರ್ದಿಷ್ಟ pieces: ನಿಯಂತ್ರಣಗಳನ್ನು summon ಮಾಡಿ, full-video mode ಸಕ್ರಿಯಗೊಳಿಸಿ, camera furniture ಕಡೆ ಹಿಡಿದು boxes ಟ್ಯಾಪ್ ಮಾಡಿ, Segment. ನಿರ್ಗಮಿಸಲು ಮೇಲೆ-ಬಲ Done ಟ್ಯಾಪ್ ಮಾಡಿ.",
    "faq.howToViewAnswer": "ಹೋಮ್ ಸ್ಕ್ರೀನ್‌ನಲ್ಲಿ ಯಾವುದೇ ಕೊಠಡಿ ಟ್ಯಾಪ್ ಮಾಡಿ 3D viewer ತೆರೆಯಿರಿ. ನೋಡಲು ಎಳೆಯಿರಿ, ಜೂಮ್ ಮಾಡಲು ಚಿಮುಡಿಸಿ. ಹೆಚ್ಚಿನ ನಿಯಂತ್ರಣಗಳಿಗಾಗಿ (recenter, ruler, Capture) ಕೊಠಡಿ ಟ್ಯಾಪ್ ಮಾಡಿ. ಕೆಳಗೆ-ಬಲದ ಚಿನ್ನದ Fit ಬಟನ್ ಯಾವಾಗಲೂ ಕಾಣಿಸುತ್ತದೆ.",
    "faq.howToNavigateAnswer": "ಕೊಠಡಿಯ ಮೇಲೆ ಎಳೆದು camera orbit ಮಾಡಿ, ಚಿಮುಡಿಸಿ ಜೂಮ್ ಮಾಡಿ. ಪರದೆ ಟ್ಯಾಪ್ ಮಾಡಿ toolbar summon ಮಾಡಿ, recenter ಗಾಗಿ viewfinder ಐಕಾನ್ ಬಳಸಿ. ಕೆಳಗೆ-ಎಡದ quiet summon disk ಮರೆಮಾಡಿದ ನಿಯಂತ್ರಣಗಳನ್ನು ಮರಳಿ ತರುತ್ತದೆ.",
    "faq.whatDoArrowsDo": "Viewer ನಿಯಂತ್ರಣಗಳನ್ನು ಹೇಗೆ ತೋರಿಸುವುದು ಅಥವಾ ಮರೆಮಾಡುವುದು?",
    "faq.whatDoArrowsDoAnswer": "Room viewer immersive mode ಬಳಸುತ್ತದೆ—ನಿಯಂತ್ರಣಗಳು ಸ್ವಚ್ಛ ನೋಟಕ್ಕಾಗಿ ಮರೆಮಾಡಲಾಗುತ್ತವೆ. toolbar summon ಮಾಡಲು ಕೊಠಡಿ ಟ್ಯಾಪ್ ಮಾಡಿ. ಕೆಳಗೆ-ಬಲದ ಚಿನ್ನದ Fit ಬಟನ್ ಯಾವಾಗಲೂ ಕಾಣಿಸುತ್ತದೆ. Fit ಸಕ್ರಿಯವಾಗಿರುವಾಗ segmentation ಬಿಡಲು ಮೇಲೆ-ಬಲ Done ಕಾಣಿಸುತ್ತದೆ.",
    "faq.adjustDimensionsAnswer": "AI room dimensions manually edit ಮಾಡಲಾಗುವುದಿಲ್ಲ. ಉಳಿಸಿದ ನಂತರ ನಿಯಂತ್ರಣಗಳನ್ನು summon ಮಾಡಿ ruler ಟ್ಯಾಪ್ ಮಾಡಿ estimated width, height, depth ನೋಡಿ. pre-save preview ನಲ್ಲಿ dimension chips ಮರೆಮಾಡಲಾಗುತ್ತವೆ—ಮೊದಲು ಉಳಿಸಿ, ನಂತರ ಮತ್ತೆ ತೆರೆದು ಅಳತೆಗಳನ್ನು ನೋಡಿ. manual rooms ನಿಮ್ಮ boundaries ಅನುಸರಿಸುತ್ತವೆ.",
    "faq.arAssistedSizingAnswer": "ಮೊದಲು Fit ಟ್ಯಾಪ್ ಮಾಡಿ furniture segment ಮಾಡಿ. room-relative scale ಗಾಗಿ summoned toolbar ನಲ್ಲಿ sizing control ಬಳಸಿ. AR/depth phones live camera depth; ಇಲ್ಲದಿದ್ದರೆ approximate.",
    "faq.resetOverlayScaleAnswer": "ಈಗ overflow menu ಇಲ್ಲ. overlay ತಪ್ಪಾಗಿ ಕಾಣಿಸಿದರೆ Done ಟ್ಯಾಪ್ ಮಾಡಿ Fit ಬಿಡಿ ಮತ್ತು ಮತ್ತೆ Fit ಟ್ಯಾಪ್ ಮಾಡಿ, ಅಥವಾ pinch ಬಳಸಿ ಗಾತ್ರ ಸರಿಪಡಿಸಿ.",
    "faq.multiplePiecesAnswer": "ಹೌದು! full-video mode ನಲ್ಲಿ ಹಲವು boxes ಟ್ಯಾಪ್ ಮಾಡಿ ಒಮ್ಮೆ Segment—ಎಲ್ಲ selected pieces ಒಟ್ಟಿಗೆ 3D room ಮೇಲೆ. default Fit flow ನಲ್ಲಿ ಒಂದೊಂದಾಗಿ segment ಮಾಡಬಹುದು.",
    "faq.whatIsPlacementIntelligenceAnswer": "ಬೆಂಬಲಿತ 3D room viewers ನಲ್ಲಿ Fit ಸಕ್ರಿಯವಾಗಿರುವಾಗ Placement Intelligence card ಪತ್ತೆಯಾದ furniture room dimensions ಒಳಗೆ ಸರಿಹೊಂದುತ್ತದೆಯೇ ಎಂದು ತೋರಿಸುತ್ತದೆ. ವಿವರಗಳಿಗಾಗಿ header ಟ್ಯಾಪ್ ಮಾಡಿ, ಸ್ಥಳ ಉಳಿಸಲು collapsed ಬಿಡಿ.",
}

HI = {
    "faq.howToCreateAnswer": "होम स्क्रीन टूलबार में Photo → 3D (या खाली स्थिति में Create Room from Photo) टैप करें, फिर अपने कमरे की फोटो लें या चुनें। 3D कमरा बनाने के दो विकल्प दिखेंगे।",
    "faq.howToTakePhotoAnswer": "कमरा बनाते समय Take a Photo चुनें। Standard (1x) या Wide Angle (0.5×) चुन सकते हैं। डिवाइस स्थिर रखकर कैप्चर टैप करें। फोटो आपके 3D कमरे के निर्माण में उपयोग होगी।",
    "faq.howToSaveRoomAnswer": "3D कमरा बनने के बाद कमरे पर टैप करके नियंत्रण बुलाएँ, फिर सेव आइकन (डाउनलोड तीर) टैप करके नाम दर्ज करें। बिना सेव किए प्रीव्यू छोड़ने पर वह कमरा हटा दिया जाता है।",
    "faq.whatIsBrainIcon": "Fit बटन क्या करता है?",
    "faq.whatIsBrainIconAnswer": "रूम व्यूअर में नीचे-दाएँ सुनहरे Fit बटन से SmartyPants फर्नीचर segmentation शुरू करें। डिफ़ॉल्ट रूप से सबसे भरोसेमंद आइटम को 3D कमरे पर पारदर्शी cutout के रूप में auto-segment करता है। बाहर निकलने के लिए ऊपर-दाएँ Done टैप करें। खास आइटम चुनने के लिए पहले नियंत्रण बुलाएँ और full-video mode चालू करें।",
    "faq.whatIsViewfinderButton": "रूम व्यूअर में full-video mode क्या करता है?",
    "faq.whatIsViewfinderButtonAnswer": "Fit चालू होने पर नियंत्रण बुलाएँ और viewfinder आइकन से full-video mode टॉगल करें। लाइव कैमरा और detection boxes—एक या अधिक आइटम टैप करें, फिर Segment। कैमरा छिप जाता है और cutouts 3D कमरे पर दिखते हैं। segment mode छोड़ने या Fit बंद करने के लिए ऊपर-दाएँ Done टैप करें।",
    "faq.howToScreenshotAnswer": "कमरे पर टैप करके नियंत्रण बुलाएँ, फिर Capture टैप करें। Fit चालू होने पर फर्नीचर overlays सहित वर्तमान दृश्य Photos लाइब्रेरी में सेव होता है।",
    "faq.whatIsSegmentationAnswer": "रूम व्यूअर में Fit बटन से फर्नीचर segmentation चलता है। डिफ़ॉल्ट रूप में एक primary आइटम auto-segment होता है। full-video mode में पहले आइटम चुनें; Segment केवल चुने गए pieces के cutouts बनाता है।",
    "faq.howToSegmentAnswer": "रूम व्यूअर में Fit टैप करें। जल्दी जाँच के लिए मुख्य आइटम auto-segment होता है। खास pieces के लिए: नियंत्रण बुलाएँ, full-video mode चालू करें, कैमरा फर्नीचर की ओर करें, boxes टैप करें, फिर Segment। बाहर निकलने के लिए ऊपर-दाएँ Done टैप करें।",
    "faq.howToViewAnswer": "होम स्क्रीन पर किसी भी कमरे को टैप करके 3D व्यूअर खोलें। देखने के लिए खींचें, ज़ूम के लिए पिंच करें। अतिरिक्त नियंत्रण (recenter, ruler, Capture) के लिए कमरे पर टैप करें। नीचे-दाएँ सुनहरा Fit बटन हमेशा दिखता है।",
    "faq.howToNavigateAnswer": "कमरे पर खींचकर कैमरा घुमाएँ और पिंच से ज़ूम करें। टूलबार बुलाने के लिए स्क्रीन टैप करें, recenter के लिए viewfinder आइकन उपयोग करें। नीचे-बाएँ quiet summon disk छिपे नियंत्रण वापस लाता है।",
    "faq.whatDoArrowsDo": "व्यूअर नियंत्रण कैसे दिखाएँ या छिपाएँ?",
    "faq.whatDoArrowsDoAnswer": "रूम व्यूअर immersive mode उपयोग करता है—साफ़ दृश्य के लिए नियंत्रण छिपे रहते हैं। टूलबार बुलाने के लिए कमरे पर टैप करें। नीचे-दाएँ सुनहरा Fit बटन हमेशा दिखता है। Fit चालू होने पर segmentation छोड़ने के लिए ऊपर-दाएँ Done दिखता है।",
    "faq.adjustDimensionsAnswer": "AI कमरे के आयाम मैन्युअल संपादित नहीं कर सकते। सेव के बाद नियंत्रण बुलाकर ruler से अनुमानित चौड़ाई, ऊँचाई, गहराई देखें। pre-save प्रीव्यू में dimension chips छिपे रहते हैं—पहले सेव करें, फिर दोबारा खोलकर माप देखें। manual कमरे आपकी सीमाओं का पालन करते हैं।",
    "faq.arAssistedSizingAnswer": "पहले Fit से फर्नीचर segment करें। room-relative स्केल के लिए summoned toolbar में sizing control उपयोग करें। AR/depth फ़ोन live camera depth उपयोग करते हैं; बिना depth माप अनुमानित रहते हैं।",
    "faq.resetOverlayScaleAnswer": "अब overflow menu नहीं है। overlay गलत लगे तो Done से Fit बंद करके फिर Fit टैप करें, या pinch से आकार ठीक करें।",
    "faq.multiplePiecesAnswer": "हाँ! full-video mode में कई detection boxes टैप करके एक बार Segment—सभी चुने pieces एक साथ 3D कमरे पर। default Fit flow से एक-एक करके भी segment कर सकते हैं।",
    "faq.whatIsPlacementIntelligenceAnswer": "समर्थित 3D room viewers में Fit चालू होने पर Placement Intelligence card दिखाता है कि पता चला फर्नीचर कमरे के आयामों में फिट होता है या नहीं। विवरण के लिए header टैप करें, जगह बचाने के लिए collapsed रखें।",
}

DE = {
    "faq.whatIsBrainIcon": "Was macht die Fit-Taste?",
    "faq.whatIsBrainIconAnswer": "Tippen Sie im Raum-Viewer unten rechts auf die goldene Fit-Taste, um SmartyPants-Möbelsegmentierung zu starten. Standardmäßig wird das Objekt mit höchster Konfidenz als transparenter Ausschnitt über Ihrem 3D-Raum segmentiert. Tippen Sie oben rechts auf Fertig zum Beenden. Für die Auswahl bestimmter Stücke zuerst Steuerung einblenden und Full-Video-Modus aktivieren.",
    "faq.whatIsViewfinderButton": "Was macht der Full-Video-Modus im Raum-Viewer?",
    "faq.whatIsViewfinderButtonAnswer": "Während Fit aktiv ist, Steuerung einblenden und das Sucher-Symbol tippen, um den Full-Video-Modus umzuschalten. Live-Kamera mit Erkennungsrahmen—ein oder mehrere Objekte tippen, dann Segmentieren. Die Kamera wird ausgeblendet und transparente Ausschnitte werden über den 3D-Raum gelegt. Oben rechts auf Fertig tippen, um den Segmentmodus zu verlassen oder Fit zu beenden.",
    "faq.howToScreenshotAnswer": "Tippen Sie auf den Raum, um die Steuerung einzublenden, dann auf Aufnahme. Speichert die aktuelle Ansicht (einschließlich Möbel-Overlays bei aktivem Fit) in Ihrer Fotomediathek.",
    "faq.howToSegmentAnswer": "Tippen Sie im Raum-Viewer auf Fit. Für eine schnelle Prüfung wird das Hauptobjekt automatisch segmentiert. Für bestimmte Stücke: Steuerung einblenden, Full-Video-Modus aktivieren, Kamera auf Möbel richten, Rahmen tippen, dann Segmentieren. Oben rechts auf Fertig tippen zum Beenden.",
    "faq.howToViewAnswer": "Tippen Sie auf einen Raum auf dem Startbildschirm, um den 3D-Viewer zu öffnen. Ziehen zum Umsehen, kneifen zum Zoomen. Tippen Sie auf den Raum für weitere Steuerung (Zentrieren, Lineal, Aufnahme). Die goldene Fit-Taste bleibt unten rechts sichtbar.",
    "faq.howToNavigateAnswer": "Ziehen Sie auf dem Raum, um die Kamera zu drehen, und kneifen Sie zum Zoomen. Tippen Sie auf den Bildschirm, um die Symbolleiste einzublenden, dann das Sucher-Symbol zum Zentrieren. Eine ruhige Einblendscheibe unten links bringt versteckte Steuerung zurück.",
    "faq.whatDoArrowsDo": "Wie blende ich Viewer-Steuerung ein oder aus?",
    "faq.whatDoArrowsDoAnswer": "Der Raum-Viewer nutzt den Immersionsmodus—Steuerung ist für eine klare Ansicht ausgeblendet. Tippen Sie auf den Raum, um die Symbolleiste einzublenden. Die goldene Fit-Taste unten rechts bleibt immer sichtbar. Bei aktivem Fit erscheint oben rechts Fertig zum Beenden der Segmentierung.",
    "faq.howToTakePhotoAnswer": "Beim Erstellen eines Raums wählen Sie Foto aufnehmen. Standard (1x) oder Weitwinkel (0.5×). Gerät ruhig halten und aufnehmen. Das Foto wird für Ihren 3D-Raum verwendet.",
    "faq.howToSaveRoomAnswer": "Nach der 3D-Erstellung auf den Raum tippen, Steuerung einblenden, dann Speichern-Symbol (Download-Pfeil) und Namen eingeben. Ohne Speichern verlassen wird die Vorschau verworfen.",
    "faq.resetOverlayScaleAnswer": "Es gibt kein Überlaufmenü mehr. Wenn das Overlay falsch wirkt, tippen Sie auf Fertig, um Fit zu beenden, und erneut auf Fit, oder passen Sie die Größe per Kneifen an.",
    "faq.whatIsPlacementIntelligenceAnswer": "Wenn Fit in unterstützten 3D-Raum-Viewern aktiv ist, zeigt eine Placement-Intelligence-Karte, ob erkanntes Möbel in die Raummaße passt. Tippen Sie auf die Überschrift für Details oder lassen Sie sie eingeklappt.",
}

FR = {
    "faq.whatIsBrainIcon": "À quoi sert le bouton Fit ?",
    "faq.whatIsBrainIconAnswer": "Appuyez sur le bouton Fit doré (en bas à droite) dans le visualiseur pour lancer la segmentation SmartyPants. Par défaut, l'objet le plus fiable est découpé en transparence sur votre pièce 3D. Appuyez sur Terminé (en haut à droite) pour quitter. Pour choisir des pièces précises, affichez d'abord les contrôles et activez le mode full-video.",
    "faq.howToScreenshotAnswer": "Appuyez sur la pièce pour afficher les contrôles, puis sur Capture. Enregistre la vue actuelle (y compris les overlays meubles si Fit est actif) dans Photos.",
    "faq.howToViewAnswer": "Appuyez sur une pièce de l'accueil pour ouvrir le visualiseur 3D. Glissez pour regarder, pincez pour zoomer. Appuyez sur la pièce pour plus de contrôles (recentrer, règle, Capture). Le bouton Fit doré reste visible en bas à droite.",
    "faq.whatDoArrowsDo": "Comment afficher ou masquer les contrôles ?",
    "faq.whatDoArrowsDoAnswer": "Le visualiseur est immersif—les contrôles se masquent pour une vue épurée. Appuyez sur la pièce pour afficher la barre d'outils. Le bouton Fit doré en bas à droite reste toujours visible. Quand Fit est actif, Terminé apparaît en haut à droite pour quitter la segmentation.",
}

ES = {
    "faq.whatIsBrainIcon": "¿Qué hace el botón Fit?",
    "faq.whatIsBrainIconAnswer": "Toca el botón Fit dorado (abajo a la derecha) en el visor para iniciar la segmentación SmartyPants. Por defecto segmenta automáticamente el objeto de mayor confianza como recorte transparente sobre tu habitación 3D. Toca Hecho (arriba a la derecha) para salir. Para elegir piezas concretas, muestra los controles y activa el modo full-video primero.",
    "faq.howToScreenshotAnswer": "Toca la habitación para mostrar controles y luego Capturar. Guarda la vista actual (incluidos overlays de muebles con Fit activo) en Fotos.",
    "faq.howToViewAnswer": "Toca cualquier habitación en inicio para abrir el visor 3D. Arrastra para mirar y pellizca para zoom. Toca la habitación para más controles (recentrar, regla, Capturar). El botón Fit dorado permanece visible abajo a la derecha.",
    "faq.whatDoArrowsDo": "¿Cómo mostrar u ocultar los controles del visor?",
    "faq.whatDoArrowsDoAnswer": "El visor usa modo inmersivo—los controles se ocultan para una vista limpia. Toca la habitación para mostrar la barra. El botón Fit dorado abajo a la derecha siempre está visible. Con Fit activo, Hecho aparece arriba a la derecha para salir de la segmentación.",
}

ZH_HANS = {
    "faq.whatIsBrainIcon": "Fit 按钮有什么作用？",
    "faq.whatIsBrainIconAnswer": "在房间查看器中点击右下角金色 Fit 按钮即可启动 SmartyPants 家具分割。默认会自动分割置信度最高的物品，并以透明抠图叠加在 3D 房间上。点击右上角完成退出。要选择特定物品，请先唤出控件并开启 full-video 模式。",
    "faq.howToScreenshotAnswer": "点击房间唤出控件，然后点按 Capture。可将当前视图（Fit 开启时含家具叠加层）保存到照片图库。",
    "faq.howToViewAnswer": "在主屏幕点击任意房间打开 3D 查看器。拖动环视，捏合缩放。点击房间可唤出更多控件（重新居中、尺子、Capture）。右下角金色 Fit 按钮始终可见。",
    "faq.whatDoArrowsDo": "如何显示或隐藏查看器控件？",
    "faq.whatDoArrowsDoAnswer": "房间查看器为沉浸模式—控件会隐藏以保持画面简洁。点击房间唤出工具栏。右下角金色 Fit 按钮始终可见。Fit 激活时，右上角会显示完成以退出分割。",
}

ZH_HANT = {
    "faq.whatIsBrainIcon": "Fit 按鈕有什麼作用？",
    "faq.whatIsBrainIconAnswer": "在房間檢視器中點按右下角金色 Fit 按鈕即可啟動 SmartyPants 家具分割。預設會自動分割置信度最高的物品，並以透明去背疊加在 3D 房間上。點按右上角完成退出。若要選擇特定物品，請先喚出控制項並開啟 full-video 模式。",
    "faq.howToScreenshotAnswer": "點按房間喚出控制項，然後點按 Capture。可將目前檢視（Fit 開啟時含家具疊加層）儲存到照片圖庫。",
    "faq.howToViewAnswer": "在主畫面點按任意房間開啟 3D 檢視器。拖動環視，捏合縮放。點按房間可喚出更多控制項（重新置中、尺規、Capture）。右下角金色 Fit 按鈕始終可見。",
    "faq.whatDoArrowsDo": "如何顯示或隱藏檢視器控制項？",
    "faq.whatDoArrowsDoAnswer": "房間檢視器為沉浸模式—控制項會隱藏以保持畫面簡潔。點按房間喚出工具列。右下角金色 Fit 按鈕始終可見。Fit 啟用時，右上角會顯示完成以退出分割。",
}

TRANSLATIONS: dict[str, dict[str, str]] = {
    "kn": _bundle(**KN),
    "hi": _bundle(**HI),
    "de": _bundle(**DE),
    "fr": _bundle(**FR),
    "es": _bundle(**ES),
    "es-MX": _bundle(**ES),
    "zh-Hans": _bundle(**ZH_HANS),
    "zh-Hant": _bundle(**ZH_HANT),
}

for _locale in ("ar", "bn", "ml", "ta", "te"):
    TRANSLATIONS[_locale] = _load_en()
