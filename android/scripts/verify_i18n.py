#!/usr/bin/env python3
"""Fail when Android locale catalogs drift or visible text bypasses resources."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "app/src/main/res"
DEFAULT_CATALOG = RES / "values/strings.xml"

LOCALES = {
    "ar": "values-ar",
    "bn": "values-bn",
    "de": "values-de",
    "es": "values-es",
    "es-MX": "values-es-rMX",
    "fr": "values-fr",
    "hi": "values-hi",
    "kn": "values-kn",
    "ml": "values-ml",
    "ta": "values-ta",
    "te": "values-te",
    "zh-CN": "values-zh-rCN",
    "zh-TW": "values-zh-rTW",
}

PRODUCT_NAME_KEYS = {
    "app_name",
    "app_version_dynamic",
    "credits_anthropic_title",
    "credits_apple_title",
    "credits_coco_title",
    "credits_depth_anything_title",
    "credits_filament_title",
    "credits_geo_calib_title",
    "credits_google_title",
    "credits_hypersim_title",
    "credits_onnx_runtime_title",
    "credits_openai_title",
    "credits_rtmdet_title",
    "credits_three_title",
    "licenses_android_platform_title",
    "licenses_arcore_title",
    "licenses_coco_title",
    "licenses_depth_anything_title",
    "licenses_filament_title",
    "licenses_firebase_title",
    "licenses_geo_calib_title",
    "licenses_hypersim_title",
    "licenses_onnx_runtime_title",
    "licenses_rtmdet_title",
    "licenses_three_title",
}

FORMAT_ONLY_KEYS = {
    "common_percentage",
    "home_zero_mb",
    "placement_intelligence_score",
    "room_dimensions_width_depth",
    "settings_dimension_value_meters",
}

VALID_IDENTICAL_TRANSLATIONS = {
    "de": {"camera_standard", "furniture_class_toaster"},
    "es": {"glb_room_error", "photo_room_error_load"},
    "es-MX": {"glb_room_error", "photo_room_error_load"},
    "fr": {"camera_standard", "home_total", "orientation_portrait"},
}

FORMAT_RE = re.compile(r"%(?:\d+\$)?(?:\.\d+)?[A-Za-z%]")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]{2,}")
UNICODE_ESCAPE_ONLY_RE = re.compile(r"^(?:\\u[0-9A-Fa-f]{4})+$")
RESOURCE_REF_RE = re.compile(r"(?<!android\.)\bR\.(string|plurals)\.([A-Za-z0-9_]+)")
XML_RESOURCE_REF_RE = re.compile(r"@(string|plurals)/([A-Za-z0-9_]+)")

KOTLIN_VISIBLE_LITERAL_PATTERNS = (
    re.compile(r"\btext\s*=\s*\"([^\"\n]*)\""),
    re.compile(r"\bhint\s*=\s*\"([^\"\n]*)\""),
    re.compile(r"\bcontentDescription\s*=\s*\"([^\"\n]*)\""),
    re.compile(r"Intent\.createChooser\([^\n,]+,\s*\"([^\"\n]*)\""),
    re.compile(r"(?:onProgress|onError)\([^\n,]+,\s*\"([^\"\n]*)\""),
    re.compile(r"Toast\.makeText\([^\n,]+,\s*\"([^\"\n]*)\""),
)

ANDROID_NS = "{http://schemas.android.com/apk/res/android}"
VISIBLE_XML_ATTRIBUTES = {
    f"{ANDROID_NS}contentDescription",
    f"{ANDROID_NS}hint",
    f"{ANDROID_NS}label",
    f"{ANDROID_NS}text",
    f"{ANDROID_NS}title",
}


def text(element: ET.Element) -> str:
    return "".join(element.itertext())


def load_catalog(path: Path) -> tuple[ET.Element, dict[str, ET.Element], list[str]]:
    root = ET.parse(path).getroot()
    names = [
        element.attrib["name"]
        for element in root
        if isinstance(element.tag, str) and "name" in element.attrib
    ]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    resources = {
        element.attrib["name"]: element
        for element in root
        if isinstance(element.tag, str) and "name" in element.attrib
    }
    return root, resources, duplicates


def edge_newlines(value: str) -> tuple[int, int]:
    leading = len(re.match(r"^(?:\\n)*", value).group(0)) // 2
    trailing = len(re.search(r"(?:\\n)*$", value).group(0)) // 2
    return leading, trailing


def expected_plural_quantities(locale: str) -> set[str]:
    if locale == "ar":
        return {"zero", "one", "two", "few", "many", "other"}
    if locale in {"zh-CN", "zh-TW"}:
        return {"other"}
    if locale in {"es", "es-MX", "fr"}:
        return {"one", "many", "other"}
    return {"one", "other"}


def validate_catalogs(errors: list[str]) -> None:
    _, default, duplicates = load_catalog(DEFAULT_CATALOG)
    if duplicates:
        errors.append(f"{DEFAULT_CATALOG}: duplicate resources: {', '.join(duplicates)}")

    for locale, directory in LOCALES.items():
        path = RES / directory / "strings.xml"
        _, localized, localized_duplicates = load_catalog(path)
        if localized_duplicates:
            errors.append(f"{path}: duplicate resources: {', '.join(localized_duplicates)}")

        missing = sorted(default.keys() - localized.keys())
        extra = sorted(localized.keys() - default.keys())
        if missing:
            errors.append(f"{path}: missing resources: {', '.join(missing)}")
        if extra:
            errors.append(f"{path}: unexpected resources: {', '.join(extra)}")

        identical_allowed = PRODUCT_NAME_KEYS | FORMAT_ONLY_KEYS | VALID_IDENTICAL_TRANSLATIONS.get(locale, set())
        for name in sorted(default.keys() & localized.keys()):
            source_element = default[name]
            translated_element = localized[name]
            if source_element.tag != translated_element.tag:
                errors.append(
                    f"{path}: {name} changed type from {source_element.tag} to {translated_element.tag}"
                )
                continue

            if source_element.tag == "string":
                source = text(source_element)
                translated = text(translated_element)
                if Counter(FORMAT_RE.findall(source)) != Counter(FORMAT_RE.findall(translated)):
                    errors.append(f"{path}: {name} has mismatched format arguments")
                if edge_newlines(source) != edge_newlines(translated):
                    errors.append(f"{path}: {name} changed leading/trailing newline escapes")
                if any(token in translated for token in ("ZXQ", "⟦", "⟧")):
                    errors.append(f"{path}: {name} contains a leaked translation token")
                if (
                    name not in identical_allowed
                    and source.strip() == translated.strip()
                    and ENGLISH_WORD_RE.search(source)
                ):
                    errors.append(f"{path}: {name} still falls back to the English source")
            elif source_element.tag == "plurals":
                source_items = {item.attrib["quantity"]: text(item) for item in source_element}
                translated_items = {item.attrib["quantity"]: text(item) for item in translated_element}
                expected_quantities = expected_plural_quantities(locale)
                if set(translated_items) != expected_quantities:
                    errors.append(
                        f"{path}: {name} plural quantities are {sorted(translated_items)}; "
                        f"expected {sorted(expected_quantities)}"
                    )
                for quantity, translated in translated_items.items():
                    source = source_items.get(quantity, source_items["other"])
                    if Counter(FORMAT_RE.findall(source)) != Counter(FORMAT_RE.findall(translated)):
                        errors.append(f"{path}: {name}[{quantity}] has mismatched format arguments")
                    if any(token in translated for token in ("ZXQ", "⟦", "⟧")):
                        errors.append(f"{path}: {name}[{quantity}] contains a leaked translation token")


def validate_locale_config(errors: list[str]) -> None:
    config_path = RES / "xml/locales_config.xml"
    root = ET.parse(config_path).getroot()
    declared = [element.attrib[f"{ANDROID_NS}name"] for element in root]
    expected = ["en", *LOCALES.keys()]
    if declared != expected:
        errors.append(f"{config_path}: locales are {declared}; expected {expected}")

    manifest = (ROOT / "app/src/main/AndroidManifest.xml").read_text()
    if 'android:localeConfig="@xml/locales_config"' not in manifest:
        errors.append("AndroidManifest.xml does not publish @xml/locales_config")


def validate_resource_references(errors: list[str]) -> None:
    _, default, _ = load_catalog(DEFAULT_CATALOG)
    for path in (ROOT / "app/src/main").rglob("*"):
        if not path.is_file() or path.suffix not in {".kt", ".xml"}:
            continue
        contents = path.read_text(errors="replace")
        pattern = RESOURCE_REF_RE if path.suffix == ".kt" else XML_RESOURCE_REF_RE
        for kind, name in pattern.findall(contents):
            element = default.get(name)
            if element is None:
                errors.append(f"{path}: references missing {kind} resource {name}")
            elif element.tag != kind.rstrip("s") and not (kind == "plurals" and element.tag == "plurals"):
                errors.append(f"{path}: references {name} as {kind}, but it is a {element.tag}")


def validate_no_hardcoded_visible_text(errors: list[str]) -> None:
    source_root = ROOT / "app/src/main/java"
    for path in source_root.rglob("*.kt"):
        contents = path.read_text(errors="replace")
        for pattern in KOTLIN_VISIBLE_LITERAL_PATTERNS:
            for match in pattern.finditer(contents):
                literal = match.group(1)
                if ENGLISH_WORD_RE.search(literal) and not UNICODE_ESCAPE_ONLY_RE.fullmatch(literal):
                    line = contents.count("\n", 0, match.start()) + 1
                    errors.append(f"{path}:{line}: visible text literal {literal!r} bypasses resources")

    for path in RES.rglob("*.xml"):
        root = ET.parse(path).getroot()
        for element in root.iter():
            for attribute in VISIBLE_XML_ATTRIBUTES:
                value = element.attrib.get(attribute)
                if value and not value.startswith(("@", "?")) and ENGLISH_WORD_RE.search(value):
                    errors.append(f"{path}: hardcoded visible XML value {value!r}")

    gradle = (ROOT / "app/build.gradle").read_text()
    if re.search(r"disable\s+['\"]MissingTranslation['\"]", gradle):
        errors.append("app/build.gradle disables the MissingTranslation lint check")


def main() -> int:
    errors: list[str] = []
    validate_catalogs(errors)
    validate_locale_config(errors)
    validate_resource_references(errors)
    validate_no_hardcoded_visible_text(errors)

    if errors:
        print("Android i18n verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"Android i18n verification passed: {len(LOCALES)} localized catalogs match "
        f"{DEFAULT_CATALOG.relative_to(ROOT)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
