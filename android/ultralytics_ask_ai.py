#!/usr/bin/env python3
"""
Open Ultralytics deployment docs, click Ask AI, and submit a SHARP/Executorch-focused question.
Use --gradle for the Gradle build failure (wildcard IP / sysconf on macOS).

Uses Playwright (same pattern as abap-ai-toolkit browser_export.py).
- Prompt kept under 200 lines (chat limit); no attachments.
- Question text steers the assistant toward ViT/TinyViT/Executorch deployment, not YOLO.

Response is printed and saved to ultralytics_response.txt. To use it in Cursor:
  "Use android/ultralytics_response.txt (or the Ultralytics AI response) and follow those
   suggestions. Before applying any changes, ask me once here in Cursor."
The assistant will summarize the suggestions and ask for your confirmation before editing code.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEPLOYMENT_URL = "https://docs.ultralytics.com/guides/model-deployment-practices/#model-deployment-options"

# Default: current problem. Switch to BLUISH_QUESTION for color fix.
CURRENT_PROBLEM_QUESTION = """We run a ViT-based 3D reconstruction model (SHARP) on Android with ExecuTorch INT8. Part4b (image encoder B + decoder + Gaussian head) takes ~80 seconds per inference; Part1+2 ~31s, Part3 ~1.4s, Part4a chunks ~3.3s. We need to reduce Part4b latency without Vulkan. Please give concrete changes: (1) Python: export or model changes for SDPA/fused attention, memory planning, or quantization when exporting to ExecuTorch .pte. (2) Android/Kotlin: runtime or build changes to use optimized attention or XNNPACK settings when loading and running the Part4b module. Include code snippets or step-by-step edits for both Python and Android. Focus on ExecuTorch and ViT decoder on mobile CPU, not YOLO."""

# Room rendering looks bluish: likely color order or SH DC interpretation.
BLUISH_QUESTION = """We have a 3D Gaussian splatting model (SHARP, ViT-based) running on Android with ExecuTorch. The model outputs Gaussian parameters including spherical harmonic DC coefficients for color (f_dc_0, f_dc_1, f_dc_2). We write PLY with (params - 0.5) / SH_C0 for these three. Input preprocessing uses RGB order (R=argb>>16, G=argb>>8, B=argb). The rendered room looks globally bluish. Could this be: (1) BGR vs RGB in input or in the model's output channel order? (2) Wrong interpretation of SH DC (sign, scale, or channel order in the 14-param Gaussian layout)? (3) Color space (sRGB vs linear) mismatch between training and our PLY? Please suggest a concrete fix (e.g. swap channels, or correct formula for f_dc) for deployment. We are not using YOLO; this is Gaussian splatting / 3D reconstruction."""

# Progress bar: keep percentage ticking during long inference so user sees progress.
PROGRESS_BAR_QUESTION = """We have an Android app that runs a long (~2 minute) ML inference pipeline (ExecuTorch, ViT-based 3D room generation). The progress bar currently jumps from 20% to 100% at the end because we only report at start and completion. We want the percentage to tick smoothly so the user feels progress is happening. The pipeline has distinct phases: encoder patches (~45s), image encoder (~1.5s), decoder chunks (~3s), one long blocking forward (~80s), then file write (~10s). We can report progress at the end of each patch and phase, but the 80s forward is a single blocking call with no mid-forward callbacks. What are best practices to keep the progress bar moving? Options we're considering: (1) Run the blocking call in a background thread and on a timer (e.g. every 2s) update progress from 55% to 90% based on elapsed time. (2) Show an indeterminate spinner for the long phase. (3) Subdivide the work so we have more granular callbacks. We want concrete UX advice: should we use time-based estimated progress during the blocking call, and what copy/messaging keeps users engaged (e.g. "This may take a minute", "Adding the finishing touches")? Not YOLO; this is 3D reconstruction / Gaussian splatting."""

# Camera: position virtual camera at back wall looking at front wall in 3D room viewer.
CAMERA_POSITION_QUESTION = """We have an Android WebView 3D room viewer using Three.js and OrbitControls. The room is a Gaussian splatting mesh (PLY) with a bounding box (Box3). We want the initial camera to be placed just inside the imaginary back wall, looking directly at the front wall, for both portrait and landscape photo orientations. We set camera.position to (center.x, eyeHeight, backWallZ - padding) and controls.target to (center.x, center.y, frontWallZ) with backWallZ = box.max.z and frontWallZ = box.min.z, but the camera position does not change from the previous default (camera was at center minus distance along Z, target at center). Why might the new position/target not take effect? Possible causes: (1) autoFrameRoom runs before the mesh has final geometry? (2) OrbitControls or something else resetting camera after we set it? (3) Need to set initialCameraPosition/initialControlsTarget differently? (4) For portrait vs landscape we need different axes (e.g. back/front along X for portrait)? We need the same behaviour for both orientations. Not YOLO; this is 3D reconstruction / room viewer."""

# Orientation: portrait and landscape photos both showing as landscape in 3D viewer.
ORIENTATION_QUESTION = """We have an Android app that shows a 3D Gaussian-splat room in a WebView (Three.js). The room is generated from a single photo. We pass photo_orientation (portrait or landscape) from the activity intent and inject it into the WebView as isPortrait. We use isPortrait to map Box3 size to roomWidth/roomHeight (portrait: width=X height=Y; landscape: width=Y height=X). We do NOT apply any rotation to the splat mesh (comment says "No rotation - see raw PLY first"). The problem: both portrait and landscape photos result in the room being displayed as landscape only; orientation is not correctly set. What is the right approach? (1) Should we rotate the splat mesh (e.g. 90° around Y) when isPortrait so the room aspect matches the photo? (2) Is the PLY always exported in a fixed coordinate frame (e.g. landscape) so we must apply a rotation in the viewer for portrait? (3) Could the intent extra "photo_orientation" be wrong or not passed, and how should we verify? We lock the activity to SCREEN_ORIENTATION_PORTRAIT or LANDSCAPE based on photo_orientation, but the 3D content still looks landscape. Not YOLO; this is 3D room reconstruction."""

# Aspect distortion: landscape room squeezed, portrait room expanded (snake-like edges).
ASPECT_SQUEEZE_QUESTION = """We display a 3D Gaussian splatting room (PLY) in a WebView with Three.js. The room comes from a single photo; we know photo_orientation (portrait or landscape) and pass room dimensions (roomWidth, roomHeight, roomDepth in meters). For portrait we apply a 90° rotation around Y to the splat mesh so the room aspect matches the photo. We map the mesh bounding box (Box3) to roomWidth/roomHeight: portrait uses roomWidth=size.x, roomHeight=size.y; landscape uses roomWidth=size.y, roomHeight=size.x; roomDepth=size.z. The problem: in landscape the room looks squeezed (narrow). In portrait it looks expanded with snake-like wavy edges on objects and walls. We use PerspectiveCamera with aspect = window.innerWidth/window.innerHeight and lock the activity to portrait or landscape. What is the correct way to (1) map Box3 axes to room width/height/depth after a 90° Y rotation for portrait so aspect ratio matches the photo? (2) Avoid squeeze in landscape and stretch/snake edges in portrait—could this be wrong width/height swap or camera frustum vs room aspect mismatch? We follow Ultralytics-style orientation (display dimensions: landscape = width > height, portrait = height > width). Not YOLO; this is 3D room reconstruction viewer."""

DEFAULT_QUESTION = CURRENT_PROBLEM_QUESTION

# Other canned questions (use -q with paste or a file).
DEPLOYMENT_QUESTION = """I'm deploying a ViT-based 3D reconstruction model (SHARP) to Android with ExecuTorch—not YOLO. The encoder outputs (B, 577, 1024) with a CLS token; we reshape to spatial [C, 24, 24] for the decoder and must skip the CLS token to avoid buffer overrun at index 589824. What deployment best practices do you recommend for this kind of vision transformer on edge: export format (ExecuTorch vs ONNX), quantization, and memory layout for TinyViT-style (B, HW+1, C) outputs? Please focus on ViT/TinyViT and ExecuTorch deployment, not object detection."""

# Use with: python ultralytics_ask_ai.py -q "$(cat android/ultralytics_question_576_vs_577.txt)"
QUESTION_576_VS_577 = """For TinyViT / PatchEmbed: when processing a single 384x384 patch, can the encoder output (B, N, C) have N=576 (spatial only) instead of N=577 (CLS + 24x24)? We get ArrayIndexOutOfBoundsException at index 589824 when reshaping Part2 output for that single-patch path—our buffer is 576*1024. Should reshapeToSpatial assume 577 tokens and skip CLS, or 576 tokens with no CLS depending on tensor size? Please focus on TinyViT single-patch output shape."""

# Aspect distortion + full INT8 ExecuTorch code (read from file to keep prompt under limit).
QUESTION_ASPECT_INT8_FILE = Path(__file__).resolve().parent / "ultralytics_question_aspect_int8.txt"

# Jagged / screwed-up 3D output (full prompt in file).
QUESTION_JAGGED_OUTPUT_FILE = Path(__file__).resolve().parent / "ultralytics_question_jagged_output.txt"

# 0.5x overlay not in middle (Furniture Fit centering at scale).
QUESTION_05X_CENTERING_FILE = Path(__file__).resolve().parent / "ultralytics_question_05x_centering.txt"

# Position still wrong after center-crop + raw coords (room 8.78x7.62x4.17).
QUESTION_POSITION_WRONG_FILE = Path(__file__).resolve().parent / "ultralytics_question_position_wrong.txt"

# Validate input: resize-fit + edge pad vs center crop (lenient, no black bars).
QUESTION_VALIDATE_INPUT_FILE = Path(__file__).resolve().parent / "ultralytics_question_validate_input.txt"

# Jagged output with 114-gray letterbox (whole pic, still jagged).
QUESTION_JAGGED_LETTERBOX114_FILE = Path(__file__).resolve().parent / "ultralytics_question_jagged_letterbox114.txt"

# More image / less crop without screwing quality (describe center-crop, ask for alternatives).
QUESTION_MORE_IMAGE_KEEP_QUALITY_FILE = Path(__file__).resolve().parent / "ultralytics_question_more_image_keep_quality.txt"

# Why Swift (stretch-to-square) didn't cause jagged output vs Android letterbox.
QUESTION_SWIFT_VS_ANDROID_FILE = Path(__file__).resolve().parent / "ultralytics_question_swift_vs_android_preprocess.txt"

# Best way to do Lanczos resize on Android for 1536x1536 model input.
QUESTION_LANCZOS_RESIZE_FILE = Path(__file__).resolve().parent / "ultralytics_question_lanczos_resize.txt"

# Gradle/Android build failure: wildcard IP and sysconf on macOS.
GRADLE_BUILD_QUESTION = """When building an Android app with Gradle on macOS (darwin), the build fails with: (1) "Could not determine a usable wildcard IP for this machine" and (2) "xargs: sysconf(_SC_ARG_MAX) failed". The project uses AGP 8.5.2 and Kotlin. Building with --no-daemon works. What is the root cause and the recommended fix so that a normal daemon build works? Should we set org.gradle.daemon=false, or -Djava.net.preferIPv4Stack=true in gradle.properties, or something else? We need a concrete fix for Gradle/JVM or macOS environment."""

RESPONSE_FILE = Path(__file__).resolve().parent / "ultralytics_response.txt"
POPUP_DOM_FILE = Path(__file__).resolve().parent / "ultralytics_popup_dom.html"
ANSWER_DOM_FILE = Path(__file__).resolve().parent / "ultralytics_answer_dom.html"


def extract_text_from_html(html: str) -> str:
    """Extract plain text from HTML: strip script/style, remove tags, normalize whitespace."""
    if not html or not html.strip():
        return ""
    s = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    s = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def ask_ultralytics_ai(
    question: str | None = None,
    headless: bool = False,
    timeout_ms: int = 30000,
    save_response_path: Path | None = RESPONSE_FILE,
) -> str | None:
    """
    Open deployment docs, click Ask AI, type question in the chat input.
    Waits for response, saves it to save_response_path, and returns it.

    Returns the response text if captured, else None. Caller can use it to "follow" suggestions (after asking user).
    """
    if sync_playwright is None:
        logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return None

    question = (question or DEFAULT_QUESTION).strip()
    max_chars = getattr(ask_ultralytics_ai, "_max_question_chars", 12000)
    if len(question) > max_chars:
        logger.warning("Question is long; Ultralytics chat may truncate. Trimming to first %s chars.", max_chars)
        question = question[:max_chars]

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                viewport={"width": 1280, "height": 900},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            )
            page = context.new_page()
            page.goto(DEPLOYMENT_URL, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(2000)

            # Click "Ask AI" (floating button, bottom-right on docs page)
            ask_ai_selectors = [
                'button:has-text("Ask AI")',
                'a:has-text("Ask AI")',
                '[data-testid*="ask"]',
                'text=Ask AI',
            ]
            ask_ai_clicked = False
            for sel in ask_ai_selectors:
                loc = page.locator(sel).first
                if loc.count() > 0:
                    try:
                        loc.click(timeout=5000)
                        ask_ai_clicked = True
                        break
                    except Exception:
                        continue
            if not ask_ai_clicked:
                logger.error("Could not find or click 'Ask AI' button.")
                browser.close()
                return None
            page.wait_for_timeout(2500)

            # Chat modal: input with placeholder containing "Ask"
            chat_input_selectors = [
                'input[placeholder*="Ask"]',
                'textarea[placeholder*="Ask"]',
                '[contenteditable="true"]',
            ]
            chat_input = None
            for sel in chat_input_selectors:
                loc = page.locator(sel).first
                if loc.count() > 0:
                    try:
                        loc.wait_for(state="visible", timeout=5000)
                        chat_input = loc
                        break
                    except Exception:
                        continue
            if chat_input is None:
                logger.error("Could not find chat input (Ask anything...).")
                browser.close()
                return None
            chat_input.fill(question)
            page.wait_for_timeout(500)
            chat_input.press("Enter")
            logger.info("Question submitted. Waiting for AI to finish responding, then reading from chat HTML...")

            # 1) Wait for the reply. Prefer chat container last message so we don't capture the docs page.
            reply_markers = ["The Fix", "skip the CLS", "tokenIdx", "outBase", "out: [C, H, W]", "XNNPACK", "Vulkan", "delegate", "quantization", "latency", "ExecuTorch", "fusion", "best practices", "Gradle", "daemon", "gradle.properties", "JVM", "wildcard", "IPv4", "aspect", "Box3", "rotation", "width", "height", "squeeze", "portrait", "landscape", "jagged", "letterbox", "Gaussian", "coordinate", "scale", "center", "ScaleGestureDetector", "focus", "pivot", "origin", "bbox", "translate", "centered", "edge", "padding", "resize", "crop", "ViT", "artifact"]
            # Reject content that looks like the docs page (not the AI reply)
            DOC_PAGE_PHRASES = ("YOLO26", "Model Deployment Options", "Cloud Deployment", "Edge Deployment", "Local Deployment", "Docker", "TensorFlow Lite", "SageMaker", "Kubernetes", "TLS (Transport Layer Security)", "MFA", "RBAC", "model obfuscation", "TEE")
            def is_likely_docs_page(text: str) -> bool:
                if not text or len(text) < 300:
                    return False
                return sum(1 for p in DOC_PAGE_PHRASES if p in text) >= 2

            answer_html: str | None = None
            popup_html: str | None = None
            question_preview = question[:80].replace("\n", " ")

            def capture_block(block_el) -> tuple[str | None, str | None]:
                out = block_el.evaluate("""
                    (node) => {
                        const block = node.closest('[class*="message"], [class*="assistant"], [class*="response"], [class*="markdown"], [role="log"] > *') || node.closest('div[class*="content"]') || node.parentElement?.parentElement?.parentElement || node;
                        if (!block) return { text: node.innerText, html: node.outerHTML };
                        return { text: block.innerText, html: block.outerHTML };
                    }
                """)
                return (out.get("text"), out.get("html"))

            for attempt in range(50):  # 50 * 3s = 150s max
                page.wait_for_timeout(3000)
                # Prefer assistant message in chat (below question): Ultralytics DOM has .ult-message-group[data-role="assistant"] or .ult-message.assistant
                try:
                    for assistant_sel in [
                        '.ult-message-group[data-role="assistant"] .ult-message',
                        '.ult-message.assistant',
                        '[data-role="assistant"] .ult-message',
                        'div.ult-message.assistant',
                    ]:
                        loc = page.locator(assistant_sel).last
                        if loc.count() > 0 and loc.is_visible():
                            t = loc.inner_text(timeout=2000)
                            if t and len(t.strip()) > 150 and not is_likely_docs_page(t) and question_preview[:50] not in t:
                                answer_html = loc.evaluate("node => node.outerHTML")
                                logger.info("Captured assistant message from chat DOM (below question)")
                                break
                        if answer_html:
                            break
                except Exception:
                    pass
                if answer_html:
                    break
                for marker in reply_markers:
                    try:
                        el = page.locator(f"text=/{marker}/").last
                        if el.count() > 0 and el.is_visible():
                            text, html = capture_block(el)
                            if text and html:
                                t = text.strip()
                                if len(t) > 100 and question_preview not in t and not t.startswith("For TinyViT") and not t.startswith("We run a ViT") and not is_likely_docs_page(t):
                                    answer_html = html
                                    logger.info("Captured answer block (marker '%s')", marker)
                                    break
                    except Exception:
                        continue
                if answer_html:
                    break
                for marker in ["589824", "CLS token", "ArrayIndexOutOfBoundsException"]:
                    try:
                        el = page.locator(f"text=/{marker}/").last
                        if el.count() > 0 and el.is_visible():
                            text, html = capture_block(el)
                            if text and html:
                                t = text.strip()
                                if len(t) > 200 and question_preview not in t and ("The Fix" in t or "tokenIdx" in t or "outBase" in t):
                                    answer_html = html
                                    break
                    except Exception:
                        continue
                if answer_html:
                    break
                try:
                    for container in ["[role='log']", "[class*='messages']", "dialog [class*='message']", "[class*='chat']"]:
                        loc = page.locator(container)
                        if loc.count() > 0:
                            children = loc.locator("> *")
                            n = children.count()
                            if n >= 2:
                                last_msg = children.nth(n - 1)
                                if last_msg.is_visible():
                                    t = last_msg.inner_text(timeout=2000)
                                    if t and 200 < len(t.strip()) < 50000 and not is_likely_docs_page(t):
                                        if any(m in t for m in ["Fix", "CLS", "token", "delegate", "latency", "ExecuTorch", "quantization", "XNNPACK", "Vulkan", "edge", "padding", "crop", "center", "ViT", "resize", "artifact", "recommend", "suggest", "approach"]):
                                            answer_html = last_msg.evaluate("node => node.outerHTML")
                                            logger.info("Captured last message DOM in chat container")
                                            break
                        if answer_html:
                            break
                except Exception:
                    pass
                if answer_html:
                    break
                try:
                    for frame in page.frames:
                        if frame != page.main_frame:
                            for marker in reply_markers:
                                try:
                                    el = frame.locator(f"text=/{marker}/").last
                                    if el.count() > 0 and el.is_visible():
                                        text, html = capture_block(el)
                                        if html and text and len(text.strip()) > 100:
                                            answer_html = html
                                            break
                                except Exception:
                                    continue
                            if answer_html:
                                break
                except Exception:
                    pass

            page.wait_for_timeout(2000)  # Let streaming finish
            # Capture whole popup DOM (dialog or chat wrapper)
            try:
                for popup_sel in ["dialog", "[role='dialog']", "[class*='modal']", "[class*='chat'][class*='container']", "div[class*='ask-ai']"]:
                    pop = page.locator(popup_sel).first
                    if pop.count() > 0 and pop.is_visible():
                        popup_html = pop.evaluate("node => node.outerHTML")
                        if popup_html and len(popup_html) > 500:
                            break
            except Exception:
                pass
            response_text = None
            if popup_html:
                POPUP_DOM_FILE.write_text(popup_html, encoding="utf-8")
                logger.info("Saved popup DOM to %s", POPUP_DOM_FILE)
                # Extract full assistant reply from popup DOM (below question; in case answer_html was truncated)
                try:
                    match = re.search(
                        r'<div[^>]*class="[^"]*ult-message[^"]*assistant[^"]*"[^>]*>([\s\S]*?)</div>\s*(?=<div|$)',
                        popup_html,
                    )
                    if not match:
                        match = re.search(r'data-role="assistant"[\s\S]*?<div[^>]*class="[^"]*ult-message[^"]*"[\s\S]*?>([\s\S]*?)</div>\s*(?=<div\s|</div>\s*</div>)', popup_html)
                    if match:
                        full_assistant_html = match.group(1)
                        full_text = extract_text_from_html(full_assistant_html).strip()
                        if full_text and len(full_text) > 200 and not is_likely_docs_page(full_text):
                            response_text = full_text
                            answer_html = answer_html or full_assistant_html
                except Exception:
                    pass
            if answer_html and not response_text:
                response_text = extract_text_from_html(answer_html)
            if response_text:
                response_text = response_text.strip()
            if not response_text and answer_html:
                response_text = re.sub(r"\s+", " ", answer_html)
                response_text = re.sub(r"<[^>]+>", " ", response_text).strip()

            if response_text:
                print("\n" + "=" * 60 + "\nULTRALYTICS AI RESPONSE\n" + "=" * 60 + "\n")
                print(response_text)
                print("\n" + "=" * 60 + "\n")
                if save_response_path:
                    save_response_path.write_text(response_text, encoding="utf-8")
                    logger.info("Response saved to %s (use this to follow suggestions; ask user before applying)", save_response_path)
            else:
                logger.warning("Could not capture response. Browser will stay open so you can read it.")

            if not headless:
                input("Press Enter to close browser...")
            browser.close()
        return response_text if response_text else None
    except Exception as e:
        logger.exception("Ask AI flow failed: %s", e)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open Ultralytics deployment docs, click Ask AI, submit a SHARP/Executorch question.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        default=None,
        help="Custom question (default: current problem = Part4b latency, no Vulkan yet)",
    )
    parser.add_argument(
        "--bluish",
        action="store_true",
        help="Use canned question for bluish room (SH DC / BGR vs RGB / color space)",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Use canned question for progress bar / percentage ticking during long inference",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use canned question for Three.js camera at back wall looking at front wall (portrait + landscape)",
    )
    parser.add_argument(
        "--orientation",
        action="store_true",
        help="Use canned question for portrait vs landscape both showing as landscape in 3D room viewer",
    )
    parser.add_argument(
        "--aspect",
        action="store_true",
        help="Use canned question for landscape squeezed, portrait expanded (snake-like edges); correct Box3/aspect mapping",
    )
    parser.add_argument(
        "--aspect-int8",
        action="store_true",
        help="Use question from ultralytics_question_aspect_int8.txt (aspect distortion + full ExecuTorch INT8 code)",
    )
    parser.add_argument(
        "--jagged",
        action="store_true",
        help="Use question from ultralytics_question_jagged_output.txt (jagged/screwed-up 3D output, 3840x2160, aspect 1.78)",
    )
    parser.add_argument(
        "--centering",
        action="store_true",
        help="Use question from ultralytics_question_05x_centering.txt (0.5x overlay/photos not in middle)",
    )
    parser.add_argument(
        "--position",
        action="store_true",
        help="Use question from ultralytics_question_position_wrong.txt (position still wrong after center-crop + raw coords)",
    )
    parser.add_argument(
        "--validate-input",
        action="store_true",
        help="Validate resize-fit + edge-pad input approach (lenient, no black bars) vs center crop; from ultralytics_question_validate_input.txt",
    )
    parser.add_argument(
        "--jagged-letterbox",
        action="store_true",
        help="Jagged output with 114-gray letterbox (whole pic); from ultralytics_question_jagged_letterbox114.txt",
    )
    parser.add_argument(
        "--more-image",
        action="store_true",
        help="Include more image without hurting quality; describe center-crop, ask for alternatives; from ultralytics_question_more_image_keep_quality.txt",
    )
    parser.add_argument(
        "--swift-vs-android",
        action="store_true",
        help="Why Swift (stretch-to-square) did not cause jagged output vs Android letterbox; from ultralytics_question_swift_vs_android_preprocess.txt",
    )
    parser.add_argument(
        "--lanczos",
        action="store_true",
        help="Best way to do Lanczos resize on Android for 1536 input; from ultralytics_question_lanczos_resize.txt",
    )
    parser.add_argument(
        "--gradle",
        action="store_true",
        help="Use canned question for Gradle build failure (wildcard IP / sysconf on macOS)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser headless",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30000,
        help="Page load timeout in ms (default 30000)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save response to ultralytics_response.txt",
    )
    args = parser.parse_args()
    ask_ultralytics_ai._max_question_chars = 12000
    if args.aspect_int8:
        if QUESTION_ASPECT_INT8_FILE.is_file():
            question = QUESTION_ASPECT_INT8_FILE.read_text(encoding="utf-8").strip()
            ask_ultralytics_ai._max_question_chars = 20000  # allow full INT8 code in prompt
        else:
            logger.error("File not found: %s", QUESTION_ASPECT_INT8_FILE)
            question = ASPECT_SQUEEZE_QUESTION
    elif args.jagged:
        if QUESTION_JAGGED_OUTPUT_FILE.is_file():
            question = QUESTION_JAGGED_OUTPUT_FILE.read_text(encoding="utf-8").strip()
        else:
            logger.error("File not found: %s", QUESTION_JAGGED_OUTPUT_FILE)
            question = ASPECT_SQUEEZE_QUESTION
    elif args.centering:
        if QUESTION_05X_CENTERING_FILE.is_file():
            question = QUESTION_05X_CENTERING_FILE.read_text(encoding="utf-8").strip()
        else:
            logger.error("File not found: %s", QUESTION_05X_CENTERING_FILE)
            question = ASPECT_SQUEEZE_QUESTION
    elif args.position:
        if QUESTION_POSITION_WRONG_FILE.is_file():
            question = QUESTION_POSITION_WRONG_FILE.read_text(encoding="utf-8").strip()
        else:
            logger.error("File not found: %s", QUESTION_POSITION_WRONG_FILE)
            question = ASPECT_SQUEEZE_QUESTION
    elif args.validate_input:
        if QUESTION_VALIDATE_INPUT_FILE.is_file():
            question = QUESTION_VALIDATE_INPUT_FILE.read_text(encoding="utf-8").strip()
        else:
            logger.error("File not found: %s", QUESTION_VALIDATE_INPUT_FILE)
            question = ASPECT_SQUEEZE_QUESTION
    elif args.jagged_letterbox:
        if QUESTION_JAGGED_LETTERBOX114_FILE.is_file():
            question = QUESTION_JAGGED_LETTERBOX114_FILE.read_text(encoding="utf-8").strip()
        else:
            logger.error("File not found: %s", QUESTION_JAGGED_LETTERBOX114_FILE)
            question = ASPECT_SQUEEZE_QUESTION
    elif args.more_image:
        if QUESTION_MORE_IMAGE_KEEP_QUALITY_FILE.is_file():
            question = QUESTION_MORE_IMAGE_KEEP_QUALITY_FILE.read_text(encoding="utf-8").strip()
        else:
            logger.error("File not found: %s", QUESTION_MORE_IMAGE_KEEP_QUALITY_FILE)
            question = ASPECT_SQUEEZE_QUESTION
    elif args.swift_vs_android:
        if QUESTION_SWIFT_VS_ANDROID_FILE.is_file():
            question = QUESTION_SWIFT_VS_ANDROID_FILE.read_text(encoding="utf-8").strip()
        else:
            logger.error("File not found: %s", QUESTION_SWIFT_VS_ANDROID_FILE)
            question = ASPECT_SQUEEZE_QUESTION
    elif args.lanczos:
        if QUESTION_LANCZOS_RESIZE_FILE.is_file():
            question = QUESTION_LANCZOS_RESIZE_FILE.read_text(encoding="utf-8").strip()
        else:
            logger.error("File not found: %s", QUESTION_LANCZOS_RESIZE_FILE)
            question = ASPECT_SQUEEZE_QUESTION
    else:
        question = (
            PROGRESS_BAR_QUESTION if args.progress_bar
            else CAMERA_POSITION_QUESTION if args.camera
            else ORIENTATION_QUESTION if args.orientation
            else ASPECT_SQUEEZE_QUESTION if args.aspect
            else BLUISH_QUESTION if args.bluish
            else GRADLE_BUILD_QUESTION if args.gradle
            else args.question
        )
    save_path = None if args.no_save else RESPONSE_FILE
    response = ask_ultralytics_ai(
        question=question,
        headless=args.headless,
        timeout_ms=args.timeout,
        save_response_path=save_path,
    )
    return 0 if response is not None else 1


if __name__ == "__main__":
    sys.exit(main())
