#!/usr/bin/env python3
"""
Open Ultralytics deployment docs, click Ask AI, and submit a SHARP/Executorch-focused question.

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

# Default: current problem (Part4b latency). Ask for concrete Python and Android changes.
CURRENT_PROBLEM_QUESTION = """We run a ViT-based 3D reconstruction model (SHARP) on Android with ExecuTorch INT8. Part4b (image encoder B + decoder + Gaussian head) takes ~80 seconds per inference; Part1+2 ~31s, Part3 ~1.4s, Part4a chunks ~3.3s. We need to reduce Part4b latency without Vulkan. Please give concrete changes: (1) Python: export or model changes for SDPA/fused attention, memory planning, or quantization when exporting to ExecuTorch .pte. (2) Android/Kotlin: runtime or build changes to use optimized attention or XNNPACK settings when loading and running the Part4b module. Include code snippets or step-by-step edits for both Python and Android. Focus on ExecuTorch and ViT decoder on mobile CPU, not YOLO."""
DEFAULT_QUESTION = CURRENT_PROBLEM_QUESTION

# Other canned questions (use -q with paste or a file).
DEPLOYMENT_QUESTION = """I'm deploying a ViT-based 3D reconstruction model (SHARP) to Android with ExecuTorch—not YOLO. The encoder outputs (B, 577, 1024) with a CLS token; we reshape to spatial [C, 24, 24] for the decoder and must skip the CLS token to avoid buffer overrun at index 589824. What deployment best practices do you recommend for this kind of vision transformer on edge: export format (ExecuTorch vs ONNX), quantization, and memory layout for TinyViT-style (B, HW+1, C) outputs? Please focus on ViT/TinyViT and ExecuTorch deployment, not object detection."""

# Use with: python ultralytics_ask_ai.py -q "$(cat android/ultralytics_question_576_vs_577.txt)"
QUESTION_576_VS_577 = """For TinyViT / PatchEmbed: when processing a single 384x384 patch, can the encoder output (B, N, C) have N=576 (spatial only) instead of N=577 (CLS + 24x24)? We get ArrayIndexOutOfBoundsException at index 589824 when reshaping Part2 output for that single-patch path—our buffer is 576*1024. Should reshapeToSpatial assume 577 tokens and skip CLS, or 576 tokens with no CLS depending on tensor size? Please focus on TinyViT single-patch output shape."""

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
    if len(question) > 12000:  # ~200 lines at ~60 chars
        logger.warning("Question is long; Ultralytics chat may truncate. Trimming to first 12000 chars.")
        question = question[:12000]

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

            # 1) Wait for the reply, then capture whole popup DOM and latest answer DOM; extract answer from HTML
            reply_markers = ["The Fix", "skip the CLS", "tokenIdx", "outBase", "out: [C, H, W]", "XNNPACK", "Vulkan", "delegate", "quantization", "latency", "ExecuTorch", "fusion", "best practices"]
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
                for marker in reply_markers:
                    try:
                        el = page.locator(f"text=/{marker}/").last
                        if el.count() > 0 and el.is_visible():
                            text, html = capture_block(el)
                            if text and html:
                                t = text.strip()
                                if len(t) > 100 and question_preview not in t and not t.startswith("For TinyViT") and not t.startswith("We run a ViT"):
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
                                    if t and 200 < len(t.strip()) < 50000 and ("Fix" in t or "CLS" in t or "589824" in t or "token" in t or "delegate" in t or "latency" in t or "ExecuTorch" in t or "quantization" in t or "XNNPACK" in t or "Vulkan" in t):
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
            if popup_html:
                POPUP_DOM_FILE.write_text(popup_html, encoding="utf-8")
                logger.info("Saved popup DOM to %s", POPUP_DOM_FILE)
            if answer_html:
                ANSWER_DOM_FILE.write_text(answer_html, encoding="utf-8")
                logger.info("Saved latest answer DOM to %s", ANSWER_DOM_FILE)
            response_text = extract_text_from_html(answer_html) if answer_html else None
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
    save_path = None if args.no_save else RESPONSE_FILE
    response = ask_ultralytics_ai(
        question=args.question,
        headless=args.headless,
        timeout_ms=args.timeout,
        save_response_path=save_path,
    )
    return 0 if response is not None else 1


if __name__ == "__main__":
    sys.exit(main())
