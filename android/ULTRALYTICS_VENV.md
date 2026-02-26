# Ultralytics Ask AI – venv and Playwright

The `ultralytics_ask_ai.py` script opens Ultralytics docs and uses Ask AI in a browser. It needs Playwright and a browser.

## 1. Use the `ultralytics` venv

From the **android** directory:

```bash
cd android
source ultralytics/bin/activate   # Linux/macOS
# or:  ultralytics\Scripts\activate   # Windows
```

## 2. Install Playwright in the venv (one-time)

With the venv activated:

```bash
pip install -r requirements_ultralytics.txt
playwright install
```

The first command installs the `playwright` Python package; the second downloads the Chromium browser so the script can run headless.

## 3. Run the script

Still in **android**, with the venv activated:

```bash
# Validate resize-fit + edge-pad input approach (no black bars)
python ultralytics_ask_ai.py --validate-input --headless

# Other options:
# python ultralytics_ask_ai.py --gradle --headless
# python ultralytics_ask_ai.py --jagged --headless
# python ultralytics_ask_ai.py -q "Your question..."
```

The reply is printed and saved to `ultralytics_response.txt`.

## If you didn’t create the venv yet

From **android**:

```bash
python3 -m venv ultralytics
source ultralytics/bin/activate
pip install -r requirements_ultralytics.txt
playwright install
```

Then run the script as above.
