# OCR Document Translator

A fully **offline** Streamlit application that performs OCR on PDF and image
files, detects and translates Russian text to English, preserves the document
layout including embedded images, and exports translated **PDF** or **DOCX**
files — all without any paid or cloud APIs.

---

## Features

| Capability | Details |
|---|---|
| **File inputs** | PDF (scanned or native), JPG, JPEG, PNG |
| **OCR engine** | Tesseract via pytesseract (word-level bounding boxes) |
| **Languages** | English, Russian, German, French, Italian, Czech |
| **Translation** | Russian → English, fully offline (Argos Translate) |
| **Layout** | Preserves text positions, multi-column detection, embedded images |
| **Output** | Translated PDF, DOCX with thumbnails, plain text, JSON metadata |
| **UI** | Streamlit with progress bars, bbox overlays, confidence scores |

---

## Screenshots (description)

### Main screen (no file loaded)
A welcome panel explains the workflow. The left sidebar contains all controls:
OCR language selector, confidence slider, DPI picker, preprocessing toggles
(grayscale, binarize, denoise, deskew), translation on/off switch, bounding-box
overlay toggle, and output format radio.

### After running OCR
A summary metrics row shows: pages processed, total words detected, words
translated, total processing time, and translation percentage.

Below the metrics, each page appears in a collapsible expander.
The **left column** shows the original page image, optionally with coloured
bounding boxes overlaid (green = kept as-is, red = translated from Russian).
The **right column** shows each text block: original Russian text in grey
italic, followed by the English translation in black, with a confidence badge.

### Download panel
Three buttons are shown side-by-side:
- **JSON** — structured OCR + translation metadata
- **Plain Text** — clean translated text
- **Translated PDF / DOCX** — reconstructed document (depending on sidebar selection)

---

## Installation

### 1. Python environment

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

### 2. Tesseract OCR

Tesseract must be installed separately and available in your system `PATH`.

#### Windows
1. Download the installer from
   <https://github.com/UB-Mannheim/tesseract/wiki>
2. During installation, tick **"Additional language data"** and select
   **Russian (rus)** and any other languages you need.
3. Add the Tesseract install directory (e.g. `C:\Program Files\Tesseract-OCR`)
   to your `PATH` environment variable.

#### macOS
```bash
brew install tesseract
brew install tesseract-lang   # installs all language packs including Russian
```

#### Linux (Debian / Ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus
# Optional extra languages:
sudo apt-get install -y tesseract-ocr-deu tesseract-ocr-fra \
                        tesseract-ocr-ita tesseract-ocr-ces
```

Verify the installation:
```bash
tesseract --version
tesseract --list-langs
```

---

### 3. Poppler (required for PDF support)

`pdf2image` depends on the Poppler PDF rendering library.

#### Windows
1. Download a pre-built binary from
   <https://github.com/oschwartz10612/poppler-windows/releases>
2. Extract the archive and add the `bin/` folder to your `PATH`.

#### macOS
```bash
brew install poppler
```

#### Linux (Debian / Ubuntu)
```bash
sudo apt-get install -y poppler-utils
```

---

### 4. Argos Translate — Russian → English model

Argos Translate ships the translation engine; the language model must be
downloaded separately (one-time, ~100 MB, requires internet).

#### Option A — Automatic (recommended)
```bash
python - <<'EOF'
import argostranslate.package

argostranslate.package.update_package_index()
available = argostranslate.package.get_available_packages()
pkg = next(
    p for p in available
    if p.from_code == "ru" and p.to_code == "en"
)
argostranslate.package.install_from_path(pkg.download())
print("RU→EN model installed successfully.")
EOF
```

#### Option B — Manual
1. Download `translate-ru_en-*.argosmodel` from
   <https://www.argosopentech.com/argospm/index/>
2. Install it:
```bash
argos-translate-cli --install translate-ru_en-*.argosmodel
```

#### Option C — CLI shortcut (if `argos-translate-cli` is available)
```bash
argos-translate-cli --update-package-index
argos-translate-cli --install-package ru en
```

Verify the model is installed:
```python
import argostranslate.translate
langs = argostranslate.translate.get_installed_languages()
ru = next(l for l in langs if l.code == "ru")
print([t.to_lang.code for t in ru.translations_to])  # should include 'en'
```

---

## Running the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Project structure

```
ocr/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── README.md
│
├── ocr/
│   ├── __init__.py
│   ├── engine.py                 # pytesseract wrapper (image_to_data)
│   ├── layout.py                 # Column detection, reading-order sort
│   └── preprocessing.py          # Grayscale, threshold, denoise, deskew
│
├── translation/
│   ├── __init__.py
│   └── translator.py             # Argos Translate wrapper + language detection
│
├── reconstruction/
│   ├── __init__.py
│   ├── pdf_builder.py            # ReportLab PDF with background + positioned text
│   └── docx_builder.py           # python-docx DOCX with thumbnails
│
└── utils/
    ├── __init__.py
    └── file_handler.py           # PDF→images, embedded-image extraction, PIL↔NumPy
```

---

## JSON export format

```json
{
  "filename": "recipe.pdf",
  "pages": [
    {
      "page_number": 1,
      "blocks": [
        {
          "bbox": [120, 45, 480, 28],
          "original_text": "Рецепт борща",
          "translated_text": "Borscht recipe",
          "confidence": 94.7
        }
      ]
    }
  ]
}
```

---

## Configuration notes

| Setting | Default | Notes |
|---|---|---|
| OCR language | `eng+rus` | Passed directly to Tesseract `--lang` |
| Confidence threshold | 30 % | Words below this are discarded |
| Rendering DPI | 200 | Higher → better quality, more RAM |
| Translation | Enabled | Detects Russian by Cyrillic character ratio |

---

## Troubleshooting

### "Tesseract not found"
Make sure `tesseract` is in your `PATH`. On Windows, restart your terminal
after updating `PATH` in System Properties.

### "Russian language pack missing"
```bash
# Linux
sudo apt-get install tesseract-ocr-rus
# macOS
brew install tesseract-lang
```

### "Could not convert PDF — Poppler not found"
Ensure `pdftoppm` (part of Poppler) is in your `PATH`.

### "RU→EN model not installed"
Follow the Argos model installation steps in section 4 above.
The app displays a warning banner with instructions when the model is missing.

### Poor OCR quality on scanned documents
- Enable **Deskew** in the sidebar preprocessing options
- Try **Binarize** for low-contrast documents
- Increase the **Rendering DPI** (300 for dense text)
- Adjust the **Confidence Threshold** downward to capture more text

---

## Technical notes

- All processing runs in-memory (no temporary disk files).
- Results are cached in Streamlit's session state between re-renders.
- The PDF builder uses pixel coordinates directly as ReportLab user-space
  units, so the output page size matches the rasterised image dimensions.
- The Cyrillic detection heuristic flags text as Russian when ≥ 35 % of
  alphabetic characters fall in the Unicode range U+0400–U+04FF.

---

## License

MIT
