"""
Auto-Installer Utilities
========================
Handles runtime installation of missing Python packages,
locates Tesseract / Poppler system binaries at common paths on all
platforms, and downloads the Argos RU→EN translation model.

Design principles
-----------------
- Python packages : install via `pip` using the *current* interpreter
  (sys.executable) so the right venv is always targeted.
- System binaries  : cannot be pip-installed; we probe common install
  locations and configure the library path accordingly. If not found
  we surface a clear, platform-specific install command.
- Argos model      : download once from the official index using the
  argostranslate package API; requires an internet connection the
  first time only.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform constants
# ---------------------------------------------------------------------------
_SYSTEM = platform.system()  # "Windows" | "Darwin" | "Linux"

# Common Tesseract installation paths (checked in order)
_TESSERACT_WINDOWS_PATHS: List[str] = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Tesseract-OCR\tesseract.exe",
    r"C:\tesseract\tesseract.exe",
    # UB-Mannheim versioned install (e.g. Tesseract-OCR 5.x)
    r"C:\Program Files\Tesseract-OCR 5\tesseract.exe",
    r"C:\Users\{user}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{user}\AppData\Local\Tesseract-OCR\tesseract.exe",
]
_TESSERACT_MACOS_PATHS: List[str] = [
    "/usr/local/bin/tesseract",          # Intel Homebrew
    "/opt/homebrew/bin/tesseract",       # Apple-Silicon Homebrew
    "/usr/bin/tesseract",
]
_TESSERACT_LINUX_PATHS: List[str] = [
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]

# Common Poppler/pdftoppm paths
_POPPLER_WINDOWS_PATHS: List[str] = [
    r"C:\Program Files\poppler\bin",
    r"C:\Program Files (x86)\poppler\bin",
    r"C:\poppler\bin",
    r"C:\poppler-23\bin",
    r"C:\poppler-24\bin",
]

# Python package name → importable name mapping
_PYTHON_PACKAGES: Dict[str, str] = {
    "pytesseract": "pytesseract",
    "pdf2image": "pdf2image",
    "Pillow": "PIL",
    "opencv-python": "cv2",
    "pymupdf": "fitz",
    "reportlab": "reportlab",
    "python-docx": "docx",
    "argostranslate": "argostranslate",
    "numpy": "numpy",
    "pandas": "pandas",
}


# ---------------------------------------------------------------------------
# Python package helpers
# ---------------------------------------------------------------------------

def is_importable(import_name: str) -> bool:
    """Return True when *import_name* can be imported."""
    return importlib.util.find_spec(import_name) is not None


def pip_install(packages: List[str]) -> Tuple[bool, str]:
    """
    Install one or more pip packages using the current Python interpreter.

    Returns
    -------
    (success: bool, message: str)
    """
    if not packages:
        return True, "Nothing to install."

    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + packages
    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return True, f"Successfully installed: {', '.join(packages)}"
        err = (result.stderr or result.stdout or "Unknown error").strip()
        return False, f"pip failed (rc={result.returncode}): {err}"
    except subprocess.TimeoutExpired:
        return False, "pip timed out after 5 minutes."
    except Exception as exc:
        return False, f"pip error: {exc}"


def install_missing_python_packages() -> List[Tuple[str, bool, str]]:
    """
    Check every required Python package and pip-install any that are missing.

    Returns
    -------
    List of (pip_name, success, message) for every package that was absent.
    """
    missing = [
        pip_name
        for pip_name, import_name in _PYTHON_PACKAGES.items()
        if not is_importable(import_name)
    ]

    if not missing:
        return []

    ok, msg = pip_install(missing)
    return [(pkg, ok, msg) for pkg in missing]


# ---------------------------------------------------------------------------
# Tesseract binary discovery & configuration
# ---------------------------------------------------------------------------

def _candidate_tesseract_paths() -> List[str]:
    user = os.environ.get("USERNAME") or os.environ.get("USER") or "user"
    if _SYSTEM == "Windows":
        paths = [p.replace("{user}", user) for p in _TESSERACT_WINDOWS_PATHS]
        # Also glob versioned Program Files entries
        for base in [Path(r"C:\Program Files"), Path(r"C:\Program Files (x86)")]:
            if base.exists():
                for entry in base.iterdir():
                    if "tesseract" in entry.name.lower():
                        exe = entry / "tesseract.exe"
                        if exe.exists():
                            paths.insert(0, str(exe))
        return paths
    if _SYSTEM == "Darwin":
        return _TESSERACT_MACOS_PATHS
    return _TESSERACT_LINUX_PATHS


def find_tesseract_binary() -> Optional[str]:
    """
    Search known install locations for the Tesseract executable.
    Returns the full path string if found, or None.
    """
    for path in _candidate_tesseract_paths():
        if Path(path).is_file():
            logger.info("Found Tesseract at: %s", path)
            return path
    return None


def configure_tesseract() -> Tuple[bool, str]:
    """
    Try to make Tesseract available to pytesseract:
      1. Test PATH first (already works on most systems).
      2. Search common install locations.
      3. Set pytesseract.pytesseract.tesseract_cmd if found.

    Returns
    -------
    (available: bool, message: str)
    """
    try:
        import pytesseract
    except ImportError:
        return False, "pytesseract not installed — run pip install pytesseract"

    # Step 1: already in PATH?
    try:
        ver = pytesseract.get_tesseract_version()
        return True, f"Tesseract {ver} found in PATH."
    except Exception:
        pass

    # Step 2: probe common locations
    found = find_tesseract_binary()
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        try:
            ver = pytesseract.get_tesseract_version()
            return True, f"Tesseract {ver} configured at: {found}"
        except Exception as exc:
            return False, f"Found binary at {found} but could not run it: {exc}"

    # Step 3: give a platform-specific install command
    install_hint = _tesseract_install_hint()
    return False, f"Tesseract not found.\n\n{install_hint}"


def _tesseract_install_hint() -> str:
    if _SYSTEM == "Windows":
        return (
            "Install Tesseract on Windows:\n"
            "  1. Download the installer from:\n"
            "     https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  2. During setup tick 'Additional language data' → Russian\n"
            "  3. Add the install folder to your PATH, OR restart this app\n"
            "     (the app will find it automatically in Program Files).\n\n"
            "  Quick install via winget (run in PowerShell as Administrator):\n"
            "     winget install UB-Mannheim.TesseractOCR\n\n"
            "  Quick install via Chocolatey:\n"
            "     choco install tesseract --pre"
        )
    if _SYSTEM == "Darwin":
        return (
            "Install Tesseract on macOS:\n"
            "  brew install tesseract\n"
            "  brew install tesseract-lang   # all language packs incl. Russian"
        )
    return (
        "Install Tesseract on Linux:\n"
        "  sudo apt-get install -y tesseract-ocr tesseract-ocr-rus\n"
        "  # Optional extra languages:\n"
        "  sudo apt-get install -y tesseract-ocr-deu tesseract-ocr-fra "
        "tesseract-ocr-ita tesseract-ocr-ces"
    )


# ---------------------------------------------------------------------------
# Poppler discovery (needed by pdf2image)
# ---------------------------------------------------------------------------

def _find_poppler_windows() -> Optional[str]:
    for path in _POPPLER_WINDOWS_PATHS:
        p = Path(path)
        if p.is_dir() and (p / "pdftoppm.exe").is_file():
            return str(p)
    # Glob search
    for base in [Path(r"C:\Program Files"), Path(r"C:\Program Files (x86)"), Path("C:\\")]:
        if not base.exists():
            continue
        for entry in base.iterdir():
            if "poppler" in entry.name.lower():
                candidate = entry / "bin"
                if (candidate / "pdftoppm.exe").is_file():
                    return str(candidate)
    return None


def configure_poppler() -> Tuple[bool, Optional[str]]:
    """
    Locate pdftoppm (Poppler) and return (found: bool, bin_path_or_None).
    On Linux/macOS Poppler is usually in PATH; on Windows we search common paths.
    """
    if _SYSTEM != "Windows":
        # Just verify pdftoppm is callable
        result = subprocess.run(
            ["pdftoppm", "-v"], capture_output=True, text=True
        )
        return result.returncode == 0, None

    # Windows: search common locations
    bin_path = _find_poppler_windows()
    if bin_path:
        # Add to os.environ PATH so pdf2image can find it
        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
        logger.info("Poppler configured at: %s", bin_path)
        return True, bin_path
    return False, None


def poppler_install_hint() -> str:
    if _SYSTEM == "Windows":
        return (
            "Install Poppler on Windows:\n"
            "  1. Download from:\n"
            "     https://github.com/oschwartz10612/poppler-windows/releases\n"
            "  2. Extract and add the bin\\ folder to your PATH.\n\n"
            "  Quick install via winget:\n"
            "     winget install poppler\n\n"
            "  Quick install via Chocolatey:\n"
            "     choco install poppler"
        )
    if _SYSTEM == "Darwin":
        return "brew install poppler"
    return "sudo apt-get install -y poppler-utils"


# ---------------------------------------------------------------------------
# Argos Translate model
# ---------------------------------------------------------------------------

def install_argos_model(
    from_code: str = "ru",
    to_code: str = "en",
    progress_callback=None,
) -> Tuple[bool, str]:
    """
    Download and install an Argos Translate language model from the official
    package index. Requires a one-time internet connection.

    Parameters
    ----------
    from_code         : ISO 639-1 source language code.
    to_code           : ISO 639-1 target language code.
    progress_callback : Optional callable(message: str) for status updates.

    Returns
    -------
    (success: bool, message: str)
    """
    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        import argostranslate.package
    except ImportError:
        return False, "argostranslate is not installed. Run: pip install argostranslate"

    try:
        _progress("Fetching Argos package index …")
        argostranslate.package.update_package_index()

        available = argostranslate.package.get_available_packages()
        pkg = next(
            (p for p in available if p.from_code == from_code and p.to_code == to_code),
            None,
        )
        if pkg is None:
            return False, (
                f"No Argos package found for {from_code}→{to_code}. "
                "Check your internet connection and try again."
            )

        _progress(f"Downloading {from_code}→{to_code} model (~100 MB) …")
        download_path = pkg.download()

        _progress("Installing model …")
        argostranslate.package.install_from_path(download_path)

        return True, f"Argos {from_code}→{to_code} model installed successfully."

    except Exception as exc:
        return False, f"Model installation failed: {exc}"


# ---------------------------------------------------------------------------
# All-in-one startup routine
# ---------------------------------------------------------------------------

class StartupResult:
    """Aggregates the results of the startup dependency check."""

    def __init__(self) -> None:
        self.python_packages_installed: List[Tuple[str, bool, str]] = []
        self.tesseract_ok: bool = False
        self.tesseract_message: str = ""
        self.poppler_ok: bool = False
        self.poppler_path: Optional[str] = None
        self.argos_ok: bool = False
        self.argos_message: str = ""

    @property
    def tesseract_missing(self) -> bool:
        return not self.tesseract_ok

    @property
    def argos_missing(self) -> bool:
        return not self.argos_ok

    @property
    def all_ok(self) -> bool:
        return self.tesseract_ok and self.argos_ok


def run_startup_checks() -> StartupResult:
    """
    Execute all dependency checks and auto-fixes at application startup.

    1. Install missing Python packages via pip.
    2. Locate / configure Tesseract.
    3. Locate / configure Poppler (Windows only).
    4. Verify the Argos RU→EN model (does NOT auto-download — needs UI consent).

    Returns a StartupResult summary.
    """
    result = StartupResult()

    # ── 1. Python packages ──────────────────────────────────────────────────
    result.python_packages_installed = install_missing_python_packages()

    # ── 2. Tesseract ────────────────────────────────────────────────────────
    result.tesseract_ok, result.tesseract_message = configure_tesseract()

    # ── 3. Poppler ──────────────────────────────────────────────────────────
    result.poppler_ok, result.poppler_path = configure_poppler()

    # ── 4. Argos model ──────────────────────────────────────────────────────
    try:
        from translation.translator import check_ru_en_model
        result.argos_ok = check_ru_en_model()
        result.argos_message = (
            "RU→EN model is installed."
            if result.argos_ok
            else "RU→EN model not found."
        )
    except Exception as exc:
        result.argos_ok = False
        result.argos_message = str(exc)

    return result
