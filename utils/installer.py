"""
Auto-Installer Utilities
========================
Handles runtime installation of missing Python packages,
locates Tesseract / Poppler system binaries using every available
detection strategy, and downloads the Argos RU→EN translation model.

Tesseract detection order (Windows)
------------------------------------
1. pytesseract default PATH check
2. `where tesseract` subprocess call  (works even if pytesseract's check fails)
3. Windows Registry  HKLM\\SOFTWARE\\Tesseract-OCR  (UB-Mannheim installer writes here)
4. `winget list` output parsing
5. Known hard-coded paths  (Program Files, AppData, …)
6. Recursive glob across Program Files / C:\\Users\\<user>\\AppData
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()   # "Windows" | "Darwin" | "Linux"

# ---------------------------------------------------------------------------
# Constant candidate lists
# ---------------------------------------------------------------------------

_TESSERACT_WINDOWS_HARDCODED: List[str] = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR 5\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR 5.3\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR 5.4\tesseract.exe",
    r"C:\Tesseract-OCR\tesseract.exe",
    r"C:\tesseract\tesseract.exe",
    r"C:\tools\tesseract\tesseract.exe",          # Chocolatey default
    # Per-user locations (filled in at runtime)
    r"C:\Users\{user}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{user}\AppData\Local\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{user}\scoop\apps\tesseract\current\tesseract.exe",  # Scoop
]

_TESSERACT_MACOS: List[str] = [
    "/usr/local/bin/tesseract",      # Homebrew Intel
    "/opt/homebrew/bin/tesseract",   # Homebrew Apple-Silicon
    "/usr/bin/tesseract",
]

_TESSERACT_LINUX: List[str] = [
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
]

_POPPLER_WINDOWS_HARDCODED: List[str] = [
    r"C:\Program Files\poppler\bin",
    r"C:\Program Files (x86)\poppler\bin",
    r"C:\poppler\bin",
    r"C:\tools\poppler\bin",                       # Chocolatey
    r"C:\Users\{user}\scoop\apps\poppler\current\bin",
]

_PYTHON_PACKAGES: Dict[str, str] = {
    "pytesseract":  "pytesseract",
    "pdf2image":    "pdf2image",
    "Pillow":       "PIL",
    "opencv-python":"cv2",
    "pymupdf":      "fitz",
    "reportlab":    "reportlab",
    "python-docx":  "docx",
    "argostranslate":"argostranslate",
    "numpy":        "numpy",
    "pandas":       "pandas",
}


# ---------------------------------------------------------------------------
# Python package helpers
# ---------------------------------------------------------------------------

def is_importable(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def pip_install(packages: List[str]) -> Tuple[bool, str]:
    if not packages:
        return True, "Nothing to install."
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + packages
    logger.info("pip install: %s", " ".join(packages))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            return True, f"Installed: {', '.join(packages)}"
        return False, (r.stderr or r.stdout or "Unknown pip error").strip()
    except subprocess.TimeoutExpired:
        return False, "pip timed out after 5 minutes."
    except Exception as exc:
        return False, str(exc)


def install_missing_python_packages() -> List[Tuple[str, bool, str]]:
    missing = [
        name for name, imp in _PYTHON_PACKAGES.items()
        if not is_importable(imp)
    ]
    if not missing:
        return []
    ok, msg = pip_install(missing)
    return [(pkg, ok, msg) for pkg in missing]


# ---------------------------------------------------------------------------
# Tesseract detection strategies
# ---------------------------------------------------------------------------

def _user() -> str:
    return os.environ.get("USERNAME") or os.environ.get("USER") or "user"


def _expand(path: str) -> str:
    return path.replace("{user}", _user())


# ── Strategy 1: pytesseract's own check (uses system PATH) ──────────────────
def _try_pytesseract_default() -> Optional[str]:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return pytesseract.pytesseract.tesseract_cmd  # usually just "tesseract"
    except Exception:
        return None


# ── Strategy 2: `where` (Windows) / `which` (Unix) ─────────────────────────
def _try_where_command() -> Optional[str]:
    """Ask the OS shell where tesseract lives."""
    cmd = "where" if _SYSTEM == "Windows" else "which"
    try:
        r = subprocess.run(
            [cmd, "tesseract"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            first_line = r.stdout.strip().splitlines()[0].strip()
            if first_line and Path(first_line).is_file():
                logger.info("'%s tesseract' → %s", cmd, first_line)
                return first_line
    except Exception:
        pass
    return None


# ── Strategy 3: Windows Registry ────────────────────────────────────────────
def _try_registry() -> Optional[str]:
    """UB-Mannheim installer writes InstallDir to the registry."""
    if _SYSTEM != "Windows":
        return None
    try:
        import winreg
        hives_and_keys = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Tesseract-OCR"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Tesseract-OCR"),
            (winreg.HKEY_CURRENT_USER,  r"SOFTWARE\Tesseract-OCR"),
        ]
        for hive, key_path in hives_and_keys:
            try:
                with winreg.OpenKey(hive, key_path) as key:
                    install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
                    exe = Path(install_dir) / "tesseract.exe"
                    if exe.is_file():
                        logger.info("Registry → %s", exe)
                        return str(exe)
            except (FileNotFoundError, OSError):
                continue
    except ImportError:
        pass
    return None


# ── Strategy 4: parse `winget list` output ───────────────────────────────────
def _try_winget_list() -> Optional[str]:
    """
    Run `winget list --name tesseract` and, if found, scan the known
    winget install root (%LOCALAPPDATA%\\Microsoft\\WinGet\\Packages).
    """
    if _SYSTEM != "Windows":
        return None
    try:
        r = subprocess.run(
            ["winget", "list", "--name", "tesseract"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0 or "tesseract" not in r.stdout.lower():
            return None
        # winget installs here by default
        winget_pkg_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkg_root.is_dir():
            for pkg_dir in winget_pkg_root.iterdir():
                if "tesseract" in pkg_dir.name.lower():
                    for exe in pkg_dir.rglob("tesseract.exe"):
                        logger.info("winget packages → %s", exe)
                        return str(exe)
    except Exception:
        pass
    return None


# ── Strategy 5: hard-coded paths list ────────────────────────────────────────
def _try_hardcoded_paths() -> Optional[str]:
    if _SYSTEM == "Windows":
        candidates = [_expand(p) for p in _TESSERACT_WINDOWS_HARDCODED]
    elif _SYSTEM == "Darwin":
        candidates = _TESSERACT_MACOS
    else:
        candidates = _TESSERACT_LINUX

    for path in candidates:
        if Path(path).is_file():
            logger.info("Hard-coded path → %s", path)
            return path
    return None


# ── Strategy 6: recursive glob in known root directories ─────────────────────
def _try_recursive_glob() -> Optional[str]:
    if _SYSTEM != "Windows":
        return None

    search_roots = [
        Path(r"C:\Program Files"),
        Path(r"C:\Program Files (x86)"),
        Path(os.environ.get("LOCALAPPDATA", r"C:\Users\Default\AppData\Local")),
        Path(os.environ.get("APPDATA",      r"C:\Users\Default\AppData\Roaming")),
        Path(os.environ.get("USERPROFILE",  rf"C:\Users\{_user()}")),
    ]

    for root in search_roots:
        if not root.is_dir():
            continue
        try:
            for exe in root.rglob("tesseract.exe"):
                logger.info("Recursive glob → %s", exe)
                return str(exe)
        except PermissionError:
            continue
    return None


# ── Master finder ─────────────────────────────────────────────────────────────
def find_tesseract_binary() -> Optional[str]:
    """
    Try every detection strategy in order; return the first working path.
    """
    for strategy in [
        _try_pytesseract_default,
        _try_where_command,
        _try_registry,
        _try_winget_list,
        _try_hardcoded_paths,
        _try_recursive_glob,
    ]:
        result = strategy()
        if result:
            return result
    return None


def configure_tesseract(custom_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Locate Tesseract using every available strategy and configure pytesseract.

    Parameters
    ----------
    custom_path : A user-supplied path to try first (from the UI input widget).

    Returns
    -------
    (available: bool, message: str)
    """
    try:
        import pytesseract
    except ImportError:
        return False, "pytesseract not installed — run: pip install pytesseract"

    # User-supplied path takes priority
    if custom_path:
        custom_path = custom_path.strip().strip('"').strip("'")
        if Path(custom_path).is_file():
            pytesseract.pytesseract.tesseract_cmd = custom_path
        elif Path(custom_path).is_dir():
            exe = Path(custom_path) / ("tesseract.exe" if _SYSTEM == "Windows" else "tesseract")
            if exe.is_file():
                pytesseract.pytesseract.tesseract_cmd = str(exe)

    # Check if the current cmd already works (covers PATH and prior custom_path)
    try:
        ver = pytesseract.get_tesseract_version()
        return True, f"Tesseract {ver} ready ({pytesseract.pytesseract.tesseract_cmd})"
    except Exception:
        pass

    # Run through all detection strategies
    found = find_tesseract_binary()
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        # Also add its directory to PATH so sub-processes can find it too
        tess_dir = str(Path(found).parent)
        os.environ["PATH"] = tess_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            ver = pytesseract.get_tesseract_version()
            return True, f"Tesseract {ver} — auto-configured at:\n{found}"
        except Exception as exc:
            return False, f"Binary found at {found} but could not execute it: {exc}"

    return False, _tesseract_not_found_message()


def _tesseract_not_found_message() -> str:
    if _SYSTEM == "Windows":
        return (
            "Tesseract was not found in any of the expected locations.\n\n"
            "Option A — winget (PowerShell, no admin needed for user install):\n"
            "  winget install UB-Mannheim.TesseractOCR\n\n"
            "Option B — Chocolatey (run PowerShell as Administrator):\n"
            "  choco install tesseract --pre\n\n"
            "Option C — Manual installer:\n"
            "  https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  ✔ Tick 'Additional language data (download)' → Russian during setup\n\n"
            "After installing, use the 'Enter path manually' box below OR click\n"
            "'🔄 Retry Detection' — the app will find it automatically."
        )
    if _SYSTEM == "Darwin":
        return (
            "  brew install tesseract\n"
            "  brew install tesseract-lang"
        )
    return (
        "  sudo apt-get install -y tesseract-ocr tesseract-ocr-rus\n"
        "  sudo apt-get install -y tesseract-ocr-deu tesseract-ocr-fra "
        "tesseract-ocr-ita tesseract-ocr-ces"
    )


# ---------------------------------------------------------------------------
# Poppler detection
# ---------------------------------------------------------------------------

def _find_poppler_windows() -> Optional[str]:
    user = _user()
    candidates = [_expand(p) for p in _POPPLER_WINDOWS_HARDCODED]

    # Hard-coded first
    for path in candidates:
        p = Path(path)
        if p.is_dir() and (p / "pdftoppm.exe").is_file():
            return str(p)

    # `where pdftoppm`
    try:
        r = subprocess.run(["where", "pdftoppm"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            exe = r.stdout.strip().splitlines()[0]
            return str(Path(exe).parent)
    except Exception:
        pass

    # Recursive glob in Program Files and AppData
    roots = [
        Path(r"C:\Program Files"),
        Path(r"C:\Program Files (x86)"),
        Path(os.environ.get("LOCALAPPDATA", rf"C:\Users\{user}\AppData\Local")),
        Path(os.environ.get("USERPROFILE",  rf"C:\Users\{user}")),
    ]
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for exe in root.rglob("pdftoppm.exe"):
                return str(exe.parent)
        except PermissionError:
            continue
    return None


def configure_poppler(custom_path: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Locate Poppler's bin directory and return (found, bin_path).
    Injects the path into os.environ['PATH'] when found on Windows.
    """
    if _SYSTEM != "Windows":
        r = subprocess.run(["pdftoppm", "-v"], capture_output=True, timeout=5)
        return r.returncode == 0, None

    # User-supplied path
    if custom_path:
        p = Path(custom_path.strip().strip('"'))
        if (p / "pdftoppm.exe").is_file():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
            return True, str(p)

    bin_path = _find_poppler_windows()
    if bin_path:
        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
        logger.info("Poppler configured at: %s", bin_path)
        return True, bin_path
    return False, None


def poppler_install_hint() -> str:
    if _SYSTEM == "Windows":
        return (
            "winget install poppler\n"
            "# — or —\n"
            "choco install poppler\n"
            "# — or manual —\n"
            "# 1. Download from https://github.com/oschwartz10612/poppler-windows/releases\n"
            "# 2. Extract anywhere, then enter the path to the bin\\ folder below."
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
    Download and install the Argos Translate model for *from_code* → *to_code*.

    Returns
    -------
    (success: bool, message: str)
        *success* is True only when the package is installed **and** can be
        verified via ``check_ru_en_model()`` immediately afterwards.
        If the package was extracted correctly but the language graph hasn't
        updated yet (rare, version-dependent), success is still True and the
        message notes that a restart may be needed.
    """
    def _cb(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        import argostranslate.package
    except ImportError:
        return False, "argostranslate not installed — run: pip install argostranslate"

    try:
        _cb("Fetching Argos package index …")
        argostranslate.package.update_package_index()

        available = argostranslate.package.get_available_packages()
        pkg = next(
            (p for p in available if p.from_code == from_code and p.to_code == to_code),
            None,
        )
        if pkg is None:
            return False, (
                f"Package {from_code}→{to_code} not found in the Argos index. "
                "Check your internet connection and try again."
            )

        _cb(f"Downloading {from_code}→{to_code} model (~100 MB) …")
        path = pkg.download()

        _cb("Installing …")
        argostranslate.package.install_from_path(path)

        # ── Verify the installation is detectable ────────────────────────────
        _cb("Verifying …")
        from translation.translator import check_ru_en_model

        if check_ru_en_model():
            return True, f"Argos {from_code}→{to_code} model installed and verified."

        # Package extracted but language graph not refreshed yet — this is
        # harmless; the model will be available after st.rerun() or a restart.
        logger.warning(
            "Package installed on disk but not yet visible to the translate "
            "layer — will resolve after app restart."
        )
        return True, (
            f"Argos {from_code}→{to_code} model installed. "
            "If the warning persists after the page reloads, "
            "please restart the app once — the model will then be available."
        )

    except Exception as exc:
        logger.exception("install_argos_model failed")
        return False, f"Installation failed: {exc}"


# ---------------------------------------------------------------------------
# StartupResult + orchestrator
# ---------------------------------------------------------------------------

class StartupResult:
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


def run_startup_checks(
    custom_tesseract_path: Optional[str] = None,
    custom_poppler_path: Optional[str] = None,
) -> StartupResult:
    """
    Run all dependency checks.  Call with custom paths when the user has
    entered them manually in the UI.
    """
    result = StartupResult()

    result.python_packages_installed = install_missing_python_packages()
    result.tesseract_ok, result.tesseract_message = configure_tesseract(custom_tesseract_path)
    result.poppler_ok, result.poppler_path = configure_poppler(custom_poppler_path)

    try:
        from translation.translator import check_ru_en_model
        result.argos_ok = check_ru_en_model()
        result.argos_message = (
            "RU→EN model ready." if result.argos_ok else "RU→EN model not found."
        )
    except Exception as exc:
        result.argos_ok = False
        result.argos_message = str(exc)

    return result
