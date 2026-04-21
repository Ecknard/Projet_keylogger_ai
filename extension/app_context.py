"""
extension/app_context.py — Extension B : Reconnaissance de contexte applicatif
TP1 — Intelligence Artificielle & Cybersécurité

Objectif : enrichir les logs avec le nom de l'application active au moment de chaque frappe,
pour contextualiser les données.

Compatibilité : Windows (pywin32), Linux (xdotool), macOS (AppKit) + fallback pygetwindow.
"""

import platform
import subprocess
from datetime import datetime

# ---------------------------------------------------------------------------
# Détection de l'OS et import conditionnel
# ---------------------------------------------------------------------------
OS = platform.system()  # "Windows", "Linux", "Darwin"

_win32_available  = False
_wnk_available    = False
_appkit_available = False

if OS == "Windows":
    try:
        import win32gui
        import win32process
        import psutil
        _win32_available = True
    except ImportError:
        pass

if OS in ("Linux", "Darwin"):
    try:
        import pygetwindow as gw
        _wnk_available = True
    except ImportError:
        pass

if OS == "Darwin":
    try:
        from AppKit import NSWorkspace
        _appkit_available = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Fonctions par OS
# ---------------------------------------------------------------------------

def _get_active_window_windows() -> dict:
    """Windows : récupère la fenêtre active via win32gui."""
    if not _win32_available:
        return {"title": "N/A", "process": "N/A", "pid": -1}
    try:
        hwnd    = win32gui.GetForegroundWindow()
        title   = win32gui.GetWindowText(hwnd)
        _, pid  = win32process.GetWindowThreadProcessId(hwnd)
        process = psutil.Process(pid).name()
        return {"title": title, "process": process, "pid": pid}
    except Exception as e:
        return {"title": "Erreur", "process": str(e), "pid": -1}


def _get_active_window_linux() -> dict:
    """Linux (X11) : récupère la fenêtre active via xdotool."""
    try:
        window_id = subprocess.check_output(
            ["xdotool", "getactivewindow"], stderr=subprocess.DEVNULL
        ).decode().strip()
        title = subprocess.check_output(
            ["xdotool", "getwindowname", window_id], stderr=subprocess.DEVNULL
        ).decode().strip()
        pid = subprocess.check_output(
            ["xdotool", "getwindowpid", window_id], stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"title": title, "process": f"PID:{pid}", "pid": int(pid)}
    except Exception:
        # Fallback pygetwindow
        if _wnk_available:
            try:
                wins = gw.getActiveWindow()
                if wins:
                    return {"title": wins.title, "process": "N/A", "pid": -1}
            except Exception:
                pass
        return {"title": "N/A", "process": "N/A", "pid": -1}


def _get_active_window_macos() -> dict:
    """macOS : récupère l'application active via AppKit."""
    if _appkit_available:
        try:
            workspace = NSWorkspace.sharedWorkspace()
            app = workspace.frontmostApplication()
            return {
                "title": app.localizedName(),
                "process": app.bundleIdentifier(),
                "pid": app.processIdentifier(),
            }
        except Exception:
            pass
    # Fallback pygetwindow
    if _wnk_available:
        try:
            wins = gw.getActiveWindow()
            if wins:
                return {"title": wins.title, "process": "N/A", "pid": -1}
        except Exception:
            pass
    return {"title": "N/A", "process": "N/A", "pid": -1}


# ---------------------------------------------------------------------------
# Interface publique
# ---------------------------------------------------------------------------

def get_active_window() -> dict:
    """
    Retourne les informations sur la fenêtre / application active.

    Retour
    ------
    dict : {
        "title"   : str  — titre de la fenêtre
        "process" : str  — nom du processus
        "pid"     : int  — identifiant du processus
        "os"      : str  — système d'exploitation
        "timestamp": str — horodatage ISO
    }
    """
    if OS == "Windows":
        info = _get_active_window_windows()
    elif OS == "Linux":
        info = _get_active_window_linux()
    elif OS == "Darwin":
        info = _get_active_window_macos()
    else:
        info = {"title": "N/A", "process": "N/A", "pid": -1}

    info["os"]        = OS
    info["timestamp"] = datetime.now().isoformat()
    return info


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"OS détecté : {OS}")
    info = get_active_window()
    for k, v in info.items():
        print(f"  {k:12} : {v}")
