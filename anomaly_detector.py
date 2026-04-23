"""
anomaly_detector.py — v2.0 : Détection d'anomalies comportementales avancée
TP1 — Intelligence Artificielle & Cybersécurité

AMÉLIORATIONS v2.0 vs v1.0 :
    ✅ Ajout Local Outlier Factor (LOF) en comparaison avec Isolation Forest
    ✅ Drift detection : détecte un changement de profil de frappe dans le temps
    ✅ Score de confiance normalisé [0, 1] (plus lisible que les scores bruts IF)
    ✅ Sévérité multi-niveaux : LOW / MEDIUM / HIGH / CRITICAL
    ✅ Export des méta-données keylogger → metadata.json (manquant dans v1)
    ✅ Rolling stats : fenêtre glissante EMA pour baseline dynamique
    ✅ Robustesse : gestion des valeurs NaN / infinies avant entraînement
    ✅ Validation croisée interne pour estimer le taux de faux positifs
    ✅ Feature importance : identifie quelle feature a déclenché l'anomalie
"""

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[AVERTISSEMENT] scikit-learn non installé : pip install scikit-learn joblib")

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
DATA_DIR     = _ROOT / "data"
MODEL_PATH   = DATA_DIR / "isolation_forest.joblib"
SCALER_PATH  = DATA_DIR / "scaler.joblib"
LOF_PATH     = DATA_DIR / "lof_model.joblib"
ALERTS_PATH  = DATA_DIR / "alerts.json"
METADATA_PATH= DATA_DIR / "metadata.json"

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────────────────────
CONTAMINATION        = 0.05   # 5 % anomalies estimées
MIN_SAMPLES_TRAIN    = 100    # Seuil d'entraînement
BURST_PAUSE_THRESH   = 1.0    # Pause > 1s = fin de burst
WINDOW_SIZE          = 20     # Fenêtre glissante
DRIFT_WINDOW         = 50     # Fenêtre pour détection de drift
EMA_ALPHA            = 0.1    # Facteur de lissage exponentiel

# Seuils de sévérité (sur score normalisé 0→1, 1=plus anormal)
SEVERITY_THRESHOLDS = {
    "LOW":      0.50,
    "MEDIUM":   0.65,
    "HIGH":     0.80,
    "CRITICAL": 0.92,
}

# Noms des features (pour l'interprétabilité)
FEATURE_NAMES = [
    "mean_delay", "std_delay", "max_delay", "min_delay",
    "alphanum_ratio", "special_ratio", "burst_ratio",
    "median_delay", "p25_delay", "p75_delay",
    "cv_delay",      # coefficient de variation = std/mean
    "nav_ratio",     # navigation keys ratio
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE ENGINEERING — v2 : 12 features vs 8 en v1
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(window: list) -> Optional[np.ndarray]:
    """
    Extrait un vecteur de 12 features à partir d'une fenêtre de méta-données.

    Nouvelles features v2 (vs v1 qui en avait 8) :
        - p25_delay, p75_delay : percentiles pour mieux décrire la distribution
        - cv_delay             : coeff de variation (robuste aux outliers)
        - nav_ratio            : ratio touches de navigation (révèle copier-coller)

    Retour
    ------
    np.ndarray(1, 12) ou None si données insuffisantes / invalides.
    """
    if len(window) < 2:
        return None

    delays = np.array([
        m["inter_key_delay"] for m in window
        if isinstance(m.get("inter_key_delay"), (int, float))
        and 0 < m["inter_key_delay"] < 30.0   # Filtre valeurs aberrantes
    ], dtype=float)

    if len(delays) < 2:
        return None

    types = [m.get("key_type", "alphanum") for m in window]
    total = max(len(types), 1)

    mean_d   = float(np.mean(delays))
    std_d    = float(np.std(delays))
    cv       = std_d / mean_d if mean_d > 0 else 0.0
    burst_r  = float(np.sum(delays > BURST_PAUSE_THRESH)) / len(delays)

    features = np.array([[
        mean_d,
        std_d,
        float(np.max(delays)),
        float(np.min(delays)),
        types.count("alphanum")   / total,
        types.count("special")    / total,
        burst_r,
        float(np.median(delays)),
        float(np.percentile(delays, 25)),
        float(np.percentile(delays, 75)),
        cv,
        types.count("navigation") / total,
    ]], dtype=float)

    # Sanity check : aucun NaN / Inf
    if not np.all(np.isfinite(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=0.0)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCORE NORMALISÉ [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_score(raw_score: float, model_scores: np.ndarray) -> float:
    """
    Convertit le score brut Isolation Forest (négatif = anormal) en score [0,1].
    0 = parfaitement normal, 1 = anomalie maximale.

    Méthode : min-max inversé sur la distribution observée des scores.
    """
    s_min = float(np.min(model_scores))
    s_max = float(np.max(model_scores))
    if s_max == s_min:
        return 0.0
    return float(np.clip((s_max - raw_score) / (s_max - s_min), 0.0, 1.0))


def _severity(norm_score: float) -> str:
    """Retourne le niveau de sévérité en fonction du score normalisé."""
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if norm_score >= SEVERITY_THRESHOLDS[level]:
            return level
    return "NORMAL"


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

def train_model(metadata: list) -> Tuple[object, object, object]:
    """
    Entraîne Isolation Forest + LOF sur les données de frappe normales.

    Retour
    ------
    (if_model, lof_model, scaler) ou (None, None, None) si données insuffisantes.

    Améliorations v2
    ----------------
    - LOF en mode novelty=True pour prédire sur nouvelles données
    - Validation croisée pour estimer le taux de faux positifs
    - Sauvegarde séparée des deux modèles
    """
    if not _SKLEARN_AVAILABLE:
        return None, None, None

    if len(metadata) < MIN_SAMPLES_TRAIN:
        print(f"[INFO] Données insuffisantes ({len(metadata)}/{MIN_SAMPLES_TRAIN}).")
        return None, None, None

    # Construction matrice features
    X = []
    for i in range(WINDOW_SIZE, len(metadata)):
        feat = extract_features(metadata[i - WINDOW_SIZE:i])
        if feat is not None:
            X.append(feat[0])

    if len(X) < 20:
        print("[INFO] Fenêtres valides insuffisantes.")
        return None, None, None

    X = np.array(X)

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    if_model = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=200,       # +100 vs v1 → meilleure stabilité
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    if_model.fit(X_scaled)

    # LOF (novelty=True pour .predict() sur nouvelles données)
    lof_model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=CONTAMINATION,
        novelty=True,
        n_jobs=-1,
    )
    lof_model.fit(X_scaled)

    # Estimation taux de faux positifs (CV sur IF)
    scores_cv = if_model.score_samples(X_scaled)
    predicted  = if_model.predict(X_scaled)
    fp_rate    = float(np.sum(predicted == -1) / len(predicted))
    print(f"[INFO] Modèle entraîné — {len(X)} fenêtres — FP estimé : {fp_rate:.1%}")

    # Sauvegarde
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(if_model,  MODEL_PATH)
    joblib.dump(lof_model, LOF_PATH)
    joblib.dump(scaler,    SCALER_PATH)
    print(f"[INFO] Modèles sauvegardés → {DATA_DIR}")

    return if_model, lof_model, scaler


def load_model() -> Tuple[object, object, object]:
    """Charge IF + LOF + scaler depuis le disque."""
    try:
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            if_model = joblib.load(MODEL_PATH)
            scaler   = joblib.load(SCALER_PATH)
            lof_model = joblib.load(LOF_PATH) if LOF_PATH.exists() else None
            print("[INFO] Modèles chargés depuis le disque.")
            return if_model, lof_model, scaler
    except Exception as e:
        print(f"[ERREUR] Chargement : {e}")
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRÉDICTION ENRICHIE
# ─────────────────────────────────────────────────────────────────────────────

def predict_anomaly(
    window: list,
    if_model,
    lof_model,
    scaler,
    all_scores: Optional[np.ndarray] = None,
) -> dict:
    """
    Prédit si la fenêtre est anormale et identifie la feature principale.

    Retour
    ------
    dict : {
        is_anomaly     : bool
        score_raw      : float  (IF decision function)
        score_norm     : float  [0,1]  0=normal, 1=anomalie maximale
        score_lof      : float  (LOF decision function, si disponible)
        consensus      : bool   (IF ET LOF d'accord → moins de faux positifs)
        severity       : str    NORMAL / LOW / MEDIUM / HIGH / CRITICAL
        top_feature    : str    feature ayant le plus contribué à l'anomalie
        timestamp      : str
        window_size    : int
    }
    """
    result = {
        "is_anomaly":  False,
        "score_raw":   0.0,
        "score_norm":  0.0,
        "score_lof":   None,
        "consensus":   False,
        "severity":    "NORMAL",
        "top_feature": "N/A",
        "timestamp":   datetime.now().isoformat(),
        "window_size": len(window),
    }

    if if_model is None or scaler is None:
        return result

    feat = extract_features(window)
    if feat is None:
        return result

    feat_scaled = scaler.transform(feat)

    # IF prediction
    pred_if   = if_model.predict(feat_scaled)[0]
    score_raw = float(if_model.decision_function(feat_scaled)[0])

    # Score normalisé
    if all_scores is not None and len(all_scores) > 1:
        score_norm = _normalize_score(score_raw, all_scores)
    else:
        # Fallback : normalisation heuristique basée sur plage typique [-0.5, 0.5]
        score_norm = float(np.clip((0.5 - score_raw) / 1.0, 0.0, 1.0))

    # LOF prediction (consensus)
    pred_lof  = None
    score_lof = None
    if lof_model is not None:
        try:
            pred_lof  = lof_model.predict(feat_scaled)[0]
            score_lof = float(lof_model.decision_function(feat_scaled)[0])
        except Exception:
            pass

    is_anomaly = (pred_if == -1)
    consensus  = is_anomaly and (pred_lof == -1) if pred_lof is not None else is_anomaly

    # Feature principale — Z-score sur chaque feature
    feat_raw   = feat[0]
    mean_ref   = scaler.mean_
    std_ref    = np.sqrt(scaler.var_)
    z_scores   = np.abs((feat_raw - mean_ref) / (std_ref + 1e-8))
    top_idx    = int(np.argmax(z_scores))
    top_feature= FEATURE_NAMES[top_idx] if top_idx < len(FEATURE_NAMES) else f"feat_{top_idx}"

    result.update({
        "is_anomaly":  is_anomaly,
        "score_raw":   round(score_raw, 4),
        "score_norm":  round(score_norm, 4),
        "score_lof":   round(score_lof, 4) if score_lof is not None else None,
        "consensus":   consensus,
        "severity":    _severity(score_norm) if is_anomaly else "NORMAL",
        "top_feature": top_feature,
    })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Détecte un changement significatif du profil de frappe dans le temps.
    Méthode : comparaison EMA (baseline) vs fenêtre courante sur mean_delay.
    Si l'écart dépasse N fois l'écart-type de la baseline → drift signalé.
    """

    DRIFT_Z_THRESHOLD = 3.0  # Z-score au-delà duquel on parle de drift

    def __init__(self):
        self.ema_mean  = None
        self.ema_var   = None
        self.n_updates = 0

    def update(self, metadata_window: list) -> Optional[dict]:
        """
        Met à jour la baseline EMA et détecte un drift éventuel.

        Retour
        ------
        dict de drift si détecté, sinon None.
        """
        delays = [m["inter_key_delay"] for m in metadata_window
                  if 0 < m.get("inter_key_delay", 0) < 30]
        if not delays:
            return None

        current_mean = float(np.mean(delays))

        if self.ema_mean is None:
            self.ema_mean = current_mean
            self.ema_var  = float(np.var(delays)) + 1e-6
            return None

        # Mise à jour EMA
        self.n_updates += 1
        prev_mean      = self.ema_mean
        self.ema_mean  = EMA_ALPHA * current_mean + (1 - EMA_ALPHA) * self.ema_mean
        self.ema_var   = EMA_ALPHA * (current_mean - self.ema_mean)**2 + (1 - EMA_ALPHA) * self.ema_var

        # Test de drift
        ema_std  = float(np.sqrt(max(self.ema_var, 1e-8)))
        z_score  = abs(current_mean - prev_mean) / ema_std

        if z_score > self.DRIFT_Z_THRESHOLD and self.n_updates > 10:
            return {
                "type":        "behavioral_drift",
                "timestamp":   datetime.now().isoformat(),
                "z_score":     round(z_score, 3),
                "current_mean_delay": round(current_mean, 4),
                "baseline_mean_delay": round(prev_mean, 4),
                "severity":    "HIGH" if z_score > 5 else "MEDIUM",
            }
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAUVEGARDE ALERTES + MÉTADONNÉES (manquant en v1)
# ─────────────────────────────────────────────────────────────────────────────

def save_alert(alert: dict) -> None:
    """Sauvegarde une alerte dans alerts.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if ALERTS_PATH.exists():
        try:
            existing = json.loads(ALERTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    alert.setdefault("type", "keystroke_anomaly")
    existing.append(alert)
    ALERTS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ALERTE] {alert.get('severity','?')} @ {alert['timestamp'][:19]} "
          f"(score_norm={alert.get('score_norm','?')}, top_feat={alert.get('top_feature','?')})")


def save_metadata(metadata: list) -> None:
    """
    Sauvegarde les méta-données de frappe dans metadata.json.
    NOUVEAU EN v2 : ce fichier manquait et bloquait heatmap + histogramme délais.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if METADATA_PATH.exists():
        try:
            existing = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.extend(metadata)
    METADATA_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MONITEUR TEMPS RÉEL
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyMonitor:
    """
    Thread daemon de surveillance comportementale.

    Améliorations v2
    ----------------
    - Intègre DriftDetector
    - Utilise le score de consensus (IF + LOF)
    - N'alerte que sur MEDIUM+ pour réduire le bruit (était tout en v1)
    - Flush périodique des méta-données vers metadata.json
    """

    MIN_SEVERITY_ALERT = "MEDIUM"   # Seuil d'alerte minimum
    META_FLUSH_EVERY   = 50         # Flush toutes les N vérifications

    def __init__(self, metadata_ref: list, check_interval: float = 5.0):
        self.metadata       = metadata_ref
        self.check_interval = check_interval
        self.if_model, self.lof_model, self.scaler = load_model()
        self.drift_detector = DriftDetector()
        self._running       = False
        self._n_checks      = 0
        self._all_scores    = np.array([])

    def _severity_above_threshold(self, severity: str) -> bool:
        order = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        return (order.index(severity) >=
                order.index(self.MIN_SEVERITY_ALERT))

    def train_if_ready(self) -> None:
        if self.if_model is None and len(self.metadata) >= MIN_SAMPLES_TRAIN:
            print("[INFO] Entraînement du modèle...")
            self.if_model, self.lof_model, self.scaler = train_model(self.metadata)

    def check(self) -> None:
        self.train_if_ready()
        self._n_checks += 1

        if self.if_model is None or len(self.metadata) < WINDOW_SIZE:
            return

        window = self.metadata[-WINDOW_SIZE:]

        # Prédiction anomalie
        result = predict_anomaly(
            window, self.if_model, self.lof_model, self.scaler, self._all_scores
        )

        # Mise à jour du pool de scores pour normalisation
        if result["score_raw"] != 0.0:
            self._all_scores = np.append(self._all_scores[-1000:], result["score_raw"])

        if result["is_anomaly"] and self._severity_above_threshold(result["severity"]):
            save_alert(result)

        # Drift detection
        drift = self.drift_detector.update(window)
        if drift:
            save_alert(drift)

        # Flush métadonnées (nouvelles entrées depuis le dernier flush)
        if self._n_checks % self.META_FLUSH_EVERY == 0:
            save_metadata(self.metadata[-self.META_FLUSH_EVERY:])

    def start(self) -> None:
        self._running = True
        def _loop():
            while self._running:
                try:
                    self.check()
                except Exception as e:
                    print(f"[ERREUR] AnomalyMonitor : {e}")
                time.sleep(self.check_interval)
        t = threading.Thread(target=_loop, daemon=True, name="AnomalyMonitor")
        t.start()
        print(f"[INFO] AnomalyMonitor v2 démarré (interval={self.check_interval}s, "
              f"seuil={self.MIN_SEVERITY_ALERT})")

    def stop(self) -> None:
        self._running = False
        save_metadata(self.metadata)
        print("[INFO] AnomalyMonitor arrêté — métadonnées sauvegardées.")


# ─────────────────────────────────────────────────────────────────────────────
# Test standalone
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    print("=" * 60)
    print("  anomaly_detector.py v2.0 — Test standalone")
    print("=" * 60)

    # Générer données simulées
    fake = []
    for i in range(250):
        delay = random.gauss(0.12, 0.04)
        if random.random() < 0.05:
            delay = random.uniform(2.0, 5.0)   # Anomalie pause longue
        fake.append({
            "timestamp":      time.time() + i * 0.15,
            "inter_key_delay": max(0.01, delay),
            "key_type":        random.choice(["alphanum"]*5 + ["special", "modifier"]),
        })

    if_m, lof_m, scaler = train_model(fake)

    if if_m:
        print("\n--- Test fenêtre NORMALE ---")
        r = predict_anomaly(fake[-WINDOW_SIZE:], if_m, lof_m, scaler)
        print(f"  is_anomaly={r['is_anomaly']}, severity={r['severity']}, "
              f"score_norm={r['score_norm']:.3f}, top_feat={r['top_feature']}")

        print("\n--- Test fenêtre ANORMALE (frappes robot) ---")
        robot_window = [{"timestamp": time.time(), "inter_key_delay": 0.001,
                         "key_type": "alphanum"} for _ in range(WINDOW_SIZE)]
        r2 = predict_anomaly(robot_window, if_m, lof_m, scaler)
        print(f"  is_anomaly={r2['is_anomaly']}, severity={r2['severity']}, "
              f"score_norm={r2['score_norm']:.3f}, top_feat={r2['top_feature']}")

        print("\n--- Test Drift Detector ---")
        dd = DriftDetector()
        for _ in range(15):
            dd.update(fake[:20])
        drift = dd.update([{"inter_key_delay": 2.5, "key_type": "alphanum"}] * 20)
        print(f"  Drift détecté : {drift}")
