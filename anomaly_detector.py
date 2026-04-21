"""
anomaly_detector.py — Partie II, Tâche 7 : Détection d'anomalies sur les patterns de frappe
TP1 — Intelligence Artificielle & Cybersécurité

Algorithme choisi : Isolation Forest (sklearn)
Raison : efficace sur données multidimensionnelles, pas besoin de labels,
         rapide à entraîner, contamination paramétrable.

Pipeline ML :
    1. Collecte des méta-données (keylogger.py → keystroke_metadata)
    2. Feature engineering
    3. Normalisation (StandardScaler)
    4. Entraînement Isolation Forest
    5. Prédiction en temps réel
    6. Alerte JSON horodatée
"""

import json
import os
import time
from datetime import datetime
from typing import Optional

import joblib
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[AVERTISSEMENT] scikit-learn non installé. "
          "Exécutez : pip install scikit-learn joblib")

# ---------------------------------------------------------------------------
# Chemins des fichiers persistants
# ---------------------------------------------------------------------------
MODEL_PATH   = os.path.join("data", "isolation_forest.joblib")
SCALER_PATH  = os.path.join("data", "scaler.joblib")
ALERTS_PATH  = os.path.join("data", "alerts.json")

# ---------------------------------------------------------------------------
# Paramètres du modèle
# ---------------------------------------------------------------------------
CONTAMINATION       = 0.05   # 5 % des données considérées comme anomalies
MIN_SAMPLES_TRAIN   = 100    # Nombre minimum d'événements avant entraînement
BURST_PAUSE_THRESHOLD = 1.0  # Pause > 1s = fin de burst (en secondes)
WINDOW_SIZE         = 20     # Taille de la fenêtre glissante pour la prédiction


# ---------------------------------------------------------------------------
# 1. Feature Engineering (Tâche 7.1)
# ---------------------------------------------------------------------------
def extract_features(metadata_window: list) -> Optional[np.ndarray]:
    """
    Construit un vecteur de features à partir d'une fenêtre de méta-données.

    Features extraites
    ------------------
    - mean_delay      : délai inter-touches moyen
    - std_delay       : écart-type des délais (mesure la régularité)
    - max_delay       : délai maximum (détecte les longues pauses)
    - min_delay       : délai minimum (détecte la sur-vitesse)
    - alphanum_ratio  : proportion de touches alphanumériques
    - special_ratio   : proportion de touches spéciales
    - burst_count     : nombre de bursts détectés
    - median_delay    : médiane des délais (robuste aux outliers)

    Retour
    ------
    np.ndarray de shape (1, 8) ou None si données insuffisantes.
    """
    if len(metadata_window) < 2:
        return None

    delays = [m["inter_key_delay"] for m in metadata_window if m["inter_key_delay"] > 0]
    if not delays:
        return None

    delays_arr = np.array(delays)

    # Ratios par type de touche
    types = [m["key_type"] for m in metadata_window]
    total = len(types)
    alphanum_ratio = types.count("alphanum") / total
    special_ratio  = types.count("special") / total

    # Comptage des bursts (séquences de frappes rapides sans pause > seuil)
    burst_count = sum(1 for d in delays if d > BURST_PAUSE_THRESHOLD)

    features = np.array([[
        np.mean(delays_arr),
        np.std(delays_arr),
        np.max(delays_arr),
        np.min(delays_arr),
        alphanum_ratio,
        special_ratio,
        burst_count / max(len(delays), 1),
        np.median(delays_arr),
    ]])

    return features


# ---------------------------------------------------------------------------
# 2. Entraînement du modèle (Tâche 7.2)
# ---------------------------------------------------------------------------
def train_model(metadata: list) -> tuple:
    """
    Entraîne un Isolation Forest sur les données de frappe normales.

    Paramètres
    ----------
    metadata : list de dict (keystroke_metadata de keylogger.py)

    Retour
    ------
    (model, scaler) ou (None, None) si données insuffisantes.

    Questions guidantes du TP
    -------------------------
    - Collecte : idéalement 30–60 minutes de frappe normale pour un profil fiable.
    - Normalisation : essentielle car les features ont des échelles très différentes
      (délais en ms vs ratios entre 0–1).
    - Contamination : 5 % est un bon point de départ ; à ajuster selon le taux
      de faux positifs observés en production.
    """
    if not _SKLEARN_AVAILABLE:
        return None, None

    if len(metadata) < MIN_SAMPLES_TRAIN:
        print(f"[INFO] Données insuffisantes ({len(metadata)}/{MIN_SAMPLES_TRAIN}). "
              "Continuez la collecte.")
        return None, None

    # Construire la matrice de features par fenêtres glissantes
    X = []
    for i in range(WINDOW_SIZE, len(metadata)):
        window = metadata[i - WINDOW_SIZE:i]
        features = extract_features(window)
        if features is not None:
            X.append(features[0])

    if len(X) < 10:
        print("[INFO] Pas assez de fenêtres valides pour l'entraînement.")
        return None, None

    X = np.array(X)

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entraînement Isolation Forest
    model = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Sauvegarde du modèle et du scaler
    os.makedirs("data", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Modèle sauvegardé → {MODEL_PATH}")

    return model, scaler


def load_model() -> tuple:
    """Charge le modèle et le scaler depuis le disque."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model  = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("[INFO] Modèle chargé depuis le disque.")
            return model, scaler
        except Exception as e:
            print(f"[ERREUR] Chargement modèle : {e}")
    return None, None


# ---------------------------------------------------------------------------
# 3. Prédiction et alertes en temps réel (Tâche 7.3)
# ---------------------------------------------------------------------------
def predict_anomaly(metadata_window: list, model, scaler) -> dict:
    """
    Prédit si la fenêtre courante est une anomalie.

    Retour
    ------
    dict avec : is_anomaly (bool), score (float), timestamp (str)
    """
    result = {
        "is_anomaly": False,
        "score": 0.0,
        "timestamp": datetime.now().isoformat(),
        "window_size": len(metadata_window),
    }

    if model is None or scaler is None:
        return result

    features = extract_features(metadata_window)
    if features is None:
        return result

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]      # 1 = normal, -1 = anomalie
    score = model.decision_function(features_scaled)[0]  # Plus négatif = plus anormal

    result["is_anomaly"] = (prediction == -1)
    result["score"] = round(float(score), 4)

    return result


def save_alert(alert_data: dict) -> None:
    """
    Sauvegarde une alerte dans le fichier JSON d'alertes (Tâche 7.3).

    Structure
    ---------
    [
      {
        "timestamp": "...",
        "type": "keystroke_anomaly",
        "score": -0.42,
        "is_anomaly": true,
        "window_size": 20
      }
    ]
    """
    os.makedirs("data", exist_ok=True)

    existing = []
    if os.path.exists(ALERTS_PATH):
        try:
            with open(ALERTS_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    alert_data["type"] = "keystroke_anomaly"
    existing.append(alert_data)

    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"[ALERTE] Anomalie détectée à {alert_data['timestamp']} "
          f"(score={alert_data['score']})")


class AnomalyMonitor:
    """
    Moniteur en temps réel : surveille keystroke_metadata et déclenche les alertes.
    S'exécute dans un thread séparé pour ne pas bloquer la capture.
    """

    def __init__(self, metadata_ref: list, check_interval: float = 5.0):
        self.metadata = metadata_ref
        self.check_interval = check_interval
        self.model, self.scaler = load_model()
        self._running = False

    def train_if_ready(self) -> None:
        """Lance l'entraînement si suffisamment de données sont disponibles."""
        if self.model is None and len(self.metadata) >= MIN_SAMPLES_TRAIN:
            print("[INFO] Données suffisantes — entraînement du modèle...")
            self.model, self.scaler = train_model(self.metadata)

    def check(self) -> None:
        """Vérifie la fenêtre courante et émet une alerte si anomalie."""
        self.train_if_ready()

        if self.model is None:
            return

        window = self.metadata[-WINDOW_SIZE:]
        result = predict_anomaly(window, self.model, self.scaler)

        if result["is_anomaly"]:
            save_alert(result)

    def start(self) -> None:
        """Démarre la surveillance dans un thread daemon."""
        self._running = True

        def _loop():
            while self._running:
                try:
                    self.check()
                except Exception as e:
                    print(f"[ERREUR] AnomalyMonitor : {e}")
                time.sleep(self.check_interval)

        import threading
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        print(f"[INFO] AnomalyMonitor démarré (intervalle={self.check_interval}s).")

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    print("=== Génération de données de frappe simulées ===")
    fake_metadata = []
    for i in range(200):
        delay = random.gauss(0.12, 0.04)  # Frappe normale ~120ms
        if random.random() < 0.05:
            delay = random.uniform(2.0, 5.0)  # Anomalie : pause longue
        fake_metadata.append({
            "timestamp": time.time() + i * 0.15,
            "inter_key_delay": max(0.01, delay),
            "key_type": random.choice(["alphanum", "alphanum", "alphanum", "special", "modifier"]),
        })

    model, scaler = train_model(fake_metadata)

    if model:
        window = fake_metadata[-WINDOW_SIZE:]
        result = predict_anomaly(window, model, scaler)
        print(f"Résultat prédiction fenêtre normale : {result}")

        # Simuler une anomalie (frappes très rapides)
        anomaly_window = []
        for i in range(WINDOW_SIZE):
            anomaly_window.append({
                "timestamp": time.time(),
                "inter_key_delay": 0.001,  # Frappe robot
                "key_type": "alphanum",
            })
        result_anomaly = predict_anomaly(anomaly_window, model, scaler)
        print(f"Résultat prédiction fenêtre anormale : {result_anomaly}")
