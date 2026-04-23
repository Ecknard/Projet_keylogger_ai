"""
sensitive_detector.py — Partie III : Classification de données sensibles
TP1 — Intelligence Artificielle & Cybersécurité

Approche hybride :
    1. Regex → détection des formats structurés (email, CB, téléphone, numéro de sécu)
    2. ML (Random Forest) → détection des mots de passe probables et cas non structurés
    3. Masquage / chiffrement → protection avant écriture sur disque

Métriques d'évaluation prioritaires : Rappel (recall) > Précision
Raison : mieux avoir un faux positif que rater une vraie donnée sensible.
"""

import hashlib
import json
import math
import os
import re
import string
from datetime import datetime
from typing import Optional

try:
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[AVERTISSEMENT] scikit-learn non installé : pip install scikit-learn joblib numpy")

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ML_MODEL_PATH  = os.path.join("data", "sensitive_classifier.joblib")
ML_SCALER_PATH = os.path.join("data", "sensitive_scaler.joblib")
DETECTIONS_LOG = os.path.join("data", "detections.json")


# ---------------------------------------------------------------------------
# Partie 1 : Détection par expressions régulières (Tâche 8.1)
# ---------------------------------------------------------------------------

PATTERNS = {
    "email": re.compile(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
    ),
    "carte_bancaire": re.compile(
        r'\b(?:\d{4}[\s\-]){3}\d{4}\b'
    ),
    "telephone_fr": re.compile(
        r'\b(?:(?:\+33|0033)\s?[67]|0[67])(?:[\s.\-]?\d{2}){4}\b'
    ),
    "numero_secu_fr": re.compile(
        r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b'
    ),
    "iban_fr": re.compile(
        r'\bFR\d{2}[\s]?(?:\d{4}[\s]?){5}\d{3}\b',
        re.IGNORECASE
    ),
    "ipv4": re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    ),
}


def detect_with_regex(text: str) -> list:
    """
    Cherche tous les patterns sensibles dans le texte.

    Retour
    ------
    list de dict : [{"type": str, "value": str, "start": int, "end": int}]
    """
    detections = []
    for data_type, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            detections.append({
                "type": data_type,
                "value": match.group(),
                "start": match.start(),
                "end": match.end(),
                "method": "regex",
            })
    return detections


# ---------------------------------------------------------------------------
# Partie 2 : Détection ML pour les mots de passe (Tâche 8.2)
# ---------------------------------------------------------------------------

def compute_entropy(s: str) -> float:
    """Calcule l'entropie de Shannon d'une chaîne (mesure de complexité)."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    entropy = -sum((f / len(s)) * math.log2(f / len(s)) for f in freq.values())
    return round(entropy, 4)


def extract_string_features(token: str) -> list:
    """
    Extrait les features d'un token pour la classification ML.

    Features
    --------
    [0] longueur
    [1] entropie de Shannon
    [2] ratio de majuscules
    [3] ratio de chiffres
    [4] ratio de caractères spéciaux
    [5] longueur > 8 (bool)
    [6] contient majuscule ET chiffre ET spécial (bool)
    [7] ratio de caractères uniques
    """
    if not token:
        return [0.0] * 8

    length = len(token)
    upper  = sum(1 for c in token if c.isupper())
    digits = sum(1 for c in token if c.isdigit())
    special = sum(1 for c in token if c in string.punctuation)
    unique  = len(set(token))

    has_all = int(upper > 0 and digits > 0 and special > 0)

    return [
        length,
        compute_entropy(token),
        upper  / length,
        digits / length,
        special / length,
        int(length > 8),
        has_all,
        unique / length,
    ]


def _generate_training_data() -> tuple:
    """
    Génère un dataset synthétique pour entraîner le classificateur.

    Classes
    -------
    1 = mot de passe probable
    0 = texte ordinaire

    Dataset réel recommandé :
    - Mots de passe : RockYou dataset (filtré), Have I Been Pwned
    - Non sensible : corpus de texte ordinaire (Wikipedia, news)
    """
    passwords = [
        # Mots de passe forts
        "P@ssw0rd123!", "MyS3cur3P@ss!", "Tr0ub4dor&3", "correct-horse-battery-staple",
        "abc123XYZ!", "Admin@2024!", "Summer2024#", "W1nter!2023",
        "Qwerty@123", "Dragon#2024", "P@$$w0rd!", "L0gin_Secure!",
        "C0mpl3x!Pass", "Secure#Pass1", "My!Pass2024", "Test@Pass99",
        "Hunter2#Safe", "Monkey!123X", "Root@Linux1!", "Admin_Pass!2",
        # Variantes courantes
        "password123", "letmein!", "welcome1!", "master2024",
        "ninja@2024!", "shadow#1!", "iloveyou2!", "sunshine99!",
    ]

    normal_words = [
        # Mots ordinaires, phrases, tokens non sensibles
        "bonjour", "hello", "monde", "world", "chat", "maison", "voiture",
        "informatique", "python", "programmation", "exercice", "cours",
        "université", "étudiant", "projet", "travail", "réunion",
        "lundi", "mardi", "mercredi", "jeudi", "vendredi",
        "janvier", "février", "mars", "avril", "mai", "juin",
        "rapport", "analyse", "résultat", "données", "modèle",
        "sklearn", "pandas", "numpy", "matplotlib", "jupyter",
        "localhost", "http", "https", "www", "html", "css",
    ]

    X, y = [], []
    for pw in passwords:
        X.append(extract_string_features(pw))
        y.append(1)
    for word in normal_words:
        X.append(extract_string_features(word))
        y.append(0)

    return X, y


def train_ml_classifier() -> tuple:
    """
    Entraîne un Random Forest pour détecter les mots de passe.

    Choix Random Forest vs alternatives
    ------------------------------------
    - Naive Bayes : rapide mais suppose indépendance des features (incorrect ici)
    - SVM         : bon mais plus lent et sensible au scaling
    - Random Forest : gère bien les features mixtes, robuste, interprétable via feature_importance
    → Choix : Random Forest

    Métrique prioritaire : Rappel (recall)
    Raison : un faux négatif (mot de passe non détecté) est plus coûteux
             qu'un faux positif (mot ordinaire signalé comme sensible).
    """
    if not _SKLEARN_AVAILABLE:
        return None, None

    X, y = _generate_training_data()
    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight={0: 1, 1: 2},  # Pénalise davantage les faux négatifs
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    print("\n=== Rapport de classification (détecteur ML) ===")
    print(classification_report(y_test, y_pred, target_names=["ordinaire", "sensible"]))

    os.makedirs("data", exist_ok=True)
    joblib.dump(model, ML_MODEL_PATH)
    joblib.dump(scaler, ML_SCALER_PATH)
    print(f"[INFO] Modèle ML sauvegardé → {ML_MODEL_PATH}")

    return model, scaler


def load_ml_classifier() -> tuple:
    """Charge le classifieur ML depuis le disque."""
    if os.path.exists(ML_MODEL_PATH) and os.path.exists(ML_SCALER_PATH):
        try:
            return joblib.load(ML_MODEL_PATH), joblib.load(ML_SCALER_PATH)
        except Exception:
            pass
    return None, None


def detect_password_ml(token: str, model, scaler) -> dict:
    """Prédit si un token est un mot de passe probable."""
    if model is None or len(token) < 4:
        return {"is_sensitive": False, "probability": 0.0}

    features = np.array([extract_string_features(token)])
    features_scaled = scaler.transform(features)
    proba = model.predict_proba(features_scaled)[0][1]  # Proba classe "sensible"

    return {
        "is_sensitive": proba >= 0.5,
        "probability": round(float(proba), 4),
        "type": "mot_de_passe_probable",
        "method": "ml",
    }


# ---------------------------------------------------------------------------
# Partie 3 : Analyse complète d'un texte (Tâche 8.3)
# ---------------------------------------------------------------------------

def analyze_text(text: str, ml_model=None, ml_scaler=None) -> dict:
    """
    Analyse complète : regex + ML.

    Retour
    ------
    dict : {
        "text": str,
        "timestamp": str,
        "detections": list,
        "masked_text": str,
        "has_sensitive": bool,
    }
    """
    detections = detect_with_regex(text)

    # Analyse ML token par token
    if ml_model is not None:
        for token in text.split():
            ml_result = detect_password_ml(token, ml_model, ml_scaler)
            if ml_result["is_sensitive"]:
                start = text.find(token)
                detections.append({
                    "type": ml_result["type"],
                    "value": token,
                    "start": start,
                    "end": start + len(token),
                    "method": "ml",
                    "probability": ml_result["probability"],
                })

    masked = mask_sensitive(text, detections)

    return {
        "text": text,
        "timestamp": datetime.now().isoformat(),
        "detections": detections,
        "masked_text": masked,
        "has_sensitive": len(detections) > 0,
    }


def mask_sensitive(text: str, detections: list, mask_char: str = "*") -> str:
    """
    Remplace les données sensibles détectées par des caractères de masquage.

    Paramètres
    ----------
    text       : texte original
    detections : liste des détections (chaque dict doit avoir 'start' et 'end')
    mask_char  : caractère de remplacement (défaut : '*')

    Retour
    ------
    str : texte masqué

    Niveaux de protection disponibles (voir tableau TP)
    ---------------------------------------------------
    - Masquage simple (****) : irréversible, perd la donnée → cette implémentation
    - Chiffrement AES        : réversible avec clé → voir Extension C
    - Hachage SHA-256        : empreinte non réversible
    - Tokenisation           : remplace par token lié à un vault séparé
    """
    if not detections:
        return text

    # Trier par position décroissante pour ne pas décaler les indices
    sorted_detections = sorted(detections, key=lambda d: d["start"], reverse=True)
    result = list(text)

    for det in sorted_detections:
        start, end = det["start"], det["end"]
        length = end - start
        # Conserver la longueur originale pour ne pas trahir la nature de la donnée
        mask = mask_char * length
        result[start:end] = list(mask)

    return "".join(result)


def hash_sensitive(value: str) -> str:
    """Retourne le hash SHA-256 d'une valeur sensible (empreinte non réversible)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_detections(results: list, path: str = DETECTIONS_LOG) -> None:
    """Sauvegarde les résultats de détection dans un fichier JSON."""
    os.makedirs("data", exist_ok=True)
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    for r in results:
        # Ne pas sauvegarder les valeurs en clair — uniquement le type et le hash
        safe_detections = []
        for det in r.get("detections", []):
            safe_detections.append({
                "type": det["type"],
                "method": det.get("method", "regex"),
                "hash_sha256": hash_sensitive(det["value"]),
                "length": len(det["value"]),
            })
        existing.append({
            "timestamp": r["timestamp"],
            "masked_text": r["masked_text"],
            "has_sensitive": r["has_sensitive"],
            "detections": safe_detections,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Entraînement du classificateur ML ===")
    model, scaler = train_ml_classifier()

    test_texts = [
        "Mon email est alice@example.com, appelez-moi au 06 12 34 56 78",
        "Numéro de carte : 4532 1234 5678 9012, expire 12/26",
        "Connexion réussie avec le mot de passe P@ssw0rd123!",
        "La réunion est prévue lundi matin pour discuter du projet",
        "Mon numéro de sécu : 1 85 12 75 123 456 78",
        "Iban : FR76 3000 6000 0112 3456 7890 189",
    ]

    print("\n=== Tests de détection ===")
    for text in test_texts:
        result = analyze_text(text, model, scaler)
        print(f"\n📝 Original  : {text}")
        print(f"🔒 Masqué    : {result['masked_text']}")
        print(f"🎯 Détections: {[d['type'] for d in result['detections']]}")
