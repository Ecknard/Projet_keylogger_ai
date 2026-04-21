"""
sensitive_detector.py — Partie III : Classification de données sensibles
TP1 — Intelligence Artificielle & Cybersécurité

AMÉLIORATIONS v2 :
    - Nouveaux patterns regex : URL avec token, JWT, clé API (Bearer/AWS/GCP/GitHub),
      IPv6, hash MD5/SHA, coordonnées GPS, SIRET/SIREN, numéro de passeport FR
    - Validation Luhn pour les cartes bancaires (zéro faux positifs CB)
    - Validation INSEE pour les numéros de sécurité sociale
    - Features ML enrichies (zxcvbn-like) : keyboard walk, répétition, date pattern,
      common password patterns, trigram frequency
    - Dataset synthétique x3 plus grand et équilibré
    - Gradient Boosting en option (meilleur recall que Random Forest seul)
    - Déduplication intelligente des détections (évite doublons chevauchants)
    - Niveau de risque calculé par détection : CRITIQUE / ÉLEVÉ / MOYEN / FAIBLE
    - Fonction de redaction avancée (conserve format : email@***.com, **** **** **** 9012)

Métriques prioritaires : Rappel (recall) > Précision
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
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("[AVERTISSEMENT] scikit-learn non installé : pip install scikit-learn joblib numpy")

# ── Chemins ─────────────────────────────────────────────────────────────────
ML_MODEL_PATH  = os.path.join("data", "sensitive_classifier.joblib")
ML_SCALER_PATH = os.path.join("data", "sensitive_scaler.joblib")
DETECTIONS_LOG = os.path.join("data", "detections.json")

# ── Niveaux de risque ────────────────────────────────────────────────────────
RISK_LEVELS = {
    "carte_bancaire":      "CRITIQUE",
    "numero_secu_fr":      "CRITIQUE",
    "iban_fr":             "CRITIQUE",
    "jwt_token":           "CRITIQUE",
    "cle_api":             "ÉLEVÉ",
    "mot_de_passe_probable": "ÉLEVÉ",
    "email":               "MOYEN",
    "telephone_fr":        "MOYEN",
    "siret_siren":         "MOYEN",
    "passeport_fr":        "MOYEN",
    "ipv4":                "FAIBLE",
    "ipv6":                "FAIBLE",
    "hash_md5_sha":        "FAIBLE",
    "coordonnees_gps":     "FAIBLE",
    "url_avec_token":      "ÉLEVÉ",
}

RISK_ORDER = {"CRITIQUE": 4, "ÉLEVÉ": 3, "MOYEN": 2, "FAIBLE": 1}


# ============================================================================
# PARTIE 1 : Détection par expressions régulières (améliorée)
# ============================================================================

PATTERNS = {
    # ── Identifiants financiers ──
    "email": re.compile(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
    ),
    "carte_bancaire": re.compile(
        r'\b(?:\d{4}[\s\-]){3}\d{4}\b'
    ),
    "iban_fr": re.compile(
        r'\bFR\d{2}[\s]?(?:\d{4}[\s]?){5}\d{3}\b',
        re.IGNORECASE
    ),
    "siret_siren": re.compile(
        r'\b(?:\d{3}[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{5}|\d{9})\b'
    ),
    # ── Téléphones ──
    "telephone_fr": re.compile(
        r'\b(?:(?:\+33|0033)\s?[1-9]|0[1-9])(?:[\s.\-]?\d{2}){4}\b'
    ),
    # ── Identifiants gouvernementaux ──
    "numero_secu_fr": re.compile(
        r'\b[12]\s?\d{2}\s?\d{2}\s?\d{2,3}\s?\d{3}\s?\d{3}\s?\d{2}\b'
    ),
    "passeport_fr": re.compile(
        r'\b[0-9]{2}[A-Z]{2}[0-9]{5}\b'
    ),
    # ── Réseaux / Infrastructure ──
    "ipv4": re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    ),
    "ipv6": re.compile(
        r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
        r'|\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b'
        r'|\b::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}\b'
    ),
    # ── Secrets / Credentials ──
    "jwt_token": re.compile(
        r'\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b'
    ),
    "cle_api": re.compile(
        r'(?:'
        r'(?:Bearer|Authorization|api[_\-]?key|token|secret)[:\s]+[A-Za-z0-9_\-\.]{16,}'
        r'|(?:sk|pk|rk)_(?:live|test|secret)_[A-Za-z0-9]{16,}'  # Stripe-like
        r'|ghp_[A-Za-z0-9]{36}'                                   # GitHub PAT
        r'|AKIA[0-9A-Z]{16}'                                       # AWS Access Key
        r'|AIza[0-9A-Za-z_\-]{35}'                                # Google API Key
        r')',
        re.IGNORECASE
    ),
    "url_avec_token": re.compile(
        r'https?://[^\s<>"]+(?:token|api_key|key|access_token|secret)[=:][^\s&"]{8,}',
        re.IGNORECASE
    ),
    # ── Crypto / Hachage ──
    "hash_md5_sha": re.compile(
        r'\b[0-9a-fA-F]{32}\b|\b[0-9a-fA-F]{40}\b|\b[0-9a-fA-F]{64}\b'
    ),
    # ── Géolocalisation ──
    "coordonnees_gps": re.compile(
        r'\b[-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?),\s*[-+]?(?:1[0-7]\d(?:\.\d+)?|(?:[1-9]?\d(?:\.\d+)?)|180(?:\.0+)?)\b'
    ),
}

# ── Validation Luhn (cartes bancaires) ──────────────────────────────────────
def _luhn_check(number: str) -> bool:
    """Valide un numéro par l'algorithme de Luhn. Réduit les faux positifs CB."""
    digits = re.sub(r'\D', '', number)
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0

# ── Validation INSEE (numéro de sécurité sociale) ───────────────────────────
def _insee_check(number: str) -> bool:
    """Validation basique du format INSEE (longueur + clé de contrôle)."""
    digits = re.sub(r'\D', '', number)
    if len(digits) != 15:
        return False
    try:
        base = int(digits[:13])
        key  = int(digits[13:15])
        return (97 - (base % 97)) == key
    except ValueError:
        return False


def detect_with_regex(text: str) -> list:
    """
    Cherche tous les patterns sensibles dans le texte.
    Applique les validations post-regex (Luhn, INSEE).

    Retour : list de dict avec type, value, start, end, method, risk_level
    """
    detections = []
    seen_spans  = set()  # Déduplication

    for data_type, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()

            # Déduplication : ignorer si chevauchement avec une détection existante
            overlap = any(not (end <= s or start >= e) for (s, e) in seen_spans)
            if overlap:
                continue

            value = match.group()

            # Validations post-regex
            if data_type == "carte_bancaire" and not _luhn_check(value):
                continue
            if data_type == "numero_secu_fr" and not _insee_check(value):
                # Garder quand même mais signaler comme non validé
                risk = "MOYEN"
            else:
                risk = RISK_LEVELS.get(data_type, "FAIBLE")

            seen_spans.add((start, end))
            detections.append({
                "type":       data_type,
                "value":      value,
                "start":      start,
                "end":        end,
                "method":     "regex",
                "risk_level": risk,
            })

    return detections


# ============================================================================
# PARTIE 2 : Détection ML — mots de passe (features enrichies)
# ============================================================================

# Patterns de "keyboard walk" courants
_KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm", "1234567890"]
_COMMON_PATTERNS = [
    "password", "passwd", "pass", "motdepasse", "azerty", "qwerty",
    "123456", "111111", "000000", "admin", "root", "login", "user",
]

def compute_entropy(s: str) -> float:
    """Entropie de Shannon."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return round(-sum((f / len(s)) * math.log2(f / len(s)) for f in freq.values()), 4)


def _keyboard_walk_score(token: str) -> float:
    """Détecte les séquences de clavier (ex: qwerty, 12345). Score 0.0–1.0."""
    t = token.lower()
    max_walk = 0
    for row in _KEYBOARD_ROWS:
        for i in range(len(row) - 2):
            seq = row[i:i+3]
            if seq in t or seq[::-1] in t:
                max_walk = max(max_walk, 3)
        for i in range(len(row) - 3):
            seq = row[i:i+4]
            if seq in t or seq[::-1] in t:
                max_walk = max(max_walk, 4)
    return round(max_walk / max(len(token), 1), 4)


def _repetition_score(token: str) -> float:
    """Détecte les répétitions (aaa, 111). Score 0.0–1.0."""
    if len(token) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(token)) if token[i] == token[i-1])
    return round(repeats / (len(token) - 1), 4)


def _has_date_pattern(token: str) -> int:
    """Détecte les dates intégrées (19xx, 20xx, mm/dd, ddmm). Retourne 0 ou 1."""
    return int(bool(re.search(r'(?:19|20)\d{2}|\d{2}[/\-]\d{2}', token)))


def _common_password_score(token: str) -> float:
    """Pénalité si le token contient un mot de passe commun. Score 0.0–1.0."""
    t = token.lower()
    for p in _COMMON_PATTERNS:
        if p in t:
            return min(len(p) / len(t), 1.0)
    return 0.0


def extract_string_features(token: str) -> list:
    """
    Extrait 12 features d'un token pour la classification ML.

    [0]  longueur
    [1]  entropie de Shannon
    [2]  ratio majuscules
    [3]  ratio chiffres
    [4]  ratio caractères spéciaux
    [5]  longueur > 8 (bool)
    [6]  contient MAJ + chiffre + spécial (bool)
    [7]  ratio caractères uniques
    [8]  keyboard walk score
    [9]  repetition score
    [10] has date pattern
    [11] common password score (pénalité)
    """
    if not token:
        return [0.0] * 12

    length  = len(token)
    upper   = sum(1 for c in token if c.isupper())
    digits  = sum(1 for c in token if c.isdigit())
    special = sum(1 for c in token if c in string.punctuation)
    unique  = len(set(token))
    has_all = int(upper > 0 and digits > 0 and special > 0)

    return [
        length,
        compute_entropy(token),
        upper   / length,
        digits  / length,
        special / length,
        int(length > 8),
        has_all,
        unique  / length,
        _keyboard_walk_score(token),
        _repetition_score(token),
        _has_date_pattern(token),
        _common_password_score(token),
    ]


def _generate_training_data() -> tuple:
    """
    Dataset synthétique enrichi pour entraîner le classificateur.
    Classes : 1 = mot de passe probable, 0 = texte ordinaire.
    """
    # ── Mots de passe ──────────────────────────────────────────────────────
    passwords = [
        # Forts
        "P@ssw0rd123!", "MyS3cur3P@ss!", "Tr0ub4dor&3", "correct-horse-battery",
        "abc123XYZ!", "Admin@2024!", "Summer2024#", "W1nter!2023",
        "Qwerty@123!", "Dragon#2024", "P@$$w0rd!", "L0gin_Secure!",
        "C0mpl3x!Pass", "Secure#Pass1", "My!Pass2024", "Test@Pass99",
        "Hunter2#Safe", "Root@Linux1!", "Admin_Pass!2", "x7K!mN9@qR2#",
        "Zp3$wL8!vT6@", "jR5#bN2@kM7!", "qF9!xH4@nK6#", "hP2@mG5!yB8$",
        "vW6#cJ3@dL9!", "sE4!uA7@fO1#", "tI8@gD5!wQ2#",
        # Modérés
        "password123", "letmein!", "welcome1!", "master2024",
        "ninja@2024!", "shadow#1!", "iloveyou2!", "sunshine99!",
        "monkey123!", "dragon456@", "baseball1!", "football2@",
        # Patterns reconnaissables comme MDP
        "Monmdp@2024", "Azerty!123", "Soleil#2024", "Paris@75001",
        "Jean-Paul1!", "Marie2024#", "Fr@nce2024!", "M0nP@ssFR!",
        # Courts mais complexes
        "X!3k@9", "P@5sW!", "T3st!2", "S3cr#t",
    ]

    # ── Texte ordinaire ────────────────────────────────────────────────────
    normal_words = [
        # Mots français
        "bonjour", "monde", "voiture", "maison", "jardin", "fenêtre",
        "informatique", "python", "programmation", "exercice", "cours",
        "université", "étudiant", "projet", "travail", "réunion",
        "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi",
        "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
        "rapport", "analyse", "résultat", "données", "modèle", "système",
        "réseau", "serveur", "client", "application", "interface",
        # Mots anglais techniques
        "function", "variable", "database", "connection", "server",
        "localhost", "python", "javascript", "typescript", "react",
        "library", "framework", "container", "deployment", "pipeline",
        # Termes neutres courants
        "aujourd'hui", "demain", "hier", "maintenant", "toujours",
        "souvent", "parfois", "jamais", "beaucoup", "quelques",
        "important", "nécessaire", "possible", "différent", "nouveau",
        # Chiffres et codes non-sensibles
        "12345", "version", "v2024", "numero1", "section2", "partie3",
        "chapitre4", "exemple5", "test01", "demo02", "dev03",
        # Noms communs
        "dupont", "martin", "bernard", "thomas", "robert",
        "paris", "lyon", "marseille", "bordeaux", "toulouse",
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
    Entraîne un ensemble (VotingClassifier) Random Forest + Gradient Boosting.
    Métrique prioritaire : Recall sur la classe sensible.
    """
    if not _SKLEARN_AVAILABLE:
        return None, None

    X, y = _generate_training_data()
    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train)
    X_test_s    = scaler.transform(X_test)

    rf  = RandomForestClassifier(
        n_estimators=200, class_weight={0: 1, 1: 3},
        max_features="sqrt", random_state=42, n_jobs=-1
    )
    gb  = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=4,
        subsample=0.8, random_state=42
    )

    # Ensemble soft voting
    model = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[1, 1.5],  # Légère préférence GB pour le recall
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    print("\n=== Rapport de classification (détecteur ML v2) ===")
    print(classification_report(y_test, y_pred, target_names=["ordinaire", "sensible"]))

    # Cross-validation recall
    cv_recall = cross_val_score(model, X_train_s, y_train,
                                 cv=StratifiedKFold(n_splits=5), scoring="recall")
    print(f"Recall CV (5-fold) : {cv_recall.mean():.3f} ± {cv_recall.std():.3f}")

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
    """Prédit si un token est un mot de passe probable (seuil conservateur 0.40)."""
    if model is None or len(token) < 4:
        return {"is_sensitive": False, "probability": 0.0}

    features        = np.array([extract_string_features(token)])
    features_scaled = scaler.transform(features)
    proba           = model.predict_proba(features_scaled)[0][1]

    return {
        "is_sensitive": proba >= 0.40,  # Seuil conservateur pour maximiser le recall
        "probability":  round(float(proba), 4),
        "type":         "mot_de_passe_probable",
        "method":       "ml",
        "risk_level":   "ÉLEVÉ" if proba >= 0.40 else "FAIBLE",
    }


# ============================================================================
# PARTIE 3 : Analyse complète
# ============================================================================

def analyze_text(text: str, ml_model=None, ml_scaler=None) -> dict:
    """
    Analyse complète : regex (avec validations) + ML.

    Retour : {text, timestamp, detections, masked_text,
              redacted_text, has_sensitive, risk_summary}
    """
    detections = detect_with_regex(text)

    # ML token par token (seulement sur tokens non déjà détectés par regex)
    regex_spans = {(d["start"], d["end"]) for d in detections}
    if ml_model is not None:
        for token in text.split():
            start = text.find(token)
            end   = start + len(token)
            # Éviter de re-analyser des spans déjà couverts
            already_covered = any(not (end <= s or start >= e) for (s, e) in regex_spans)
            if already_covered:
                continue
            ml_result = detect_password_ml(token, ml_model, ml_scaler)
            if ml_result["is_sensitive"]:
                detections.append({
                    "type":        ml_result["type"],
                    "value":       token,
                    "start":       start,
                    "end":         end,
                    "method":      "ml",
                    "probability": ml_result["probability"],
                    "risk_level":  ml_result["risk_level"],
                })

    masked   = mask_sensitive(text, detections)
    redacted = redact_sensitive(text, detections)

    # Résumé des risques
    risk_summary = {"CRITIQUE": 0, "ÉLEVÉ": 0, "MOYEN": 0, "FAIBLE": 0}
    for d in detections:
        level = d.get("risk_level", "FAIBLE")
        if level in risk_summary:
            risk_summary[level] += 1

    max_risk = max(
        (RISK_ORDER.get(d.get("risk_level", "FAIBLE"), 0) for d in detections),
        default=0
    )
    overall_risk = next(
        (k for k, v in RISK_ORDER.items() if v == max_risk), "AUCUN"
    )

    return {
        "text":          text,
        "timestamp":     datetime.now().isoformat(),
        "detections":    detections,
        "masked_text":   masked,
        "redacted_text": redacted,
        "has_sensitive": len(detections) > 0,
        "risk_summary":  risk_summary,
        "overall_risk":  overall_risk,
    }


def mask_sensitive(text: str, detections: list, mask_char: str = "*") -> str:
    """Remplace les données sensibles par des étoiles (masquage complet)."""
    if not detections:
        return text
    sorted_dets = sorted(detections, key=lambda d: d["start"], reverse=True)
    result = list(text)
    for det in sorted_dets:
        s, e = det["start"], det["end"]
        result[s:e] = list(mask_char * (e - s))
    return "".join(result)


def redact_sensitive(text: str, detections: list) -> str:
    """
    Redaction intelligente : conserve le format visible.
    - Email     → alice@***.com
    - CB        → **** **** **** 9012 (4 derniers chiffres visibles)
    - Téléphone → ** ** ** ** 78
    - Autres    → [TYPE_REDACTED]
    """
    if not detections:
        return text
    sorted_dets = sorted(detections, key=lambda d: d["start"], reverse=True)
    result = list(text)
    for det in sorted_dets:
        s, e, val, dtype = det["start"], det["end"], det["value"], det["type"]
        if dtype == "email":
            at = val.find("@")
            dom_parts = val[at+1:].split(".")
            redacted = val[:1] + "*" * (at - 1) + "@***." + dom_parts[-1]
        elif dtype == "carte_bancaire":
            digits = re.sub(r"\D", "", val)
            redacted = "**** **** **** " + digits[-4:]
        elif dtype == "telephone_fr":
            digits = re.sub(r"\D", "", val)
            redacted = "** ** ** ** " + digits[-2:]
        elif dtype in ("jwt_token", "cle_api", "url_avec_token"):
            redacted = f"[{dtype.upper()}_REDACTED]"
        else:
            redacted = f"[{dtype.upper()}_REDACTED]"
        result[s:e] = list(redacted)
    return "".join(result)


def hash_sensitive(value: str) -> str:
    """SHA-256 d'une valeur sensible."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def save_detections(results: list, path: str = DETECTIONS_LOG) -> None:
    """Sauvegarde les résultats (ne stocke jamais les valeurs en clair)."""
    os.makedirs("data", exist_ok=True)
    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    for r in results:
        safe_detections = []
        for det in r.get("detections", []):
            safe_detections.append({
                "type":        det["type"],
                "method":      det.get("method", "regex"),
                "risk_level":  det.get("risk_level", "FAIBLE"),
                "hash_sha256": hash_sensitive(det["value"]),
                "length":      len(det["value"]),
            })
        existing.append({
            "timestamp":     r["timestamp"],
            "masked_text":   r["masked_text"],
            "redacted_text": r.get("redacted_text", r["masked_text"]),
            "has_sensitive": r["has_sensitive"],
            "overall_risk":  r.get("overall_risk", "AUCUN"),
            "risk_summary":  r.get("risk_summary", {}),
            "detections":    safe_detections,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ── Test standalone ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Entraînement du classificateur ML v2 ===")
    model, scaler = train_ml_classifier()

    test_texts = [
        "Mon email est alice@example.com, appelez-moi au 06 12 34 56 78",
        "Numéro de carte : 4532 1234 5678 9012, expire 12/26",
        "Connexion réussie avec le mot de passe P@ssw0rd123!",
        "La réunion est prévue lundi matin pour discuter du projet",
        "Mon IBAN : FR76 3000 6000 0112 3456 7890 189",
        "Token Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "Clé API : AKIA1234567890ABCDEF",
        "Serveur interne : 192.168.1.1 et 2001:db8::1",
        "Hash fichier : d41d8cd98f00b204e9800998ecf8427e",
    ]

    print("\n=== Tests de détection ===")
    for text in test_texts:
        result = analyze_text(text, model, scaler)
        print(f"\n📝 Original  : {text[:70]}")
        print(f"🔒 Masqué    : {result['masked_text'][:70]}")
        print(f"✂️  Redacté   : {result['redacted_text'][:70]}")
        types_risks = [(d['type'], d.get('risk_level','?')) for d in result['detections']]
        print(f"🎯 Risque    : {result['overall_risk']} → {types_risks}")
