"""
sentiment_analyzer.py — Partie II, Tâche 6 : Analyse de sentiments
TP1 — Intelligence Artificielle & Cybersécurité

AMÉLIORATIONS v2 :
    - Détection automatique de la langue (langdetect) avec fallback anglais
    - Support natif du français via lexique VADER étendu + TextBlob fallback
    - Seuils adaptatifs basés sur la distribution glissante des 50 derniers scores
    - Score de confiance [0.0–1.0] basé sur magnitude, longueur et polarité nette
    - Intensificateurs / atténuateurs / négations français reconnus par VADER
    - Émojis textuels pris en charge dans le scoring
    - Classification fine 5 niveaux : très_positif / positif / neutre / négatif / très_négatif
    - Pipeline de nettoyage amélioré (timestamps, balises clavier, séparateurs)
    - Statistiques agrégées : tendance, distribution, langue dominante
"""

import json
import os
import re
from collections import Counter, deque
from datetime import datetime
from typing import Optional

# ── Imports optionnels ──────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False
    print("[AVERTISSEMENT] vaderSentiment non installé : pip install vaderSentiment")

try:
    from langdetect import detect as _langdetect
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False

# ── Singleton analyseur VADER ───────────────────────────────────────────────
_analyzer: Optional[object] = None

def _get_analyzer():
    global _analyzer
    if _analyzer is None and _VADER_AVAILABLE:
        _analyzer = SentimentIntensityAnalyzer()
        _inject_french_lexicon(_analyzer)
    return _analyzer


# ── Lexique français étendu pour VADER ─────────────────────────────────────
FRENCH_LEXICON = {
    # Positifs forts
    "excellent": 3.0, "magnifique": 2.8, "parfait": 2.7, "génial": 2.6,
    "formidable": 2.5, "fantastique": 2.5, "merveilleux": 2.4, "superbe": 2.3,
    "brillant": 2.2, "extraordinaire": 2.6, "incroyable": 2.2, "sublime": 2.5,
    "exceptionnel": 2.4, "remarquable": 2.1, "impressionnant": 2.0,
    # Positifs modérés
    "bien": 1.5, "bon": 1.5, "bonne": 1.5, "content": 1.8, "heureux": 2.0,
    "heureuse": 2.0, "satisfait": 1.7, "agréable": 1.6, "sympa": 1.5,
    "cool": 1.4, "super": 1.8, "top": 1.6, "bravo": 2.0, "merci": 0.8,
    "adore": 2.3, "aime": 1.8, "aimer": 1.8, "réussi": 1.7, "succès": 1.8,
    "réussite": 1.8, "amélioration": 1.4, "progrès": 1.3, "efficace": 1.4,
    # Négatifs forts
    "horrible": -3.0, "terrible": -2.8, "affreux": -2.7, "nul": -2.2,
    "catastrophique": -2.9, "désastreux": -2.6, "atroce": -2.9, "ignoble": -2.5,
    "insupportable": -2.4, "détestable": -2.3, "abominable": -2.7,
    # Négatifs modérés
    "mauvais": -1.8, "mauvaise": -1.8, "triste": -1.8, "malheureux": -2.0,
    "malheureuse": -2.0, "déçu": -1.9, "décevant": -1.8, "problème": -1.3,
    "erreur": -1.4, "bug": -1.3, "cassé": -1.6, "brisé": -1.5, "raté": -2.0,
    "échoué": -2.0, "impossible": -1.4, "difficile": -0.8, "pénible": -1.7,
    "énervant": -1.8, "agaçant": -1.7, "frustrant": -1.9, "stressant": -1.6,
    "inquiet": -1.5, "inquiète": -1.5, "peur": -1.4, "crainte": -1.2,
    "colère": -2.0, "furieux": -2.5, "énervé": -2.0, "irrité": -1.8,
    # Intensificateurs
    "très": 1.3, "trop": 1.2, "vraiment": 1.25, "extrêmement": 1.5,
    "totalement": 1.3, "absolument": 1.35, "complètement": 1.3,
    "particulièrement": 1.15, "incroyablement": 1.4, "terriblement": 1.3,
    # Atténuateurs
    "peu": 0.7, "légèrement": 0.6, "plutôt": 0.85, "assez": 0.9,
    "relativement": 0.8, "peut-être": 0.7, "parfois": 0.75,
    # Négations
    "pas": -0.5, "ne": -0.3, "jamais": -0.5, "rien": -0.4, "aucun": -0.3,
    "sans": -0.2, "ni": -0.25,
    # Émojis textuels
    ":)": 2.0, ":-)": 2.0, ":D": 2.5, ":(": -2.0, ":-(": -2.0,
    "^^": 1.5, "<3": 2.5, "</3": -2.0,
}

def _inject_french_lexicon(vader_instance) -> None:
    try:
        for word, score in FRENCH_LEXICON.items():
            vader_instance.lexicon[word] = score
    except Exception:
        pass


# ── Seuils et classification ────────────────────────────────────────────────
VERY_POSITIVE_THRESHOLD =  0.50
POSITIVE_THRESHOLD      =  0.05
NEGATIVE_THRESHOLD      = -0.05
VERY_NEGATIVE_THRESHOLD = -0.50
MIN_WORDS               =  3

_score_history: deque = deque(maxlen=50)  # Fenêtre glissante pour seuils adaptatifs


def _adaptive_label(compound: float) -> str:
    """Classification 5 niveaux avec seuils adaptatifs."""
    if len(_score_history) >= 20:
        mean = sum(_score_history) / len(_score_history)
        std  = (sum((x - mean) ** 2 for x in _score_history) / len(_score_history)) ** 0.5
        pos_thresh   = min(mean + 0.5 * std,  0.1)
        neg_thresh   = max(mean - 0.5 * std, -0.1)
        v_pos_thresh = min(mean + 1.5 * std,  0.5)
        v_neg_thresh = max(mean - 1.5 * std, -0.5)
    else:
        pos_thresh   = POSITIVE_THRESHOLD
        neg_thresh   = NEGATIVE_THRESHOLD
        v_pos_thresh = VERY_POSITIVE_THRESHOLD
        v_neg_thresh = VERY_NEGATIVE_THRESHOLD

    if compound >= v_pos_thresh:
        return "très_positif"
    elif compound >= pos_thresh:
        return "positif"
    elif compound <= v_neg_thresh:
        return "très_négatif"
    elif compound <= neg_thresh:
        return "négatif"
    return "neutre"


def _compute_confidence(text: str, scores: dict, word_count: int) -> float:
    """Score de confiance [0.0–1.0] basé sur magnitude, longueur et polarité nette."""
    compound_mag = abs(scores.get("compound", 0))
    length_bonus = min(word_count / 20, 1.0)
    polarity_gap = abs(scores.get("pos", 0) - scores.get("neg", 0))
    confidence   = (compound_mag * 0.5) + (length_bonus * 0.3) + (polarity_gap * 0.2)
    return round(min(confidence, 1.0), 3)


# ── Nettoyage ───────────────────────────────────────────────────────────────
_RE_TIMESTAMP = re.compile(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]')
_RE_DASH_LINE = re.compile(r'—{3,}')
_RE_KEY_TAG   = re.compile(r'\[(?:BACK|TAB|ENTER|CTRL|ALT|SHIFT)\]', re.IGNORECASE)
_RE_MULTI_SP  = re.compile(r' {2,}')

def _clean_text(text: str) -> str:
    text = _RE_TIMESTAMP.sub('', text)
    text = _RE_DASH_LINE.sub('', text)
    text = _RE_KEY_TAG.sub(' ', text)
    text = _RE_MULTI_SP.sub(' ', text)
    return text.strip()


# ── Détection de langue ─────────────────────────────────────────────────────
def _detect_language(text: str) -> str:
    if not _LANGDETECT_AVAILABLE or len(text.split()) < 4:
        return "unknown"
    try:
        lang = _langdetect(text)
        return lang if lang in ("fr", "en") else "other"
    except Exception:
        return "unknown"


# ── Fallback TextBlob français ──────────────────────────────────────────────
def _analyze_with_textblob_fr(text: str) -> dict:
    if not _TEXTBLOB_AVAILABLE:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "method": "none"}
    try:
        blob     = TextBlob(text)
        compound = round(float(blob.sentiment.polarity), 4)
        subj     = round(float(blob.sentiment.subjectivity), 4)
        return {
            "compound": compound,
            "pos":      max(compound, 0),
            "neu":      1 - subj,
            "neg":      abs(min(compound, 0)),
            "method":   "textblob_fr",
        }
    except Exception:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "method": "error"}


# ── Fonction principale ─────────────────────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    """
    Analyse le sentiment d'un texte (FR / EN / mixte).

    Retour : dict avec score, label, confidence, language, method,
             timestamp, text, details, word_count.
    """
    ts         = datetime.now().isoformat()
    text_clean = _clean_text(text)
    word_count = len(text_clean.split())

    base = {
        "score": 0.0, "label": "trop_court", "confidence": 0.0,
        "language": "unknown", "method": "none",
        "timestamp": ts, "text": text_clean, "details": {}, "word_count": word_count,
    }

    if word_count < MIN_WORDS:
        return base

    language    = _detect_language(text_clean)
    base["language"] = language

    analyzer = _get_analyzer()
    if analyzer is not None:
        scores   = analyzer.polarity_scores(text_clean)
        compound = scores["compound"]
        method   = "vader"
    elif _TEXTBLOB_AVAILABLE:
        scores_fb = _analyze_with_textblob_fr(text_clean)
        compound  = scores_fb["compound"]
        scores    = scores_fb
        method    = "textblob_fr"
    else:
        return {**base, "label": "erreur_librairie"}

    _score_history.append(compound)

    return {
        "score":      round(compound, 4),
        "label":      _adaptive_label(compound),
        "confidence": _compute_confidence(text_clean, scores, word_count),
        "language":   language,
        "method":     method,
        "timestamp":  ts,
        "text":       text_clean,
        "details": {
            "neg": scores.get("neg", 0), "neu": scores.get("neu", 1),
            "pos": scores.get("pos", 0), "compound": compound,
        },
        "word_count": word_count,
    }


def analyze_sentences_from_log(log_text: str) -> list:
    """Découpe le log en phrases et analyse chacune."""
    lines = []
    for raw in log_text.split("\n"):
        line = _clean_text(raw)
        if line and not line.startswith("—") and len(line) > 5:
            sub = re.split(r'[.!?;]+', line)
            lines.extend(s.strip() for s in sub if s.strip())
    return [analyze_sentiment(s) for s in lines if s]


def save_sentiment_results(results: list, output_path: str = "data/sentiments.json") -> None:
    """Sauvegarde les résultats (mode append, filtre les trop_court)."""
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    existing = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    for r in results:
        if r.get("label") == "trop_court":
            continue
        existing.append({
            "timestamp":  r["timestamp"],
            "text":       r["text"],
            "sentiment":  r["label"],
            "score":      r["score"],
            "confidence": r.get("confidence", 0.0),
            "language":   r.get("language", "unknown"),
            "method":     r.get("method", "vader"),
            "details":    r.get("details", {}),
            "word_count": r.get("word_count", 0),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def get_sentiment_stats(results: list) -> dict:
    """Statistiques agrégées : tendance, distribution, langue dominante."""
    if not results:
        return {"total": 0, "avg_score": 0, "avg_confidence": 0,
                "distribution": {}, "dominant_language": "unknown", "trend": "stable"}

    scores      = [r.get("score", 0) for r in results]
    confidences = [r.get("confidence", 0) for r in results]
    labels      = [r.get("sentiment", r.get("label", "neutre")) for r in results]
    languages   = [r.get("language", "unknown") for r in results]

    label_dist = dict(Counter(labels))
    lang_dist  = Counter(languages)

    half  = len(scores) // 2
    trend = "stable"
    if half > 0:
        diff = (sum(scores[half:]) / (len(scores) - half)) - (sum(scores[:half]) / half)
        trend = "hausse" if diff > 0.1 else ("baisse" if diff < -0.1 else "stable")

    return {
        "total":             len(results),
        "avg_score":         round(sum(scores) / len(scores), 4),
        "avg_confidence":    round(sum(confidences) / len(confidences), 3),
        "distribution":      label_dist,
        "dominant_language": lang_dist.most_common(1)[0][0] if lang_dist else "unknown",
        "trend":             trend,
    }


# ── Test standalone ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "I am so happy today, everything is going great!",
        "This is terrible, I hate this broken computer",
        "Je suis vraiment content de ce projet, c'est génial !",
        "Ce logiciel est horrible, j'en ai vraiment marre.",
        "La réunion s'est bien passée.",
        "Erreur critique : le système a complètement planté !",
        "Super :) vraiment top ce projet !",
        "Je ne suis pas du tout satisfait de cette solution.",
        "Hi",
    ]
    print(f"\n{'Texte':<52} {'Langue':<8} {'Label':<15} {'Score':>7} {'Conf':>6}")
    print("─" * 95)
    for s in samples:
        r = analyze_sentiment(s)
        print(f"{s[:50]:<52} {r['language']:<8} {r['label']:<15} "
              f"{r['score']:>7.4f} {r['confidence']:>6.3f}")
    stats = get_sentiment_stats([analyze_sentiment(s) for s in samples])
    print(f"\n📊 Stats : {stats}")
