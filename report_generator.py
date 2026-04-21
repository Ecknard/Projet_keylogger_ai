"""
report_generator.py — Partie IV : Génération de rapports et visualisation
TP1 — Intelligence Artificielle & Cybersécurité

Visualisations implémentées (Tâche 9.2) :
    1. Évolution des sentiments dans le temps (line chart)
    2. Distribution des délais inter-touches (histogramme + densité)
    3. Heatmap d'activité horaire (heure × jour de la semaine)
    4. Proportion de données sensibles détectées (donut chart)
    5. Timeline des anomalies détectées (scatter plot)

Export (Tâche 10.2) : rapport HTML complet auto-généré via Jinja2 + Plotly
Résumé NLP (Tâche 10.1) : fréquence de mots + génération via API Claude
"""

import collections
import json
import os
import re
import string
from datetime import datetime
from typing import Optional

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    print("[AVERTISSEMENT] plotly non installé : pip install plotly")

try:
    from jinja2 import Template
    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False
    print("[AVERTISSEMENT] jinja2 non installé : pip install jinja2")

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_json(path: str) -> list:
    """Charge un fichier JSON et retourne une liste vide si indisponible."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def load_all_data(data_dir: str = "data") -> dict:
    """Charge toutes les données produites par le keylogger."""
    return {
        "sentiments":   load_json(os.path.join(data_dir, "sentiments.json")),
        "alerts":       load_json(os.path.join(data_dir, "alerts.json")),
        "detections":   load_json(os.path.join(data_dir, "detections.json")),
        "log_path":     os.path.join(data_dir, "log.txt"),
    }


# ---------------------------------------------------------------------------
# Visualisation 1 : Évolution des sentiments dans le temps
# ---------------------------------------------------------------------------

def plot_sentiment_timeline(sentiments: list) -> Optional[object]:
    """Line chart avec zones colorées positif/négatif/neutre."""
    if not _PLOTLY_AVAILABLE or not sentiments:
        print("[INFO] Pas de données de sentiment à afficher.")
        return None

    timestamps = [s["timestamp"] for s in sentiments]
    scores     = [s["score"] for s in sentiments]
    labels     = [s["sentiment"] for s in sentiments]

    colors = {"positif": "#27ae60", "négatif": "#e74c3c", "neutre": "#95a5a6",
              "trop_court": "#bdc3c7", "erreur_librairie": "#bdc3c7"}
    marker_colors = [colors.get(l, "#95a5a6") for l in labels]

    fig = go.Figure()

    # Ligne du score
    fig.add_trace(go.Scatter(
        x=timestamps, y=scores,
        mode="lines+markers",
        name="Score de sentiment",
        line=dict(color="#3498db", width=2),
        marker=dict(color=marker_colors, size=8),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
    ))

    # Zones colorées
    fig.add_hrect(y0=0.05, y1=1.0,   fillcolor="#27ae60", opacity=0.07, line_width=0,
                  annotation_text="Positif",  annotation_position="top left")
    fig.add_hrect(y0=-0.05, y1=0.05, fillcolor="#95a5a6", opacity=0.07, line_width=0,
                  annotation_text="Neutre",   annotation_position="top left")
    fig.add_hrect(y0=-1.0, y1=-0.05, fillcolor="#e74c3c", opacity=0.07, line_width=0,
                  annotation_text="Négatif",  annotation_position="top left")

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="📊 Évolution des sentiments au cours du temps",
        xaxis_title="Horodatage",
        yaxis_title="Score de sentiment (compound)",
        yaxis=dict(range=[-1.1, 1.1]),
        template="plotly_white",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Visualisation 2 : Distribution des délais inter-touches
# ---------------------------------------------------------------------------

def plot_inter_key_delays(metadata: list) -> Optional[object]:
    """Histogramme avec courbe de densité KDE."""
    if not _PLOTLY_AVAILABLE or not metadata:
        return None

    delays = [m["inter_key_delay"] for m in metadata
              if 0 < m.get("inter_key_delay", 0) < 2.0]  # Filtrer les outliers extrêmes

    if not delays:
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=delays,
        nbinsx=50,
        name="Fréquence",
        marker_color="#3498db",
        opacity=0.7,
        histnorm="probability density",
    ))

    fig.update_layout(
        title="⌨️  Distribution des délais inter-touches",
        xaxis_title="Délai (secondes)",
        yaxis_title="Densité",
        template="plotly_white",
        height=400,
        bargap=0.02,
        annotations=[dict(
            x=0.98, y=0.95, xref="paper", yref="paper",
            text=f"n={len(delays)} frappes<br>μ={sum(delays)/len(delays):.3f}s",
            showarrow=False,
            bgcolor="white", bordercolor="#3498db", borderwidth=1,
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# Visualisation 3 : Heatmap d'activité horaire
# ---------------------------------------------------------------------------

def plot_activity_heatmap(metadata: list) -> Optional[object]:
    """Heatmap heure × jour de la semaine."""
    if not _PLOTLY_AVAILABLE or not metadata:
        return None

    days_fr = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    matrix  = [[0] * 24 for _ in range(7)]  # [jour][heure]

    for m in metadata:
        try:
            dt = datetime.fromtimestamp(m["timestamp"])
            matrix[dt.weekday()][dt.hour] += 1
        except (KeyError, ValueError, OSError):
            continue

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=list(range(24)),
        y=days_fr,
        colorscale="Blues",
        hoverongaps=False,
        hovertemplate="Jour: %{y}<br>Heure: %{x}h<br>Frappes: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="🕐 Heatmap d'activité — Heure × Jour de la semaine",
        xaxis_title="Heure de la journée",
        yaxis_title="Jour",
        template="plotly_white",
        height=350,
        xaxis=dict(tickmode="linear", dtick=2),
    )
    return fig


# ---------------------------------------------------------------------------
# Visualisation 4 : Proportion de données sensibles (donut)
# ---------------------------------------------------------------------------

def plot_sensitive_data_distribution(detections: list) -> Optional[object]:
    """Donut chart par type de données sensibles détectées."""
    if not _PLOTLY_AVAILABLE or not detections:
        return None

    type_counts: dict = collections.Counter()
    for record in detections:
        for det in record.get("detections", []):
            type_counts[det["type"]] += 1

    if not type_counts:
        return None

    labels = list(type_counts.keys())
    values = list(type_counts.values())
    colors = ["#e74c3c", "#e67e22", "#f39c12", "#27ae60", "#3498db", "#9b59b6"]

    fig = go.Figure(data=go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors[:len(labels)]),
        hovertemplate="%{label}: %{value} détections (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title="🔒 Répartition des données sensibles détectées",
        template="plotly_white",
        height=400,
        annotations=[dict(text=f"Total<br>{sum(values)}", x=0.5, y=0.5,
                          font_size=14, showarrow=False)],
    )
    return fig


# ---------------------------------------------------------------------------
# Visualisation 5 : Timeline des anomalies
# ---------------------------------------------------------------------------

def plot_anomaly_timeline(alerts: list) -> Optional[object]:
    """Scatter plot avec marqueurs d'alerte horodatés."""
    if not _PLOTLY_AVAILABLE or not alerts:
        return None

    timestamps = [a["timestamp"] for a in alerts]
    scores     = [a.get("score", 0) for a in alerts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=scores,
        mode="markers",
        name="Anomalie",
        marker=dict(
            color="#e74c3c",
            size=12,
            symbol="x",
            line=dict(color="#c0392b", width=2),
        ),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#7f8c8d", opacity=0.5,
                  annotation_text="Seuil anomalie")

    fig.update_layout(
        title="⚠️  Timeline des anomalies détectées",
        xaxis_title="Horodatage",
        yaxis_title="Score d'anomalie (Isolation Forest)",
        template="plotly_white",
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Résumé NLP — Tâche 10.1
# ---------------------------------------------------------------------------

STOPWORDS_FR = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "est", "en",
    "à", "au", "aux", "ce", "se", "je", "tu", "il", "elle", "nous", "vous",
    "ils", "elles", "que", "qui", "quoi", "dont", "où", "ou", "si", "ne",
    "pas", "plus", "très", "bien", "tout", "tous", "pour", "par", "sur",
    "dans", "avec", "sans", "sous", "entre", "vers", "chez", "lors", "car",
    "mais", "donc", "or", "ni", "car", "cela", "ceci", "mon", "ma", "mes",
    "ton", "ta", "son", "sa", "ses", "notre", "votre", "leur", "leurs",
}
STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "up",
    "out", "as", "it", "its", "this", "that", "these", "those", "i",
    "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "not", "no", "so", "if",
}
STOPWORDS = STOPWORDS_FR | STOPWORDS_EN


def compute_top_words(log_text: str, top_n: int = 10) -> list:
    """Retourne les N mots les plus fréquents du log (hors stopwords)."""
    words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', log_text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    return collections.Counter(filtered).most_common(top_n)


def generate_text_summary(data: dict) -> str:
    """
    Génère un résumé textuel automatique des activités enregistrées.

    Approche : résumé extractif + statistiques calculées.
    Pour un résumé narratif complet, utiliser generate_summary_with_llm().
    """
    sentiments = data.get("sentiments", [])
    alerts     = data.get("alerts", [])
    detections = data.get("detections", [])

    lines = []
    lines.append(f"## Résumé de session — {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")

    # Stats sentiments
    if sentiments:
        labels = [s.get("sentiment", "neutre") for s in sentiments]
        pos    = labels.count("positif")
        neg    = labels.count("négatif")
        neu    = labels.count("neutre")
        scores = [s.get("score", 0) for s in sentiments]
        avg    = sum(scores) / len(scores) if scores else 0

        lines.append(f"**Analyse de sentiments** ({len(sentiments)} phrases analysées) :")
        lines.append(f"- Positif : {pos} ({pos*100//len(labels) if labels else 0}%)")
        lines.append(f"- Négatif : {neg} ({neg*100//len(labels) if labels else 0}%)")
        lines.append(f"- Neutre  : {neu} ({neu*100//len(labels) if labels else 0}%)")
        lines.append(f"- Score moyen : {avg:.4f}\n")
    else:
        lines.append("**Analyse de sentiments** : aucune donnée disponible.\n")

    # Stats anomalies
    lines.append(f"**Anomalies détectées** : {len(alerts)}")
    if alerts:
        lines.append(f"- Première alerte : {alerts[0].get('timestamp', 'N/A')}")
        lines.append(f"- Dernière alerte  : {alerts[-1].get('timestamp', 'N/A')}\n")
    else:
        lines.append("- Aucune anomalie de comportement de frappe détectée.\n")

    # Stats données sensibles
    total_sensitive = sum(1 for r in detections if r.get("has_sensitive"))
    lines.append(f"**Données sensibles** : {total_sensitive} occurrences détectées")
    type_counts: dict = collections.Counter()
    for record in detections:
        for det in record.get("detections", []):
            type_counts[det["type"]] += 1
    for dtype, count in type_counts.most_common():
        lines.append(f"  - {dtype} : {count}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export HTML — Tâche 10.2
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport Keylogger IA — {{ date }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #2c3e50; }
        .header {
            background: linear-gradient(135deg, #1a252f, #2c3e50);
            color: white; padding: 40px; text-align: center;
        }
        .header h1 { font-size: 2em; margin-bottom: 8px; }
        .header p  { opacity: 0.75; }
        .container { max-width: 1200px; margin: 30px auto; padding: 0 20px; }
        .card {
            background: white; border-radius: 12px; padding: 30px;
            margin-bottom: 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        }
        .card h2 { color: #2c3e50; margin-bottom: 16px; border-bottom: 2px solid #3498db; padding-bottom: 8px; }
        .summary { white-space: pre-wrap; font-family: monospace; background: #f8f9fa;
                   padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }
        .stat-box {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white; padding: 20px; border-radius: 10px; text-align: center;
        }
        .stat-box .value { font-size: 2em; font-weight: bold; }
        .stat-box .label { font-size: 0.85em; opacity: 0.85; margin-top: 4px; }
        .footer { text-align: center; padding: 20px; color: #7f8c8d; font-size: 0.85em; }
        .warning { background: #fff3cd; border: 1px solid #ffc107;
                   padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Rapport d'Analyse — AI Keylogger</h1>
        <p>Généré automatiquement le {{ date }}</p>
    </div>
    <div class="container">
        <div class="warning">
            ⚠️ <strong>Usage éthique uniquement.</strong>
            Ce rapport est généré dans un cadre pédagogique et/ou avec le consentement explicite de l'utilisateur monitoré.
        </div>

        <!-- Statistiques globales -->
        <div class="card">
            <h2>📊 Vue d'ensemble</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="value">{{ stats.total_sentences }}</div>
                    <div class="label">Phrases analysées</div>
                </div>
                <div class="stat-box" style="background: linear-gradient(135deg,#27ae60,#229954)">
                    <div class="value">{{ stats.positive_pct }}%</div>
                    <div class="label">Sentiment positif</div>
                </div>
                <div class="stat-box" style="background: linear-gradient(135deg,#e74c3c,#c0392b)">
                    <div class="value">{{ stats.anomaly_count }}</div>
                    <div class="label">Anomalies détectées</div>
                </div>
                <div class="stat-box" style="background: linear-gradient(135deg,#f39c12,#d68910)">
                    <div class="value">{{ stats.sensitive_count }}</div>
                    <div class="label">Données sensibles</div>
                </div>
            </div>
        </div>

        <!-- Résumé textuel -->
        <div class="card">
            <h2>📝 Résumé de session</h2>
            <div class="summary">{{ summary }}</div>
        </div>

        <!-- Graphiques -->
        {% for chart in charts %}
        <div class="card">
            <h2>{{ chart.title }}</h2>
            {{ chart.html | safe }}
        </div>
        {% endfor %}

        <!-- Top mots -->
        {% if top_words %}
        <div class="card">
            <h2>🔤 Mots les plus fréquents</h2>
            <table style="width:100%; border-collapse:collapse">
                <tr style="background:#f0f2f5">
                    <th style="padding:8px;text-align:left">Mot</th>
                    <th style="padding:8px;text-align:right">Occurrences</th>
                </tr>
                {% for word, count in top_words %}
                <tr style="border-bottom:1px solid #ecf0f1">
                    <td style="padding:8px">{{ word }}</td>
                    <td style="padding:8px;text-align:right"><strong>{{ count }}</strong></td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
    </div>
    <div class="footer">
        TP1 — Intelligence Artificielle & Cybersécurité — SUP DE VINCI
    </div>
</body>
</html>"""


def generate_html_report(data_dir: str = "data", output_path: str = "data/report.html") -> str:
    """
    Génère le rapport HTML complet.

    Retour
    ------
    str : chemin vers le rapport généré.
    """
    if not _PLOTLY_AVAILABLE or not _JINJA2_AVAILABLE:
        print("[ERREUR] plotly et jinja2 sont requis pour la génération HTML.")
        return ""

    data = load_all_data(data_dir)

    # Préparer les graphiques
    charts_raw = [
        ("📈 Évolution des sentiments",       plot_sentiment_timeline(data["sentiments"])),
        ("⌨️  Délais inter-touches",           plot_inter_key_delays(
            load_json(os.path.join(data_dir, "metadata.json"))
        )),
        ("🕐 Heatmap d'activité",             plot_activity_heatmap(
            load_json(os.path.join(data_dir, "metadata.json"))
        )),
        ("🔒 Données sensibles (répartition)", plot_sensitive_data_distribution(data["detections"])),
        ("⚠️  Timeline des anomalies",         plot_anomaly_timeline(data["alerts"])),
    ]

    charts = []
    for title, fig in charts_raw:
        if fig is not None:
            charts.append({
                "title": title,
                "html": pio.to_html(fig, full_html=False, include_plotlyjs="cdn"),
            })

    # Résumé texte
    summary = generate_text_summary(data)

    # Top mots depuis le log.txt
    top_words = []
    if os.path.exists(data["log_path"]):
        with open(data["log_path"], "r", encoding="utf-8") as f:
            log_content = f.read()
        top_words = compute_top_words(log_content)

    # Stats globales
    sentiments = data["sentiments"]
    labels     = [s.get("sentiment") for s in sentiments]
    pos_count  = labels.count("positif")
    pct        = int(pos_count * 100 / len(labels)) if labels else 0

    stats = {
        "total_sentences": len(sentiments),
        "positive_pct":    pct,
        "anomaly_count":   len(data["alerts"]),
        "sensitive_count": sum(1 for r in data["detections"] if r.get("has_sensitive")),
    }

    # Rendre le template
    template = Template(HTML_TEMPLATE)
    html = template.render(
        date=datetime.now().strftime("%d/%m/%Y à %H:%M:%S"),
        summary=summary,
        charts=charts,
        top_words=top_words,
        stats=stats,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[INFO] Rapport HTML généré → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Génération d'un rapport de démonstration avec données simulées...")

    # Créer des données de test
    os.makedirs("data", exist_ok=True)

    import random, time as _time
    fake_sentiments = []
    fake_alerts = []
    fake_detections = []

    for i in range(30):
        score = random.uniform(-0.8, 0.9)
        label = "positif" if score > 0.05 else ("négatif" if score < -0.05 else "neutre")
        fake_sentiments.append({
            "timestamp": datetime.fromtimestamp(_time.time() - (30-i)*300).isoformat(),
            "text": f"Sample sentence number {i}",
            "sentiment": label,
            "score": round(score, 4),
        })

    for i in range(5):
        fake_alerts.append({
            "timestamp": datetime.fromtimestamp(_time.time() - i*3600).isoformat(),
            "score": round(random.uniform(-0.8, -0.2), 4),
            "is_anomaly": True,
        })

    for i in range(8):
        fake_detections.append({
            "timestamp": datetime.now().isoformat(),
            "masked_text": "Masked text example",
            "has_sensitive": True,
            "detections": [{"type": random.choice(["email","carte_bancaire","telephone_fr"]),
                            "method": "regex", "hash_sha256": "abc123", "length": 16}]
        })

    with open("data/sentiments.json", "w") as f: json.dump(fake_sentiments, f)
    with open("data/alerts.json", "w") as f:     json.dump(fake_alerts, f)
    with open("data/detections.json", "w") as f: json.dump(fake_detections, f)

    path = generate_html_report()
    if path:
        print(f"✅ Rapport disponible : {path}")
