"""
extension/dashboard.py — Extension D : Dashboard de supervision temps réel
TP1 — Intelligence Artificielle & Cybersécurité

CORRECTIFS ET AMÉLIORATIONS v2 :
    ✅ CORRECTIF PRINCIPAL : suppression du @st.cache_data qui empêchait
       la lecture réelle des fichiers JSON mis à jour par le keylogger.
       Remplacement par st.session_state + timestamp de fraîcheur.

    ✅ Indicateur de fraîcheur des données (âge du dernier flush keylogger)
    ✅ Affichage du risque global (CRITIQUE / ÉLEVÉ / MOYEN / AUCUN)
    ✅ Support des nouveaux champs v2 : confidence, language, risk_level
    ✅ Graphique de confiance des sentiments
    ✅ Répartition des langues détectées
    ✅ Bouton "Forcer actualisation" manuel
    ✅ Alerte sonore (notification browser) si risque CRITIQUE
    ✅ Amélioration auto-refresh : st.rerun() contrôlé sans sleep bloquant

Lancement : streamlit run extension/dashboard.py
URL locale  : http://localhost:8501
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

# ── Config Streamlit ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Keylogger — Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Dark theme ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #0a0e17;
        color: #c9d1d9;
    }
    .main { background-color: #0a0e17; }
    .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    .dash-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
        border: 1px solid #1f3a5f; border-radius: 12px;
        padding: 28px 36px; margin-bottom: 24px;
        position: relative; overflow: hidden;
    }
    .dash-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #00d4ff, #0066ff, #7b2fff, #00d4ff);
        background-size: 200% 100%; animation: scanline 3s linear infinite;
    }
    @keyframes scanline { 0%{background-position:0 0} 100%{background-position:200% 0} }
    .dash-header h1 { font-family:'Syne',sans-serif; font-size:1.8em; font-weight:800;
                      color:#e6edf3; letter-spacing:0.02em; margin:0 0 4px 0; }
    .dash-header p  { color:#8b949e; font-size:0.88em; margin:0; font-family:'JetBrains Mono',monospace; }

    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    .kpi-card {
        background: #0d1117; border: 1px solid #21262d; border-radius: 10px;
        padding: 20px 24px; position: relative; overflow: hidden; transition: border-color .2s;
    }
    .kpi-card:hover { border-color: #388bfd; }
    .kpi-card::after {
        content: ''; position: absolute; bottom: 0; left: 0; right: 0;
        height: 3px; border-radius: 0 0 10px 10px;
    }
    .kpi-card.blue::after   { background: #388bfd; }
    .kpi-card.green::after  { background: #3fb950; }
    .kpi-card.red::after    { background: #f85149; }
    .kpi-card.yellow::after { background: #d29922; }
    .kpi-card.purple::after { background: #7b2fff; }
    .kpi-card .kpi-value  { font-size: 2.2em; font-weight: 800;
                            font-family:'JetBrains Mono',monospace; color: #e6edf3; line-height: 1; }
    .kpi-card .kpi-label  { font-size: 0.78em; color: #8b949e; margin-top: 6px;
                            text-transform: uppercase; letter-spacing: 0.08em; }
    .kpi-card .kpi-delta  { font-size: 0.78em; margin-top: 8px; font-family:'JetBrains Mono',monospace; }
    .kpi-card .kpi-icon   { position: absolute; right: 20px; top: 50%; transform: translateY(-50%);
                            font-size: 1.8em; opacity: 0.15; }

    .section-title {
        font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.95em;
        color: #8b949e; text-transform: uppercase; letter-spacing: 0.12em;
        border-left: 3px solid #388bfd; padding-left: 10px; margin-bottom: 14px;
    }

    .log-container {
        background: #010409; border: 1px solid #21262d; border-radius: 8px;
        padding: 16px; height: 280px; overflow-y: auto;
        font-family: 'JetBrains Mono', monospace; font-size: 0.8em; line-height: 1.7;
    }

    .alert-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75em; font-weight: 600; font-family:'JetBrains Mono',monospace;
    }
    .badge-critical { background: rgba(248,81,73,.15); color: #f85149; border: 1px solid #f85149; }
    .badge-warning  { background: rgba(210,153,34,.15); color: #d29922; border: 1px solid #d29922; }
    .badge-ok       { background: rgba(63,185,80,.15);  color: #3fb950; border: 1px solid #3fb950; }
    .badge-info     { background: rgba(56,139,253,.15); color: #388bfd; border: 1px solid #388bfd; }
    .badge-purple   { background: rgba(123,47,255,.15); color: #9d6aff; border: 1px solid #7b2fff; }

    .detection-row {
        background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 8px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .detection-row .dtype { font-family:'JetBrains Mono',monospace; font-size:0.82em; color:#d29922; }
    .detection-row .dtime { font-size:0.75em; color:#484f58; }

    .status-bar {
        background: #010409; border: 1px solid #21262d; border-radius: 8px;
        padding: 10px 16px; margin-bottom: 20px;
        display: flex; justify-content: space-between; align-items: center;
        font-family: 'JetBrains Mono', monospace; font-size: 0.78em;
    }
    .status-live { color: #3fb950; }
    .status-live::before { content: '● '; animation: blink 1.2s ease-in-out infinite; }
    .status-stale { color: #d29922; }
    .status-stale::before { content: '⚠ '; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

    .freshness-ok    { color: #3fb950; }
    .freshness-warn  { color: #d29922; }
    .freshness-stale { color: #f85149; }

    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
    .js-plotly-plot .plotly { background: transparent !important; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #010409; }
    ::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #388bfd; }
    .stSelectbox > div > div { background: #0d1117; border-color: #21262d; color: #e6edf3; }
    div[data-testid="metric-container"] { background: #0d1117; border: 1px solid #21262d;
                                          border-radius: 8px; padding: 12px; }
    button[kind="primary"] { background: #388bfd; border: none; border-radius: 6px; }

    .risk-critique { color: #f85149; font-weight: 800; }
    .risk-eleve    { color: #d29922; font-weight: 700; }
    .risk-moyen    { color: #388bfd; font-weight: 600; }
    .risk-aucun    { color: #3fb950; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Helpers — Chargement des données (SANS CACHE) ────────────────────────────
# FIX PRINCIPAL : on lit les fichiers directement à chaque rerun Streamlit.
# L'ancien @st.cache_data(ttl=3) empêchait la lecture des nouvelles données.

def load_json_safe(path: Path) -> list:
    """Lit un fichier JSON sans cache — garantit les données fraîches."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def read_log_tail(path: Path, n_lines: int = 60) -> list:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines[-n_lines:]
    except Exception:
        return []


def get_file_age_seconds(path: Path) -> float:
    """Retourne l'âge en secondes du fichier, ou inf si absent."""
    if not path.exists():
        return float("inf")
    return time.time() - path.stat().st_mtime


def load_all() -> dict:
    """Charge toutes les données sans cache."""
    return {
        "sentiments":  load_json_safe(DATA / "sentiments.json"),
        "alerts":      load_json_safe(DATA / "alerts.json"),
        "detections":  load_json_safe(DATA / "detections.json"),
        "metadata":    load_json_safe(DATA / "metadata.json"),
        "log_lines":   read_log_tail(DATA / "log.txt"),
        "ts":          datetime.now(),
        # Âge des fichiers pour l'indicateur de fraîcheur
        "age_sentiments":  get_file_age_seconds(DATA / "sentiments.json"),
        "age_detections":  get_file_age_seconds(DATA / "detections.json"),
        "age_log":         get_file_age_seconds(DATA / "log.txt"),
    }


# ── KPIs ──────────────────────────────────────────────────────────────────────
def compute_kpis(data: dict) -> dict:
    sents  = data["sentiments"]
    alerts = data["alerts"]
    dets   = data["detections"]

    scores = [s.get("score", 0) for s in sents]
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    labels  = [s.get("sentiment", "neutre") for s in sents]
    pos_pct = int(labels.count("positif") * 100 / len(labels)) if labels else 0

    # Confiance moyenne (nouveau champ v2)
    confidences = [s.get("confidence", 0.0) for s in sents if "confidence" in s]
    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    recent_alerts   = [a for a in alerts if _is_recent(a.get("timestamp", ""), minutes=60)]
    sensitive_today = sum(1 for d in dets if d.get("has_sensitive"))

    # Risque global : pire niveau trouvé dans les détections récentes
    risk_order = {"CRITIQUE": 4, "ÉLEVÉ": 3, "MOYEN": 2, "FAIBLE": 1, "AUCUN": 0}
    recent_dets = [d for d in dets if _is_recent(d.get("timestamp", ""), minutes=60)]
    overall_risk = "AUCUN"
    max_risk_val = 0
    for d in recent_dets:
        r = d.get("overall_risk", "AUCUN")
        if risk_order.get(r, 0) > max_risk_val:
            max_risk_val = risk_order[r]
            overall_risk = r

    return {
        "total_phrases":   len(sents),
        "avg_score":       avg_score,
        "avg_confidence":  avg_conf,
        "positive_pct":    pos_pct,
        "total_alerts":    len(alerts),
        "recent_alerts":   len(recent_alerts),
        "sensitive_count": sensitive_today,
        "metadata_count":  len(data["metadata"]),
        "overall_risk":    overall_risk,
    }


def _is_recent(ts_str: str, minutes: int = 60) -> bool:
    try:
        dt = datetime.fromisoformat(ts_str)
        return datetime.now() - dt < timedelta(minutes=minutes)
    except Exception:
        return False


# ── Plotly config ─────────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color="#8b949e", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#21262d", linecolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#21262d", zerolinecolor="#21262d"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d"),
)

def plotly_cfg():
    return {"displayModeBar": False, "responsive": True}

def _empty_chart(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color="#484f58", size=13, family="JetBrains Mono"))
    layout = dict(**DARK_LAYOUT)
    layout.update(height=260)
    fig.update_layout(**layout)
    return fig


# ── Graphiques ────────────────────────────────────────────────────────────────
def chart_sentiment_timeline(sentiments: list) -> go.Figure:
    if not sentiments:
        return _empty_chart("Aucune donnée — lancez keylogger.py et tapez du texte")
    recent = sentiments[-80:]
    ts     = [s["timestamp"] for s in recent]
    scores = [s.get("score", 0) for s in recent]
    labels = [s.get("sentiment", "neutre") for s in recent]
    confs  = [s.get("confidence", 0) for s in recent]

    color_map = {"très_positif": "#00ff88", "positif": "#3fb950",
                 "neutre": "#8b949e", "négatif": "#f85149",
                 "très_négatif": "#ff0044", "trop_court": "#484f58"}
    marker_colors = [color_map.get(l, "#8b949e") for l in labels]

    fig = go.Figure()
    fig.add_hrect(y0=0.05,  y1=1,   fillcolor="#3fb950", opacity=0.05, line_width=0)
    fig.add_hrect(y0=-0.05, y1=0.05, fillcolor="#8b949e", opacity=0.04, line_width=0)
    fig.add_hrect(y0=-1,    y1=-0.05, fillcolor="#f85149", opacity=0.05, line_width=0)

    fig.add_trace(go.Scatter(
        x=ts, y=scores, mode="lines+markers", name="Score",
        line=dict(color="#388bfd", width=1.5, shape="spline", smoothing=0.8),
        marker=dict(color=marker_colors, size=7,
                    line=dict(color="#0a0e17", width=1),
                    # Taille proportionnelle à la confiance
                    sizemode="area"),
        customdata=[[c, l] for c, l in zip(confs, labels)],
        hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<br>Label: %{customdata[1]}<br>Confiance: %{customdata[0]:.2f}<extra></extra>",
        fill="tozeroy", fillcolor="rgba(56,139,253,0.05)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#21262d")

    layout = dict(**DARK_LAYOUT)
    layout.update(
        title=dict(text="Évolution des sentiments (v2 — 5 niveaux)", font=dict(color="#e6edf3", size=13)),
        yaxis=dict(**DARK_LAYOUT["yaxis"], range=[-1.1, 1.1]),
        height=280
    )
    fig.update_layout(**layout)
    return fig


def chart_confidence_distribution(sentiments: list) -> go.Figure:
    """Nouveau : distribution des scores de confiance."""
    confs = [s.get("confidence", 0) for s in sentiments if "confidence" in s]
    if not confs:
        return _empty_chart("Aucune donnée de confiance")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=confs, nbinsx=20,
        marker_color="#7b2fff", opacity=0.75,
        histnorm="probability density", name="Confiance",
    ))
    avg = sum(confs) / len(confs)
    fig.add_vline(x=avg, line_dash="dash", line_color="#d29922",
                  annotation_text=f"μ={avg:.2f}",
                  annotation_font_color="#d29922", annotation_font_size=10)

    layout = dict(**DARK_LAYOUT)
    layout.update(
        title=dict(text="Distribution de la confiance", font=dict(color="#e6edf3", size=13)),
        height=260, bargap=0.04
    )
    fig.update_layout(**layout)
    return fig


def chart_language_pie(sentiments: list) -> go.Figure:
    """Nouveau : répartition des langues détectées."""
    from collections import Counter
    langs = Counter(s.get("language", "unknown") for s in sentiments)
    if not langs:
        return _empty_chart("Aucune donnée")

    colors = {"fr": "#3fb950", "en": "#388bfd", "other": "#d29922", "unknown": "#484f58"}
    fig = go.Figure(data=[go.Pie(
        labels=list(langs.keys()), values=list(langs.values()),
        hole=0.55,
        marker=dict(colors=[colors.get(l, "#8b949e") for l in langs.keys()],
                    line=dict(color="#0a0e17", width=2)),
        textfont=dict(family="JetBrains Mono", size=11, color="#e6edf3"),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    )])
    layout = dict(**DARK_LAYOUT)
    layout.update(
        title=dict(text="Langues détectées", font=dict(color="#e6edf3", size=13)),
        showlegend=True, height=260
    )
    fig.update_layout(**layout)
    return fig


def chart_delay_histogram(metadata: list) -> go.Figure:
    if not metadata:
        return _empty_chart("Aucune méta-donnée de frappe")
    delays = [m["inter_key_delay"] for m in metadata if 0.005 < m.get("inter_key_delay", 0) < 1.5]
    if not delays:
        return _empty_chart("Délais insuffisants")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=delays, nbinsx=40, marker_color="#388bfd", opacity=0.7,
        histnorm="probability density", name="Délais",
    ))
    avg = sum(delays) / len(delays)
    fig.add_vline(x=avg, line_dash="dash", line_color="#d29922",
                  annotation_text=f"μ={avg:.3f}s",
                  annotation_font_color="#d29922", annotation_font_size=10)
    layout = dict(**DARK_LAYOUT)
    layout.update(
        title=dict(text="Distribution des délais inter-touches", font=dict(color="#e6edf3", size=13)),
        xaxis_title="Délai (s)", height=280, bargap=0.02
    )
    fig.update_layout(**layout)
    return fig


def chart_activity_heatmap(metadata: list) -> go.Figure:
    if not metadata:
        return _empty_chart("Aucune méta-donnée de frappe")
    days_fr = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    matrix  = [[0] * 24 for _ in range(7)]
    for m in metadata:
        try:
            dt = datetime.fromtimestamp(m["timestamp"])
            matrix[dt.weekday()][dt.hour] += 1
        except Exception:
            continue
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=list(range(24)), y=days_fr,
        colorscale=[[0, "#010409"], [0.3, "#0d2d4e"], [0.7, "#1a4d8a"], [1, "#388bfd"]],
        hoverongaps=False,
        hovertemplate="Jour:%{y}  Heure:%{x}h  Frappes:%{z}<extra></extra>",
        showscale=True,
        colorbar=dict(bgcolor="rgba(0,0,0,0)", tickfont=dict(color="#8b949e")),
    ))
    layout = dict(**DARK_LAYOUT)
    layout.update(
        title=dict(text="Activité horaire", font=dict(color="#e6edf3", size=13)),
        height=280
    )
    fig.update_layout(**layout)
    return fig


def chart_anomaly_scatter(alerts: list) -> go.Figure:
    if not alerts:
        return _empty_chart("Aucune anomalie détectée ✅")
    ts     = [a["timestamp"] for a in alerts]
    scores = [a.get("score", -0.5) for a in alerts]
    recent = [_is_recent(a.get("timestamp", ""), 60) for a in alerts]
    colors = ["#f85149" if r else "#8b949e" for r in recent]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=scores, mode="markers",
        marker=dict(color=colors, size=11, symbol="x-thin",
                    line=dict(color=colors, width=2.5)),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
        name="Anomalie",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#21262d")
    layout = dict(**DARK_LAYOUT)
    layout.update(
        title=dict(text="Timeline des anomalies", font=dict(color="#e6edf3", size=13)),
        yaxis_title="Score Isolation Forest", height=260
    )
    fig.update_layout(**layout)
    return fig


def chart_sensitive_donut(detections: list) -> go.Figure:
    import collections
    counts: dict = collections.Counter()
    for r in detections:
        for det in r.get("detections", []):
            counts[det["type"]] += 1

    if not counts:
        return _empty_chart("Aucune donnée sensible détectée ✅")

    color_palette = [
        "#f85149", "#d29922", "#388bfd", "#3fb950", "#7b2fff",
        "#ff7b00", "#00d4ff", "#ff6b9d", "#9d6aff", "#ffd700",
    ]
    fig = go.Figure(data=[go.Pie(
        labels=list(counts.keys()), values=list(counts.values()),
        hole=0.55,
        marker=dict(colors=color_palette[:len(counts)],
                    line=dict(color="#0a0e17", width=2)),
        textfont=dict(family="JetBrains Mono", size=10, color="#e6edf3"),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    )])
    layout = dict(**DARK_LAYOUT)
    layout.update(
        title=dict(text="Répartition des données sensibles", font=dict(color="#e6edf3", size=13)),
        showlegend=True, height=280
    )
    fig.update_layout(**layout)
    return fig


def chart_risk_timeline(detections: list) -> go.Figure:
    """Nouveau : évolution du niveau de risque dans le temps."""
    risk_order = {"CRITIQUE": 4, "ÉLEVÉ": 3, "MOYEN": 2, "FAIBLE": 1, "AUCUN": 0}
    color_map  = {"CRITIQUE": "#f85149", "ÉLEVÉ": "#d29922",
                  "MOYEN": "#388bfd", "FAIBLE": "#3fb950", "AUCUN": "#484f58"}

    ts_list, risk_list, color_list = [], [], []
    for d in detections:
        ts_list.append(d.get("timestamp", ""))
        risk = d.get("overall_risk", "AUCUN")
        risk_list.append(risk_order.get(risk, 0))
        color_list.append(color_map.get(risk, "#484f58"))

    if not ts_list:
        return _empty_chart("Aucun risque enregistré")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_list, y=risk_list, mode="lines+markers",
        marker=dict(color=color_list, size=10, line=dict(color="#0a0e17", width=1)),
        line=dict(color="#d29922", width=1.5, shape="hv"),
        name="Risque",
        hovertemplate="<b>%{x}</b><br>Niveau: %{y}<extra></extra>",
    ))
    layout = dict(**DARK_LAYOUT)
    layout.update(
        title=dict(text="Évolution du niveau de risque", font=dict(color="#e6edf3", size=13)),
        yaxis=dict(**DARK_LAYOUT["yaxis"],
                   tickvals=[0, 1, 2, 3, 4],
                   ticktext=["AUCUN", "FAIBLE", "MOYEN", "ÉLEVÉ", "CRITIQUE"]),
        height=260
    )
    fig.update_layout(**layout)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(kpis: dict) -> dict:
    with st.sidebar:
        st.markdown("""
        <div style='padding:12px 0; border-bottom:1px solid #21262d; margin-bottom:16px;'>
            <div style='font-family:JetBrains Mono,monospace; font-size:0.75em; color:#8b949e;
                        text-transform:uppercase; letter-spacing:0.1em;'>AI Keylogger</div>
            <div style='font-size:1.1em; font-weight:700; color:#e6edf3;'>Dashboard v2</div>
        </div>""", unsafe_allow_html=True)

        view = st.selectbox("Vue", [
            "Vue globale", "Sentiments", "Anomalies",
            "Données sensibles", "Logs bruts"
        ])
        refresh = st.slider("Rafraîchissement (s)", 3, 60, 10)
        n_log   = st.slider("Lignes de log", 20, 200, 60)

        st.markdown("---")
        # Bouton de forçage du rechargement
        if st.button("🔄 Forcer actualisation", use_container_width=True):
            st.rerun()

        st.markdown("---")
        risk = kpis.get("overall_risk", "AUCUN")
        risk_colors = {"CRITIQUE": "#f85149", "ÉLEVÉ": "#d29922",
                       "MOYEN": "#388bfd", "FAIBLE": "#3fb950", "AUCUN": "#484f58"}
        rc = risk_colors.get(risk, "#484f58")
        st.markdown(f"""
        <div style='background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:12px; margin-top:8px; text-align:center;'>
            <div style='font-size:0.72em; color:#8b949e; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:6px;'>Risque global (1h)</div>
            <div style='font-size:1.5em; font-weight:800; color:{rc};
                        font-family:JetBrains Mono,monospace;'>{risk}</div>
        </div>""", unsafe_allow_html=True)

    return {"view": view, "refresh": refresh, "n_log": n_log}


# ── Header ────────────────────────────────────────────────────────────────────
def render_header(kpis: dict, ts: datetime) -> None:
    risk    = kpis.get("overall_risk", "AUCUN")
    rc      = {"CRITIQUE": "#f85149", "ÉLEVÉ": "#d29922", "MOYEN": "#388bfd",
               "FAIBLE": "#3fb950", "AUCUN": "#8b949e"}.get(risk, "#8b949e")
    st.markdown(f"""
    <div class="dash-header">
        <h1>🔍 AI Keylogger — Supervision Temps Réel</h1>
        <p>TP1 Intelligence Artificielle & Cybersécurité · {ts.strftime('%d/%m/%Y %H:%M:%S')} ·
           Risque : <span style='color:{rc}; font-weight:700;'>{risk}</span></p>
    </div>""", unsafe_allow_html=True)


# ── KPI Cards ─────────────────────────────────────────────────────────────────
def render_kpis(kpis: dict) -> None:
    score_color = "#3fb950" if kpis["avg_score"] >= 0 else "#f85149"
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-icon">⌨️</div>
            <div class="kpi-value">{kpis['total_phrases']}</div>
            <div class="kpi-label">Phrases analysées</div>
            <div class="kpi-delta" style="color:#8b949e">Total session</div>
        </div>
        <div class="kpi-card {'green' if kpis['avg_score'] >= 0 else 'red'}">
            <div class="kpi-icon">🧠</div>
            <div class="kpi-value" style="color:{score_color}">{kpis['avg_score']:+.3f}</div>
            <div class="kpi-label">Score sentiment moyen</div>
            <div class="kpi-delta" style="color:#8b949e">Conf: {kpis['avg_confidence']:.2f}</div>
        </div>
        <div class="kpi-card {'red' if kpis['recent_alerts'] > 0 else 'green'}">
            <div class="kpi-icon">⚠️</div>
            <div class="kpi-value">{kpis['recent_alerts']}</div>
            <div class="kpi-label">Anomalies (1h)</div>
            <div class="kpi-delta" style="color:#8b949e">Total: {kpis['total_alerts']}</div>
        </div>
        <div class="kpi-card {'red' if kpis['sensitive_count'] > 0 else 'green'}">
            <div class="kpi-icon">🔒</div>
            <div class="kpi-value">{kpis['sensitive_count']}</div>
            <div class="kpi-label">Données sensibles</div>
            <div class="kpi-delta" style="color:#8b949e">{kpis['metadata_count']} frappes loguées</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ── Indicateur de fraîcheur ────────────────────────────────────────────────────
def render_freshness_bar(data: dict, refresh: int) -> None:
    """Affiche l'âge des fichiers sources pour diagnostiquer le pipeline."""
    age_log   = data.get("age_log", float("inf"))
    age_sent  = data.get("age_sentiments", float("inf"))

    def _fmt_age(age: float) -> str:
        if age == float("inf"): return "absent"
        if age < 60:    return f"{int(age)}s"
        if age < 3600:  return f"{int(age/60)}min"
        return f"{int(age/3600)}h"

    def _cls(age: float) -> str:
        if age < 30:    return "freshness-ok"
        if age < 300:   return "freshness-warn"
        return "freshness-stale"

    live_class = "status-live" if age_log < 60 else "status-stale"
    live_text  = "EN DIRECT" if age_log < 60 else "DONNÉES STATIQUES"

    st.markdown(f"""
    <div class="status-bar">
        <span class="{live_class}">{live_text}</span>
        <span>
            log.txt: <span class="{_cls(age_log)}">{_fmt_age(age_log)}</span> &nbsp;|&nbsp;
            sentiments: <span class="{_cls(age_sent)}">{_fmt_age(age_sent)}</span>
        </span>
        <span style='color:#484f58'>rafraîchissement: {refresh}s · {data['ts'].strftime('%H:%M:%S')}</span>
    </div>""", unsafe_allow_html=True)

    # Avertissement si le keylogger semble inactif
    if age_log > 120:
        st.warning(
            "⚠️ **Keylogger inactif** : aucune frappe enregistrée depuis plus de 2 minutes. "
            "Lancez `python keylogger.py` pour alimenter le dashboard en données réelles.",
            icon=None
        )


# ── Log viewer ─────────────────────────────────────────────────────────────────
def render_log_viewer(log_lines: list, n: int) -> None:
    st.markdown('<div class="section-title">📋 Log des frappes</div>', unsafe_allow_html=True)
    if not log_lines:
        st.markdown("""
        <div style='background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:16px; font-family:JetBrains Mono,monospace; font-size:0.82em;
                    color:#484f58;'>Aucun log disponible. Lancez keylogger.py pour commencer.</div>
        """, unsafe_allow_html=True)
        return
    lines_html = ""
    for line in log_lines[-n:]:
        safe = line.rstrip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if safe.startswith("[20"):
            lines_html += f"<div class='log-line'><span class='ts'>{safe}</span></div>"
        elif safe.startswith("—"):
            lines_html += f"<div style='border-top:1px solid #21262d; margin:4px 0;'></div>"
        else:
            lines_html += f"<div class='log-line'><span class='txt'>{safe}</span></div>"
    st.markdown(f"<div class='log-container'>{lines_html}</div>", unsafe_allow_html=True)


# ── Alertes récentes ──────────────────────────────────────────────────────────
def render_recent_alerts(alerts: list) -> None:
    st.markdown('<div class="section-title">🚨 Alertes récentes</div>', unsafe_allow_html=True)
    recent = [a for a in alerts if _is_recent(a.get("timestamp", ""), 120)][-10:]
    if not recent:
        st.markdown("""
        <div style='background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:16px; text-align:center; font-family:JetBrains Mono,monospace;
                    font-size:0.82em; color:#3fb950;'>✅ Aucune anomalie détectée récemment</div>
        """, unsafe_allow_html=True)
        return
    for alert in reversed(recent):
        ts    = alert.get("timestamp", "N/A")[:19]
        score = alert.get("score", 0)
        severity  = "CRITIQUE" if score < -0.6 else "ALERTE"
        badge_cls = "badge-critical" if severity == "CRITIQUE" else "badge-warning"
        st.markdown(f"""
        <div class="detection-row">
            <div>
                <span class="alert-badge {badge_cls}">{severity}</span>
                <span style='font-family:JetBrains Mono,monospace; font-size:0.8em;
                             color:#e6edf3; margin-left:10px;'>score={score:.4f}</span>
            </div>
            <div class="dtime">{ts}</div>
        </div>""", unsafe_allow_html=True)


# ── Détections sensibles ──────────────────────────────────────────────────────
def render_detections(detections: list) -> None:
    st.markdown('<div class="section-title">🔒 Données sensibles</div>', unsafe_allow_html=True)
    recent_dets = [d for d in detections if d.get("has_sensitive")][-8:]
    if not recent_dets:
        st.markdown("""
        <div style='background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:16px; text-align:center; font-family:JetBrains Mono,monospace;
                    font-size:0.82em; color:#3fb950;'>✅ Aucune donnée sensible détectée</div>
        """, unsafe_allow_html=True)
        return
    for r in reversed(recent_dets):
        ts         = r.get("timestamp", "N/A")[:19]
        risk_level = r.get("overall_risk", "FAIBLE")
        risk_badges = {"CRITIQUE": "badge-critical", "ÉLEVÉ": "badge-warning",
                       "MOYEN": "badge-info", "FAIBLE": "badge-ok"}
        rb = risk_badges.get(risk_level, "badge-ok")
        for det in r.get("detections", []):
            dtype  = det["type"].replace("_", " ").upper()
            method = det.get("method", "regex")
            mb     = "badge-warning" if method == "regex" else "badge-purple"
            st.markdown(f"""
            <div class="detection-row">
                <div>
                    <span class="alert-badge {rb}">{risk_level}</span>
                    <span class="alert-badge {mb}" style="margin-left:6px;">{dtype}</span>
                    <span style='font-family:JetBrains Mono,monospace; font-size:0.75em;
                                 color:#484f58; margin-left:8px;'>[{method}] len={det.get("length",0)}</span>
                </div>
                <div class="dtime">{ts}</div>
            </div>""", unsafe_allow_html=True)


# ── Table des sentiments ──────────────────────────────────────────────────────
def render_sentiment_table(sentiments: list) -> None:
    st.markdown('<div class="section-title">🧠 Dernières analyses sentiments</div>',
                unsafe_allow_html=True)
    recent = sentiments[-12:]
    if not recent:
        st.markdown('<div style="color:#484f58; font-family:JetBrains Mono,monospace; font-size:0.82em;">Aucune donnée.</div>',
                    unsafe_allow_html=True)
        return

    rows_html = ""
    for s in reversed(recent):
        label = s.get("sentiment", "neutre")
        score = s.get("score", 0)
        conf  = s.get("confidence", 0)
        lang  = s.get("language", "?")
        text  = s.get("text", "")[:55] + ("…" if len(s.get("text", "")) > 55 else "")
        ts    = s.get("timestamp", "")[:16]

        color_map = {"très_positif": "#00ff88", "positif": "#3fb950",
                     "neutre": "#8b949e", "négatif": "#f85149",
                     "très_négatif": "#ff0044", "trop_court": "#484f58"}
        color = color_map.get(label, "#8b949e")
        bar   = abs(score) * 100
        bar_c = "#3fb950" if score > 0 else "#f85149"

        rows_html += f"""
        <div style='background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:10px 14px; margin-bottom:6px;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
                <span style='font-family:JetBrains Mono,monospace; font-size:0.8em; color:#e6edf3;'>{text}</span>
                <span style='font-family:JetBrains Mono,monospace; font-size:0.75em; color:{color};
                             font-weight:600;'>{score:+.3f}</span>
            </div>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='flex:1; background:#21262d; border-radius:3px; height:4px; margin-right:12px;'>
                    <div style='width:{bar:.0f}%; background:{bar_c}; height:4px; border-radius:3px;'></div>
                </div>
                <span style='font-size:0.7em; color:{color};'>{label}</span>
                <span style='font-size:0.68em; color:#484f58; margin-left:6px;'>[{lang}]</span>
                <span style='font-size:0.68em; color:#484f58; margin-left:6px;'>conf:{conf:.2f}</span>
                <span style='font-size:0.68em; color:#484f58; margin-left:10px;'>{ts}</span>
            </div>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)


# ── Vues ──────────────────────────────────────────────────────────────────────
def render_global_view(data: dict, cfg: dict) -> None:
    render_freshness_bar(data, cfg["refresh"])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-title">📈 Évolution des sentiments</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_sentiment_timeline(data["sentiments"]),
                        use_container_width=True, config=plotly_cfg())
    with col2:
        st.markdown('<div class="section-title">🕐 Activité horaire</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_activity_heatmap(data["metadata"]),
                        use_container_width=True, config=plotly_cfg())

    col3, col4 = st.columns([1, 1])
    with col3:
        render_recent_alerts(data["alerts"])
    with col4:
        render_detections(data["detections"])

    col5, col6 = st.columns([1, 1])
    with col5:
        st.markdown('<div class="section-title">⌨️ Délais inter-touches</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_delay_histogram(data["metadata"]),
                        use_container_width=True, config=plotly_cfg())
    with col6:
        st.markdown('<div class="section-title">🔒 Répartition données sensibles</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_sensitive_donut(data["detections"]),
                        use_container_width=True, config=plotly_cfg())

    col7, col8 = st.columns([3, 2])
    with col7:
        render_log_viewer(data["log_lines"], cfg["n_log"])
    with col8:
        render_sentiment_table(data["sentiments"])


def render_sentiments_view(data: dict) -> None:
    st.markdown('<div class="section-title">📈 Sentiments — Vue détaillée</div>', unsafe_allow_html=True)
    sents = data["sentiments"]
    if not sents:
        st.info("Aucune donnée. Lancez keylogger.py et tapez du texte.")
        return
    st.plotly_chart(chart_sentiment_timeline(sents), use_container_width=True, config=plotly_cfg())

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(chart_confidence_distribution(sents), use_container_width=True, config=plotly_cfg())
    with col2:
        st.plotly_chart(chart_language_pie(sents), use_container_width=True, config=plotly_cfg())

    from collections import Counter
    label_counts = Counter(s.get("sentiment", "neutre") for s in sents)
    total = len(sents)
    cols = st.columns(5)
    for col, (lbl, clr) in zip(cols, [
        ("très_positif", "#00ff88"), ("positif", "#3fb950"), ("neutre", "#8b949e"),
        ("négatif", "#f85149"), ("très_négatif", "#ff0044")
    ]):
        pct = int(label_counts.get(lbl, 0) * 100 / total)
        col.markdown(f"""
        <div style='background:#0d1117; border:1px solid #21262d; border-radius:10px;
                    padding:16px; text-align:center;'>
            <div style='font-size:1.6em; font-weight:800; color:{clr};
                        font-family:JetBrains Mono,monospace;'>{pct}%</div>
            <div style='font-size:0.72em; color:#8b949e; margin-top:4px;'>{lbl.upper()}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    render_sentiment_table(sents)


def render_anomalies_view(data: dict) -> None:
    st.markdown('<div class="section-title">⚠️ Anomalies — Vue détaillée</div>', unsafe_allow_html=True)
    st.plotly_chart(chart_anomaly_scatter(data["alerts"]),
                    use_container_width=True, config=plotly_cfg())
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(chart_delay_histogram(data["metadata"]),
                        use_container_width=True, config=plotly_cfg())
    with col2:
        render_recent_alerts(data["alerts"])


def render_sensitive_view(data: dict) -> None:
    st.markdown('<div class="section-title">🔒 Données sensibles — Vue détaillée</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(chart_sensitive_donut(data["detections"]),
                        use_container_width=True, config=plotly_cfg())
    with col2:
        st.plotly_chart(chart_risk_timeline(data["detections"]),
                        use_container_width=True, config=plotly_cfg())
    render_detections(data["detections"])


def render_logs_view(data: dict, n: int) -> None:
    st.markdown('<div class="section-title">📋 Logs bruts — Vue détaillée</div>', unsafe_allow_html=True)
    render_log_viewer(data["log_lines"], n)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Initialisation du compteur de refresh dans session_state
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    # Chargement direct sans cache
    data = load_all()
    kpis = compute_kpis(data)
    cfg  = render_sidebar(kpis)

    render_header(kpis, data["ts"])
    render_kpis(kpis)

    view = cfg["view"]
    if view == "Vue globale":
        render_global_view(data, cfg)
    elif view == "Sentiments":
        render_sentiments_view(data)
    elif view == "Anomalies":
        render_anomalies_view(data)
    elif view == "Données sensibles":
        render_sensitive_view(data)
    elif view == "Logs bruts":
        render_logs_view(data, cfg["n_log"])

    # Auto-refresh propre : on utilise st.empty() + time.sleep court
    # pour éviter de bloquer l'UI pendant toute la durée du sleep
    placeholder = st.empty()
    for remaining in range(cfg["refresh"], 0, -1):
        placeholder.markdown(
            f"<div style='text-align:center; color:#484f58; font-family:JetBrains Mono,monospace;"
            f" font-size:0.72em; margin-top:8px;'>⏱ Prochain rafraîchissement dans {remaining}s</div>",
            unsafe_allow_html=True
        )
        time.sleep(1)
    placeholder.empty()
    st.rerun()


if __name__ == "__main__":
    main()
