"""
extension/dashboard.py — Extension D : Dashboard de supervision temps réel
TP1 — Intelligence Artificielle & Cybersécurité

Interface web locale (Streamlit) pour superviser en temps réel :
    - Logs de frappes horodatés
    - Évolution des sentiments
    - Anomalies comportementales
    - Données sensibles détectées
    - Métriques de session en direct

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

# ---------------------------------------------------------------------------
# Résolution des chemins (fonctionne quelle que soit la CWD)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Configuration Streamlit — DOIT être le premier appel st.*
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Keylogger — Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Dark theme industriel / cybersec
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #0a0e17;
        color: #c9d1d9;
    }
    .main { background-color: #0a0e17; }
    .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    /* ── Header ── */
    .dash-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
        border: 1px solid #1f3a5f;
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .dash-header::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #00d4ff, #0066ff, #7b2fff, #00d4ff);
        background-size: 200% 100%;
        animation: scanline 3s linear infinite;
    }
    @keyframes scanline { 0%{background-position:0 0} 100%{background-position:200% 0} }
    .dash-header h1 { font-family:'Syne',sans-serif; font-size:1.8em; font-weight:800;
                      color:#e6edf3; letter-spacing:0.02em; margin:0 0 4px 0; }
    .dash-header p  { color:#8b949e; font-size:0.88em; margin:0; font-family:'JetBrains Mono',monospace; }

    /* ── KPI Cards ── */
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    .kpi-card {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
        transition: border-color .2s;
    }
    .kpi-card:hover { border-color: #388bfd; }
    .kpi-card::after {
        content: '';
        position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
        border-radius: 0 0 10px 10px;
    }
    .kpi-card.blue::after   { background: #388bfd; }
    .kpi-card.green::after  { background: #3fb950; }
    .kpi-card.red::after    { background: #f85149; }
    .kpi-card.yellow::after { background: #d29922; }
    .kpi-card .kpi-value  { font-size: 2.2em; font-weight: 800; font-family:'JetBrains Mono',monospace;
                            color: #e6edf3; line-height: 1; }
    .kpi-card .kpi-label  { font-size: 0.78em; color: #8b949e; margin-top: 6px;
                            text-transform: uppercase; letter-spacing: 0.08em; }
    .kpi-card .kpi-delta  { font-size: 0.78em; margin-top: 8px; font-family:'JetBrains Mono',monospace; }
    .kpi-card .kpi-icon   { position: absolute; right: 20px; top: 50%; transform: translateY(-50%);
                            font-size: 1.8em; opacity: 0.15; }

    /* ── Section titles ── */
    .section-title {
        font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.95em;
        color: #8b949e; text-transform: uppercase; letter-spacing: 0.12em;
        border-left: 3px solid #388bfd; padding-left: 10px;
        margin-bottom: 14px;
    }

    /* ── Log viewer ── */
    .log-container {
        background: #010409; border: 1px solid #21262d; border-radius: 8px;
        padding: 16px; height: 280px; overflow-y: auto;
        font-family: 'JetBrains Mono', monospace; font-size: 0.8em; line-height: 1.7;
    }
    .log-line { color: #8b949e; }
    .log-line .ts  { color: #388bfd; }
    .log-line .txt { color: #e6edf3; }

    /* ── Alert badges ── */
    .alert-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75em; font-weight: 600; font-family:'JetBrains Mono',monospace;
    }
    .badge-critical { background: rgba(248,81,73,.15); color: #f85149; border: 1px solid #f85149; }
    .badge-warning  { background: rgba(210,153,34,.15); color: #d29922; border: 1px solid #d29922; }
    .badge-ok       { background: rgba(63,185,80,.15);  color: #3fb950; border: 1px solid #3fb950; }
    .badge-info     { background: rgba(56,139,253,.15); color: #388bfd; border: 1px solid #388bfd; }

    /* ── Sensitive detections ── */
    .detection-row {
        background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 8px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .detection-row .dtype { font-family:'JetBrains Mono',monospace; font-size:0.82em; color:#d29922; }
    .detection-row .dtime { font-size:0.75em; color:#484f58; }

    /* ── Status bar ── */
    .status-bar {
        background: #010409; border: 1px solid #21262d; border-radius: 8px;
        padding: 10px 16px; margin-bottom: 20px;
        display: flex; justify-content: space-between; align-items: center;
        font-family: 'JetBrains Mono', monospace; font-size: 0.78em;
    }
    .status-live { color: #3fb950; }
    .status-live::before { content: '● '; animation: blink 1.2s ease-in-out infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
    [data-testid="stSidebar"] .stMarkdown { color: #8b949e; }

    /* ── Plotly charts ── */
    .js-plotly-plot .plotly { background: transparent !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #010409; }
    ::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #388bfd; }

    /* ── Streamlit overrides ── */
    .stSelectbox > div > div { background: #0d1117; border-color: #21262d; color: #e6edf3; }
    .stSlider > div { color: #e6edf3; }
    div[data-testid="metric-container"] { background: #0d1117; border: 1px solid #21262d;
                                          border-radius: 8px; padding: 12px; }
    .stAlert { border-radius: 8px; }
    button[kind="primary"] { background: #388bfd; border: none; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers — Chargement des données
# ---------------------------------------------------------------------------

def load_json_safe(path: Path) -> list:
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


def load_all() -> dict:
    return {
        "sentiments":  load_json_safe(DATA / "sentiments.json"),
        "alerts":      load_json_safe(DATA / "alerts.json"),
        "detections":  load_json_safe(DATA / "detections.json"),
        "metadata":    load_json_safe(DATA / "metadata.json"),
        "log_lines":   read_log_tail(DATA / "log.txt"),
        "ts":          datetime.now(),
    }


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_kpis(data: dict) -> dict:
    sents = data["sentiments"]
    alerts = data["alerts"]
    dets   = data["detections"]

    scores = [s.get("score", 0) for s in sents]
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0

    labels = [s.get("sentiment", "neutre") for s in sents]
    pos_pct = int(labels.count("positif") * 100 / len(labels)) if labels else 0

    recent_alerts = [a for a in alerts
                     if _is_recent(a.get("timestamp", ""), minutes=60)]

    sensitive_today = sum(1 for d in dets if d.get("has_sensitive"))

    return {
        "total_phrases":     len(sents),
        "avg_score":         avg_score,
        "positive_pct":      pos_pct,
        "total_alerts":      len(alerts),
        "recent_alerts":     len(recent_alerts),
        "sensitive_count":   sensitive_today,
        "metadata_count":    len(data["metadata"]),
    }


def _is_recent(ts_str: str, minutes: int = 60) -> bool:
    try:
        dt = datetime.fromisoformat(ts_str)
        return datetime.now() - dt < timedelta(minutes=minutes)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Plotly helpers — dark theme unifié
# ---------------------------------------------------------------------------

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color="#8b949e", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#21262d", linecolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#21262d", zerolinecolor="#21262d"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d"),
)


def chart_sentiment_timeline(sentiments: list) -> go.Figure:
    if not sentiments:
        return _empty_chart("Aucune donnée de sentiment")

    recent = sentiments[-80:]
    ts     = [s["timestamp"] for s in recent]
    scores = [s.get("score", 0) for s in recent]
    labels = [s.get("sentiment", "neutre") for s in recent]

    color_map = {"positif": "#3fb950", "négatif": "#f85149", "neutre": "#8b949e",
                 "trop_court": "#484f58"}
    marker_colors = [color_map.get(l, "#8b949e") for l in labels]

    fig = go.Figure()
    # Zones
    fig.add_hrect(y0=0.05, y1=1,    fillcolor="#3fb950", opacity=0.05, line_width=0)
    fig.add_hrect(y0=-0.05, y1=0.05, fillcolor="#8b949e", opacity=0.04, line_width=0)
    fig.add_hrect(y0=-1, y1=-0.05,  fillcolor="#f85149", opacity=0.05, line_width=0)

    fig.add_trace(go.Scatter(
        x=ts, y=scores, mode="lines+markers",
        name="Sentiment",
        line=dict(color="#388bfd", width=1.5, shape="spline", smoothing=0.8),
        marker=dict(color=marker_colors, size=7, line=dict(color="#0a0e17", width=1)),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(56,139,253,0.05)",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#21262d")

    layout = dict(**DARK_LAYOUT)
    layout.update(title=dict(text="Évolution des sentiments", font=dict(color="#e6edf3", size=13)),
                  yaxis=dict(**DARK_LAYOUT["yaxis"], range=[-1.1, 1.1]),
                  height=280)
    fig.update_layout(**layout)
    return fig


def chart_delay_histogram(metadata: list) -> go.Figure:
    if not metadata:
        return _empty_chart("Aucune méta-donnée de frappe")

    delays = [m["inter_key_delay"] for m in metadata
              if 0.005 < m.get("inter_key_delay", 0) < 1.5]
    if not delays:
        return _empty_chart("Délais insuffisants")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=delays, nbinsx=40,
        marker_color="#388bfd", opacity=0.7,
        histnorm="probability density", name="Délais",
    ))
    avg = sum(delays) / len(delays)
    fig.add_vline(x=avg, line_dash="dash", line_color="#d29922",
                  annotation_text=f"μ={avg:.3f}s",
                  annotation_font_color="#d29922", annotation_font_size=10)

    layout = dict(**DARK_LAYOUT)
    layout.update(title=dict(text="Distribution des délais inter-touches", font=dict(color="#e6edf3", size=13)),
                  xaxis_title="Délai (s)", height=280, bargap=0.02)
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
    layout.update(title=dict(text="Activité horaire", font=dict(color="#e6edf3", size=13)),
                  xaxis=dict(**DARK_LAYOUT["xaxis"], title="Heure", dtick=3,
                             tickfont=dict(color="#8b949e", size=10)),
                  yaxis=dict(**DARK_LAYOUT["yaxis"], tickfont=dict(color="#8b949e", size=10)),
                  height=280)
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
    layout.update(title=dict(text="Timeline des anomalies", font=dict(color="#e6edf3", size=13)),
                  yaxis_title="Score Isolation Forest",
                  height=260)
    fig.update_layout(**layout)
    return fig


def chart_sensitive_donut(detections: list) -> go.Figure:
    import collections
    counts: dict = collections.Counter()
    for r in detections:
        for d in r.get("detections", []):
            counts[d["type"]] += 1

    if not counts:
        return _empty_chart("Aucune donnée sensible détectée ✅")

    colors = ["#f85149", "#d29922", "#388bfd", "#3fb950", "#7b2fff", "#ff6b6b"]
    fig = go.Figure(data=go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.55,
        marker=dict(colors=colors[:len(counts)], line=dict(color="#0a0e17", width=2)),
        textfont=dict(family="JetBrains Mono", color="#e6edf3"),
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{sum(counts.values())}</b><br><span style='font-size:10'>détections</span>",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16, color="#e6edf3", family="JetBrains Mono"))
    layout = dict(**DARK_LAYOUT)
    layout.update(title=dict(text="Données sensibles — répartition", font=dict(color="#e6edf3", size=13)),
                  height=280, showlegend=True)
    fig.update_layout(**layout)
    return fig


def _empty_chart(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=13, color="#484f58",
                                                   family="JetBrains Mono"))
    layout = dict(**DARK_LAYOUT)
    layout.update(height=260, xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.update_layout(**layout)
    return fig


def plotly_cfg() -> dict:
    return {"displayModeBar": False, "responsive": True}


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def render_sidebar(kpis: dict) -> dict:
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:16px 0 8px;'>
            <div style='font-family:Syne,sans-serif; font-size:1.1em; font-weight:800; color:#e6edf3;'>
                🔍 AI KEYLOGGER
            </div>
            <div style='font-family:JetBrains Mono,monospace; font-size:0.7em; color:#8b949e; margin-top:4px;'>
                SUPERVISION DASHBOARD v1.0
            </div>
        </div>
        <hr style='border:none; border-top:1px solid #21262d; margin:12px 0;'/>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚙️ Contrôles")
        refresh_interval = st.slider(
            "Rafraîchissement (secondes)", min_value=2, max_value=30, value=5, step=1
        )
        view_mode = st.selectbox(
            "Vue",
            ["Vue globale", "Sentiments", "Anomalies", "Données sensibles", "Logs bruts"],
            index=0,
        )
        n_log_lines = st.slider("Lignes de log à afficher", 10, 100, 40)

        st.markdown("<hr style='border:none; border-top:1px solid #21262d; margin:12px 0;'/>",
                    unsafe_allow_html=True)

        # Statut des fichiers
        st.markdown("#### 📂 Sources de données")
        files_status = {
            "log.txt":        (DATA / "log.txt").exists(),
            "sentiments.json":(DATA / "sentiments.json").exists(),
            "alerts.json":    (DATA / "alerts.json").exists(),
            "detections.json":(DATA / "detections.json").exists(),
            "metadata.json":  (DATA / "metadata.json").exists(),
        }
        for fname, ok in files_status.items():
            icon  = "🟢" if ok else "🔴"
            color = "#3fb950" if ok else "#f85149"
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace; font-size:0.75em;"
                f"color:{color}; margin:3px 0'>{icon} {fname}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border:none; border-top:1px solid #21262d; margin:12px 0;'/>",
                    unsafe_allow_html=True)

        # Actions
        st.markdown("#### 🛠️ Actions")
        if st.button("🔄 Forcer le rafraîchissement", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        if st.button("📊 Générer rapport HTML", use_container_width=True):
            try:
                from report_generator import generate_html_report
                path = generate_html_report(str(DATA))
                st.success(f"Rapport généré : {path}")
            except Exception as e:
                st.error(f"Erreur : {e}")

        st.markdown("<hr style='border:none; border-top:1px solid #21262d; margin:12px 0;'/>",
                    unsafe_allow_html=True)

        # Avertissement éthique
        st.markdown("""
        <div style='background:rgba(248,81,73,.08); border:1px solid rgba(248,81,73,.3);
                    border-radius:8px; padding:12px; font-size:0.72em; color:#8b949e;
                    font-family:JetBrains Mono,monospace; line-height:1.6;'>
            ⚠️ Usage pédagogique uniquement.<br>
            Consentement requis.<br>
            Loi Godfrain — RGPD.
        </div>
        """, unsafe_allow_html=True)

    return {"refresh": refresh_interval, "view": view_mode, "n_log": n_log_lines}


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

def render_header(kpis: dict, ts: datetime) -> None:
    alert_level = "CRITIQUE" if kpis["recent_alerts"] > 3 else (
                  "ALERTE"   if kpis["recent_alerts"] > 0 else "NOMINAL")
    alert_color = "#f85149" if alert_level == "CRITIQUE" else (
                  "#d29922"  if alert_level == "ALERTE"   else "#3fb950")

    st.markdown(f"""
    <div class="dash-header">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h1>🔍 Supervision · AI Keylogger</h1>
                <p>Dernière mise à jour : {ts.strftime('%Y-%m-%d  %H:%M:%S')}  ·  
                   {kpis['total_phrases']} phrases analysées  ·  
                   {kpis['metadata_count']} frappes capturées</p>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.75em;
                            color:#8b949e; margin-bottom:6px;">STATUT SYSTÈME</div>
                <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.2em;
                            color:{alert_color}; letter-spacing:0.1em;">{alert_level}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------------------------

def render_kpis(kpis: dict) -> None:
    score  = kpis["avg_score"]
    score_color = "#3fb950" if score > 0.05 else ("#f85149" if score < -0.05 else "#d29922")
    score_label = "positif" if score > 0.05 else ("négatif" if score < -0.05 else "neutre")

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-icon">⌨️</div>
            <div class="kpi-value">{kpis['metadata_count']:,}</div>
            <div class="kpi-label">Frappes capturées</div>
            <div class="kpi-delta" style="color:#388bfd">{kpis['total_phrases']} phrases</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-icon">🧠</div>
            <div class="kpi-value" style="color:{score_color};">{score:+.3f}</div>
            <div class="kpi-label">Score sentiment moyen</div>
            <div class="kpi-delta" style="color:{score_color}">{score_label}  ·  {kpis['positive_pct']}% positif</div>
        </div>
        <div class="kpi-card {'red' if kpis['recent_alerts'] > 0 else 'green'}">
            <div class="kpi-icon">⚠️</div>
            <div class="kpi-value" style="color:{'#f85149' if kpis['recent_alerts'] > 0 else '#3fb950'}">
                {kpis['recent_alerts']}
            </div>
            <div class="kpi-label">Alertes (dernière heure)</div>
            <div class="kpi-delta" style="color:#484f58">Total cumulé : {kpis['total_alerts']}</div>
        </div>
        <div class="kpi-card {'yellow' if kpis['sensitive_count'] > 0 else 'green'}">
            <div class="kpi-icon">🔒</div>
            <div class="kpi-value" style="color:{'#d29922' if kpis['sensitive_count'] > 0 else '#3fb950'}">
                {kpis['sensitive_count']}
            </div>
            <div class="kpi-label">Données sensibles détectées</div>
            <div class="kpi-delta" style="color:#484f58">emails · CB · téléphones</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LOG VIEWER
# ---------------------------------------------------------------------------

def render_log_viewer(log_lines: list, n: int = 40) -> None:
    st.markdown('<div class="section-title">📋 Log en temps réel</div>', unsafe_allow_html=True)

    lines_html = ""
    for line in log_lines[-n:]:
        line = line.rstrip()
        if not line:
            continue
        # Lignes d'horodatage
        if line.startswith("[20"):
            lines_html += f'<div class="log-line"><span class="ts">{line}</span></div>'
        elif line.startswith("—"):
            lines_html += f'<div class="log-line" style="color:#21262d">{line}</div>'
        else:
            # Masquer les potentielles données sensibles dans l'affichage
            safe_line = line[:120] + ("…" if len(line) > 120 else "")
            lines_html += f'<div class="log-line"><span class="txt">{safe_line}</span></div>'

    if not lines_html:
        lines_html = '<div class="log-line" style="color:#484f58; font-style:italic">En attente de données… (lancez keylogger.py)</div>'

    st.markdown(f'<div class="log-container">{lines_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ALERTES RÉCENTES
# ---------------------------------------------------------------------------

def render_recent_alerts(alerts: list) -> None:
    st.markdown('<div class="section-title">🚨 Alertes récentes</div>', unsafe_allow_html=True)

    recent = [a for a in alerts if _is_recent(a.get("timestamp", ""), 120)][-10:]

    if not recent:
        st.markdown("""
        <div style='background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:16px; text-align:center; font-family:JetBrains Mono,monospace;
                    font-size:0.82em; color:#3fb950;'>
            ✅ Aucune anomalie comportementale détectée récemment
        </div>
        """, unsafe_allow_html=True)
        return

    for alert in reversed(recent):
        ts    = alert.get("timestamp", "N/A")[:19]
        score = alert.get("score", 0)
        severity = "CRITIQUE" if score < -0.6 else "ALERTE"
        badge_cls = "badge-critical" if severity == "CRITIQUE" else "badge-warning"

        st.markdown(f"""
        <div class="detection-row">
            <div>
                <span class="alert-badge {badge_cls}">{severity}</span>
                <span style='font-family:JetBrains Mono,monospace; font-size:0.8em;
                             color:#e6edf3; margin-left:10px;'>score={score:.4f}</span>
            </div>
            <div class="dtime">{ts}</div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DÉTECTIONS SENSIBLES
# ---------------------------------------------------------------------------

def render_detections(detections: list) -> None:
    st.markdown('<div class="section-title">🔒 Données sensibles</div>', unsafe_allow_html=True)

    recent_dets = [d for d in detections if d.get("has_sensitive")][-8:]

    if not recent_dets:
        st.markdown("""
        <div style='background:#0d1117; border:1px solid #21262d; border-radius:8px;
                    padding:16px; text-align:center; font-family:JetBrains Mono,monospace;
                    font-size:0.82em; color:#3fb950;'>
            ✅ Aucune donnée sensible détectée
        </div>
        """, unsafe_allow_html=True)
        return

    for r in reversed(recent_dets):
        ts = r.get("timestamp", "N/A")[:19]
        for det in r.get("detections", []):
            dtype  = det["type"].replace("_", " ").upper()
            method = det.get("method", "regex")
            badge_cls = "badge-warning" if method == "regex" else "badge-info"

            st.markdown(f"""
            <div class="detection-row">
                <div>
                    <span class="alert-badge {badge_cls}">{dtype}</span>
                    <span style='font-family:JetBrains Mono,monospace; font-size:0.75em;
                                 color:#484f58; margin-left:8px;'>[{method}] len={det.get("length",0)}</span>
                </div>
                <div class="dtime">{ts}</div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SENTIMENTS TABLE
# ---------------------------------------------------------------------------

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
        text  = s.get("text", "")[:55] + ("…" if len(s.get("text","")) > 55 else "")
        ts    = s.get("timestamp", "")[:16]

        color_map = {"positif": "#3fb950", "négatif": "#f85149",
                     "neutre": "#8b949e", "trop_court": "#484f58"}
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
                <span style='font-size:0.68em; color:#484f58; margin-left:10px;'>{ts}</span>
            </div>
        </div>
        """

    st.markdown(rows_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# VUE GLOBALE
# ---------------------------------------------------------------------------

def render_global_view(data: dict, cfg: dict) -> None:
    # Status bar
    st.markdown(f"""
    <div class="status-bar">
        <span class="status-live">EN DIRECT</span>
        <span style='color:#484f58'>rafraîchissement toutes les {cfg['refresh']}s</span>
        <span style='color:#8b949e'>{data['ts'].strftime('%H:%M:%S')}</span>
    </div>
    """, unsafe_allow_html=True)

    # Row 1 : Sentiment timeline + Heatmap
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-title">📈 Évolution des sentiments</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_sentiment_timeline(data["sentiments"]),
                        use_container_width=True, config=plotly_cfg())
    with col2:
        st.markdown('<div class="section-title">🕐 Activité horaire</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_activity_heatmap(data["metadata"]),
                        use_container_width=True, config=plotly_cfg())

    # Row 2 : Alertes + Détections
    col3, col4 = st.columns([1, 1])
    with col3:
        render_recent_alerts(data["alerts"])
    with col4:
        render_detections(data["detections"])

    # Row 3 : Histogramme délais + Donut sensibles
    col5, col6 = st.columns([1, 1])
    with col5:
        st.markdown('<div class="section-title">⌨️ Délais inter-touches</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_delay_histogram(data["metadata"]),
                        use_container_width=True, config=plotly_cfg())
    with col6:
        st.markdown('<div class="section-title">🔒 Répartition données sensibles</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(chart_sensitive_donut(data["detections"]),
                        use_container_width=True, config=plotly_cfg())

    # Row 4 : Log + Sentiments table
    col7, col8 = st.columns([3, 2])
    with col7:
        render_log_viewer(data["log_lines"], cfg["n_log"])
    with col8:
        render_sentiment_table(data["sentiments"])


# ---------------------------------------------------------------------------
# VUE DÉDIÉE — Sentiments
# ---------------------------------------------------------------------------

def render_sentiments_view(data: dict) -> None:
    st.markdown('<div class="section-title">📈 Analyse de sentiments — Vue détaillée</div>',
                unsafe_allow_html=True)

    sents = data["sentiments"]
    if not sents:
        st.info("Aucune donnée de sentiment disponible. Lancez le keylogger et tapez du texte.")
        return

    st.plotly_chart(chart_sentiment_timeline(sents), use_container_width=True, config=plotly_cfg())

    # Répartition en barres
    import collections
    label_counts = collections.Counter(s.get("sentiment","neutre") for s in sents)
    total = len(sents)

    c1, c2, c3 = st.columns(3)
    for col, lbl, clr in zip([c1, c2, c3],
                              ["positif", "négatif", "neutre"],
                              ["#3fb950", "#f85149", "#8b949e"]):
        pct = int(label_counts.get(lbl, 0) * 100 / total)
        col.markdown(f"""
        <div style='background:#0d1117; border:1px solid #21262d; border-radius:10px;
                    padding:20px; text-align:center;'>
            <div style='font-size:2em; font-weight:800; color:{clr}; font-family:JetBrains Mono,monospace;'>
                {pct}%</div>
            <div style='font-size:0.8em; color:#8b949e; margin-top:4px;'>{lbl.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    render_sentiment_table(sents)


# ---------------------------------------------------------------------------
# VUE DÉDIÉE — Anomalies
# ---------------------------------------------------------------------------

def render_anomalies_view(data: dict) -> None:
    st.markdown('<div class="section-title">⚠️ Détection d\'anomalies — Vue détaillée</div>',
                unsafe_allow_html=True)

    st.plotly_chart(chart_anomaly_scatter(data["alerts"]),
                    use_container_width=True, config=plotly_cfg())

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(chart_delay_histogram(data["metadata"]),
                        use_container_width=True, config=plotly_cfg())
    with col2:
        render_recent_alerts(data["alerts"])


# ---------------------------------------------------------------------------
# VUE DÉDIÉE — Données sensibles
# ---------------------------------------------------------------------------

def render_sensitive_view(data: dict) -> None:
    st.markdown('<div class="section-title">🔒 Données sensibles — Vue détaillée</div>',
                unsafe_allow_html=True)

    st.plotly_chart(chart_sensitive_donut(data["detections"]),
                    use_container_width=True, config=plotly_cfg())
    render_detections(data["detections"])


# ---------------------------------------------------------------------------
# VUE DÉDIÉE — Logs bruts
# ---------------------------------------------------------------------------

def render_logs_view(data: dict, n: int) -> None:
    st.markdown('<div class="section-title">📋 Logs bruts — Vue détaillée</div>',
                unsafe_allow_html=True)
    render_log_viewer(data["log_lines"], n)


# ---------------------------------------------------------------------------
# MAIN — Point d'entrée Streamlit
# ---------------------------------------------------------------------------

def main() -> None:
    # Charger les données (avec cache court)
    @st.cache_data(ttl=3)
    def _load():
        return load_all()

    data = _load()
    kpis = compute_kpis(data)

    # Sidebar
    cfg = render_sidebar(kpis)

    # Header
    render_header(kpis, data["ts"])

    # KPIs
    render_kpis(kpis)

    # Contenu selon la vue sélectionnée
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

    # Auto-refresh via st.rerun après N secondes
    time.sleep(cfg["refresh"])
    st.rerun()


if __name__ == "__main__":
    main()
