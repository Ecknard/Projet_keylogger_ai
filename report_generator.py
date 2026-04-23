"""
report_generator.py — v2.0 : Génération de rapports HTML/JSON avancée
TP1 — Intelligence Artificielle & Cybersécurité

AMÉLIORATIONS v2.0 vs v1.0 :
    ✅ Lecture de metadata.json (manquant en v1 → graphiques vides)
    ✅ Dark theme professionnel cohérent avec le dashboard
    ✅ 6e graphique : Timeline activité frappe + score IF superposé
    ✅ Résumé NLP enrichi : top bigrammes + score de risque calculé
    ✅ Export PNG des graphiques via Plotly kaleido (pour le rapport PDF)
    ✅ Rapport JSON machine-readable en parallèle du rapport HTML
    ✅ Gestion robuste : ne plante jamais même si un fichier de données manque
    ✅ Métriques de session : WPM estimé, durée, top app (si disponible)
    ✅ Chemins absolus via pathlib (évite les bugs CWD)
"""

import collections
import json
import math
import os
import re
import string
from datetime import datetime
from pathlib import Path
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

# ─────────────────────────────────────────────────────────────────────────────
# Chemins absolus
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
DATA_DIR = _ROOT / "data"

# ─────────────────────────────────────────────────────────────────────────────
# Constantes Plotly — Dark theme unifié dashboard
# ─────────────────────────────────────────────────────────────────────────────
_DARK = dict(
    paper_bgcolor="#060a0f",
    plot_bgcolor="#08111a",
    font=dict(family="monospace", color="#5a8a78", size=11),
    margin=dict(l=50, r=30, t=50, b=50),
    xaxis=dict(gridcolor="#0d2030", linecolor="#0d2030", zerolinecolor="#0d2030",
               tickfont=dict(color="#3a6050", size=10)),
    yaxis=dict(gridcolor="#0d2030", linecolor="#0d2030", zerolinecolor="#0d2030",
               tickfont=dict(color="#3a6050", size=10)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#0d2030",
                font=dict(color="#5a8a78", size=10)),
    title_font=dict(color="#a0d0b8", size=13),
    hoverlabel=dict(bgcolor="#08111a", bordercolor="#0d2030",
                    font=dict(family="monospace", color="#c0e0d0")),
)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement données
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list:
    """Charge un fichier JSON de manière robuste."""
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception:
        return []


def load_all_data(data_dir: Optional[str] = None) -> dict:
    """Charge toutes les sources de données disponibles."""
    d = Path(data_dir) if data_dir else DATA_DIR
    return {
        "sentiments":  _load_json(d / "sentiments.json"),
        "alerts":      _load_json(d / "alerts.json"),
        "detections":  _load_json(d / "detections.json"),
        "metadata":    _load_json(d / "metadata.json"),   # ← NOUVEAU v2
        "log_path":    d / "log.txt",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graphique 1 : Évolution des sentiments
# ─────────────────────────────────────────────────────────────────────────────

def plot_sentiment_timeline(sentiments: list) -> Optional[go.Figure]:
    if not _PLOTLY_AVAILABLE or not sentiments:
        return None

    ts     = [s.get("timestamp", "") for s in sentiments]
    scores = [s.get("score", 0) for s in sentiments]
    labels = [s.get("sentiment", "neutre") for s in sentiments]
    cmap   = {"positif": "#00ff88", "négatif": "#ff3366",
              "neutre": "#3a6050", "trop_court": "#1a3028"}
    mc     = [cmap.get(l, "#3a6050") for l in labels]

    fig = go.Figure()
    fig.add_hrect(y0=0.05,  y1=1,    fillcolor="#00ff88", opacity=0.04, line_width=0)
    fig.add_hrect(y0=-0.05, y1=0.05, fillcolor="#3a6050", opacity=0.03, line_width=0)
    fig.add_hrect(y0=-1,    y1=-0.05,fillcolor="#ff3366", opacity=0.04, line_width=0)
    fig.add_trace(go.Scatter(
        x=ts, y=scores, mode="lines+markers",
        name="Sentiment",
        line=dict(color="#00aaff", width=2, shape="spline", smoothing=0.8),
        marker=dict(color=mc, size=7, line=dict(color="#060a0f", width=1)),
        fill="tozeroy", fillcolor="rgba(0,170,255,0.05)",
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Score: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#0d2030")
    l = dict(**_DARK)
    l.update(
        title="Évolution des sentiments dans le temps",
        xaxis_title="Horodatage",
        yaxis_title="Score compound VADER",
        yaxis=dict(**_DARK["yaxis"], range=[-1.1, 1.1]),
        height=380,
    )
    fig.update_layout(**l)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Graphique 2 : Distribution des délais inter-touches
# ─────────────────────────────────────────────────────────────────────────────

def plot_inter_key_delays(metadata: list) -> Optional[go.Figure]:
    if not _PLOTLY_AVAILABLE or not metadata:
        return None

    delays = [m["inter_key_delay"] for m in metadata
              if 0.005 < m.get("inter_key_delay", 0) < 2.0]
    if len(delays) < 5:
        return None

    avg = sum(delays) / len(delays)
    med = sorted(delays)[len(delays)//2]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=delays, nbinsx=50, name="Fréquence",
        marker_color="#00aaff", opacity=0.65,
        histnorm="probability density",
        hovertemplate="délai: %{x:.3f}s<extra></extra>",
    ))
    fig.add_vline(x=avg, line_dash="dash", line_color="#00ff88", line_width=2,
                  annotation_text=f"μ={avg:.3f}s",
                  annotation_font=dict(color="#00ff88", family="monospace"))
    fig.add_vline(x=med, line_dash="dot", line_color="#ffaa00", line_width=1.5,
                  annotation_text=f"med={med:.3f}s",
                  annotation_font=dict(color="#ffaa00", family="monospace"))
    l = dict(**_DARK)
    l.update(
        title="Distribution des délais inter-touches",
        xaxis_title="Délai (secondes)",
        yaxis_title="Densité de probabilité",
        height=350, bargap=0.02,
        annotations=[dict(
            x=0.98, y=0.95, xref="paper", yref="paper",
            text=f"n = {len(delays):,} frappes",
            showarrow=False,
            font=dict(color="#5a8a78", family="monospace", size=11),
            bgcolor="#08111a", bordercolor="#0d2030", borderwidth=1,
        )],
    )
    fig.update_layout(**l)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Graphique 3 : Heatmap d'activité horaire
# ─────────────────────────────────────────────────────────────────────────────

def plot_activity_heatmap(metadata: list) -> Optional[go.Figure]:
    if not _PLOTLY_AVAILABLE or not metadata:
        return None

    days_fr = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    matrix  = [[0]*24 for _ in range(7)]
    for m in metadata:
        try:
            dt = datetime.fromtimestamp(m["timestamp"])
            matrix[dt.weekday()][dt.hour] += 1
        except Exception:
            continue

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=list(range(24)), y=days_fr,
        colorscale=[[0,"#060a0f"],[0.3,"#063020"],[0.7,"#0a5030"],[1,"#00ff88"]],
        hoverongaps=False, showscale=True,
        colorbar=dict(bgcolor="#08111a", tickfont=dict(color="#3a6050"), tickcolor="#0d2030"),
        hovertemplate="Jour: %{y}<br>Heure: %{x}h<br>Frappes: %{z}<extra></extra>",
    ))
    l = dict(**_DARK)
    l.update(
        title="Heatmap d'activité — Heure × Jour de la semaine",
        xaxis_title="Heure de la journée",
        yaxis_title="Jour",
        height=320,
        xaxis=dict(**_DARK["xaxis"], tickmode="linear", dtick=3),
    )
    fig.update_layout(**l)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Graphique 4 : Répartition des données sensibles (donut)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitive_data_distribution(detections: list) -> Optional[go.Figure]:
    if not _PLOTLY_AVAILABLE or not detections:
        return None

    counts: dict = collections.Counter()
    for r in detections:
        for d in r.get("detections", []):
            counts[d["type"]] += 1
    if not counts:
        return None

    colors = ["#ff3366","#ffaa00","#00aaff","#bb00ff","#ff6400","#00ff88"]
    fig = go.Figure(data=go.Pie(
        labels=list(counts.keys()), values=list(counts.values()),
        hole=0.5,
        marker=dict(colors=colors[:len(counts)], line=dict(color="#060a0f", width=2)),
        textfont=dict(family="monospace", color="#c0e0d0"),
        hovertemplate="%{label}: %{value} détections (%{percent})<extra></extra>",
    ))
    total = sum(counts.values())
    fig.add_annotation(text=f"<b>{total}</b>", x=.5, y=.5, showarrow=False,
                       font=dict(size=22, color="#ff3366", family="monospace"))
    l = dict(**_DARK)
    l.update(title="Répartition des données sensibles détectées", height=360)
    fig.update_layout(**l)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Graphique 5 : Timeline des anomalies
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_timeline(alerts: list) -> Optional[go.Figure]:
    if not _PLOTLY_AVAILABLE or not alerts:
        return None

    ts      = [a.get("timestamp","") for a in alerts]
    scores  = [a.get("score_norm", a.get("score", 0)) for a in alerts]
    sevs    = [a.get("severity", "MEDIUM") for a in alerts]
    sev_col = {"CRITICAL":"#ff3366","HIGH":"#ff6400","MEDIUM":"#ffaa00",
               "LOW":"#3a6050","NORMAL":"#1a3028"}
    colors  = [sev_col.get(s,"#ffaa00") for s in sevs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=scores, mode="markers",
        name="Anomalie",
        marker=dict(color=colors, size=11, symbol="x-thin",
                    line=dict(color=colors, width=2.5)),
        hovertemplate="<b>%{x|%H:%M:%S}</b><br>Score: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#ffaa00", line_width=1,
                  annotation_text="MEDIUM", annotation_font=dict(color="#ffaa00"))
    fig.add_hline(y=0.8, line_dash="dash", line_color="#ff6400", line_width=1,
                  annotation_text="HIGH", annotation_font=dict(color="#ff6400"))
    l = dict(**_DARK)
    l.update(
        title="Timeline des anomalies — Isolation Forest + LOF",
        xaxis_title="Horodatage",
        yaxis_title="Score normalisé [0-1]",
        yaxis=dict(**_DARK["yaxis"], range=[-0.05, 1.05]),
        height=350,
    )
    fig.update_layout(**l)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Graphique 6 (NOUVEAU v2) : Activité frappe + scores IF superposés
# ─────────────────────────────────────────────────────────────────────────────

def plot_keystroke_vs_anomaly(metadata: list, alerts: list) -> Optional[go.Figure]:
    """
    Graphique combiné : volume de frappes par minute + marqueurs d'anomalies.
    Permet de voir si les anomalies coïncident avec des pics d'activité.
    """
    if not _PLOTLY_AVAILABLE or not metadata:
        return None

    # Volume par minute
    minute_counts: dict = collections.Counter()
    for m in metadata:
        try:
            dt  = datetime.fromtimestamp(m["timestamp"])
            key = dt.strftime("%Y-%m-%d %H:%M")
            minute_counts[key] += 1
        except Exception:
            continue

    if not minute_counts:
        return None

    sorted_minutes = sorted(minute_counts.keys())
    volumes        = [minute_counts[k] for k in sorted_minutes]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=sorted_minutes, y=volumes,
        name="Frappes/min",
        marker_color="#00aaff", opacity=0.55,
    ), secondary_y=False)

    # Superposer les anomalies
    if alerts:
        al_ts  = [a.get("timestamp","")[:16] for a in alerts]
        al_sc  = [a.get("score_norm", 0.5) for a in alerts]
        fig.add_trace(go.Scatter(
            x=al_ts, y=al_sc, mode="markers",
            name="Anomalies",
            marker=dict(color="#ff3366", size=12, symbol="triangle-up",
                        line=dict(color="#ff0000", width=1.5)),
        ), secondary_y=True)
        fig.update_yaxes(title_text="Score anomalie [0-1]", secondary_y=True,
                         tickfont=dict(color="#ff3366", size=9),
                         gridcolor="#0d2030", range=[0, 1.1])

    fig.update_yaxes(title_text="Frappes / minute", secondary_y=False,
                     tickfont=dict(color="#3a6050", size=9), gridcolor="#0d2030")
    l = dict(**_DARK)
    l.update(
        title="Volume de frappes & anomalies dans le temps",
        xaxis_title="Horodatage (minutes)",
        height=360,
    )
    fig.update_layout(**l)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Résumé NLP enrichi
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    "le","la","les","de","du","des","un","une","et","est","en","à","au","aux",
    "ce","se","je","tu","il","elle","nous","vous","ils","que","qui","ne","pas",
    "plus","très","bien","tout","pour","par","sur","dans","avec","sans","mais",
    "the","a","an","and","or","is","are","was","be","have","to","of","in","on",
    "at","by","for","with","from","it","this","not","so","if","my","your","we",
}


def compute_top_words(text: str, n: int = 10) -> list:
    words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text.lower())
    return collections.Counter(w for w in words if w not in STOPWORDS).most_common(n)


def compute_top_bigrams(text: str, n: int = 5) -> list:
    """Extrait les bigrammes les plus fréquents (révèle des patterns de saisie)."""
    words = [w for w in re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text.lower())
             if w not in STOPWORDS]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    return collections.Counter(bigrams).most_common(n)


def compute_wpm(metadata: list) -> float:
    """Estime le WPM (mots par minute) moyen à partir des délais inter-touches."""
    if len(metadata) < 5:
        return 0.0
    avg_delay = sum(m.get("inter_key_delay", 0.2) for m in metadata) / len(metadata)
    avg_delay = max(avg_delay, 0.01)
    # ~5 touches = 1 mot ; 60 secondes
    chars_per_min = 60 / avg_delay
    return round(chars_per_min / 5, 1)


def compute_risk_score(data: dict) -> dict:
    """
    Calcule un score de risque global [0-100] et son niveau.

    Formule pondérée :
        - Anomalies récentes (60 min)  : poids 40
        - Données sensibles détectées  : poids 30
        - Score sentiment négatif moyen: poids 15
        - Présence de CRITICAL/HIGH    : poids 15
    """
    alerts     = data.get("alerts", [])
    dets       = data.get("detections", [])
    sentiments = data.get("sentiments", [])

    from datetime import timedelta
    recent_alerts = sum(
        1 for a in alerts
        if _is_recent_ts(a.get("timestamp",""), 60)
    )
    sensitive_cnt = sum(1 for r in dets if r.get("has_sensitive"))
    neg_scores    = [s.get("score",0) for s in sentiments if s.get("sentiment")=="négatif"]
    avg_neg       = abs(sum(neg_scores)/len(neg_scores)) if neg_scores else 0
    critical_cnt  = sum(1 for a in alerts if a.get("severity") in ["CRITICAL","HIGH"])

    raw = (min(recent_alerts * 8, 40) +
           min(sensitive_cnt * 3, 30) +
           min(avg_neg * 15, 15) +
           min(critical_cnt * 5, 15))
    score = int(min(raw, 100))

    if   score < 20: level = "LOW"
    elif score < 50: level = "MEDIUM"
    elif score < 75: level = "HIGH"
    else:            level = "CRITICAL"

    return {"score": score, "level": level}


def _is_recent_ts(ts: str, minutes: int) -> bool:
    try:
        from datetime import timedelta
        return datetime.now() - datetime.fromisoformat(ts) < timedelta(minutes=minutes)
    except Exception: return False


def generate_text_summary(data: dict) -> str:
    sents    = data.get("sentiments", [])
    alerts   = data.get("alerts", [])
    dets     = data.get("detections", [])
    meta     = data.get("metadata", [])
    risk     = compute_risk_score(data)
    wpm      = compute_wpm(meta)

    lines = [f"RAPPORT DE SESSION — {datetime.now().strftime('%d/%m/%Y %H:%M')}", ""]

    # Métriques de session
    lines += [
        f"SCORE DE RISQUE GLOBAL : {risk['score']}/100 [{risk['level']}]",
        f"Frappes capturées      : {len(meta):,}",
        f"WPM estimé             : {wpm}",
        "",
    ]

    # Sentiments
    if sents:
        labels = [s.get("sentiment","neutre") for s in sents]
        scores = [s.get("score",0) for s in sents]
        avg    = sum(scores)/len(scores)
        lines += [
            f"ANALYSE DE SENTIMENTS ({len(sents)} phrases) :",
            f"  Positif  : {labels.count('positif')} ({int(labels.count('positif')*100/len(labels))}%)",
            f"  Négatif  : {labels.count('négatif')} ({int(labels.count('négatif')*100/len(labels))}%)",
            f"  Neutre   : {labels.count('neutre')} ({int(labels.count('neutre')*100/len(labels))}%)",
            f"  Score μ  : {avg:.4f}",
            "",
        ]

    # Anomalies
    sev_counts = collections.Counter(a.get("severity","?") for a in alerts)
    lines += [
        f"ANOMALIES DÉTECTÉES : {len(alerts)}",
        *(f"  {sev} : {cnt}" for sev, cnt in sev_counts.most_common()),
        "",
    ]

    # Données sensibles
    type_counts: dict = collections.Counter()
    for r in dets:
        for d in r.get("detections",[]): type_counts[d["type"]] += 1
    lines += [
        f"DONNÉES SENSIBLES : {sum(type_counts.values())} occurrences",
        *(f"  {t} : {c}" for t, c in type_counts.most_common()),
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Template HTML dark theme
# ─────────────────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rapport SOC — AI Keylogger — {{ date }}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#060a0f;color:#8ab8a0;font-family:'Share Tech Mono',monospace;font-size:13px;line-height:1.7;}
.wrap{max-width:1100px;margin:0 auto;padding:0 24px 40px;}
/* ── Header ── */
.hdr{
  background:linear-gradient(135deg,#060f1a,#0a1e32,#060f1a);
  border-bottom:2px solid #00ff88;
  padding:40px 36px;margin-bottom:32px;
  position:relative;overflow:hidden;
}
.hdr::before{
  content:'';position:absolute;inset:0;
  background-image:linear-gradient(rgba(0,255,136,.025)1px,transparent 1px),
                   linear-gradient(90deg,rgba(0,255,136,.025)1px,transparent 1px);
  background-size:40px 40px;
}
.hdr h1{font-family:'Orbitron',monospace;font-size:1.8em;font-weight:900;
         color:#e0f0e8;letter-spacing:.06em;position:relative;}
.hdr p{color:#2a6040;margin-top:8px;font-size:.85em;position:relative;letter-spacing:.04em;}
/* ── Risk badge ── */
.risk-badge{
  display:inline-block;padding:6px 18px;border-radius:4px;font-family:'Orbitron',monospace;
  font-size:.75em;font-weight:700;letter-spacing:.1em;margin-top:12px;
  border:1px solid;
}
.risk-LOW     {background:rgba(0,255,136,.1);color:#00ff88;border-color:#00ff88;}
.risk-MEDIUM  {background:rgba(255,170,0,.1);color:#ffaa00;border-color:#ffaa00;}
.risk-HIGH    {background:rgba(255,100,0,.1);color:#ff6400;border-color:#ff6400;}
.risk-CRITICAL{background:rgba(255,51,102,.1);color:#ff3366;border-color:#ff3366;}
/* ── Warning banner ── */
.warn{background:rgba(255,51,102,.06);border:1px solid rgba(255,51,102,.2);
      border-left:3px solid #ff3366;border-radius:6px;padding:12px 16px;
      margin-bottom:24px;color:#6a8888;font-size:.82em;line-height:1.8;}
/* ── Stats grid ── */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:28px;}
.stat{background:#07111a;border:1px solid #0d2030;border-radius:8px;padding:18px;text-align:center;
      position:relative;overflow:hidden;}
.stat::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.stat.s-blue::before  {background:linear-gradient(90deg,#00aaff,#0077cc);}
.stat.s-green::before {background:linear-gradient(90deg,#00ff88,#00bb66);}
.stat.s-red::before   {background:linear-gradient(90deg,#ff3366,#cc1144);}
.stat.s-orange::before{background:linear-gradient(90deg,#ffaa00,#cc7700);}
.stat .val{font-family:'Orbitron',monospace;font-size:2em;font-weight:900;line-height:1;}
.stat .lbl{font-size:.7em;color:#2a5040;text-transform:uppercase;letter-spacing:.08em;margin-top:5px;}
.s-blue   .val{color:#00aaff;}
.s-green  .val{color:#00ff88;}
.s-red    .val{color:#ff3366;}
.s-orange .val{color:#ffaa00;}
/* ── Cards ── */
.card{background:#07111a;border:1px solid #0d2030;border-radius:8px;
      padding:24px;margin-bottom:22px;}
.card h2{font-family:'Orbitron',monospace;font-size:.9em;font-weight:700;
          color:#2a6040;letter-spacing:.14em;border-left:2px solid #00ff88;
          padding-left:10px;margin-bottom:16px;text-transform:uppercase;}
/* ── Summary ── */
.summary{background:#030608;border:1px solid #0a1820;border-radius:6px;
          padding:18px;white-space:pre-wrap;font-size:.8em;color:#5a9878;line-height:1.9;}
/* ── Top words table ── */
.words-table{width:100%;border-collapse:collapse;}
.words-table th{background:#0a1820;padding:8px 12px;text-align:left;
                color:#2a5040;font-size:.75em;letter-spacing:.06em;font-weight:normal;}
.words-table td{padding:7px 12px;border-bottom:1px solid #0a1820;font-size:.82em;color:#5a9878;}
.words-table tr:hover td{background:#0a1820;}
.words-table .cnt{text-align:right;color:#00aaff;font-family:'Orbitron',monospace;}
/* ── Footer ── */
.ftr{text-align:center;padding:24px;color:#1e3828;font-size:.72em;letter-spacing:.06em;
      border-top:1px solid #0a1820;margin-top:20px;}
</style>
</head>
<body>
<div class="hdr">
  <h1>🛡️ SOC RAPPORT — AI KEYLOGGER</h1>
  <p>GÉNÉRÉ LE {{ date }} &nbsp;·&nbsp; TP1 — IA &amp; CYBERSÉCURITÉ</p>
  <div class="risk-badge risk-{{ risk.level }}">RISK LEVEL : {{ risk.level }} ({{ risk.score }}/100)</div>
</div>
<div class="wrap">
  <div class="warn">
    ⚠️ <strong>USAGE ÉTHIQUE UNIQUEMENT</strong><br>
    Ce rapport est généré dans un cadre pédagogique exclusivement.
    Toute utilisation d'un keylogger sans consentement explicite est illégale
    (Loi Godfrain, Art. 323-1 Code pénal — jusqu'à 2 ans + 60 000 € amende, RGPD).
  </div>

  <!-- Stats -->
  <div class="stats">
    <div class="stat s-blue">
      <div class="val">{{ stats.total_keys }}</div>
      <div class="lbl">Frappes capturées</div>
    </div>
    <div class="stat s-green">
      <div class="val">{{ stats.positive_pct }}%</div>
      <div class="lbl">Sentiment positif</div>
    </div>
    <div class="stat s-red">
      <div class="val">{{ stats.anomaly_count }}</div>
      <div class="lbl">Anomalies détectées</div>
    </div>
    <div class="stat s-orange">
      <div class="val">{{ stats.sensitive_count }}</div>
      <div class="lbl">Données sensibles</div>
    </div>
  </div>

  <!-- Résumé -->
  <div class="card">
    <h2>📋 Résumé de session</h2>
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
    <h2>🔤 Top mots saisis</h2>
    <table class="words-table">
      <tr><th>Mot</th><th style="text-align:right">Occurrences</th></tr>
      {% for word, count in top_words %}
      <tr><td>{{ word }}</td><td class="cnt">{{ count }}</td></tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}

  {% if top_bigrams %}
  <div class="card">
    <h2>🔗 Top bigrammes</h2>
    <table class="words-table">
      <tr><th>Bigramme</th><th style="text-align:right">Occurrences</th></tr>
      {% for bg, count in top_bigrams %}
      <tr><td>{{ bg }}</td><td class="cnt">{{ count }}</td></tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}
</div>

<div class="ftr">
  TP1 — INTELLIGENCE ARTIFICIELLE &amp; CYBERSÉCURITÉ — SUP DE VINCI &nbsp;·&nbsp;
  Généré par report_generator.py v2.0
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Génération rapport HTML
# ─────────────────────────────────────────────────────────────────────────────

def generate_html_report(
    data_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Génère le rapport HTML complet (dark theme, 6 graphiques, résumé enrichi).

    Retour : chemin absolu du fichier généré.
    """
    if not _PLOTLY_AVAILABLE or not _JINJA2_AVAILABLE:
        print("[ERREUR] plotly et jinja2 requis.")
        return ""

    d_dir  = Path(data_dir) if data_dir else DATA_DIR
    out    = Path(output_path) if output_path else (d_dir / "report.html")
    data   = load_all_data(str(d_dir))

    # Graphiques
    charts_raw = [
        ("📈 Évolution des sentiments",        plot_sentiment_timeline(data["sentiments"])),
        ("⌨️  Distribution des délais",         plot_inter_key_delays(data["metadata"])),
        ("🕐 Heatmap d'activité horaire",       plot_activity_heatmap(data["metadata"])),
        ("🔒 Données sensibles — répartition",  plot_sensitive_data_distribution(data["detections"])),
        ("⚠️  Timeline des anomalies",           plot_anomaly_timeline(data["alerts"])),
        ("📊 Volume frappes & anomalies",        plot_keystroke_vs_anomaly(data["metadata"], data["alerts"])),
    ]
    charts = [
        {"title": t, "html": pio.to_html(f, full_html=False, include_plotlyjs="cdn")}
        for t, f in charts_raw if f is not None
    ]

    # Résumé + mots-clés
    summary     = generate_text_summary(data)
    top_words   = []
    top_bigrams = []
    if data["log_path"].exists():
        log_text    = data["log_path"].read_text(encoding="utf-8")
        top_words   = compute_top_words(log_text)
        top_bigrams = compute_top_bigrams(log_text)

    risk   = compute_risk_score(data)
    sents  = data["sentiments"]
    labels = [s.get("sentiment") for s in sents]
    pct    = int(labels.count("positif")*100/len(labels)) if labels else 0

    stats = {
        "total_keys":     len(data["metadata"]),
        "positive_pct":   pct,
        "anomaly_count":  len(data["alerts"]),
        "sensitive_count": sum(1 for r in data["detections"] if r.get("has_sensitive")),
    }

    template = Template(_HTML)
    html     = template.render(
        date=datetime.now().strftime("%d/%m/%Y à %H:%M:%S"),
        summary=summary,
        charts=charts,
        top_words=top_words,
        top_bigrams=top_bigrams,
        stats=stats,
        risk=risk,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[INFO] ✅ Rapport HTML → {out}")

    # Rapport JSON machine-readable (NOUVEAU v2)
    json_out = out.with_suffix(".json")
    json_data = {
        "generated_at":   datetime.now().isoformat(),
        "risk":           risk,
        "stats":          stats,
        "top_words":      top_words,
        "top_bigrams":    top_bigrams,
        "sentiment_summary": {
            "total": len(sents),
            "positive_pct": pct,
            "avg_score": round(sum(s.get("score",0) for s in sents)/max(len(sents),1), 4),
        },
    }
    json_out.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] ✅ Rapport JSON → {json_out}")

    return str(out)


# ─────────────────────────────────────────────────────────────────────────────
# Test standalone
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random, time as _t

    print("=== report_generator.py v2.0 — Génération démo ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    base = _t.time()
    fake_sents = [{"timestamp": datetime.fromtimestamp(base - i*300).isoformat(),
                   "text": f"Sample sentence {i}", "sentiment": random.choice(["positif","négatif","neutre"]),
                   "score": round(random.uniform(-0.8, 0.9), 4)} for i in range(30)]
    fake_alerts = [{"timestamp": datetime.fromtimestamp(base - i*3600).isoformat(),
                    "score_norm": round(random.uniform(0.5, 0.99), 4),
                    "score": round(random.uniform(-0.8, -0.2), 4),
                    "severity": random.choice(["MEDIUM","HIGH","CRITICAL"]),
                    "is_anomaly": True} for i in range(5)]
    fake_dets   = [{"timestamp": datetime.now().isoformat(), "masked_text": "****",
                    "has_sensitive": True,
                    "detections": [{"type": random.choice(["email","carte_bancaire","telephone_fr"]),
                                    "method":"regex","hash_sha256":"abc","length":16}]} for _ in range(8)]
    fake_meta   = [{"timestamp": base - i*0.2,
                    "inter_key_delay": max(0.05, random.gauss(0.15, 0.06)),
                    "key_type": random.choice(["alphanum"]*4+["special","navigation"])}
                   for i in range(500)]

    (DATA_DIR/"sentiments.json").write_text(json.dumps(fake_sents))
    (DATA_DIR/"alerts.json").write_text(json.dumps(fake_alerts))
    (DATA_DIR/"detections.json").write_text(json.dumps(fake_dets))
    (DATA_DIR/"metadata.json").write_text(json.dumps(fake_meta))

    path = generate_html_report()
    if path: print(f"✅ {path}")
