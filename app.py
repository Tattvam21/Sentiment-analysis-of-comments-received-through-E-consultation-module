# app.py — DARK THEME (high contrast, modern)
import os, glob, re
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image
from collections import Counter

# ---------------- Page & Dark Style ----------------
st.set_page_config(page_title="MCA Sentiment Dashboard", layout="wide")

st.markdown("""
<style>
:root {
  --bg:#0B0F19;          /* page background */
  --card:#141A26;        /* card background */
  --muted:#A3B1C6;       /* muted grey-blue text */
  --text:#F3F7FF;        /* primary text */
  --border:#1F2735;      /* borders */
  --shadow: 0 2px 14px rgba(0,0,0,.35);
}

html, body, [class*="css"]  {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Segoe UI Emoji";
}

.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

.card {
  background: var(--card);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: var(--shadow);
  border: 1px solid var(--border);
}

.kpi-label {
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .06em;
}

.kpi {
  color: var(--text);         /* make KPI value fully visible */
  font-size: 34px;
  font-weight: 800;
  margin-top: 8px;
}

h1,h2,h3,h4 { color: var(--text); margin-bottom: .4rem; }
.small { color: var(--muted); font-size: 13px;}
hr { border:none; border-top:1px solid var(--border); margin:8px 0 18px 0; }

/* Fix file uploader text contrast */
[data-testid="stFileUploader"] * { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("### Sentiment Analysis")
st.markdown('<div class="small">MCA e-Consultation — AI insights on stakeholder comments</div>', unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------- Upload first ----------------
st.markdown("#### Upload CSV")
uploaded = st.file_uploader(
    "Upload a CSV with a `comment_text` column (optional: `created_at`, `section_title`, `stakeholder_type`).",
    type=["csv"]
)
if not uploaded:
    st.info("Upload a CSV to see predictions and dashboard.")
    st.stop()

df = pd.read_csv(uploaded)
if "comment_text" not in df.columns:
    st.error("CSV must contain a 'comment_text' column.")
    st.stop()
df["comment_text"] = df["comment_text"].astype(str)
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# ---------------- Model (lazy load) ----------------
device = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))

@st.cache_resource(show_spinner=False)
def load_model(model_dir: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.to(device).eval()
    id2label = mdl.config.id2label
    id2label = {int(k): v for k, v in id2label.items()} if isinstance(id2label, dict) else {i: lab for i, lab in enumerate(id2label)}
    return mdl, tok, id2label

candidates = sorted(glob.glob("model_artifacts/sentiment_*"))
if not candidates:
    st.error("No model folders in `model_artifacts/`.")
    st.stop()
MODEL_DIR = candidates[-1]

with st.spinner(f"Loading model: {os.path.basename(MODEL_DIR)}"):
    model, tokenizer, id2label = load_model(MODEL_DIR, device)

MAX_LEN = 160
COLORS = {
    "positive":  "#22C55E",   # green
    "negative":  "#F43F5E",   # red
    "neutral":   "#60A5FA",   # blue
    "suggestion":"#F59E0B"    # amber
}

def predict_batch(texts, batch=128):
    out_labels, out_conf = [], []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tokenizer(chunk, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        idx = probs.argmax(axis=1)
        out_labels.extend([id2label[int(j)] for j in idx])
        out_conf.extend(probs.max(axis=1))
    return out_labels, out_conf

with st.spinner("Scoring comments…"):
    df["pred_label"], df["pred_confidence"] = predict_batch(df["comment_text"].tolist())

# ---------------- KPIs (dark cards) ----------------
total = int(len(df))
pct_pos = round((df["pred_label"] == "positive").mean()*100, 2)
pct_neg = round((df["pred_label"] == "negative").mean()*100, 2)
pct_sug = round((df["pred_label"] == "suggestion").mean()*100, 2)

k1, k2, k3, k4 = st.columns(4)
for col, label, value in [
    (k1, "Total Comments", f"{total:,}"),
    (k2, "% Positive", f"{pct_pos} %"),
    (k3, "% Negative", f"{pct_neg} %"),
    (k4, "% Suggestions", f"{pct_sug} %"),
]:
    with col:
        st.markdown(f'<div class="card"><div class="kpi-label">{label}</div><div class="kpi">{value}</div></div>', unsafe_allow_html=True)

st.write("")

# ---------------- Row 1: Donut + Word Cloud ----------------
r1c1, r1c2 = st.columns([1.1, 1])

with r1c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Sentiment Overview")

    dist = df["pred_label"].value_counts(normalize=True).mul(100).round(2).reset_index()
    dist.columns = ["label", "percent"]
    order = ["positive", "negative", "neutral", "suggestion"]
    dist["label"] = pd.Categorical(dist["label"], categories=order, ordered=True)
    dist = dist.sort_values("label")

    fig_donut = px.pie(
        dist, names="label", values="percent", hole=0.50,
        template="plotly_dark", color="label",
        color_discrete_map=COLORS
    )
    fig_donut.update_traces(textinfo="label+percent", textfont_size=16,
                            marker=dict(line=dict(color="#0B0F19", width=2)))
    fig_donut.update_layout(
        margin=dict(t=4,l=0,r=0,b=0),
        legend_title=None,
        font=dict(color="#F3F7FF", size=16),
        paper_bgcolor="#141A26",
        plot_bgcolor="#141A26"
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def top_terms_wordcloud(text_series: pd.Series, max_words=80, width=800, height=360):
    # tokenize & clean
    tokens = re.findall(r"[A-Za-z][A-Za-z\\-]{2,}", " ".join(text_series.astype(str).tolist()).lower())
    stop = {
        "comment","comments","id","amp","https","http","neutral","positive","negative","suggestion",
        "stakeholder","please","kindly","also","like","would","could","should","may","might","need","needs",
        "draft","legislation","provision","proposal","amendment","change","changes"
    }
    tokens = [t for t in tokens if t not in stop]
    freq = Counter(tokens).most_common(max_words)
    wc = WordCloud(
        width=width, height=height, background_color="#141A26",  # dark bg
        colormap="Set3", max_words=max_words, collocations=False,
        prefer_horizontal=0.95, relative_scaling=0.3
    ).generate_from_frequencies(dict(freq))
    return wc.to_image()

with r1c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Keyword Cloud")
    wc_img = top_terms_wordcloud(df["comment_text"], max_words=80, width=900, height=360)
    st.image(wc_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Row 2: Section Heatmap + Trends ----------------
r2c1, r2c2 = st.columns([1.2, 1])

with r2c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Section-wise Sentiment")
    if "section_title" in df.columns:
        pivot = df.pivot_table(index="section_title", columns="pred_label",
                               values=df.columns[0], aggfunc="count", fill_value=0)
        pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns])
        hm = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale="Blues", colorbar=dict(title="Count")
        ))
        hm.update_layout(
            template="plotly_dark", margin=dict(t=8,l=0,r=0,b=0),
            font=dict(color="#F3F7FF", size=15),
            paper_bgcolor="#141A26", plot_bgcolor="#141A26"
        )
        st.plotly_chart(hm, use_container_width=True)
    else:
        st.info("No `section_title` column in CSV.")
    st.markdown('</div>', unsafe_allow_html=True)

with r2c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Trends Over Time")
    if "created_at" in df.columns and df["created_at"].notna().any():
        ts = df.dropna(subset=["created_at"]).copy()
        ts["date"] = ts["created_at"].dt.to_period("M").dt.to_timestamp()
        trend = ts.groupby(["date","pred_label"]).size().reset_index(name="count")
        fig_line = px.line(trend, x="date", y="count", color="pred_label",
                           template="plotly_dark", color_discrete_map=COLORS)
        fig_line.update_layout(
            margin=dict(t=8,l=0,r=0,b=0), legend_title=None,
            font=dict(color="#F3F7FF", size=15),
            paper_bgcolor="#141A26", plot_bgcolor="#141A26"
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No time information to plot.")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------------- Stakeholder Comparison ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### Stakeholder Comparison")
if "stakeholder_type" in df.columns:
    stake = df.groupby(["stakeholder_type","pred_label"]).size().reset_index(name="count")
    fig_bar = px.bar(
        stake, x="stakeholder_type", y="count", color="pred_label",
        barmode="group", template="plotly_dark", color_discrete_map=COLORS
    )
    fig_bar.update_layout(
        margin=dict(t=6,l=0,r=0,b=0), legend_title=None,
        font=dict(color="#F3F7FF", size=15),
        xaxis_title=None, yaxis_title=None,
        paper_bgcolor="#141A26", plot_bgcolor="#141A26"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No `stakeholder_type` column in CSV.")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------------- Sample Comments ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### Sample Comments")
cols = [c for c in ["created_at","section_title","stakeholder_type","pred_label","pred_confidence","comment_text"] if c in df.columns]
st.dataframe(df[cols].head(60), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
