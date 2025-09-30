# app.py — MCA e-Consultation Sentiment Dashboard (polished)

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

# ---------------- Page & Style ----------------
st.set_page_config(page_title="MCA Sentiment Dashboard", layout="wide")

# Clean, card-based UI similar to your mock
st.markdown("""
<style>
:root { --card-bg:#ffffff; --text:#0f172a; --muted:#475569; }
html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Segoe UI Emoji";}
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
.card { background:var(--card-bg); border-radius:16px; padding:18px 18px; box-shadow:0 2px 14px rgba(15,23,42,.07); border:1px solid rgba(2,6,23,.06); }
.kpi { font-size:34px; font-weight:800; margin-top:6px; }
.kpi-label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }
h1,h2,h3,h4 { color:var(--text); margin-bottom: .4rem; }
.small { color:var(--muted); font-size:13px;}
hr { border:none; border-top:1px solid rgba(2,6,23,.08); margin:8px 0 18px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("### Sentiment Analysis")
st.markdown('<div class="small">MCA e-Consultation — AI insights on stakeholder comments</div>', unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------- Uploader first (instant UI) ----------------
st.markdown("#### Upload CSV")
uploaded = st.file_uploader(
    "Upload a CSV with a `comment_text` column (optional: `created_at`, `section_title`, `stakeholder_type`).",
    type=["csv"]
)

if not uploaded:
    st.info("Upload a CSV to see predictions and dashboard.")
    st.stop()

# Read CSV early (fast)
df = pd.read_csv(uploaded)
if "comment_text" not in df.columns:
    st.error("CSV must contain a 'comment_text' column.")
    st.stop()
df["comment_text"] = df["comment_text"].astype(str)
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# ---------------- Load model lazily (after upload) ----------------
default_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(default_device)

@st.cache_resource(show_spinner=False)
def load_model(model_dir: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.to(device).eval()
    id2label = mdl.config.id2label
    id2label = {int(k): v for k, v in id2label.items()} if isinstance(id2label, dict) else {i: lab for i, lab in enumerate(id2label)}
    return mdl, tok, id2label

# Pick latest timestamped folder in model_artifacts/
candidates = sorted(glob.glob("model_artifacts/sentiment_*"))
if not candidates:
    st.error("No model folders found in `model_artifacts/` (expected e.g. model_artifacts/sentiment_YYYYMMDD_HHMM/).")
    st.stop()
MODEL_DIR = candidates[-1]

with st.spinner(f"Loading model: {os.path.basename(MODEL_DIR)}"):
    model, tokenizer, id2label = load_model(MODEL_DIR, device)

MAX_LEN = 160

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

# ---------------- KPI Cards ----------------
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

st.write("")  # spacing

# ---------------- Row 1: Donut + Word Cloud ----------------
r1c1, r1c2 = st.columns([1.1, 1])

# Donut chart (larger labels, consistent colors)
with r1c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Sentiment Overview")
    dist = df["pred_label"].value_counts(normalize=True).mul(100).round(2).reset_index()
    dist.columns = ["label", "percent"]
    order = ["positive", "negative", "neutral", "suggestion"]
    dist["label"] = pd.Categorical(dist["label"], categories=order, ordered=True)
    dist = dist.sort_values("label")

    fig_donut = px.pie(
        dist, names="label", values="percent", hole=0.50, template="plotly_white",
        color="label",
        color_discrete_map={
            "positive": "#10B981",   # green
            "negative": "#EF4444",   # red
            "neutral":  "#3B82F6",   # blue
            "suggestion": "#F59E0B"  # amber
        }
    )
    fig_donut.update_traces(textinfo="label+percent", textfont_size=16,
                            marker=dict(line=dict(color="white", width=2)))
    fig_donut.update_layout(margin=dict(t=0,l=0,r=0,b=0), legend_title=None, font=dict(size=16))
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Word cloud — top terms only, compact size, stopword cleanup
def top_terms_wordcloud(text_series: pd.Series, max_words=80, width=700, height=420):
    # tokenize & clean
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", " ".join(text_series.astype(str).tolist()).lower())
    stop = {
        "comment","comments","id","amp","https","http","neutral","positive","negative","suggestion",
        "stakeholder","please","kindly","also","like","would","could","should","may","might","need","needs",
        "draft","legislation","provision","proposal","amendment","change","changes"
    }
    tokens = [t for t in tokens if t not in stop]
    freq = Counter(tokens).most_common(max_words)
    wc = WordCloud(
        width=width, height=height, background_color="white",
        max_words=max_words, collocations=False, prefer_horizontal=0.95, relative_scaling=0.3
    ).generate_from_frequencies(dict(freq))
    return wc.to_image()

with r1c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Keyword Cloud")
    wc_img = top_terms_wordcloud(df["comment_text"], max_words=80, width=800, height=380)
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
        pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns])  # column order
        hm = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Blues",
            colorbar=dict(title="Count")
        ))
        hm.update_layout(template="plotly_white", margin=dict(t=8,l=0,r=0,b=0), font=dict(size=15))
        st.plotly_chart(hm, use_container_width=True)
    else:
        st.info("No `section_title` column in CSV.")
    st.markdown('</div>', unsafe_allow_html=True)

with r2c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Trends Over Time")
    if "created_at" in df.columns and df["created_at"].notna().any():
        ts = df.dropna(subset=["created_at"]).copy()
        ts["date"] = ts["created_at"].dt.to_period("M").dt.to_timestamp()  # monthly
        trend = ts.groupby(["date","pred_label"]).size().reset_index(name="count")
        fig_line = px.line(trend, x="date", y="count", color="pred_label", template="plotly_white",
                           color_discrete_map={
                               "positive": "#10B981", "negative": "#EF4444",
                               "neutral": "#3B82F6", "suggestion": "#F59E0B"
                           })
        fig_line.update_layout(margin=dict(t=8,l=0,r=0,b=0), legend_title=None, font=dict(size=15))
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
        stake, x="stakeholder_type", y="count", color="pred_label", barmode="group",
        template="plotly_white",
        color_discrete_map={
            "positive": "#10B981", "negative": "#EF4444",
            "neutral": "#3B82F6", "suggestion": "#F59E0B"
        }
    )
    fig_bar.update_layout(margin=dict(t=6,l=0,r=0,b=0), legend_title=None, font=dict(size=15),
                          xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No `stakeholder_type` column in CSV.")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------------- Sample Comments (table) ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### Sample Comments")
cols = [c for c in ["created_at","section_title","stakeholder_type","pred_label","pred_confidence","comment_text"] if c in df.columns]
st.dataframe(df[cols].head(60), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
