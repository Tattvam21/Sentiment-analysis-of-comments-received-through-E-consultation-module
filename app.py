import os, glob, re
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
from PIL import Image
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="MCA Sentiment Dashboard", layout="wide")

# ---- Minimal modern styling (cards, font, shadows)
st.markdown("""
<style>
:root { --card-bg:#ffffff; --text:#0f172a; --muted:#475569; --accent:#2563eb; }
html, body, [class*="css"]  { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Segoe UI Emoji"; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
.card { background:var(--card-bg); border-radius:16px; padding:18px 18px; box-shadow:0 2px 12px rgba(15,23,42,.06); border:1px solid rgba(2,6,23,.06); }
.kpi { font-size:34px; font-weight:700; margin-top:6px; }
.kpi-label { color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.06em; }
h1,h2,h3 { color:var(--text); }
.small { color:var(--muted); font-size:13px;}
hr { border: none; border-top:1px solid rgba(2,6,23,.06); margin: 8px 0 18px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("### Sentiment Analysis")
st.markdown('<div class="small">MCA e-Consultation – AI insights on stakeholder comments</div>', unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ---------------- DEVICE ----------------
default_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(default_device)

# ---------------- MODEL PICK ----------------
# Expect timestamped folder inside model_artifacts/
candidates = sorted(glob.glob("model_artifacts/sentiment_*"))
if not candidates:
    st.error("No model folders found in `model_artifacts/`. Expected e.g. model_artifacts/sentiment_YYYYMMDD_HHMM/")
    st.stop()
MODEL_DIR = candidates[-1]

@st.cache_resource(show_spinner=False)
def load_model(model_dir: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.to(device).eval()
    id2label = mdl.config.id2label
    id2label = {int(k): v for k, v in id2label.items()} if isinstance(id2label, dict) else {i: lab for i, lab in enumerate(id2label)}
    return mdl, tok, id2label

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

# ---------------- UPLOAD ----------------
st.markdown("##### Upload CSV")
uploaded = st.file_uploader("Upload a CSV with a `comment_text` column (optional: `created_at`, `section_title`, `stakeholder_type`)", type=["csv"])
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

with st.spinner("Scoring comments…"):
    df["pred_label"], df["pred_confidence"] = predict_batch(df["comment_text"].tolist())

# ---------------- KPI CARDS ----------------
total = int(len(df))
pct_pos = round((df["pred_label"] == "positive").mean()*100, 2)
pct_neg = round((df["pred_label"] == "negative").mean()*100, 2)
pct_sug = round((df["pred_label"] == "suggestion").mean()*100, 2)

c1, c2, c3, c4 = st.columns(4)
for c, label, value in [
    (c1, "Total Comments", f"{total:,}"),
    (c2, "% Positive", f"{pct_pos} %"),
    (c3, "% Negative", f"{pct_neg} %"),
    (c4, "% Suggestions", f"{pct_sug} %"),
]:
    with c:
        st.markdown(f'<div class="card"><div class="kpi-label">{label}</div><div class="kpi">{value}</div></div>', unsafe_allow_html=True)

st.write("")  # spacing

# ---------------- ROW 1: DONUT + WORD CLOUD ----------------
r1c1, r1c2 = st.columns([1.1, 1])

# Donut
with r1c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Sentiment Overview")
    dist = df["pred_label"].value_counts(normalize=True).mul(100).round(2).reset_index()
    dist.columns = ["label", "percent"]
    order = ["positive", "negative", "neutral", "suggestion"]
    dist["label"] = pd.Categorical(dist["label"], categories=order, ordered=True)
    dist = dist.sort_values("label")
    fig_donut = px.pie(
        dist, names="label", values="percent", hole=.55,
        template="plotly_white"
    )
    fig_donut.update_traces(textinfo="percent", textfont_size=14,
                            marker=dict(line=dict(color="white", width=2)))
    fig_donut.update_layout(margin=dict(t=0,l=0,r=0,b=0), legend_title=None)
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Word Cloud (top-N, tidy, responsive)
def build_wordcloud(series: pd.Series, max_words=60, width=900, height=360):
    # Use CountVectorizer for clean frequencies; remove numbers & small tokens
    stop = set(STOPWORDS) | {"comment","comments","stakeholder","id","amp","https","http"}
    vec = CountVectorizer(
        stop_words="english",
        max_features=5000,
        token_pattern=r"(?u)\\b[a-zA-Z][a-zA-Z\\-]{2,}\\b",  # >=3 letters
        ngram_range=(1,2)
    )
    try:
        X = vec.fit_transform(series.fillna(""))
        freqs = np.asarray(X.sum(axis=0)).ravel()
        vocab = np.array(vec.get_feature_names_out())
        # keep top-N
        idx = np.argsort(freqs)[::-1][:max_words]
        freq_dict = {vocab[i]: int(freqs[i]) for i in idx}
    except ValueError:
        freq_dict = {}
    wc = WordCloud(width=width, height=height, background_color="white",
                   stopwords=stop, max_words=max_words, collocations=False,
                   prefer_horizontal=0.95, relative_scaling=0.3)
    return wc.generate_from_frequencies(freq_dict)

with r1c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Keyword Cloud")
    wc_img = build_wordcloud(df["comment_text"], max_words=60, width=1000, height=360).to_image()
    st.image(wc_img, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ROW 2: SECTION HEATMAP + TRENDS + STAKEHOLDER ----------------
r2c1, r2c2 = st.columns([1.2, 1])

with r2c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Section-wise Sentiment")
    if "section_title" in df.columns:
        pivot = df.pivot_table(index="section_title", columns="pred_label",
                               values=df.columns[0], aggfunc="count", fill_value=0)
        # order columns like donut
        pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns])
        hm = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns, y=pivot.index, colorscale="Blues",
            colorbar=dict(title="Count")
        ))
        hm.update_layout(template="plotly_white", margin=dict(t=0,l=0,r=0,b=0))
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
        fig_line = px.line(trend, x="date", y="count", color="pred_label", template="plotly_white")
        fig_line.update_layout(margin=dict(t=0,l=0,r=0,b=0), legend_title=None)
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No time information to plot.")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")  # spacing

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### Stakeholder Comparison")
if "stakeholder_type" in df.columns:
    stake = df.groupby(["stakeholder_type","pred_label"]).size().reset_index(name="count")
    fig_bar = px.bar(stake, x="stakeholder_type", y="count", color="pred_label",
                     barmode="group", template="plotly_white")
    fig_bar.update_layout(margin=dict(t=0,l=0,r=0,b=0), legend_title=None,
                          xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No `stakeholder_type` column in CSV.")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### Sample Comments")
show_cols = [c for c in ["created_at","section_title","stakeholder_type","pred_label","pred_confidence","comment_text"] if c in df.columns]
st.dataframe(df[show_cols].head(50), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
