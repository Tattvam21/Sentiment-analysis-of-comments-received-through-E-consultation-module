import os, glob
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image

# ---------- Page ----------
st.set_page_config(page_title="MCA Sentiment Dashboard", layout="wide")
st.title("MCA e-Consultation Sentiment Dashboard")

# ---------- Choose device (helps if MPS is flaky) ----------
default_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
device_str = st.sidebar.selectbox("Device", [default_device, "cpu"])
device = torch.device(device_str)

# ---------- Pick latest model folder safely ----------
candidates = sorted(glob.glob("model_artifacts/sentiment_*"))
if not candidates:
    st.error("No model folders found in `model_artifacts/`. Expected e.g. model_artifacts/sentiment_YYYYMMDD_HHMM/")
    st.stop()
MODEL_DIR = candidates[-1]
st.sidebar.write(f"Model: `{os.path.basename(MODEL_DIR)}`")

# ---------- Cached model loader (prevents blank page) ----------
@st.cache_resource(show_spinner=False)
def load_model(model_dir: str, device: torch.device):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.to(device).eval()
    # Normalize id2label shape
    id2label = mdl.config.id2label
    if isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}
    else:
        id2label = {i: lab for i, lab in enumerate(id2label)}
    return mdl, tok, id2label

with st.spinner("Loading model…"):
    try:
        model, tokenizer, id2label = load_model(MODEL_DIR, device)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

MAX_LEN = 160

# ---------- Prediction helper ----------
def predict_batch(texts, batch=128):
    out_labels, out_conf = [], []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tokenizer(
            chunk, truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        idx = probs.argmax(axis=1)
        out_labels.extend([id2label[int(j)] for j in idx])
        out_conf.extend(probs.max(axis=1))
    return out_labels, out_conf

# ---------- UI: CSV Upload ----------
uploaded = st.file_uploader("Upload a CSV with a 'comment_text' column (optional: created_at, section_title, stakeholder_type)", type=["csv"])

if not uploaded:
    st.info("Upload a CSV to see predictions and dashboard.")
    st.stop()

# ---------- Load and validate CSV ----------
df = pd.read_csv(uploaded)
if "comment_text" not in df.columns:
    st.error("CSV must contain a 'comment_text' column.")
    st.stop()

df["comment_text"] = df["comment_text"].astype(str)
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# ---------- Run predictions ----------
with st.spinner("Scoring comments…"):
    df["pred_label"], df["pred_confidence"] = predict_batch(df["comment_text"].tolist())

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Comments", len(df))
c2.metric("% Positive", round((df["pred_label"]=="positive").mean()*100, 2))
c3.metric("% Negative", round((df["pred_label"]=="negative").mean()*100, 2))
c4.metric("% Suggestion", round((df["pred_label"]=="suggestion").mean()*100, 2))

# ---------- Donut: Sentiment Overview ----------
dist = df["pred_label"].value_counts(normalize=True).mul(100).round(2).reset_index()
dist.columns = ["label", "percent"]
st.subheader("Overall Sentiment")
st.plotly_chart(px.pie(dist, names="label", values="percent", hole=.45), use_container_width=True)

# ---------- Trends Over Time ----------
if "created_at" in df.columns and df["created_at"].notna().any():
    ts = df.dropna(subset=["created_at"]).copy()
    ts["date"] = ts["created_at"].dt.date
    trend = ts.groupby(["date", "pred_label"]).size().reset_index(name="count")
    st.subheader("Trends Over Time")
    st.plotly_chart(px.line(trend, x="date", y="count", color="pred_label"), use_container_width=True)

# ---------- Section-wise Sentiment ----------
if "section_title" in df.columns:
    pivot = df.pivot_table(index="section_title", columns="pred_label",
                           values=df.columns[0], aggfunc="count", fill_value=0)
    st.subheader("Section-wise Sentiment")
    st.dataframe(pivot.style.background_gradient(cmap="Blues"), use_container_width=True)

# ---------- Stakeholder Comparison ----------
if "stakeholder_type" in df.columns:
    stake = df.groupby(["stakeholder_type","pred_label"]).size().reset_index(name="count")
    st.subheader("Stakeholder Comparison")
    st.plotly_chart(px.bar(stake, x="stakeholder_type", y="count", color="pred_label", barmode="group"),
                    use_container_width=True)

# ---------- Word Cloud ----------
st.subheader("Keyword Cloud")
text_blob = " ".join(df["comment_text"].tolist())
wc = WordCloud(width=1100, height=600, background_color="white").generate(text_blob)
buf = BytesIO(); wc.to_image().save(buf, format="PNG")
st.image(Image.open(BytesIO(buf.getvalue())), use_container_width=True)

# ---------- Sample Comments ----------
st.subheader("Sample Comments")
show_cols = [c for c in ["created_at","section_title","stakeholder_type","pred_label","pred_confidence","comment_text"] if c in df.columns]
st.dataframe(df[show_cols].head(50), use_container_width=True)
