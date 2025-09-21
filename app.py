import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------- Settings --------
MODEL_DIR = "sentiment_model_mps"   # <- path to your saved folder
MAX_LEN = 160

# -------- Device (MPS if available) --------
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_dir: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    # labels saved in config during training
    id2label = model.config.id2label
    # ensure int keys sorted
    id2label = {int(k): v for k, v in id2label.items()}
    return model, tokenizer, id2label

model, tokenizer, id2label = load_model_and_tokenizer(MODEL_DIR)

st.set_page_config(page_title="SIH Sentiment Analysis", layout="centered")
st.title("💬 Sentiment Analysis of Comments")
st.caption(f"Device: **{DEVICE}** | Labels: {list(id2label.values())}")

# ---------- Single text inference ----------
st.header("Single Comment")
text = st.text_area("Type or paste a comment:", height=140, placeholder="e.g., This amendment is confusing and adds burden to SMEs.")
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter a comment.")
    else:
        with torch.no_grad():
            enc = tokenizer(text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        lid = int(np.argmax(probs))
        st.subheader(f"Prediction: **{id2label[lid]}**")
        st.write("Confidence:")
        for i, p in enumerate(probs):
            st.write(f"- {id2label[i]}: {p:.3f}")

st.markdown("---")

# ---------- Batch CSV inference ----------
st.header("Batch (CSV Upload)")
st.write("Upload a CSV with a column named **comment_text**.")
uploaded = st.file_uploader("Choose CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    if "comment_text" not in df.columns:
        st.error("CSV must contain a 'comment_text' column.")
    else:
        st.write("Preview:", df.head())
        if st.button("Run batch predictions"):
            texts = df["comment_text"].astype(str).tolist()
            preds, confs = [], []

            # Simple batching to avoid memory spikes
            BATCH = 64
            for i in range(0, len(texts), BATCH):
                chunk = texts[i:i+BATCH]
                with torch.no_grad():
                    enc = tokenizer(chunk, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
                    enc = {k: v.to(DEVICE) for k, v in enc.items()}
                    logits = model(**enc).logits
                    prob = torch.softmax(logits, dim=-1).cpu().numpy()
                    label_ids = np.argmax(prob, axis=1)
                    preds.extend([id2label[int(x)] for x in label_ids])
                    confs.extend(prob.max(axis=1))

            out = df.copy()
            out["pred_label"] = preds
            out["pred_confidence"] = confs

            st.success("Done! Showing first rows:")
            st.dataframe(out.head(20))

            # Download button
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions CSV",
                data=csv_bytes,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )
