import streamlit as st, pandas as pd, numpy as np, torch, glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="MCA Sentiment Dashboard", layout="wide")

# Load latest model
latest_model = sorted(glob.glob("model_artifacts/sentiment_*"))[-1]
device = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
model = AutoModelForSequenceClassification.from_pretrained(latest_model).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(latest_model)
id2label = {int(k): v for k,v in model.config.id2label.items()}
MAX_LEN = 160

st.title("MCA e-Consultation Sentiment Dashboard")
uploaded = st.file_uploader("Upload CSV with a 'comment_text' column", type=["csv"])

def predict_batch(texts, batch=128):
    out_labels, out_conf = [], []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        enc = tokenizer(chunk, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        idx = probs.argmax(axis=1)
        out_labels.extend([id2label[int(j)] for j in idx])
        out_conf.extend(probs.max(axis=1))
    return out_labels, out_conf

if uploaded:
    df = pd.read_csv(uploaded)
    if "comment_text" not in df.columns:
        st.error("CSV must contain a 'comment_text' column.")
        st.stop()

    df["comment_text"] = df["comment_text"].astype(str)
    df["pred_label"], df["pred_confidence"] = predict_batch(df["comment_text"].tolist())

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Comments", len(df))
    c2.metric("% Positive", round((df["pred_label"]=="positive").mean()*100,2))
    c3.metric("% Negative", round((df["pred_label"]=="negative").mean()*100,2))
    c4.metric("% Suggestion", round((df["pred_label"]=="suggestion").mean()*100,2))

    # Charts
    dist = df["pred_label"].value_counts(normalize=True).mul(100).round(2).reset_index()
    dist.columns = ["label","percent"]
    st.subheader("Sentiment Overview")
    st.plotly_chart(px.pie(dist, names="label", values="percent", hole=.45), use_container_width=True)

    if "created_at" in df.columns:
        ts = df.dropna(subset=["created_at"]).copy()
        if len(ts):
            ts["date"] = pd.to_datetime(ts["created_at"], errors="coerce").dt.date
            trend = ts.groupby(["date","pred_label"]).size().reset_index(name="count")
            st.subheader("Trend Over Time")
            st.plotly_chart(px.line(trend, x="date", y="count", color="pred_label"), use_container_width=True)

    if "section_title" in df.columns:
        pivot = df.pivot_table(index="section_title", columns="pred_label", values=df.columns[0], aggfunc="count", fill_value=0)
        st.subheader("Section-wise Sentiment")
        st.dataframe(pivot.style.background_gradient(cmap="Blues"), use_container_width=True)

    # Word cloud (in-app)
    text_blob = " ".join(df["comment_text"].tolist())
    wc = WordCloud(width=1000, height=600, background_color="white").generate(text_blob)
    buf = BytesIO(); wc.to_image().save(buf, format="PNG")
    st.subheader("Keyword Cloud")
    st.image(Image.open(BytesIO(buf.getvalue())), use_container_width=True)

    st.subheader("Sample Comments")
    st.dataframe(df.head(50), use_container_width=True)
else:
    st.info("Upload a CSV to see predictions and dashboard.")
