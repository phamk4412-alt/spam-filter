import streamlit as st, joblib
from pathlib import Path

st.set_page_config(page_title="Bộ lọc Spam (MLP)", page_icon="🧠")
st.title("🧠 Bộ lọc Email Spam – MLP (nơ-ron nhẹ)")

MODEL_PATH = Path("model.pkl")
if not MODEL_PATH.exists():
    st.error("Không thấy model.pkl trong repo. Hãy upload model.pkl rồi reload ứng dụng.")
    st.stop()

model = joblib.load(MODEL_PATH)

col1, col2 = st.columns([3,1])
with col1:
    subject = st.text_input("Tiêu đề Email")
body = st.text_area("Nội dung Email", height=200, placeholder="Dán nội dung email tiếng Việt...")

threshold = st.slider("Ngưỡng phân loại (mặc định 0.5)", 0.1, 0.9, 0.5, 0.05)

if st.button("Kiểm tra Spam"):
    text = (subject + " " + body).strip()
    if not text:
        st.warning("Vui lòng nhập tiêu đề hoặc nội dung email.")
    else:
        proba = float(model.predict_proba([text])[0, 1])
        label = "🚨 SPAM" if proba >= threshold else "✅ Không phải SPAM"
        st.markdown(f"**Kết quả:** {label}")
        st.caption(f"Xác suất spam: {proba:.3f} • Ngưỡng: {threshold:.2f}")
