import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Bộ lọc Spam Email", page_icon="📧")
st.title("📧 Bộ lọc Spam Email")

texts = [
    "Win a free iPhone now!!! Click here",
    "Limited offer! Cheap Viagra online",
    "You have been selected to receive a prize, click",
    "Meeting schedule for next week",
    "Your invoice for October",
    "Project update attached",
]
labels = [1, 1, 1, 0, 0, 0]

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("nb", MultinomialNB())
])
model.fit(texts, labels)

subject = st.text_input("Tiêu đề Email")
body = st.text_area("Nội dung Email", height=180)

if st.button("Kiểm tra Spam"):
    text = (subject + " " + body).strip()
    if not text:
        st.warning("Vui lòng nhập nội dung hoặc tiêu đề email.")
    else:
        score = float(model.predict_proba([text])[0, 1])
        label = "🚨 SPAM" if score >= 0.5 else "✅ Không phải SPAM"
        st.success(f"Kết quả: {label}")
        st.write(f"Độ tin cậy: {score:.2f}")

