from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


DATA_FILE = "email_dataset_100k.csv"


@st.cache_resource
def train_model(data_path: Path):
    df = pd.read_csv(data_path)

    if "raw_text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'raw_text' and 'label' columns.")

    df = df.dropna(subset=["raw_text", "label"])

    vectorizer = CountVectorizer()
    x_all = vectorizer.fit_transform(df["raw_text"].astype(str))
    y_all = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, vectorizer, accuracy


def predict_email(text: str, model: MultinomialNB, vectorizer: CountVectorizer):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label = "SPAM" if int(prediction) == 1 else "SAFE"
    confidence = max(probabilities)
    return label, confidence


def main():
    st.set_page_config(page_title="Email Spam Detector", layout="centered")
    st.title("Email Spam Detector")
    st.caption("Trained on email text data using CountVectorizer + MultinomialNB")

    data_path = Path(__file__).resolve().parent / DATA_FILE

    if not data_path.exists():
        st.error(f"Dataset not found: {data_path}")
        st.stop()

    try:
        model, vectorizer, accuracy = train_model(data_path)
    except Exception as exc:
        st.error(f"Could not train model: {exc}")
        st.stop()

   

    user_input = st.text_area(
        "Paste an email/message to classify",
        placeholder="Example: Winner! Claim your free gift card now!",
        height=150,
    )

    if st.button("Check Message"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            label, confidence = predict_email(user_input.strip(), model, vectorizer)

            if label == "SPAM":
                st.error(f"Result: {label}")
            else:
                st.success(f"Result: {label}")

            st.write(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
