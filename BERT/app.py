import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt

from transformers import pipeline

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 BERT Sentiment Analysis Dashboard")

st.markdown("""
This application uses a pre-trained BERT model
for sentiment analysis.
""")

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    classifier = pipeline(
        "sentiment-analysis"
    )

    return classifier

classifier = load_model()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("📚 Project Information")

st.sidebar.write("Model: BERT")
st.sidebar.write("Task: Sentiment Analysis")
st.sidebar.write("Framework: Transformers")

st.sidebar.subheader("🧠 Architecture")

st.sidebar.code("""
Input Text

↓ Tokenizer

↓ BERT Encoder

↓ Classification Head

↓ Sentiment Output
""")

# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs([
    "📊 Model Info",
    "🔍 Prediction"
])

# =====================================================
# MODEL INFO
# =====================================================

with tab1:

    st.header("About BERT")

    st.write("""
    BERT stands for:

    Bidirectional Encoder Representations
    from Transformers.

    It reads text in both directions
    and understands context better
    than traditional RNNs and LSTMs.
    """)

    st.subheader("Advantages")

    st.markdown("""
    - Bidirectional Context
    - State-of-the-Art NLP
    - Transfer Learning
    - High Accuracy
    """)

    # Example chart

    fig, ax = plt.subplots()

    models = [
        "RNN",
        "LSTM",
        "GRU",
        "BERT"
    ]

    scores = [
        75,
        82,
        84,
        92
    ]

    ax.bar(
        models,
        scores
    )

    ax.set_ylabel(
        "Approx Accuracy"
    )

    ax.set_title(
        "Model Comparison"
    )

    st.pyplot(fig)

# =====================================================
# PREDICTION
# =====================================================

with tab2:

    st.header("Analyze Sentiment")

    text = st.text_area(
        "Enter Text",
        height=150
    )

    if st.button(
        "Analyze"
    ):

        if text.strip() == "":

            st.warning(
                "Please enter text."
            )

        else:

            result = classifier(text)[0]

            label = result["label"]

            confidence = (
                result["score"]
                * 100
            )

            if label == "POSITIVE":

                st.success(
                    "😊 Positive Sentiment"
                )

            else:

                st.error(
                    "😔 Negative Sentiment"
                )

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

            df = pd.DataFrame({
                "Prediction":
                [label],

                "Confidence (%)":
                [confidence]
            })

            st.dataframe(
                df,
                use_container_width=True
            )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    "🚀 Built using BERT, Transformers and Streamlit"
)