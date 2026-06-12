import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    GRU,
    Dense
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="GRU Sentiment Analysis",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GRU Sentiment Analysis Dashboard")

st.markdown("""
This application uses a **GRU (Gated Recurrent Unit)** network
to classify IMDB movie reviews as Positive or Negative.
""")

# =====================================================
# PARAMETERS
# =====================================================

VOCAB_SIZE = 5000
MAX_LEN = 200

# =====================================================
# LOAD DATA
# =====================================================

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

X_train = X_train[:2000]
y_train = y_train[:2000]

X_test = X_test[:500]
y_test = y_test[:500]

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LEN
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LEN
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("📚 Project Information")

st.sidebar.write("Dataset: IMDB Reviews")
st.sidebar.write("Training Samples: 2000")
st.sidebar.write("Testing Samples: 500")

st.sidebar.subheader("🧠 GRU Architecture")

st.sidebar.code("""
Input Review

↓ Embedding(5000,32)

↓ GRU(32)

↓ Dense(1)

↓ Sigmoid
""")

# =====================================================
# MODEL
# =====================================================

@st.cache_resource
def train_model():

    model = Sequential([

        Embedding(
            VOCAB_SIZE,
            32,
            input_length=MAX_LEN
        ),

        GRU(32),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2,
        verbose=0
    )

    return model, history


with st.spinner("Training GRU Model..."):

    model, history = train_model()

# =====================================================
# EVALUATION
# =====================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Dataset",
    "📈 Performance",
    "🔍 Prediction"
])

# =====================================================
# DATASET
# =====================================================

with tab1:

    st.header("Dataset Overview")

    st.write(
        f"Training Samples: {len(X_train)}"
    )

    st.write(
        f"Testing Samples: {len(X_test)}"
    )

    st.subheader("Sample Encoded Review")

    st.write(X_train[0][:50])

# =====================================================
# PERFORMANCE
# =====================================================

with tab2:

    st.header("GRU Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

    col2.metric(
        "Loss",
        f"{loss:.4f}"
    )

    st.subheader("Training Accuracy")

    fig1, ax1 = plt.subplots()

    ax1.plot(
        history.history["accuracy"],
        marker="o",
        label="Training"
    )

    ax1.plot(
        history.history["val_accuracy"],
        marker="o",
        label="Validation"
    )

    ax1.legend()

    st.pyplot(fig1)

    st.subheader("Training Loss")

    fig2, ax2 = plt.subplots()

    ax2.plot(
        history.history["loss"],
        marker="o",
        label="Training"
    )

    ax2.plot(
        history.history["val_loss"],
        marker="o",
        label="Validation"
    )

    ax2.legend()

    st.pyplot(fig2)

# =====================================================
# PREDICTION
# =====================================================

with tab3:

    st.header("Predict Review Sentiment")

    review_index = st.slider(
        "Choose Review",
        0,
        len(X_test)-1,
        0
    )

    sample = X_test[
        review_index
    ].reshape(
        1,
        MAX_LEN
    )

    actual = (
        "Positive"
        if y_test[review_index] == 1
        else "Negative"
    )

    st.info(
        f"Actual Sentiment: {actual}"
    )

    if st.button(
        "Predict"
    ):

        pred = model.predict(
            sample,
            verbose=0
        )[0][0]

        sentiment = (
            "Positive"
            if pred > 0.5
            else "Negative"
        )

        confidence = (
            pred * 100
            if pred > 0.5
            else (1-pred)*100
        )

        st.success(
            f"Prediction: {sentiment}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.markdown(
    "🚀 Built using TensorFlow, GRU and Streamlit"
)