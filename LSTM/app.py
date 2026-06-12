import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="LSTM Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

st.title("😊 LSTM Sentiment Analysis Dashboard")

st.markdown("""
This application uses an **LSTM (Long Short-Term Memory)** network
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

# Small subset for fast training

X_train = X_train[:2000]
y_train = y_train[:2000]

X_test = X_test[:500]
y_test = y_test[:500]

# Padding

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
st.sidebar.write("Vocabulary Size: 5000")

st.sidebar.subheader("🧠 LSTM Architecture")

st.sidebar.code("""
Input Review

↓ Embedding(5000,32)

↓ LSTM(32)

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
            input_dim=VOCAB_SIZE,
            output_dim=32,
            input_length=MAX_LEN
        ),

        LSTM(32),

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


with st.spinner("Training LSTM Model..."):

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
# DATASET TAB
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

    st.write(
        X_train[0][:50]
    )

    st.subheader("Sample Label")

    st.write(
        "Positive"
        if y_train[0] == 1
        else "Negative"
    )

# =====================================================
# PERFORMANCE TAB
# =====================================================

with tab2:

    st.header("Model Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

    col2.metric(
        "Loss",
        f"{loss:.4f}"
    )

    # --------------------------------

    st.subheader("Training Accuracy")

    fig1, ax1 = plt.subplots(
        figsize=(8,4)
    )

    ax1.plot(
        history.history["accuracy"],
        marker="o",
        linewidth=2,
        label="Training"
    )

    ax1.plot(
        history.history["val_accuracy"],
        marker="o",
        linewidth=2,
        label="Validation"
    )

    ax1.set_title(
        "Accuracy vs Epoch"
    )

    ax1.set_xlabel(
        "Epoch"
    )

    ax1.set_ylabel(
        "Accuracy"
    )

    ax1.legend()

    st.pyplot(fig1)

    # --------------------------------

    st.subheader("Training Loss")

    fig2, ax2 = plt.subplots(
        figsize=(8,4)
    )

    ax2.plot(
        history.history["loss"],
        marker="o",
        linewidth=2,
        label="Training"
    )

    ax2.plot(
        history.history["val_loss"],
        marker="o",
        linewidth=2,
        label="Validation"
    )

    ax2.set_title(
        "Loss vs Epoch"
    )

    ax2.set_xlabel(
        "Epoch"
    )

    ax2.set_ylabel(
        "Loss"
    )

    ax2.legend()

    st.pyplot(fig2)

    # --------------------------------

    st.subheader("History Values")

    st.dataframe(
        {
            "Epoch":
            list(
                range(
                    1,
                    len(
                        history.history["accuracy"]
                    ) + 1
                )
            ),

            "Train Accuracy":
            history.history["accuracy"],

            "Validation Accuracy":
            history.history["val_accuracy"],

            "Train Loss":
            history.history["loss"],

            "Validation Loss":
            history.history["val_loss"]
        }
    )

# =====================================================
# PREDICTION TAB
# =====================================================

with tab3:

    st.header("Predict Review Sentiment")

    review_index = st.slider(
        "Choose Review Index",
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
        if y_test[
            review_index
        ] == 1
        else "Negative"
    )

    st.info(
        f"Actual Sentiment: {actual}"
    )

    if st.button(
        "Predict Sentiment"
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
    "🚀 Built using TensorFlow, LSTM and Streamlit"
)