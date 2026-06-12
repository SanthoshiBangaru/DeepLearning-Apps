import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense
)
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="CNN Fashion-MNIST Classifier",
    page_icon="👕",
    layout="wide"
)

st.title("👕 CNN Fashion-MNIST Dashboard")

st.markdown("""
### Convolutional Neural Network (CNN)

This project classifies Fashion-MNIST images using a CNN.

Classes:
- T-shirt
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot
""")

# =====================================================
# CLASS NAMES
# =====================================================

class_names = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# =====================================================
# LOAD DATA
# =====================================================

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Small subset for fast training

X_train = X_train[:2000]
y_train = y_train[:2000]

X_test = X_test[:500]
y_test = y_test[:500]

# Normalize

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Reshape

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One Hot Encoding

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("📚 CNN Information")

st.sidebar.write("Dataset: Fashion-MNIST")
st.sidebar.write("Training Samples: 2,000")
st.sidebar.write("Testing Samples: 500")
st.sidebar.write("Classes: 10")

st.sidebar.subheader("Architecture")

st.sidebar.code("""
Input (28x28x1)

↓ Conv2D(8)

↓ MaxPooling

↓ Flatten

↓ Dense(32)

↓ Dense(10)
""")

# =====================================================
# MODEL
# =====================================================

@st.cache_resource
def train_model():

    model = Sequential([

        Conv2D(
            8,
            (3, 3),
            activation="relu",
            input_shape=(28, 28, 1)
        ),

        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            10,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train_cat,
        epochs=1,
        batch_size=64,
        validation_split=0.2,
        verbose=0
    )

    return model, history


with st.spinner("Training CNN..."):

    model, history = train_model()

# =====================================================
# EVALUATION
# =====================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

predictions = model.predict(
    X_test,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset",
    "📈 Visualizations",
    "🧠 Performance",
    "🔍 Prediction"
])

# =====================================================
# DATASET TAB
# =====================================================

with tab1:

    st.header("Dataset Overview")

    st.write(
        f"Training Images: {len(X_train)}"
    )

    st.write(
        f"Testing Images: {len(X_test)}"
    )

    st.subheader("Sample Images")

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(10, 5)
    )

    for i, ax in enumerate(axes.flat):

        ax.imshow(
            X_train[i].reshape(28, 28),
            cmap="gray"
        )

        ax.set_title(
            class_names[y_train[i]],
            fontsize=8
        )

        ax.axis("off")

    st.pyplot(fig)

# =====================================================
# VISUALIZATION TAB
# =====================================================

with tab2:

    st.header("Visualizations")

    st.subheader("Class Distribution")

    counts = np.bincount(y_train)

    fig2, ax2 = plt.subplots()

    ax2.bar(
        class_names,
        counts
    )

    plt.xticks(rotation=45)

    st.pyplot(fig2)

    st.subheader("Random Fashion Item")

    idx = np.random.randint(
        0,
        len(X_test)
    )

    fig3, ax3 = plt.subplots()

    ax3.imshow(
        X_test[idx].reshape(28, 28),
        cmap="gray"
    )

    ax3.set_title(
        class_names[y_test[idx]]
    )

    ax3.axis("off")

    st.pyplot(fig3)

# =====================================================
# PERFORMANCE TAB
# =====================================================

with tab3:

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

    st.subheader(
        "Training Accuracy"
    )

    fig4, ax4 = plt.subplots()

    ax4.plot(
        history.history["accuracy"],
        marker="o"
    )

    ax4.set_ylabel(
        "Accuracy"
    )

    ax4.set_xlabel(
        "Epoch"
    )

    st.pyplot(fig4)

    st.subheader(
        "Training Loss"
    )

    fig5, ax5 = plt.subplots()

    ax5.plot(
        history.history["loss"],
        marker="o"
    )

    ax5.set_ylabel(
        "Loss"
    )

    ax5.set_xlabel(
        "Epoch"
    )

    st.pyplot(fig5)

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        predicted_classes
    )

    fig6, ax6 = plt.subplots(
        figsize=(8, 8)
    )

    im = ax6.imshow(cm)

    plt.colorbar(im)

    st.pyplot(fig6)

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        predicted_classes
    )

    st.text(report)

# =====================================================
# PREDICTION TAB
# =====================================================

with tab4:

    st.header(
        "Predict Fashion Item"
    )

    sample_idx = st.slider(
        "Select Image",
        0,
        len(X_test)-1,
        0
    )

    image = X_test[sample_idx]

    st.image(
        image,
        width=250
    )

    if st.button(
        "Predict"
    ):

        pred = model.predict(
            image.reshape(
                1,
                28,
                28,
                1
            ),
            verbose=0
        )

        predicted_class = np.argmax(
            pred
        )

        confidence = (
            np.max(pred)
            * 100
        )

        st.success(
            f"Prediction: {class_names[predicted_class]}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        prob_df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)":
            pred[0] * 100
        })

        st.dataframe(
            prob_df,
            use_container_width=True
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    "🚀 CNN using Fashion-MNIST, TensorFlow and Streamlit"
)