import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

import matplotlib.pyplot as plt

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="ANN Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# ==================================================
# TITLE
# ==================================================

st.title("🌸 ANN Iris Flower Classification Dashboard")
st.markdown(
    """
    This project demonstrates an **Artificial Neural Network (ANN)** using
    Scikit-Learn's MLPClassifier on the Iris dataset.
    """
)

# ==================================================
# LOAD DATA
# ==================================================

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["Species"] = [iris.target_names[i] for i in y]

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("📚 Project Information")

st.sidebar.success("Dataset: Iris")

st.sidebar.write("Total Samples: 150")
st.sidebar.write("Features: 4")
st.sidebar.write("Classes: 3")

st.sidebar.markdown("---")

st.sidebar.subheader("🧠 ANN Architecture")

st.sidebar.code(
"""
Input Layer (4)

        ↓

Hidden Layer 1
16 Neurons (ReLU)

        ↓

Hidden Layer 2
8 Neurons (ReLU)

        ↓

Output Layer
3 Neurons
"""
)

# ==================================================
# TRAIN MODEL
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

predicted_classes = model.predict(X_test)
prediction_probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(
    y_test,
    predicted_classes
)

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dataset",
        "📈 Visualizations",
        "🧠 Model Performance",
        "🌸 Prediction"
    ]
)

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.header("Dataset Overview")

    st.dataframe(df)

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Statistical Summary")

    st.dataframe(df.describe())

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.header("Data Visualizations")

    # -------------------------
    # Class Distribution
    # -------------------------

    st.subheader("Class Distribution")

    counts = [
        np.sum(y == 0),
        np.sum(y == 1),
        np.sum(y == 2)
    ]

    fig1, ax1 = plt.subplots()

    ax1.bar(
        iris.target_names,
        counts
    )

    ax1.set_title(
        "Iris Flower Class Distribution"
    )

    ax1.set_ylabel("Count")

    st.pyplot(fig1)

    # -------------------------
    # Scatter Plot
    # -------------------------

    st.subheader("Petal Length vs Petal Width")

    fig2, ax2 = plt.subplots()

    scatter = ax2.scatter(
        X[:, 2],
        X[:, 3],
        c=y
    )

    ax2.set_xlabel("Petal Length")

    ax2.set_ylabel("Petal Width")

    ax2.set_title(
        "Petal Length vs Width"
    )

    st.pyplot(fig2)

    # -------------------------
    # Correlation Heatmap
    # -------------------------

    st.subheader("Feature Correlation Heatmap")

    corr = df.iloc[:, :-1].corr()

    fig3, ax3 = plt.subplots()

    heatmap = ax3.imshow(
        corr,
        aspect="auto"
    )

    ax3.set_xticks(
        range(len(corr.columns))
    )

    ax3.set_yticks(
        range(len(corr.columns))
    )

    ax3.set_xticklabels(
        corr.columns,
        rotation=90
    )

    ax3.set_yticklabels(
        corr.columns
    )

    plt.colorbar(heatmap)

    st.pyplot(fig3)

# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.header("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

    col2.metric(
        "Training Samples",
        len(X_train)
    )

    col3.metric(
        "Testing Samples",
        len(X_test)
    )

    st.markdown("---")

    # -------------------------
    # Training Loss Curve
    # -------------------------

    st.subheader("Training Loss Curve")

    fig4, ax4 = plt.subplots()

    ax4.plot(
        model.loss_curve_
    )

    ax4.set_title(
        "ANN Training Loss"
    )

    ax4.set_xlabel(
        "Iterations"
    )

    ax4.set_ylabel(
        "Loss"
    )

    st.pyplot(fig4)

    # -------------------------
    # Confusion Matrix
    # -------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predicted_classes
    )

    fig5, ax5 = plt.subplots()

    im = ax5.imshow(cm)

    ax5.set_xlabel("Predicted")

    ax5.set_ylabel("Actual")

    ax5.set_title(
        "Confusion Matrix"
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax5.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar(im)

    st.pyplot(fig5)

    # -------------------------
    # Classification Report
    # -------------------------

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        predicted_classes,
        target_names=iris.target_names
    )

    st.text(report)

# ==================================================
# TAB 4
# ==================================================

with tab4:

    st.header(
        "Predict Iris Flower Species"
    )

    col1, col2 = st.columns(2)

    with col1:

        sepal_length = st.slider(
            "Sepal Length (cm)",
            4.0,
            8.0,
            5.1
        )

        sepal_width = st.slider(
            "Sepal Width (cm)",
            2.0,
            5.0,
            3.5
        )

    with col2:

        petal_length = st.slider(
            "Petal Length (cm)",
            1.0,
            7.0,
            1.4
        )

        petal_width = st.slider(
            "Petal Width (cm)",
            0.1,
            3.0,
            0.2
        )

    if st.button("🔍 Predict Species"):

        sample = np.array([
            [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]
        ])

        sample = scaler.transform(
            sample
        )

        prediction = model.predict(
            sample
        )

        probabilities = model.predict_proba(
            sample
        )

        predicted_class = prediction[0]

        confidence = (
            np.max(probabilities)
            * 100
        )

        st.success(
            f"Predicted Species: {iris.target_names[predicted_class].capitalize()}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        st.subheader(
            "Prediction Probabilities"
        )

        prob_df = pd.DataFrame(
            {
                "Species": iris.target_names,
                "Probability (%)":
                probabilities[0] * 100
            }
        )

        st.dataframe(
            prob_df,
            use_container_width=True
        )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    🚀 Built with Streamlit, Scikit-Learn, NumPy, Pandas and Matplotlib.
    """
)