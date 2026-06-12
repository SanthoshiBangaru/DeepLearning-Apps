import streamlit as st
import pandas as pd

from transformers import pipeline

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="GPT Text Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GPT Text Generation Dashboard")

st.markdown("""
This application uses a pre-trained GPT model
to generate text from a user prompt.
""")

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    generator = pipeline(
    "text-generation",
    model="gpt2"
    )

    return generator

generator = load_model()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("📚 Project Information")

st.sidebar.write("Model: GPT")
st.sidebar.write("Variant: DistilGPT2")
st.sidebar.write("Task: Text Generation")

st.sidebar.subheader("🧠 GPT Architecture")

st.sidebar.code("""
Input Prompt

↓ Tokenization

↓ GPT Decoder

↓ Next Token Prediction

↓ Generated Text
""")

# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs([
    "📊 About GPT",
    "✨ Generate Text"
])

# =====================================================
# ABOUT GPT
# =====================================================

with tab1:

    st.header("What is GPT?")

    st.write("""
    GPT stands for:

    Generative Pre-trained Transformer.

    GPT generates text by predicting
    the next most likely word repeatedly
    until a complete response is formed.
    """)

    st.subheader("Applications")

    st.markdown("""
    - Chatbots
    - Content Writing
    - Story Generation
    - Code Generation
    - Question Answering
    """)

    comparison = pd.DataFrame({
        "Model": [
            "RNN",
            "LSTM",
            "GRU",
            "BERT",
            "GPT"
        ],
        "Primary Task": [
            "Sequence Learning",
            "Long-Term Memory",
            "Efficient Sequence Learning",
            "Text Understanding",
            "Text Generation"
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

# =====================================================
# TEXT GENERATION
# =====================================================

with tab2:

    st.header("Generate Text")

    prompt = st.text_area(
        "Enter Prompt",
        value="Artificial Intelligence is",
        height=150
    )

    max_length = st.slider(
        "Maximum Length",
        50,
        300,
        100
    )

    if st.button("🚀 Generate"):

        with st.spinner(
            "Generating text..."
        ):

            result = generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                truncation=True
            )

            generated_text = result[0][
                "generated_text"
            ]

        st.subheader(
            "Generated Output"
        )

        st.success(
            generated_text
        )

        st.subheader(
            "Generation Statistics"
        )

        stats = pd.DataFrame({
            "Metric": [
                "Prompt Length",
                "Generated Length"
            ],
            "Value": [
                len(prompt),
                len(generated_text)
            ]
        })

        st.dataframe(
            stats,
            use_container_width=True
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    "🚀 Built using GPT, Transformers and Streamlit"
)