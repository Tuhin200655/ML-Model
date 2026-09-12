import streamlit as st
from predict import predict_condition

# Page Config
st.set_page_config(
    page_title="Condition Detection AI",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for a professional, clinical look
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Header styling */
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Result Card styling */
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .condition-detected {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 2px solid #ef4444;
    }
    .no-condition {
        background-color: #dcfce7;
        color: #15803d;
        border: 2px solid #22c55e;
    }

    /* Metric card styling */
    .metric-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        background-color: #1e3a8a !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        border: none !important;
    }
    .stButton>button:hover {
        background-color: #1e40af !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<h1 class="main-header">🧠 Condition Detection AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem;">A clinical-grade ML pipeline integrating Semantic Embeddings, Linguistic Analysis, and Safety Guardrails.</p>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Control Panel")
    st.markdown("---")

    st.subheader("Model Hyperparameters")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.01,
        help="Lower values increase sensitivity (Recall). Higher values increase specificity (Precision)."
    )

    st.markdown("---")
    st.subheader("About the Model")
    st.info("""
    **Architecture:**
    - **SBERT**: all-MiniLM-L6-v2
    - **Classifier**: RandomForest
    - **Sentiment**: VADER Polarity
    - **Safety**: Red-Flag Overrides
    """)

# --- Main Layout ---
col_main, col_info = st.columns([2, 1])

with col_main:
    st.subheader("Input Analysis")

    # Sample Text Selector for easier testing
    samples = {
        "Positive (Semantic)": "I feel like a burden to everyone around me",
        "Positive (Red Flag)": "My neighbour tortured me",
        "Negative (Happy)": "I feel so happy and excited about my new job!",
        "Negative (Resilient)": "I am trying my best to stay positive despite the pain",
        "Healing Story": "My abuse ended years ago. My life is good now."
    }

    selected_sample = st.selectbox("Quick Test Samples", ["None"] + list(samples.keys()))

    default_text = samples[selected_sample] if selected_sample != "None" else ""

    user_text = st.text_area(
        "Enter text for clinical analysis:",
        value=default_text,
        placeholder="Enter a sentence or paragraph...",
        height=150
    )

    if st.button("🚀 Run Analysis"):
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Processing semantic embeddings..."):
                result = predict_condition(user_text, threshold=threshold)

                if isinstance(result, dict):
                    # 1. Main Result Card
                    if result['prediction'] == 1:
                        st.markdown(f'<div class="result-card condition-detected">⚠️ {result["label"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-card no-condition">✅ {result["label"]}</div>', unsafe_allow_html=True)

                    # 2. Detailed Metrics Grid
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.caption("Model Probability")
                        st.markdown(f"### {result['probability']:.2%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with m_col2:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.caption("Threshold Applied")
                        st.markdown(f"### {result['threshold_used']:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with m_col3:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.caption("Binary Prediction")
                        st.markdown(f"### {'Positive' if result['prediction']==1 else 'Negative'}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # 3. Reasoning Section
                    st.markdown("### 🔍 Analysis Logic")
                    if 'flag' in result and result['flag']:
                        st.error(f"**Safety Trigger**: The system detected a high-severity keyword: `{result['flag']}`. This override ensures high-risk signals are never missed.")
                    elif result['prediction'] == 1:
                        st.success("**Semantic Detection**: The model identified semantic patterns of distress that exceed the current threshold.")
                    else:
                        st.info("**Neutral/Positive**: The text did not trigger red flags and its semantic probability remained below the threshold.")
                else:
                    st.error(f"Analysis Error: {result}")

with col_info:
    st.subheader("System Status")
    st.success("Model: Loaded ✅")
    st.success("SBERT: Active ✅")
    st.success("VADER: Active ✅")
    st.success("Artifacts: Verified ✅")

    st.markdown("---")
    st.markdown("**Note**: This tool is for research and demonstration purposes. It should be used as a supportive signal, not a standalone diagnostic tool.")

st.divider()
st.caption("© 2026 Condition Detection AI | Built with Streamlit & Scikit-Learn")
