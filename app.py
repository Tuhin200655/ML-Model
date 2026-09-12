import streamlit as st
import os
from predict import predict_condition, assess_vulnerability

# Page Config
st.set_page_config(
    page_title="NHAA Stress Assessment AI",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for a professional, clinical look
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .condition-detected { background-color: #fee2e2; color: #b91c1c; border: 2px solid #ef4444; }
    .no-condition { background-color: #dcfce7; color: #15803d; border: 2px solid #22c55e; }
    .metric-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
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
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<h1 class="main-header">🧠 NHAA Stress & Trauma Assessment Module</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem;">AI-enabled real-time assessment for victims/complainants accessing NHAA (14566).</p>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Control Panel")
    st.markdown("---")
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.45, 0.01)
    st.markdown("---")
    st.subheader("About the Module")
    st.info("""
    **Multimodal Analysis:**
    - **Text**: SBERT Semantic Embeddings
    - **Audio**: Pitch, Energy, and Pause Analysis
    - **Linguistic**: LIWC-style feature fusion
    - **Emotion**: VADER Sentiment Polarity
    """)

# --- Main Layout ---
col_main, col_info = st.columns([2, 1])

with col_main:
    st.subheader("Victim Assessment Interface")

    # Input Section
    tab1, tab2 = st.tabs(["Text Analysis", "Multimodal (Text + Audio)"])

    with tab1:
        user_text_1 = st.text_area("Enter textual narrative:", placeholder="Describe the experience...", height=150, key="text1")
        analyze_text = st.button("🚀 Run Text Analysis", key="btn1")

    with tab2:
        user_text_2 = st.text_area("Enter textual narrative:", placeholder="Describe the experience...", height=150, key="text2")
        audio_file = st.file_uploader("Upload Audio Interaction (.wav, .mp3)", type=["wav", "mp3"])
        analyze_multimodal = st.button("🚀 Run Multimodal Analysis", key="btn2")

    # Execution Logic
    result = None
    if analyze_text and user_text_1:
        result = assess_vulnerability(user_text_1, threshold=threshold)
    elif analyze_multimodal and user_text_2:
        # Save uploaded file temporarily
        audio_path = None
        if audio_file:
            audio_path = f"temp_{audio_file.name}"
            with open(audio_path, "wb") as f:
                f.write(audio_file.getbuffer())

        result = assess_vulnerability(user_text_2, audio_path=audio_path, threshold=threshold)
        # Cleanup temp file
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

    if result and isinstance(result, dict):
        # 1. Result Card
        risk = result['risk_category']
        if risk in ["High", "Critical"]:
            st.markdown(f'<div class="result-card condition-detected">⚠️ Risk Level: {risk}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card no-condition">✅ Risk Level: {risk}</div>', unsafe_allow_html=True)

        # 2. Metrics Grid
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.caption("SVI Score")
            st.markdown(f"### {result['svi']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with m_col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.caption("ML Probability")
            st.markdown(f"### {result['ml_probability']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        with m_col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.caption("Sentiment")
            st.markdown(f"### {result['sentiment_score']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with m_col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.caption("Audio Stress")
            audio_s = result.get('audio_score', 0.0)
            st.markdown(f"### {audio_s:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        # 3. Recommendation Section
        st.markdown("### 📋 Recommended Action Plan")
        rec = result['recommendation']
        st.warning(f"**{rec['action']}** (Priority: {rec['priority']})")
        st.info(rec['details'])

        # 4. Detailed Analysis
        with st.expander("View Technical Analysis"):
            st.write(f"**Triggered Red Flags:** {', '.join(result['triggered_flags']) if result['triggered_flags'] else 'None'}")
            st.write(f"**Recovery Indicators Detected:** {'Yes' if result['is_recovery'] else 'No'}")
            if result.get('audio_metrics'):
                st.write("**Audio Metrics:**")
                st.json(result['audio_metrics'])

with col_info:
    st.subheader("System Status")
    st.success("SVI Engine: Active ✅")
    st.success("Audio Processor: Active ✅")
    st.success("Multilingual SBERT: Active ✅")
    st.success("Red Flag Net: Active ✅")
    st.markdown("---")
    st.markdown("**Legal Disclaimer**")
    st.caption("This AI module provides a vulnerability assessment based on linguistic and acoustic patterns. It is intended to support decision-making for mental health professionals and law enforcement and does not replace a clinical diagnosis.")

st.divider()
st.caption("NHAA (14566) Integrated Portal AI Module | Ministry of Social Justice and Empowerment")
