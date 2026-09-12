import streamlit as st
from predict import predict_condition

# Page Config
st.set_page_config(
    page_title="Condition Detection Dashboard",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .condition-detected {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .no-condition {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🧠 Condition Detection Dashboard")
st.markdown("""
    This dashboard uses a hybrid ML model (SBERT + Linguistic Features + VADER)
    to detect specific conditions from text. It includes a **Red Flag** system
    for high-risk keywords and a **Contextual Guardrail** to filter out healing stories.
""")

st.divider()

# Sidebar for Settings
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Detection Threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.01)
st.sidebar.info("A lower threshold increases sensitivity (higher recall), while a higher threshold reduces false positives.")

# Main Input Section
st.subheader("Analyze Text")
user_text = st.text_area("Enter the text you want to analyze:", placeholder="Type here...", height=150)

if st.button("Analyze Text"):
    if not user_text.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            # Call the model
            result = predict_condition(user_text, threshold=threshold)

            if isinstance(result, dict):
                # Display Result Box
                if result['prediction'] == 1:
                    st.markdown(f'<div class="result-box condition-detected">{result["label"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box no-condition">{result["label"]}</div>', unsafe_allow_html=True)

                # Detailed Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probability", f"{result['probability']:.2%}")
                with col2:
                    st.metric("Threshold Used", f"{result['threshold_used']:.2f}")

                # Red Flag Info
                if 'flag' in result and result['flag']:
                    st.error(f"🚩 **Red Flag Triggered**: {result['flag']}")
                else:
                    st.success("✅ No red flags triggered.")

            else:
                st.error(f"Error: {result}")

# Footer
st.divider()
st.caption("Condition Detection Model | Developed with SBERT & RandomForest")
