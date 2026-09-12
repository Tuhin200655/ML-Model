# 🧠 AI-Based Real-Time Stress and Trauma Assessment Module (NHAA-14566)

An advanced, multimodal AI system developed for the **National Helpline Against Atrocities (NHAA - 14566)** and the Integrated Portal of the **Ministry of Social Justice and Empowerment (MoSJE)**. This module is designed to assess the psychological stress, trauma, and vulnerability levels of victims/complainants in real-time.

## 🎯 Problem Statement Alignment
The system addresses the need for a standardized mechanism to assess psychological conditions of victims interacting via digital platforms (Helplines, Chatbots, Mobile Apps, IVRS). It focuses on identifying signs of trauma, fear, anxiety, and extreme vulnerability to prioritize critical support.

## 🚀 Core Capabilities

- **Multimodal Analysis**: Integrates textual narratives with acoustic speech patterns.
- **Stress Vulnerability Index (SVI)**: A scientific scoring mechanism (0.0 - 1.0) that quantifies the level of distress.
- **Emotion AI**: Utilizes VADER sentiment polarity and SBERT semantic embeddings to detect hidden trauma.
- **Acoustic Analytics**: Analyzes pitch variation, pause durations, and energy levels to identify physiological signs of stress.
- **Automated Recommendations**: Maps risk levels to specific institutional actions (Legal Aid, Police Intervention, etc.).
- **Safety Guardrails**: Implements a specialized "Red Flag" system for atrocity-specific markers and "Recovery Anchors" to prevent false positives in healing narratives.

## 🛠️ Technical Architecture

### 1. SVI Calculation Formula
The system calculates the **Stress Vulnerability Index (SVI)** using a weighted fusion of four signals:
$$\text{SVI} = (\text{ML Probability} \times 0.4) + (\text{Sentiment Negativity} \times 0.1) + (\text{Red Flags} \times 0.2) + (\text{Audio Stress} \times 0.3)$$

### 2. Risk Categorization
The SVI is mapped to one of four risk categories:
- **Low (0.0 - 0.3)** $\rightarrow$ Standard Support & Resource Guides.
- **Moderate (0.3 - 0.5)** $\rightarrow$ Prioritized Counselling.
- **High (0.5 - 0.8)** $\rightarrow$ Urgent Legal Aid & Medical Assistance.
- **Critical (0.8 - 1.0)** $\rightarrow$ Immediate Police Intervention & Witness Protection.

### 3. Feature Stack
- **Text**: `Sentence-BERT (all-MiniLM-L6-v2)` for semantic trauma detection.
- **Audio**: `Librosa` for F0 Pitch, RMS Energy, and Silence/Pause Ratio.
- **Sentiment**: `vaderSentiment` for emotional valence.
- **Linguistic**: LIWC/DAL based feature fusion.

## 📂 Project Structure

```text
.
├── Data/                   # Training and Test datasets
├── model_artifacts/        # Saved model, scaler, and neutral means
├── red_flags.json          # Atrocity-specific high-severity keywords
├── train.py                # Training pipeline with data augmentation
├── predict.py               # Core Multimodal Assessment Engine
├── audio_processor.py       # Acoustic stress feature extraction
├── recommendation_engine.py # Risk-to-Action mapping logic
├── server.py               # FastAPI REST server for portal integration
├── app.py                  # Streamlit Visual Dashboard for operators
└── evaluate_system.py      # System performance and accuracy evaluator
```

## ⚙️ Setup & Execution

```bash
# 1. Install dependencies
pip install pandas numpy sentence-transformers scikit-learn joblib vaderSentiment fastapi uvicorn streamlit librosa soundfile

# 2. Train the model
py train.py

# 3. Launch the visual assessment dashboard
py -m streamlit run app.py

# 4. Start the API server for portal integration
py server.py
```

## 📈 Evaluation Results
The system is tuned for **High Recall** to ensure no critical victim is overlooked.
- **Recall (Condition Detection)**: ~87%
- **Overall Accuracy**: ~74%
- **F1-Score**: ~77%

## 🛡️ Ethics & Privacy
- **Anonymization**: Designed to be integrated with PII scrubbing modules.
- **Informed Consent**: Dashboard includes prompts for victim consent.
- **Clinical Support**: The tool is a support system for professionals, not a replacement for clinical diagnosis.
