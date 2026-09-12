# 🧠 Condition Detection ML System

A high-sensitivity machine learning pipeline designed to detect specific emotional or psychological conditions from text and linguistic features. This system combines semantic embeddings, linguistic analysis, and a safety-first "Red Flag" override system to ensure high recall for high-risk signals.

## 🚀 Key Features

- **Semantic Understanding**: Uses `Sentence-BERT (all-MiniLM-L6-v2)` to understand the meaning of phrases, solving the "Out-of-Vocabulary" (OOV) problem.
- **Linguistic Fusion**: Integrates dense linguistic features (LIWC, DAL) with text embeddings for a holistic analysis.
- **Sentiment Guardrails**: Integrates `vaderSentiment` to adjust predictions based on emotional polarity.
- **Red Flag System**: A JSON-based override system that forces a "Condition Detected" result for critical high-severity keywords.
- **Context-Aware Filtering**: Implements "Recovery Anchors" to prevent False Positives in stories about healing and survival.
- **Multi-Interface Access**: Available via CLI, a REST API (FastAPI), and a visual Dashboard (Streamlit).

## 🛠️ Technical Architecture

1. **Input**: Raw text $\rightarrow$ Preprocessing.
2. **Vectorization**: 
   - Text $\rightarrow$ SBERT Embeddings (384-dim).
   - Text $\rightarrow$ LIWC/DAL Linguistic Features + VADER Sentiment.
3. **Fusion**: Dense concatenation of semantic and linguistic vectors.
4. **Classification**: Tuned `RandomForestClassifier` with balanced class weights.
5. **Override Layer**: Red Flag checks $\rightarrow$ Recovery Anchor filtering.
6. **Output**: Prediction (0/1), Probability, and Trigger Label.

## 📂 Project Structure

```text
.
├── Data/                   # Training, Validation, and Test CSVs
├── model_artifacts/        # Saved model, scaler, and means
├── red_flags.json          # Configuration for high-severity keywords
├── train.py                # Full training and tuning pipeline
├── predict.py               # Core inference engine
├── server.py               # FastAPI REST server
├── app.py                  # Streamlit Visual Dashboard
├── evaluate_system.py      # Accuracy and F1-Score evaluator
└── analyze_errors.py        # FP/FN analysis tool
```

## ⚙️ Installation

```bash
# Install dependencies
pip install pandas numpy sentence-transformers scikit-learn joblib vaderSentiment fastapi uvicorn streamlit
```

## 🏃 How to Use

### 1. Train the Model
Generate the model and save artifacts to the `model_artifacts/` folder.
```bash
py train.py
```

### 2. Run the Visual Dashboard (Recommended)
Launch the interactive web interface.
```bash
py -m streamlit run app.py
```

### 3. Run the REST API
Start the server to allow other applications to send requests.
```bash
py server.py
```
*API Endpoint: `POST http://127.0.0.1:8000/predict`*

### 4. Evaluate Performance
Run a full evaluation on the test dataset.
```bash
py evaluate_system.py
```

## 📈 Final Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Accuracy** | ~74% |
| **F1-Score** | ~77% |
| **Recall (Condition)** | **~87%** |

*Note: The system is intentionally tuned for high recall to minimize the risk of missing critical distress signals.*

## 🛡️ Safety & Configuration
You can update the critical keyword list in `red_flags.json`. Any text containing these words will be flagged as "Condition Detected" regardless of the ML probability, unless the "Recovery Anchors" in `predict.py` detect a healing context.
