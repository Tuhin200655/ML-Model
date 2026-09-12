import joblib
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load artifacts
ARTIFACTS_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'rf_model.joblib')
SBERT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.joblib')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.joblib')
LING_COLS_PATH = os.path.join(ARTIFACTS_DIR, 'ling_cols.joblib')
LING_MEANS_PATH = os.path.join(ARTIFACTS_DIR, 'ling_means.joblib')
RED_FLAGS_PATH = 'red_flags.json'

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    sbert_name = joblib.load(SBERT_MODEL_PATH)
    sbert_model = SentenceTransformer(sbert_name)
    scaler = joblib.load(SCALER_PATH)
    ling_cols = joblib.load(LING_COLS_PATH)
    ling_means = joblib.load(LING_MEANS_PATH)

    red_flags = []
    try:
        with open(RED_FLAGS_PATH, 'r') as f:
            red_flags = json.load(f).get('high_severity_keywords', [])
    except (FileNotFoundError, json.JSONDecodeError):
        print("Warning: Red flags file not found or invalid.")

    return model, sbert_model, scaler, ling_cols, ling_means, red_flags

# Initialize artifacts
try:
    model, sbert_model, scaler, ling_cols, ling_means, red_flags = load_artifacts()
except FileNotFoundError:
    print("Model artifacts not found. Please run train.py first.")
    model = sbert_model = scaler = ling_cols = ling_means = red_flags = None

def predict_condition(text, ling_features=None, threshold=0.45):
    if model is None:
        return "Error: Model not loaded."

    text_lower = text.lower()

    # --- RECOVERY/HEALING ANCHORS (Negative Red Flags) ---
    # These words suggest the person is talking about the past, healing, or advocacy
    recovery_anchors = [
        "overcame", "recovered", "healing", "life is good now", "proud of",
        "survivor", "advocate", "moving forward", "getting better",
        "found peace", "overcome", "past abuse", "ended years ago"
    ]

    is_recovery = any(anchor in text_lower for anchor in recovery_anchors)

    # --- RED FLAG OVERRIDE ---
    for flag in red_flags:
        if flag.lower() in text_lower:
            # If a red flag is triggered, but the text also indicates recovery/healing,
            # we don't automatically trigger Condition Detected. We let the ML model decide.
            if not is_recovery:
                return {
                    "prediction": 1,
                    "probability": 1.0,
                    "label": "Condition Detected (Red Flag Triggered)",
                    "threshold_used": threshold,
                    "flag": flag
                }

    # Semantic Embeddings
    text_feat = sbert_model.encode([text_lower])

    # Linguistic features
    if ling_features is None:
        ling_feat = np.array(ling_means).reshape(1, -1)
    else:
        ling_feat = np.array(ling_features).reshape(1, -1)

    # VADER sentiment analysis
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_score = analyzer.polarity_scores(text)['compound']
        if 'sentiment' in ling_cols:
            sentiment_idx = ling_cols.index('sentiment')
            ling_feat[0, sentiment_idx] = sentiment_score
    except Exception:
        pass

    # Scale linguistic features
    ling_feat_scaled = scaler.transform(ling_feat)

    # Fusion
    final_feat = np.hstack([text_feat, ling_feat_scaled])

    # Prediction
    probs = model.predict_proba(final_feat)[0]
    prob_class_1 = probs[1]

    # --- CONTEXTUAL GUARDRAIL ---
    # If sentiment is highly positive AND recovery anchors are present,
    # we cap the probability to prevent False Positives in healing stories.
    if sentiment_score > 0.6 and is_recovery:
        prob_class_1 = min(prob_class_1, 0.3)

    prediction = 1 if prob_class_1 >= threshold else 0

    return {
        "prediction": int(prediction),
        "probability": float(prob_class_1),
        "label": "Condition Detected" if prediction == 1 else "No Condition Detected",
        "threshold_used": threshold
    }

def predict_from_voice(audio_path=None):
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        if audio_path:
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
        else:
            with sr.Microphone() as source:
                print("Listening...")
                audio_data = recognizer.listen(source)
        print("Recognizing...")
        text = recognizer.recognize_google(audio_data)
        print(f"Recognized text: {text}")
        return predict_condition(text)
    except ImportError:
        return "Error: SpeechRecognition library not installed."
    except Exception as e:
        return f"Error during voice processing: {str(e)}"

if __name__ == "__main__":
    test_texts = [
        "I feel so happy and excited about my new job!",
        "I have been feeling a deep sense of doom and hopelessness for weeks.",
        "The weather is quite nice today, I might go for a walk.",
        "My neighbour tortured me",
        "I think I am useless in this world",
        "My abuse ended years ago. My life is good now. I'm about to get married to someone really wonderful.",
        "I am proud to have broken the silence about the abuse I suffered as a child."
    ]

    for t in test_texts:
        res = predict_condition(t)
        print(f"Text: {t}\nResult: {res}\n")
