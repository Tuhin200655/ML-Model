import joblib
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from recommendation_engine import map_svi_to_category, get_recommendation
from audio_processor import extract_audio_stress_features, calculate_audio_stress_score

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

def assess_vulnerability(text, ling_features=None, audio_path=None, threshold=0.45):
    if model is None:
        return "Error: Model not loaded."

    text_lower = text.lower()

    # 1. Red Flag Detection
    triggered_flags = [flag for flag in red_flags if flag.lower() in text_lower]
    red_flag_bonus = 1.0 if triggered_flags else 0.0

    # 2. Recovery/Healing Anchor Detection
    recovery_anchors = [
        "overcame", "recovered", "healing", "life is good now", "proud of",
        "survivor", "advocate", "moving forward", "getting better",
        "found peace", "overcome", "past abuse", "ended years ago"
    ]
    is_recovery = any(anchor in text_lower for anchor in recovery_anchors)

    # 3. ML Semantic Prediction
    text_feat = sbert_model.encode([text_lower])
    if ling_features is None:
        ling_feat = np.array(ling_means).reshape(1, -1)
    else:
        ling_feat = np.array(ling_features).reshape(1, -1)

    # VADER Sentiment
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_score = analyzer.polarity_scores(text)['compound']
        if 'sentiment' in ling_cols:
            sentiment_idx = ling_cols.index('sentiment')
            ling_feat[0, sentiment_idx] = sentiment_score
    except Exception:
        sentiment_score = 0.0

    # Scale and Fusion
    ling_feat_scaled = scaler.transform(ling_feat)
    final_feat = np.hstack([text_feat, ling_feat_scaled])

    # Model Probability
    probs = model.predict_proba(final_feat)[0]
    ml_prob = probs[1]

    # 4. Audio Stress Analysis
    audio_score = 0.0
    audio_metrics = None
    if audio_path:
        audio_metrics = extract_audio_stress_features(audio_path)
        if audio_metrics:
            audio_score = calculate_audio_stress_score(audio_metrics)

    # 5. SVI Calculation (Stress Vulnerability Index)
    # Weighted formula for NHAA standards:
    # 40% ML Probability + 10% Sentiment + 20% Red Flags + 30% Audio Stress
    sentiment_negativity = 1.0 - ((sentiment_score + 1.0) / 2.0)

    svi = (ml_prob * 0.4) + (sentiment_negativity * 0.1) + (red_flag_bonus * 0.2) + (audio_score * 0.3)

    # Contextual Adjustment: If it's a recovery story, reduce SVI
    if is_recovery:
        svi *= 0.7

    svi = min(1.0, max(0.0, svi))

    # 6. Risk Categorization and Recommendation
    risk_category = map_svi_to_category(svi)
    recommendation = get_recommendation(risk_category)

    return {
        "svi": float(svi),
        "risk_category": risk_category,
        "recommendation": recommendation,
        "ml_probability": float(ml_prob),
        "sentiment_score": float(sentiment_score),
        "audio_score": float(audio_score),
        "audio_metrics": audio_metrics,
        "triggered_flags": triggered_flags,
        "is_recovery": is_recovery
    }

# For backward compatibility with existing tests/apps
def predict_condition(text, ling_features=None, threshold=0.45):
    res = assess_vulnerability(text, ling_features, threshold)
    if isinstance(res, str): return res

    prediction = 1 if res['svi'] >= threshold else 0
    return {
        "prediction": prediction,
        "probability": res['svi'],
        "label": "Condition Detected" if prediction == 1 else "No Condition Detected",
        "threshold_used": threshold,
        "flag": res['triggered_flags'][0] if res['triggered_flags'] else None
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
        return assess_vulnerability(text)
    except ImportError:
        return "Error: SpeechRecognition library not installed."
    except Exception as e:
        return f"Error during voice processing: {str(e)}"

if __name__ == "__main__":
    test_texts = [
        "I feel so happy and excited about my new job!",
        "My neighbour tortured me",
        "I think I am useless in this world",
        "The village head blocked me from the water well and everyone is ignoring me",
        "My abuse ended years ago. My life is good now."
    ]

    for t in test_texts:
        res = assess_vulnerability(t)
        print(f"Text: {t}\nResult: {res}\n")
