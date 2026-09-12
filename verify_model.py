import pandas as pd
import joblib
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

ARTIFACTS_DIR = 'model_artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'rf_model.joblib')
TFIDF_PATH = os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.joblib')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.joblib')
LING_COLS_PATH = os.path.join(ARTIFACTS_DIR, 'ling_cols.joblib')
LING_MEANS_PATH = os.path.join(ARTIFACTS_DIR, 'ling_means.joblib')

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    scaler = joblib.load(SCALER_PATH)
    ling_cols = joblib.load(LING_COLS_PATH)
    ling_means = joblib.load(LING_MEANS_PATH)
    return model, tfidf, scaler, ling_cols, ling_means

model, tfidf, scaler, ling_cols, ling_means = load_artifacts()

def predict_condition(text, ling_features=None):
    text = text.lower()
    text_feat = tfidf.transform([text])
    if ling_features is None:
        ling_feat = np.array(ling_means).reshape(1, -1)
    else:
        ling_feat = np.array(ling_features).reshape(1, -1)
    ling_feat_scaled = scaler.transform(ling_feat)
    final_feat = hstack([text_feat, ling_feat_scaled])
    prediction = model.predict(final_feat)[0]
    return int(prediction)

# Load test data to get real samples
test_df = pd.read_csv('test-data.csv')
# Get linguistic columns
ling_df = test_df[ling_cols]

# Sample some cases
samples = test_df.sample(5, random_state=42)

print("Qualitative Verification with real features:\n")
for idx, row in samples.iterrows():
    text = row['text']
    true_label = row['label']
    ling_feat = ling_df.loc[idx].values
    pred_label = predict_condition(text, ling_feat)
    print(f"Text: {text[:100]}...")
    print(f"True: {true_label}, Pred: {pred_label}")
    print("-" * 20)
