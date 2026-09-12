import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import os

def load_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df

def preprocess_text(text):
    if isinstance(text, str):
        return text.lower()
    return ""

def extract_linguistic_features(df):
    liwc_cols = [col for col in df.columns if col.startswith('lex_liwc_')]
    dal_cols = [col for col in df.columns if col.startswith('lex_dal_')]
    sentiment_col = ['sentiment']
    feature_cols = liwc_cols + dal_cols + sentiment_col
    return df[feature_cols]

def main():
    # File paths
    train_path = 'Data/train_data.csv'
    val_path = 'Data/validation_data.csv'
    test_path = 'Data/test-data.csv'
    synth_path = 'Data/synthetic_distress.csv'
    artifacts_dir = 'model_artifacts'

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # --- DATA AUGMENTATION ---
    try:
        print("Integrating synthetic distress data...")
        synth_df = pd.read_csv(synth_path)

        # Identify linguistic columns from main training set
        liwc_cols = [col for col in train_df.columns if col.startswith('lex_liwc_')]
        dal_cols = [col for col in train_df.columns if col.startswith('lex_dal_')]
        sentiment_col = ['sentiment']
        ling_cols = liwc_cols + dal_cols + sentiment_col

        # Fill synthetic data with neutral means
        neutral_means = train_df[ling_cols].mean()
        for col in ling_cols:
            synth_df[col] = neutral_means[col]

        # Align other necessary columns (like subreddit, post_id etc. can be NaN)
        # We just need text and label to be present
        train_df = pd.concat([train_df, synth_df], ignore_index=True)
        print(f"Augmented training set size: {len(train_df)}")
    except FileNotFoundError:
        print("Synthetic data not found, proceeding with original data.")
    # --------------------------

    print("Preprocessing text...")
    train_df['text'] = train_df['text'].apply(preprocess_text)
    val_df['text'] = val_df['text'].apply(preprocess_text)
    test_df['text'] = test_df['text'].apply(preprocess_text)

    # 1. Semantic Embeddings instead of TF-IDF
    print("Generating Sentence-BERT embeddings (this may take a minute)...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    X_train_text = sbert_model.encode(train_df['text'].tolist())
    X_val_text = sbert_model.encode(val_df['text'].tolist())
    X_test_text = sbert_model.encode(test_df['text'].tolist())

    # 2. Extract and Scale linguistic features
    print("Extracting and scaling linguistic features...")
    X_train_ling = extract_linguistic_features(train_df)
    X_val_ling = extract_linguistic_features(val_df)
    X_test_ling = extract_linguistic_features(test_df)

    scaler = StandardScaler()
    X_train_ling_scaled = scaler.fit_transform(X_train_ling)
    X_val_ling_scaled = scaler.transform(X_val_ling)
    X_test_ling_scaled = scaler.transform(X_test_ling)

    # 3. Fusion
    print("Fusing semantic and linguistic features...")
    X_train_final = np.hstack([X_train_text, X_train_ling_scaled])
    X_val_final = np.hstack([X_val_text, X_val_ling_scaled])
    X_test_final = np.hstack([X_test_text, X_test_ling_scaled])

    y_train = train_df['label']
    y_val = val_df['label']
    y_test = test_df['label']

    # 4. Model Training
    print("Tuning RandomForestClassifier...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20, 40],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train_final, y_train)

    model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Evaluation
    val_preds = model.predict(X_val_final)
    print("\nValidation Set Performance:")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    print(f"F1-Score: {f1_score(y_val, val_preds):.4f}")
    print(classification_report(y_val, val_preds))

    test_preds = model.predict(X_test_final)
    print("\nTest Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test, test_preds):.4f}")
    print(f"F1-Score: {f1_score(y_test, test_preds):.4f}")
    print(classification_report(y_test, test_preds))

    # Save artifacts
    print("\nSaving artifacts...")
    joblib.dump(model, os.path.join(artifacts_dir, 'rf_model.joblib'))
    # We save the model name instead of the full object to avoid large files,
    # predict.py will load it from the name.
    joblib.dump('all-MiniLM-L6-v2', os.path.join(artifacts_dir, 'tfidf_vectorizer.joblib'))
    joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.joblib'))
    joblib.dump(X_train_ling.columns.tolist(), os.path.join(artifacts_dir, 'ling_cols.joblib'))
    neutral_means = train_df[train_df['label'] == 0][X_train_ling.columns].mean().tolist()
    joblib.dump(neutral_means, os.path.join(artifacts_dir, 'ling_means.joblib'))
    print("Artifacts saved successfully.")

if __name__ == "__main__":
    main()
