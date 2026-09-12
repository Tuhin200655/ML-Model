import pandas as pd
import numpy as np
from predict import predict_condition, load_artifacts
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def main():
    # Load test data
    test_path = 'Data/test-data.csv'
    try:
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_path}")
        return

    # Identify linguistic columns
    liwc_cols = [col for col in test_df.columns if col.startswith('lex_liwc_')]
    dal_cols = [col for col in test_df.columns if col.startswith('lex_dal_')]
    sentiment_col = ['sentiment']
    ling_cols = liwc_cols + dal_cols + sentiment_col

    print(f"Evaluating system on {len(test_df)} samples using actual linguistic features...")

    y_true = test_df['label'].tolist()
    y_pred = []

    # Process in batches or simply iterate
    for idx, row in test_df.iterrows():
        text = str(row['text'])

        # Extract linguistic features for this specific sample
        ling_features = row[ling_cols].tolist()

        res = predict_condition(text, ling_features=ling_features)

        if isinstance(res, dict):
            y_pred.append(res['prediction'])
        else:
            y_pred.append(-1)

    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n==========================================")
    print("   FINAL SYSTEM EVALUATION REPORT         ")
    print("==========================================\n")
    print(f"Total Samples: {len(test_df)}")
    print(f"Accuracy:       {acc:.4f}")
    print(f"F1-Score:       {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))
    print("\n==========================================")

if __name__ == "__main__":
    main()
