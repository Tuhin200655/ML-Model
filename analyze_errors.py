import pandas as pd
import numpy as np
from predict import predict_condition

def main():
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

    fps = []
    fns = []

    print(f"Analyzing errors on {len(test_df)} samples...")

    for idx, row in test_df.iterrows():
        text = str(row['text'])
        ling_features = row[ling_cols].tolist()

        res = predict_condition(text, ling_features=ling_features)

        if isinstance(res, dict):
            pred = res['prediction']
            prob = res['probability']
            actual = row['label']

            # False Positive: Actual 0, Pred 1
            if actual == 0 and pred == 1:
                fps.append({
                    'text': text,
                    'actual': actual,
                    'predicted': pred,
                    'probability': prob,
                    'label': 'FP'
                })
            # False Negative: Actual 1, Pred 0
            elif actual == 1 and pred == 0:
                fns.append({
                    'text': text,
                    'actual': actual,
                    'predicted': pred,
                    'probability': prob,
                    'label': 'FN'
                })

    # Save False Positives
    fp_df = pd.DataFrame(fps)
    fp_df.to_csv('model_artifacts/false_positives.csv', index=False)

    # Save False Negatives
    fn_df = pd.DataFrame(fns)
    fn_df.to_csv('model_artifacts/false_negatives.csv', index=False)

    print(f"Analysis complete.")
    print(f"False Positives found: {len(fps)}")
    print(f"False Negatives found: {len(fns)}")
    print("Detailed logs saved to model_artifacts/false_positives.csv and false_negatives.csv")

if __name__ == "__main__":
    main()
