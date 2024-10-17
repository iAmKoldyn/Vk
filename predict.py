import pandas as pd
import numpy as np
import joblib
from tsfresh.feature_extraction import feature_calculators
from scipy.stats import skew, kurtosis
from tqdm import tqdm

def extract_features(row):
    values = np.array(row['values'])
    features = {}
    features['id'] = row['id']
    features['mean'] = np.mean(values)
    features['std'] = np.std(values)
    features['min'] = np.min(values)
    features['max'] = np.max(values)
    features['skewness'] = skew(values)
    features['kurtosis'] = kurtosis(values)
    features['median'] = np.median(values)
    features['quantile_25'] = np.quantile(values, 0.25)
    features['quantile_75'] = np.quantile(values, 0.75)
    features['abs_energy'] = feature_calculators.abs_energy(values)
    features['cid_ce'] = feature_calculators.cid_ce(values, normalize=True)
    features['mean_abs_change'] = feature_calculators.mean_abs_change(values)
    features['mean_change'] = feature_calculators.mean_change(values)
    features['variance'] = feature_calculators.variance(values)
    features['longest_strike_above_mean'] = feature_calculators.longest_strike_above_mean(values)
    features['longest_strike_below_mean'] = feature_calculators.longest_strike_below_mean(values)
    features['count_above_mean'] = feature_calculators.count_above_mean(values)
    features['count_below_mean'] = feature_calculators.count_below_mean(values)
    return features

def main():
    # Load the test data
    test = pd.read_parquet('test.parquet')

    # Extract features
    test_features_list = []

    for idx, row in tqdm(test.iterrows(), total=test.shape[0]):
        features = extract_features(row)
        test_features_list.append(features)

    test_features = pd.DataFrame(test_features_list)
    test_features.fillna(0, inplace=True)

    # Prepare test data
    X_test = test_features.drop(['id'], axis=1)

    # Load the trained model
    model = joblib.load('trained_model.pkl')

    # Predict probabilities
    test_pred_proba = model.predict_proba(X_test)[:, 1]

    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_features['id'],
        'score': test_pred_proba
    })

    # Ensure the format matches sample_submission.csv
    submission = submission[['id', 'score']]

    # Save to CSV
    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")

if __name__ == '__main__':
    main()
