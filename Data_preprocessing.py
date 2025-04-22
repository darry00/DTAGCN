import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split


def preprocess_data(data, missing_threshold=0.3, knn_neighbors=5):
    """
    Preprocess water level and precipitation data.
    - Fill missing values (linear interpolation for short-term, KNN for long-term).
    - Detect and replace outliers (3-sigma rule + Local Outlier Factor).
    - Split data into train, validation, and test sets (6:2:2).

    Parameters:
    data (pd.DataFrame): Time series data with timestamps as index.
    missing_threshold (float): Maximum allowed missing rate for a column.
    knn_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
    train, valid, test (pd.DataFrame): Preprocessed and split datasets.
    """
    # Step 1: Handling missing values
    missing_ratio = data.isnull().mean()
    valid_columns = missing_ratio[missing_ratio < missing_threshold].index
    data = data[valid_columns]  # Drop columns with excessive missing data

    # Fill short-term missing values with linear interpolation
    data = data.interpolate(method='linear', limit_direction='both')

    # Fill long-term missing values with KNN imputation
    knn_imputer = KNNImputer(n_neighbors=knn_neighbors)
    data.iloc[:, :] = knn_imputer.fit_transform(data)

    # Step 2: Outlier detection and replacement
    lof = LocalOutlierFactor(n_neighbors=10, contamination=0.01)

    for col in data.columns:
        # Three-sigma rule
        mean, std = data[col].mean(), data[col].std()
        upper, lower = mean + 3 * std, mean - 3 * std
        data[col] = np.where((data[col] > upper) | (data[col] < lower), np.nan, data[col])

        # Local Outlier Factor detection
        mask = lof.fit_predict(data[[col]]) == -1
        data.loc[mask, col] = np.nan  # Mark outliers as NaN

        # Fill outliers with rolling median
        data[col] = data[col].fillna(data[col].rolling(5, min_periods=1).median())

    # Step 3: Train-validation-test split (6:2:2)
    train_size = int(len(data) * 0.6)
    valid_size = int(len(data) * 0.2)

    train = data.iloc[:train_size]
    valid = data.iloc[train_size:train_size + valid_size]
    test = data.iloc[train_size + valid_size:]

    return train, valid, test

# Example usage:
# df = pd.read_csv("water_level_precipitation.csv", index_col=0, parse_dates=True)
# train, valid, test = preprocess_data(df)
# train.to_csv("train.csv")
# valid.to_csv("valid.csv")
# test.to_csv("test.csv")
