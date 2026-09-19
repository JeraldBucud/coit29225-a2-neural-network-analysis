"""
COIT29225 Assessment 2 - Problem 1, Part 1

California Housing dataset analysis and preprocessing.
"""

import pandas as pd


DATA_FILE = "data/CaliforniaHousing.csv"

housing_data = pd.read_csv(DATA_FILE)

print("First five rows:")
print(housing_data.head())

print("\nDataset shape:")
print(housing_data.shape)

print("\nColumn names:")
print(housing_data.columns.tolist())

print("\nData types:")
print(housing_data.dtypes)

print("\nMissing values:")
print(housing_data.isnull().sum())

print("\nDuplicate rows:")
print(housing_data.duplicated().sum())

print("\nDescriptive statistics:")
print(housing_data.describe())

print("\nOcean proximity categories:")
print(housing_data["ocean_proximity"].value_counts())


# Keep the original dataset unchanged while preprocessing a working copy.
housing_clean = housing_data.copy()

# Median imputation is used because total_bedrooms contains extreme values
# that can make the mean less representative.
bedroom_median = housing_clean["total_bedrooms"].median()
housing_clean["total_bedrooms"] = housing_clean["total_bedrooms"].fillna(
    bedroom_median
)

print("\nMissing values after preprocessing:")
print(housing_clean.isnull().sum())