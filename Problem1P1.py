"""
COIT29225 Assessment 2 - Problem 1, Part 1

California Housing dataset analysis and preprocessing.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


# ---------------------------------------------------------------------------
# Median House Value Distribution by Location
# ---------------------------------------------------------------------------

# Scale population values so they can be used as readable scatter-point sizes.
population_marker_size = housing_clean["population"] / 100

plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    housing_clean["longitude"],
    housing_clean["latitude"],
    s=population_marker_size,
    c=housing_clean["median_house_value"],
    cmap="viridis",
    alpha=0.5
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("California Median House Value Distribution by Location")

colorbar = plt.colorbar(scatter)
colorbar.set_label("Median House Value ($)")

# Add reference markers so the population-based point sizes are interpretable.
for population in [1000, 3000, 5000]:
    plt.scatter(
        [],
        [],
        s=population / 100,
        c="gray",
        alpha=0.5,
        label=f"{population:,}"
    )

plt.legend(
    title="Population",
    loc="upper right"
)

plt.tight_layout()

plt.savefig(
    "outputs/problem1_part1/median_house_value_by_location.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------------------------
# Principal Component Analysis (PCA)
# ---------------------------------------------------------------------------

feature_columns = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

pca_features = housing_clean[feature_columns]

# Standardise the predictor variables because PCA is sensitive to differences
# in feature scale.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(pca_features)

pca = PCA()
pca.fit(scaled_features)

explained_variance = pca.explained_variance_ratio_

print("\nPCA explained variance ratio:")
for component_number, variance in enumerate(explained_variance, start=1):
    print(f"PC{component_number}: {variance:.4f}")

component_labels = [
    f"PC{component_number}"
    for component_number in range(1, len(explained_variance) + 1)
]

plt.figure(figsize=(9, 6))
plt.bar(component_labels, explained_variance)

plt.xlabel("Principal Component")
plt.ylabel("Proportion of Explained Variance")
plt.title("PCA Explained Variance - California Housing Dataset")

plt.tight_layout()

plt.savefig(
    "outputs/problem1_part1/pca_explained_variance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()