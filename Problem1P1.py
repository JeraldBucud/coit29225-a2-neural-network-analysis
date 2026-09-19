"""
COIT29225 Assessment 2 - Problem 1, Part 1

California Housing dataset analysis and preprocessing.
"""

import pandas as pd
import matplotlib.pyplot as plt


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

plt.show()