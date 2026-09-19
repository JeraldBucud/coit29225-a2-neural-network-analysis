"""
COIT29225 Assessment 2 - Problem 1, Part 1

California Housing dataset analysis and preprocessing.
"""

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

# ---------------------------------------------------------------------------
# K-means Clustering Analysis
# ---------------------------------------------------------------------------

kmeans_columns = [
    "longitude",
    "latitude",
    "median_house_value",
]

kmeans_features = housing_clean[kmeans_columns]

# Standardise variables before K-means because the feature scales differ.
kmeans_scaler = StandardScaler()
scaled_kmeans_features = kmeans_scaler.fit_transform(kmeans_features)


def plot_kmeans_clusters(data, scaled_data, n_clusters, output_file):
    """Fit K-means and save a geographical cluster visualisation."""

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    cluster_labels = model.fit_predict(scaled_data)

    plt.figure(figsize=(10, 8))

    markers = ["o", "^", "s", "D", "X"]

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id

        scatter = plt.scatter(
            data.loc[cluster_mask, "longitude"],
            data.loc[cluster_mask, "latitude"],
            c=data.loc[cluster_mask, "median_house_value"],
            cmap="viridis",
            vmin=data["median_house_value"].min(),
            vmax=data["median_house_value"].max(),
            marker=markers[cluster_id],
            s=20,
            alpha=0.6,
            label=f"Cluster {cluster_id}",
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(
        f"California Housing K-means Clustering - {n_clusters} Clusters"
    )

    colorbar = plt.colorbar(scatter)
    colorbar.set_label("Median House Value ($)")

    plt.legend(title="K-means Cluster")

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return cluster_labels

clusters_4 = plot_kmeans_clusters(
    housing_clean,
    scaled_kmeans_features,
    4,
    "outputs/problem1_part1/kmeans_4_clusters.png",
)

clusters_5 = plot_kmeans_clusters(
    housing_clean,
    scaled_kmeans_features,
    5,
    "outputs/problem1_part1/kmeans_5_clusters.png",
)

print("\nK-means cluster counts - 4 clusters:")
print(pd.Series(clusters_4).value_counts().sort_index())

print("\nK-means cluster counts - 5 clusters:")
print(pd.Series(clusters_5).value_counts().sort_index())


# ---------------------------------------------------------------------------
# Neural Network Data Preparation
# ---------------------------------------------------------------------------

# Predictor variables used to estimate the median house value.
model_features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

X = housing_clean[model_features]
y = housing_clean["median_house_value"]

# Reserve 20% of the data for final model evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# Fit the scaler using training data only to avoid test-data leakage.
model_scaler = StandardScaler()

X_train_scaled = model_scaler.fit_transform(X_train)
X_test_scaled = model_scaler.transform(X_test)

print("\nNeural network data preparation:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of input features: {X_train.shape[1]}")

scaler.fit_transform(X)


# ---------------------------------------------------------------------------
# Baseline Deep Neural Network Regression Model
# ---------------------------------------------------------------------------

# Use a fixed seed so the baseline experiment can be reproduced.
tf.keras.utils.set_random_seed(42)

baseline_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1),
])

baseline_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"],
)

print("\nBaseline neural network architecture:")
baseline_model.summary()

# ---------------------------------------------------------------------------
# Baseline Model Training
# ---------------------------------------------------------------------------

print("\nTraining baseline neural network...")

baseline_history = baseline_model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=64,
    verbose=1,
)

# ---------------------------------------------------------------------------
# Baseline Training Curves
# ---------------------------------------------------------------------------

epochs = range(1, len(baseline_history.history["loss"]) + 1)

# Mean Squared Error curve.
plt.figure(figsize=(9, 6))
plt.plot(epochs, baseline_history.history["loss"], label="Training MSE")
plt.plot(epochs, baseline_history.history["val_loss"], label="Validation MSE")

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Baseline DNN - MSE Training Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/problem1_part1/baseline_mse_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Mean Absolute Error curve.
plt.figure(figsize=(9, 6))
plt.plot(epochs, baseline_history.history["mae"], label="Training MAE")
plt.plot(epochs, baseline_history.history["val_mae"], label="Validation MAE")

plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error ($)")
plt.title("Baseline DNN - MAE Training Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/problem1_part1/baseline_mae_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ---------------------------------------------------------------------------
# Baseline Model Evaluation
# ---------------------------------------------------------------------------

test_predictions = baseline_model.predict(
    X_test_scaled,
    verbose=0,
).reshape(-1)

test_mse = mean_squared_error(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

print("\nBaseline test results:")
print(f"MSE: {test_mse:,.2f}")
print(f"MAE: ${test_mae:,.2f}")
print(f"R2: {test_r2:.4f}")


# Compare actual and predicted house values.
plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    test_predictions,
    alpha=0.4,
    s=20,
)

minimum_value = min(y_test.min(), test_predictions.min())
maximum_value = max(y_test.max(), test_predictions.max())

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    linestyle="--",
    label="Perfect prediction",
)

plt.xlabel("Actual Median House Value ($)")
plt.ylabel("Predicted Median House Value ($)")
plt.title("Baseline DNN - Actual vs Predicted House Values")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/problem1_part1/baseline_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\n10 example house-price predictions:")

prediction_examples = pd.DataFrame({
    "Actual House Value": y_test.iloc[:10].to_numpy(),
    "Predicted House Value": test_predictions[:10],
})

print(prediction_examples.to_string(index=False))