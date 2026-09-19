"""
COIT29225 Neural Networks & Deep Learning
Assessment 2 - Problem 1, Part 1

California Housing Dataset Analysis and Neural Network Regression.

This program:
- Loads and inspects the California Housing dataset.
- Handles missing values.
- Produces the required geographical visualisation.
- Performs Principal Component Analysis (PCA).
- Performs K-means clustering using 4 and 5 clusters.
- Prepares the data for neural network regression.
- Trains and evaluates a baseline deep neural network.
- Performs controlled neural network parameter experiments.
- Selects a final neural network using validation performance.
- Evaluates the selected model on the test dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = Path("data/CaliforniaHousing.csv")
OUTPUT_DIR = Path("outputs/problem1_part1")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

housing_data = pd.read_csv(DATA_FILE)

print("First five rows:")
print(housing_data.head())


# ---------------------------------------------------------------------------
# Initial Dataset Inspection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------------------------

# Work with a copy so that the original imported dataset remains unchanged.
housing_clean = housing_data.copy()

# Median imputation is used because total_bedrooms contains extreme values
# that can make the mean less representative.
bedroom_median = housing_clean["total_bedrooms"].median()

housing_clean["total_bedrooms"] = (
    housing_clean["total_bedrooms"].fillna(bedroom_median)
)

print("\nMedian total bedrooms:")
print(bedroom_median)

print("\nMissing values after preprocessing:")
print(housing_clean.isnull().sum())


# ---------------------------------------------------------------------------
# Median House Value Distribution by Location
# ---------------------------------------------------------------------------

# Population controls marker size while colour represents median house value.
population_marker_size = housing_clean["population"] / 100

plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    housing_clean["longitude"],
    housing_clean["latitude"],
    s=population_marker_size,
    c=housing_clean["median_house_value"],
    cmap="viridis",
    alpha=0.5,
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("California Median House Value Distribution by Location")

colorbar = plt.colorbar(scatter)
colorbar.set_label("Median House Value ($)")

# Reference marker sizes make the population encoding easier to interpret.
for population in [1000, 3000, 5000]:
    plt.scatter(
        [],
        [],
        s=population / 100,
        c="gray",
        alpha=0.5,
        label=f"{population:,}",
    )

plt.legend(
    title="Population",
    loc="upper right",
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "median_house_value_by_location.png",
    dpi=300,
    bbox_inches="tight",
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

# PCA is sensitive to feature scale, so the predictors are standardised.
pca_scaler = StandardScaler()

scaled_pca_features = pca_scaler.fit_transform(
    pca_features
)

pca = PCA()
pca.fit(scaled_pca_features)

explained_variance = pca.explained_variance_ratio_

print("\nPCA explained variance ratio:")

for component_number, variance in enumerate(
    explained_variance,
    start=1,
):
    print(
        f"PC{component_number}: "
        f"{variance:.4f}"
    )

component_labels = [
    f"PC{component_number}"
    for component_number in range(
        1,
        len(explained_variance) + 1,
    )
]

plt.figure(figsize=(9, 6))

plt.bar(
    component_labels,
    explained_variance,
)

plt.xlabel("Principal Component")
plt.ylabel("Proportion of Explained Variance")
plt.title("PCA Explained Variance - California Housing Dataset")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "pca_explained_variance.png",
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

kmeans_features = housing_clean[
    kmeans_columns
]

# Standardisation prevents house value from dominating the Euclidean
# distance calculation because of its much larger numerical scale.
kmeans_scaler = StandardScaler()

scaled_kmeans_features = (
    kmeans_scaler.fit_transform(
        kmeans_features
    )
)


def plot_kmeans_clusters(
    data,
    scaled_data,
    n_clusters,
    output_file,
):
    """Fit K-means and save a geographical cluster visualisation."""

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    cluster_labels = model.fit_predict(
        scaled_data
    )

    plt.figure(figsize=(10, 8))

    markers = [
        "o",
        "^",
        "s",
        "D",
        "X",
    ]

    for cluster_id in range(n_clusters):
        cluster_mask = (
            cluster_labels == cluster_id
        )

        scatter = plt.scatter(
            data.loc[
                cluster_mask,
                "longitude",
            ],
            data.loc[
                cluster_mask,
                "latitude",
            ],
            c=data.loc[
                cluster_mask,
                "median_house_value",
            ],
            cmap="viridis",
            vmin=data[
                "median_house_value"
            ].min(),
            vmax=data[
                "median_house_value"
            ].max(),
            marker=markers[cluster_id],
            s=20,
            alpha=0.6,
            label=f"Cluster {cluster_id}",
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.title(
        "California Housing K-means Clustering "
        f"- {n_clusters} Clusters"
    )

    colorbar = plt.colorbar(scatter)
    colorbar.set_label(
        "Median House Value ($)"
    )

    plt.legend(
        title="K-means Cluster"
    )

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
    OUTPUT_DIR / "kmeans_4_clusters.png",
)

clusters_5 = plot_kmeans_clusters(
    housing_clean,
    scaled_kmeans_features,
    5,
    OUTPUT_DIR / "kmeans_5_clusters.png",
)

print(
    "\nK-means cluster counts - 4 clusters:"
)

print(
    pd.Series(
        clusters_4
    )
    .value_counts()
    .sort_index()
)

print(
    "\nK-means cluster counts - 5 clusters:"
)

print(
    pd.Series(
        clusters_5
    )
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------------------------
# Neural Network Data Preparation
# ---------------------------------------------------------------------------

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

X = housing_clean[
    model_features
]

y = housing_clean[
    "median_house_value"
]

# Reserve 20% of observations for final model evaluation.
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )
)

# Scale the target during neural-network training to improve numerical
# stability. Predictions are converted back into dollars for evaluation.
TARGET_SCALE = 100_000.0

y_train_nn = (
    y_train / TARGET_SCALE
)

# Fit the predictor scaler only to the training data to prevent leakage.
model_scaler = StandardScaler()

X_train_scaled = (
    model_scaler.fit_transform(
        X_train
    )
)

X_test_scaled = (
    model_scaler.transform(
        X_test
    )
)

print(
    "\nNeural network data preparation:"
)

print(
    f"Training samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples: "
    f"{len(X_test)}"
)

print(
    f"Number of input features: "
    f"{X_train.shape[1]}"
)


# ---------------------------------------------------------------------------
# Baseline Deep Neural Network
# ---------------------------------------------------------------------------

tf.keras.utils.set_random_seed(42)

baseline_model = Sequential(
    [
        Input(
            shape=(
                X_train_scaled.shape[1],
            )
        ),
        Dense(
            32,
            activation="relu",
        ),
        Dense(
            16,
            activation="relu",
        ),
        Dense(1),
    ]
)

baseline_model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"],
)

print(
    "\nBaseline neural network architecture:"
)

baseline_model.summary()


# ---------------------------------------------------------------------------
# Baseline Model Training
# ---------------------------------------------------------------------------

print(
    "\nTraining baseline neural network..."
)

baseline_history = baseline_model.fit(
    X_train_scaled,
    y_train_nn,
    validation_split=0.20,
    epochs=100,
    batch_size=64,
    verbose=1,
)


# ---------------------------------------------------------------------------
# Baseline Training Curves
# ---------------------------------------------------------------------------

baseline_epochs = range(
    1,
    len(
        baseline_history.history[
            "loss"
        ]
    )
    + 1,
)

# Convert the scaled metrics back into dollar-based units.
baseline_training_mse = [
    value * TARGET_SCALE**2
    for value
    in baseline_history.history[
        "loss"
    ]
]

baseline_validation_mse = [
    value * TARGET_SCALE**2
    for value
    in baseline_history.history[
        "val_loss"
    ]
]

baseline_training_mae = [
    value * TARGET_SCALE
    for value
    in baseline_history.history[
        "mae"
    ]
]

baseline_validation_mae = [
    value * TARGET_SCALE
    for value
    in baseline_history.history[
        "val_mae"
    ]
]


# Baseline MSE curve.
plt.figure(figsize=(9, 6))

plt.plot(
    baseline_epochs,
    baseline_training_mse,
    label="Training MSE",
)

plt.plot(
    baseline_epochs,
    baseline_validation_mse,
    label="Validation MSE",
)

plt.xlabel("Epoch")
plt.ylabel(
    "Mean Squared Error"
)
plt.title(
    "Baseline DNN - MSE Training Curve"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_mse_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Baseline MAE curve.
plt.figure(figsize=(9, 6))

plt.plot(
    baseline_epochs,
    baseline_training_mae,
    label="Training MAE",
)

plt.plot(
    baseline_epochs,
    baseline_validation_mae,
    label="Validation MAE",
)

plt.xlabel("Epoch")

plt.ylabel(
    "Mean Absolute Error ($)"
)

plt.title(
    "Baseline DNN - MAE Training Curve"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "baseline_mae_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------------
# Baseline Model Evaluation
# ---------------------------------------------------------------------------

baseline_predictions = (
    baseline_model.predict(
        X_test_scaled,
        verbose=0,
    ).reshape(-1)
    * TARGET_SCALE
)

baseline_test_mse = (
    mean_squared_error(
        y_test,
        baseline_predictions,
    )
)

baseline_test_mae = (
    mean_absolute_error(
        y_test,
        baseline_predictions,
    )
)

baseline_test_r2 = (
    r2_score(
        y_test,
        baseline_predictions,
    )
)

print(
    "\nBaseline test results:"
)

print(
    f"MSE: "
    f"{baseline_test_mse:,.2f}"
)

print(
    f"MAE: "
    f"${baseline_test_mae:,.2f}"
)

print(
    f"R2: "
    f"{baseline_test_r2:.4f}"
)


# Baseline actual versus predicted graph.
plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    baseline_predictions,
    alpha=0.4,
    s=20,
)

minimum_value = min(
    y_test.min(),
    baseline_predictions.min(),
)

maximum_value = max(
    y_test.max(),
    baseline_predictions.max(),
)

plt.plot(
    [
        minimum_value,
        maximum_value,
    ],
    [
        minimum_value,
        maximum_value,
    ],
    linestyle="--",
    label="Perfect prediction",
)

plt.xlabel(
    "Actual Median House Value ($)"
)

plt.ylabel(
    "Predicted Median House Value ($)"
)

plt.title(
    "Baseline DNN - Actual vs Predicted House Values"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "baseline_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Show 10 baseline prediction examples.
baseline_prediction_examples = (
    pd.DataFrame(
        {
            "Actual House Value":
                y_test.iloc[
                    :10
                ].to_numpy(),

            "Predicted House Value":
                baseline_predictions[
                    :10
                ],
        }
    )
)

print(
    "\n10 baseline house-price predictions:"
)

print(
    baseline_prediction_examples.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------------
# Neural Network Experiment Utilities
# ---------------------------------------------------------------------------

def build_regression_model(
    hidden_layers,
    learning_rate,
    optimizer_name,
):
    """Build and compile a configurable DNN regression model."""

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    model = Sequential()

    model.add(
        Input(
            shape=(
                X_train_scaled.shape[1],
            )
        )
    )

    for neurons in hidden_layers:
        model.add(
            Dense(
                neurons,
                activation="relu",
            )
        )

    model.add(
        Dense(1)
    )

    if optimizer_name == "adam":
        optimizer = Adam(
            learning_rate=learning_rate
        )

    elif optimizer_name == "sgd":
        optimizer = SGD(
            learning_rate=learning_rate
        )

    else:
        raise ValueError(
            "Unsupported optimizer: "
            f"{optimizer_name}"
        )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )

    return model


def run_experiment(
    name,
    hidden_layers,
    batch_size,
    epochs,
    learning_rate,
    optimizer_name,
):
    """Train and evaluate one neural-network configuration."""

    print(
        f"\nRunning experiment: "
        f"{name}"
    )

    model = build_regression_model(
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
    )

    history = model.fit(
        X_train_scaled,
        y_train_nn,
        validation_split=0.20,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    # Convert validation metrics back into dollar-based units.
    final_val_mse = (
        history.history[
            "val_loss"
        ][-1]
        * TARGET_SCALE**2
    )

    final_val_mae = (
        history.history[
            "val_mae"
        ][-1]
        * TARGET_SCALE
    )

    predictions = (
        model.predict(
            X_test_scaled,
            verbose=0,
        ).reshape(-1)
        * TARGET_SCALE
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print(
        f"{name}: "
        f"MSE={mse:,.2f}, "
        f"MAE=${mae:,.2f}, "
        f"R2={r2:.4f}"
    )

    return {
        "Experiment":
            name,

        "Hidden Layers":
            len(hidden_layers),

        "Neurons":
            "-".join(
                map(
                    str,
                    hidden_layers,
                )
            ),

        "Batch Size":
            batch_size,

        "Epochs":
            epochs,

        "Learning Rate":
            learning_rate,

        "Optimizer":
            optimizer_name.upper(),

        "Validation MSE":
            final_val_mse,

        "Validation MAE":
            final_val_mae,

        "Test MSE":
            mse,

        "Test MAE":
            mae,

        "Test R2":
            r2,
    }


# ---------------------------------------------------------------------------
# Controlled Neural Network Experiments
# ---------------------------------------------------------------------------

experiment_results = [
    {
        "Experiment":
            "Baseline",

        "Hidden Layers":
            2,

        "Neurons":
            "32-16",

        "Batch Size":
            64,

        "Epochs":
            100,

        "Learning Rate":
            0.001,

        "Optimizer":
            "ADAM",

        "Validation MSE":
            baseline_history.history[
                "val_loss"
            ][-1]
            * TARGET_SCALE**2,

        "Validation MAE":
            baseline_history.history[
                "val_mae"
            ][-1]
            * TARGET_SCALE,

        "Test MSE":
            baseline_test_mse,

        "Test MAE":
            baseline_test_mae,

        "Test R2":
            baseline_test_r2,
    }
]


experiment_configs = [
    {
        "name":
            "1 Hidden Layer",

        "hidden_layers":
            [32],

        "batch_size":
            64,

        "epochs":
            100,

        "learning_rate":
            0.001,

        "optimizer_name":
            "adam",
    },
    {
        "name":
            "3 Hidden Layers",

        "hidden_layers":
            [50, 32, 16],

        "batch_size":
            64,

        "epochs":
            100,

        "learning_rate":
            0.001,

        "optimizer_name":
            "adam",
    },
    {
        "name":
            "4 Hidden Layers",

        "hidden_layers":
            [50, 40, 30, 20],

        "batch_size":
            64,

        "epochs":
            100,

        "learning_rate":
            0.001,

        "optimizer_name":
            "adam",
    },
    {
        "name":
            "5 Hidden Layers",

        "hidden_layers":
            [50, 40, 30, 20, 10],

        "batch_size":
            64,

        "epochs":
            100,

        "learning_rate":
            0.001,

        "optimizer_name":
            "adam",
    },
    {
        "name":
            "Batch Size 128",

        "hidden_layers":
            [32, 16],

        "batch_size":
            128,

        "epochs":
            100,

        "learning_rate":
            0.001,

        "optimizer_name":
            "adam",
    },
    {
        "name":
            "200 Epochs",

        "hidden_layers":
            [32, 16],

        "batch_size":
            64,

        "epochs":
            200,

        "learning_rate":
            0.001,

        "optimizer_name":
            "adam",
    },
    {
        "name":
            "400 Epochs",

        "hidden_layers":
            [32, 16],

        "batch_size":
            64,

        "epochs":
            400,

        "learning_rate":
            0.001,

        "optimizer_name":
            "adam",
    },
    {
        "name":
            "Learning Rate 0.0005",

        "hidden_layers":
            [32, 16],

        "batch_size":
            64,

        "epochs":
            100,

        "learning_rate":
            0.0005,

        "optimizer_name":
            "adam",
    },
    {
        "name":
            "Learning Rate 0.0001",

        "hidden_layers":
            [32, 16],

        "batch_size":
            64,

        "epochs":
            100,

        "learning_rate":
            0.0001,

        "optimizer_name":
            "adam",
    },
    {
        "name":
            "SGD Optimizer",

        "hidden_layers":
            [32, 16],

        "batch_size":
            64,

        "epochs":
            100,

        # Same learning rate as the Adam 0.0001 experiment,
        # allowing a controlled optimizer comparison.
        "learning_rate":
            0.0001,

        "optimizer_name":
            "sgd",
    },
    {
        "name":
            "10 Neurons Per Layer",

        "hidden_layers":
            [10, 10],

        "batch_size":
            64,

        "epochs":
            100,

        "learning_rate":
            0.001,

        "optimizer_name":
            "adam",
    },
    {
        "name":
            "50 Neurons Per Layer",

        "hidden_layers":
            [50, 50],

        "batch_size":
            64,

        "epochs":
            100,

        "learning_rate":
            0.001,

        "optimizer_name":
            "adam",
    },
]


for config in experiment_configs:
    result = run_experiment(
        **config
    )

    experiment_results.append(
        result
    )


# Convert all experiment results into one table.
experiment_results_df = pd.DataFrame(
    experiment_results
)

print(
    "\nNeural network experiment results:"
)

print(
    experiment_results_df.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:,.4f}"
        ),
    )
)

experiment_results_df.to_csv(
    OUTPUT_DIR
    / "dnn_experiment_results.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# Final Selected Neural Network Model
# ---------------------------------------------------------------------------

print(
    "\n--- Final Selected Neural Network Model ---"
)

# The final model is selected using validation MSE from the controlled
# experiments. The 50-32-16 architecture achieved the lowest validation
# MSE among the tested configurations.
final_model = build_regression_model(
    hidden_layers=[
        50,
        32,
        16,
    ],
    learning_rate=0.001,
    optimizer_name="adam",
)

print(
    "\nFinal model architecture:"
)

final_model.summary()

final_history = final_model.fit(
    X_train_scaled,
    y_train_nn,
    validation_split=0.20,
    epochs=100,
    batch_size=64,
    verbose=1,
)


# ---------------------------------------------------------------------------
# Final Model Evaluation
# ---------------------------------------------------------------------------

# Convert the network output back into house values in dollars.
final_predictions = (
    final_model.predict(
        X_test_scaled,
        verbose=0,
    ).reshape(-1)
    * TARGET_SCALE
)

final_mse = mean_squared_error(
    y_test,
    final_predictions,
)

final_mae = mean_absolute_error(
    y_test,
    final_predictions,
)

final_r2 = r2_score(
    y_test,
    final_predictions,
)

print(
    "\nFinal model test results:"
)

print(
    f"MSE: "
    f"{final_mse:,.2f}"
)

print(
    f"MAE: "
    f"${final_mae:,.2f}"
)

print(
    f"R2: "
    f"{final_r2:.4f}"
)


# ---------------------------------------------------------------------------
# Final Model Training Curves
# ---------------------------------------------------------------------------

final_epochs = range(
    1,
    len(
        final_history.history[
            "loss"
        ]
    )
    + 1,
)

final_training_mse = [
    value * TARGET_SCALE**2
    for value
    in final_history.history[
        "loss"
    ]
]

final_validation_mse = [
    value * TARGET_SCALE**2
    for value
    in final_history.history[
        "val_loss"
    ]
]

final_training_mae = [
    value * TARGET_SCALE
    for value
    in final_history.history[
        "mae"
    ]
]

final_validation_mae = [
    value * TARGET_SCALE
    for value
    in final_history.history[
        "val_mae"
    ]
]


# Final MSE curve.
plt.figure(figsize=(9, 6))

plt.plot(
    final_epochs,
    final_training_mse,
    label="Training MSE",
)

plt.plot(
    final_epochs,
    final_validation_mse,
    label="Validation MSE",
)

plt.xlabel("Epoch")
plt.ylabel(
    "Mean Squared Error"
)

plt.title(
    "Final DNN - MSE Training Curve"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_mse_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Final MAE curve.
plt.figure(figsize=(9, 6))

plt.plot(
    final_epochs,
    final_training_mae,
    label="Training MAE",
)

plt.plot(
    final_epochs,
    final_validation_mae,
    label="Validation MAE",
)

plt.xlabel("Epoch")
plt.ylabel(
    "Mean Absolute Error ($)"
)

plt.title(
    "Final DNN - MAE Training Curve"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_mae_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------------
# Final Actual vs Predicted House Values
# ---------------------------------------------------------------------------

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    final_predictions,
    alpha=0.4,
    s=20,
)

minimum_value = min(
    y_test.min(),
    final_predictions.min(),
)

maximum_value = max(
    y_test.max(),
    final_predictions.max(),
)

plt.plot(
    [
        minimum_value,
        maximum_value,
    ],
    [
        minimum_value,
        maximum_value,
    ],
    linestyle="--",
    label="Perfect prediction",
)

plt.xlabel(
    "Actual Median House Value ($)"
)

plt.ylabel(
    "Predicted Median House Value ($)"
)

plt.title(
    "Final DNN - Actual vs Predicted House Values"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "final_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------------
# Final Prediction Examples
# ---------------------------------------------------------------------------

prediction_results = pd.DataFrame(
    {
        "Actual House Value":
            y_test.reset_index(
                drop=True
            ),

        "Predicted House Value":
            final_predictions,
    }
)

# Select 10 reproducible examples from the test dataset.
final_prediction_examples = (
    prediction_results.sample(
        n=10,
        random_state=42,
    )
)

print(
    "\n10 example predictions from the final model:"
)

print(
    final_prediction_examples.to_string(
        index=False,
        formatters={
            "Actual House Value":
                lambda value: (
                    f"${value:,.2f}"
                ),

            "Predicted House Value":
                lambda value: (
                    f"${value:,.2f}"
                ),
        },
    )
)

final_prediction_examples.to_csv(
    OUTPUT_DIR
    / "final_prediction_examples.csv",
    index=False,
)