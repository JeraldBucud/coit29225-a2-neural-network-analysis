"""
COIT29225 Neural Networks & Deep Learning
Assessment 2 - Problem 1, Part 2

Custom Brisbane House Price Dataset and Neural Network Regression.

This program:
- Generates a customised synthetic Brisbane house price dataset.
- Defines 13 property features labelled x1 to x13.
- Represents property location using Cartesian coordinates relative to Brisbane CBD.
- Models Australian property characteristics such as property type, floor area,
  transport accessibility, flood risk, shopping accessibility, body corporate
  fees, solar capacity, outdoor area, and walkability.
- Generates synthetic house prices using a combination of linear, nonlinear,
  property-type, location, and random market effects.
- Saves the generated dataset as BrisbaneHousePrice.csv.
- Performs dataset quality and distribution checks.
- Performs K-means clustering using location and house price information.
- Prepares the customised dataset for neural network regression.
- Trains and evaluates a baseline deep neural network.
- Performs controlled neural network parameter experiments.
- Compares hidden layers, neuron counts, batch sizes, epochs, learning rates,
  and optimizers.
- Selects a final neural network using validation performance.
- Evaluates the selected model on the test dataset using MSE, MAE, and R2.
- Produces training-error and actual-versus-predicted visualisations.
- Generates 10 example house price predictions.
- Performs permutation feature importance analysis on the final model.
- Allows a user to enter property feature values and receive a predicted
  Brisbane house price.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.cluster import KMeans
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

OUTPUT_FILE = Path("data/BrisbaneHousePrice.csv")
OUTPUT_DIR = Path("outputs/problem1_part2")

NUMBER_OF_PROPERTIES = 2000
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

rng = np.random.default_rng(
    RANDOM_SEED
)


# ---------------------------------------------------------------------------
# Location Variables
# ---------------------------------------------------------------------------

# Generate approximate property locations around Brisbane CBD.
#
# x1:
#   Negative = west of Brisbane CBD
#   Positive = east of Brisbane CBD
#
# x2:
#   Negative = south of Brisbane CBD
#   Positive = north of Brisbane CBD
#
# Locations are represented using Cartesian coordinates measured in kilometres.

distance_from_cbd = rng.uniform(
    1.0,
    35.0,
    NUMBER_OF_PROPERTIES,
)

location_angle = rng.uniform(
    0.0,
    2.0 * np.pi,
    NUMBER_OF_PROPERTIES,
)

x1 = (
    distance_from_cbd
    * np.cos(location_angle)
)

x2 = (
    distance_from_cbd
    * np.sin(location_angle)
)


# ---------------------------------------------------------------------------
# Property Type
# ---------------------------------------------------------------------------

# Property type codes:
# 0 = Apartment
# 1 = Townhouse
# 2 = Detached house

x3 = rng.choice(
    [0, 1, 2],
    size=NUMBER_OF_PROPERTIES,
    p=[0.35, 0.25, 0.40],
)


# ---------------------------------------------------------------------------
# Internal Floor Area
# ---------------------------------------------------------------------------

# Internal floor area differs by property type.

apartment_area = rng.normal(
    85,
    25,
    NUMBER_OF_PROPERTIES,
)

townhouse_area = rng.normal(
    140,
    30,
    NUMBER_OF_PROPERTIES,
)

house_area = rng.normal(
    210,
    60,
    NUMBER_OF_PROPERTIES,
)

x4 = np.where(
    x3 == 0,
    apartment_area,
    np.where(
        x3 == 1,
        townhouse_area,
        house_area,
    ),
)

x4 = np.clip(
    x4,
    40,
    450,
)


# ---------------------------------------------------------------------------
# Bathrooms
# ---------------------------------------------------------------------------

x5 = np.where(
    x3 == 0,
    rng.choice(
        [1, 2],
        NUMBER_OF_PROPERTIES,
        p=[0.65, 0.35],
    ),
    np.where(
        x3 == 1,
        rng.choice(
            [1, 2, 3],
            NUMBER_OF_PROPERTIES,
            p=[0.20, 0.65, 0.15],
        ),
        rng.choice(
            [1, 2, 3, 4],
            NUMBER_OF_PROPERTIES,
            p=[0.10, 0.45, 0.35, 0.10],
        ),
    ),
)


# ---------------------------------------------------------------------------
# Car Parking Spaces
# ---------------------------------------------------------------------------

x6 = np.where(
    x3 == 0,
    rng.choice(
        [0, 1, 2],
        NUMBER_OF_PROPERTIES,
        p=[0.20, 0.65, 0.15],
    ),
    np.where(
        x3 == 1,
        rng.choice(
            [1, 2],
            NUMBER_OF_PROPERTIES,
            p=[0.55, 0.45],
        ),
        rng.choice(
            [1, 2, 3],
            NUMBER_OF_PROPERTIES,
            p=[0.20, 0.60, 0.20],
        ),
    ),
)


# ---------------------------------------------------------------------------
# Public Transport Accessibility
# ---------------------------------------------------------------------------

# Properties closer to the CBD are generally assigned stronger transport
# accessibility, with random variation added to prevent a deterministic pattern.

x7 = (
    10
    - distance_from_cbd / 4
    + rng.normal(
        0,
        1.3,
        NUMBER_OF_PROPERTIES,
    )
)

x7 = np.clip(
    x7,
    0,
    10,
)


# ---------------------------------------------------------------------------
# Flood Risk
# ---------------------------------------------------------------------------

# Most properties receive low-to-moderate flood risk while a smaller number
# receive higher scores.

x8 = (
    rng.beta(
        2,
        5,
        NUMBER_OF_PROPERTIES,
    )
    * 10
)


# ---------------------------------------------------------------------------
# Shopping Accessibility
# ---------------------------------------------------------------------------

# Shopping accessibility generally decreases with distance from Brisbane CBD,
# while random variation represents suburban shopping centres.

x9 = (
    9
    - distance_from_cbd / 6
    + rng.normal(
        0,
        1.5,
        NUMBER_OF_PROPERTIES,
    )
)

x9 = np.clip(
    x9,
    0,
    10,
)


# ---------------------------------------------------------------------------
# Body Corporate Fees
# ---------------------------------------------------------------------------

# Detached houses generally have no body-corporate fee.
# Apartments typically have the highest annual fees.

apartment_body_corporate = rng.normal(
    5500,
    1800,
    NUMBER_OF_PROPERTIES,
)

townhouse_body_corporate = rng.normal(
    3000,
    1200,
    NUMBER_OF_PROPERTIES,
)

x10 = np.where(
    x3 == 0,
    apartment_body_corporate,
    np.where(
        x3 == 1,
        townhouse_body_corporate,
        0,
    ),
)

x10 = np.clip(
    x10,
    0,
    12000,
)


# ---------------------------------------------------------------------------
# Solar Capacity
# ---------------------------------------------------------------------------

# Detached houses are more likely to support larger rooftop solar systems.

x11 = np.where(
    x3 == 0,
    rng.uniform(
        0,
        3,
        NUMBER_OF_PROPERTIES,
    ),
    np.where(
        x3 == 1,
        rng.uniform(
            0,
            6,
            NUMBER_OF_PROPERTIES,
        ),
        rng.uniform(
            0,
            13,
            NUMBER_OF_PROPERTIES,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Private Outdoor Area
# ---------------------------------------------------------------------------

x12 = np.where(
    x3 == 0,
    rng.uniform(
        5,
        40,
        NUMBER_OF_PROPERTIES,
    ),
    np.where(
        x3 == 1,
        rng.uniform(
            20,
            150,
            NUMBER_OF_PROPERTIES,
        ),
        rng.uniform(
            80,
            900,
            NUMBER_OF_PROPERTIES,
        ),
    ),
)


# ---------------------------------------------------------------------------
# Walkability
# ---------------------------------------------------------------------------

# Walkability tends to be higher closer to the CBD but includes substantial
# local variation.

x13 = (
    10
    - distance_from_cbd / 5
    + rng.normal(
        0,
        1.5,
        NUMBER_OF_PROPERTIES,
    )
)

x13 = np.clip(
    x13,
    0,
    10,
)


# ---------------------------------------------------------------------------
# Synthetic House Price Model
# ---------------------------------------------------------------------------

# Property-type adjustments represent broad market differences.
property_type_adjustment = np.select(
    [
        x3 == 0,
        x3 == 1,
        x3 == 2,
    ],
    [
        0,
        90_000,
        180_000,
    ],
)


# Properties near Brisbane CBD receive a nonlinear location premium.
cbd_location_premium = (
    220_000
    * np.exp(
        -distance_from_cbd / 10
    )
)


# Random market variation prevents the target from being an exact deterministic
# function of the generated predictor variables.
market_noise = rng.normal(
    0,
    50_000,
    NUMBER_OF_PROPERTIES,
)


house_price = (
    180_000
    + property_type_adjustment
    + x4 * 3_200
    + x5 * 30_000
    + x6 * 35_000
    + x7 * 15_000
    - x8 * 18_000
    + x9 * 12_000
    - x10 * 5
    + x11 * 4_000
    + x12 * 300
    + x13 * 10_000
    + cbd_location_premium
    - distance_from_cbd * 3_000
    + market_noise
)


# Prevent unrealistic negative or extremely low property prices.
house_price = np.clip(
    house_price,
    250_000,
    3_000_000,
)


# ---------------------------------------------------------------------------
# Create Dataset
# ---------------------------------------------------------------------------

brisbane_data = pd.DataFrame(
    {
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x5": x5,
        "x6": x6,
        "x7": x7,
        "x8": x8,
        "x9": x9,
        "x10": x10,
        "x11": x11,
        "x12": x12,
        "x13": x13,
        "house_price": house_price,
    }
)


# Round continuous values to improve dataset readability.
brisbane_data = brisbane_data.round(
    {
        "x1": 2,
        "x2": 2,
        "x4": 2,
        "x7": 2,
        "x8": 2,
        "x9": 2,
        "x10": 2,
        "x11": 2,
        "x12": 2,
        "x13": 2,
        "house_price": 2,
    }
)


# ---------------------------------------------------------------------------
# Save Dataset
# ---------------------------------------------------------------------------

brisbane_data.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ---------------------------------------------------------------------------
# Initial Verification
# ---------------------------------------------------------------------------

print(
    "Brisbane House Price dataset created successfully."
)

print("\nDataset shape:")
print(
    brisbane_data.shape
)

print("\nFirst five rows:")
print(
    brisbane_data.head()
)

print("\nMissing values:")
print(
    brisbane_data.isnull().sum()
)

print("\nDescriptive statistics:")
print(
    brisbane_data.describe()
)

print(
    "\nDataset saved to:"
)

print(
    OUTPUT_FILE
)


# ---------------------------------------------------------------------------
# Dataset Quality Checks
# ---------------------------------------------------------------------------

print(
    "\nProperty type distribution:"
)

print(
    brisbane_data["x3"]
    .value_counts()
    .sort_index()
)


print(
    "\nHouse price summary by property type:"
)

print(
    brisbane_data.groupby(
        "x3"
    )["house_price"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)


print(
    "\nPrice clipping checks:"
)

minimum_price_count = (
    brisbane_data[
        "house_price"
    ]
    == 250_000
).sum()

maximum_price_count = (
    brisbane_data[
        "house_price"
    ]
    == 3_000_000
).sum()

print(
    f"Properties at $250,000 minimum: "
    f"{minimum_price_count}"
)

print(
    f"Properties at $3,000,000 maximum: "
    f"{maximum_price_count}"
)


# ---------------------------------------------------------------------------
# K-means Clustering Analysis
# ---------------------------------------------------------------------------

kmeans_columns = [
    "x1",
    "x2",
    "house_price",
]

kmeans_features = brisbane_data[
    kmeans_columns
]


# Standardise the variables so house price does not dominate the Euclidean
# distance calculation simply because it has a larger numerical scale.
kmeans_scaler = StandardScaler()

scaled_kmeans_features = (
    kmeans_scaler.fit_transform(
        kmeans_features
    )
)


kmeans_model = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10,
)

cluster_labels = (
    kmeans_model.fit_predict(
        scaled_kmeans_features
    )
)


print(
    "\nK-means cluster counts:"
)

print(
    pd.Series(
        cluster_labels
    )
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------------------------
# K-means Visualisation
# ---------------------------------------------------------------------------

plt.figure(
    figsize=(10, 8)
)

markers = [
    "o",
    "^",
    "s",
    "D",
    "X",
]

for cluster_id in range(5):
    cluster_mask = (
        cluster_labels
        == cluster_id
    )

    scatter = plt.scatter(
        brisbane_data.loc[
            cluster_mask,
            "x1",
        ],
        brisbane_data.loc[
            cluster_mask,
            "x2",
        ],
        c=brisbane_data.loc[
            cluster_mask,
            "house_price",
        ],
        cmap="viridis",
        vmin=brisbane_data[
            "house_price"
        ].min(),
        vmax=brisbane_data[
            "house_price"
        ].max(),
        marker=markers[
            cluster_id
        ],
        s=35,
        alpha=0.7,
        label=f"Cluster {cluster_id}",
    )


plt.axhline(
    y=0,
    linewidth=0.8,
)

plt.axvline(
    x=0,
    linewidth=0.8,
)

plt.xlabel(
    "East/West Distance from Brisbane CBD (km)"
)

plt.ylabel(
    "North/South Distance from Brisbane CBD (km)"
)

plt.title(
    "Brisbane House Price K-means Clustering"
)

colorbar = plt.colorbar(
    scatter
)

colorbar.set_label(
    "House Price ($)"
)

plt.legend(
    title="K-means Cluster"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "brisbane_kmeans_clusters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------------
# Neural Network Data Preparation
# ---------------------------------------------------------------------------

feature_columns = [
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "x7",
    "x8",
    "x9",
    "x10",
    "x11",
    "x12",
    "x13",
]

X = brisbane_data[
    feature_columns
]

y = brisbane_data[
    "house_price"
]


# Reserve 20% of the generated dataset for final model evaluation.
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )
)


# Fit the feature scaler only on training data to prevent test-data leakage.
nn_scaler = StandardScaler()

X_train_scaled = (
    nn_scaler.fit_transform(
        X_train
    )
)

X_test_scaled = (
    nn_scaler.transform(
        X_test
    )
)


# Brisbane property values are approximately measured in millions of dollars.
# Scaling the target improves numerical stability during neural-network
# training. Predictions are converted back into dollars for evaluation.
TARGET_SCALE = 1_000_000.0

y_train_scaled = (
    y_train
    / TARGET_SCALE
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

tf.keras.utils.set_random_seed(
    RANDOM_SEED
)

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
    "\nBaseline Brisbane neural network architecture:"
)

baseline_model.summary()


# ---------------------------------------------------------------------------
# Baseline Model Training
# ---------------------------------------------------------------------------

print(
    "\nTraining baseline Brisbane neural network..."
)

baseline_history = baseline_model.fit(
    X_train_scaled,
    y_train_scaled,
    validation_split=0.20,
    epochs=100,
    batch_size=64,
    verbose=1,
)


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


baseline_mse = mean_squared_error(
    y_test,
    baseline_predictions,
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions,
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions,
)


print(
    "\nBaseline Brisbane model test results:"
)

print(
    f"MSE: "
    f"{baseline_mse:,.2f}"
)

print(
    f"MAE: "
    f"${baseline_mae:,.2f}"
)

print(
    f"R2: "
    f"{baseline_r2:.4f}"
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


# Convert scaled metrics back into dollar-based units.
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
plt.figure(
    figsize=(9, 6)
)

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

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Mean Squared Error"
)

plt.title(
    "Brisbane Baseline DNN - MSE Training Curve"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "baseline_mse_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Baseline MAE curve.
plt.figure(
    figsize=(9, 6)
)

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

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Mean Absolute Error ($)"
)

plt.title(
    "Brisbane Baseline DNN - MAE Training Curve"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "baseline_mae_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------------
# Baseline Actual vs Predicted House Prices
# ---------------------------------------------------------------------------

plt.figure(
    figsize=(8, 8)
)

plt.scatter(
    y_test,
    baseline_predictions,
    alpha=0.5,
    s=25,
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
    "Actual Brisbane House Price ($)"
)

plt.ylabel(
    "Predicted Brisbane House Price ($)"
)

plt.title(
    "Brisbane Baseline DNN - Actual vs Predicted Prices"
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


# ---------------------------------------------------------------------------
# Baseline Prediction Examples
# ---------------------------------------------------------------------------

baseline_prediction_results = pd.DataFrame(
    {
        "Actual House Price":
            y_test.reset_index(
                drop=True
            ),
        "Predicted House Price":
            baseline_predictions,
    }
)

baseline_prediction_examples = (
    baseline_prediction_results.sample(
        n=10,
        random_state=42,
    )
)


print(
    "\n10 baseline Brisbane house-price predictions:"
)

print(
    baseline_prediction_examples.to_string(
        index=False,
        formatters={
            "Actual House Price":
                lambda value: (
                    f"${value:,.2f}"
                ),
            "Predicted House Price":
                lambda value: (
                    f"${value:,.2f}"
                ),
        },
    )
)


# ---------------------------------------------------------------------------
# Neural Network Experiment Utilities
# ---------------------------------------------------------------------------

def build_brisbane_regression_model(
    hidden_layers,
    learning_rate,
    optimizer_name,
):
    """Build and compile a configurable DNN for Brisbane price regression."""

    # Clear the previous Keras model before creating the next experiment.
    tf.keras.backend.clear_session()

    # Use a fixed seed so experiment results are reproducible.
    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    model = Sequential()

    model.add(
        Input(
            shape=(
                X_train_scaled.shape[1],
            )
        )
    )

    # Add the requested hidden-layer architecture.
    for neurons in hidden_layers:
        model.add(
            Dense(
                neurons,
                activation="relu",
            )
        )

    # Regression requires one continuous output value.
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
            f"Unsupported optimizer: "
            f"{optimizer_name}"
        )


    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )

    return model


def run_brisbane_experiment(
    name,
    hidden_layers,
    batch_size,
    epochs,
    learning_rate,
    optimizer_name,
):
    """Train and evaluate one Brisbane neural-network configuration."""

    print(
        f"\nRunning experiment: "
        f"{name}"
    )

    model = build_brisbane_regression_model(
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
    )

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_split=0.20,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )


    # Validation metrics are stored in scaled target units.
    # Convert them back into dollar-based values for reporting.
    validation_mse = (
        history.history[
            "val_loss"
        ][-1]
        * TARGET_SCALE**2
    )

    validation_mae = (
        history.history[
            "val_mae"
        ][-1]
        * TARGET_SCALE
    )


    # Convert model predictions back into dollars.
    predictions = (
        model.predict(
            X_test_scaled,
            verbose=0,
        ).reshape(-1)
        * TARGET_SCALE
    )

    test_mse = mean_squared_error(
        y_test,
        predictions,
    )

    test_mae = mean_absolute_error(
        y_test,
        predictions,
    )

    test_r2 = r2_score(
        y_test,
        predictions,
    )


    print(
        f"{name}: "
        f"MSE={test_mse:,.2f}, "
        f"MAE=${test_mae:,.2f}, "
        f"R2={test_r2:.4f}"
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
            validation_mse,

        "Validation MAE":
            validation_mae,

        "Test MSE":
            test_mse,

        "Test MAE":
            test_mae,

        "Test R2":
            test_r2,
    }


# ---------------------------------------------------------------------------
# Controlled Neural Network Experiments
# ---------------------------------------------------------------------------

# Include the existing baseline as the reference configuration.
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
            (
                baseline_history.history[
                    "val_loss"
                ][-1]
                * TARGET_SCALE**2
            ),

        "Validation MAE":
            (
                baseline_history.history[
                    "val_mae"
                ][-1]
                * TARGET_SCALE
            ),

        "Test MSE":
            baseline_mse,

        "Test MAE":
            baseline_mae,

        "Test R2":
            baseline_r2,
    }
]


experiment_configs = [
    # -----------------------------------------------------------------------
    # Hidden-layer experiments
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Batch-size experiment
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Epoch experiments
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Learning-rate experiments
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Optimizer experiment
    # -----------------------------------------------------------------------
    {
        "name":
            "SGD Optimizer",

        "hidden_layers":
            [32, 16],

        "batch_size":
            64,

        "epochs":
            100,

        # Use the same learning rate as the Adam 0.0001 experiment so the
        # optimizer itself is the main controlled difference.
        "learning_rate":
            0.0001,

        "optimizer_name":
            "sgd",
    },

    # -----------------------------------------------------------------------
    # Neuron-count experiments
    # -----------------------------------------------------------------------
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


# Run every controlled experiment.
for config in experiment_configs:
    result = run_brisbane_experiment(
        **config
    )

    experiment_results.append(
        result
    )


# ---------------------------------------------------------------------------
# Experiment Results Table
# ---------------------------------------------------------------------------

experiment_results_df = pd.DataFrame(
    experiment_results
)


print(
    "\nBrisbane neural network experiment results:"
)

print(
    experiment_results_df.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:,.4f}"
        ),
    )
)


# Save the results for the technical report.
experiment_results_df.to_csv(
    OUTPUT_DIR
    / "brisbane_dnn_experiment_results.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# Final Selected Brisbane Neural Network Model
# ---------------------------------------------------------------------------

print(
    "\n--- Final Selected Brisbane Neural Network Model ---"
)


# The final configuration is selected using validation MSE from the
# controlled experiments. The 32-16 architecture trained for 400 epochs
# achieved the lowest validation MSE among the tested configurations.
final_model = build_brisbane_regression_model(
    hidden_layers=[
        32,
        16,
    ],
    learning_rate=0.001,
    optimizer_name="adam",
)


print(
    "\nFinal Brisbane model architecture:"
)

final_model.summary()


final_history = final_model.fit(
    X_train_scaled,
    y_train_scaled,
    validation_split=0.20,
    epochs=400,
    batch_size=64,
    verbose=1,
)


# ---------------------------------------------------------------------------
# Final Model Evaluation
# ---------------------------------------------------------------------------

# Convert scaled model predictions back into Australian dollar values.
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
    "\nFinal Brisbane model test results:"
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


# Convert scaled metrics back into dollar-based units.
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
plt.figure(
    figsize=(9, 6)
)

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

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Mean Squared Error"
)

plt.title(
    "Brisbane Final DNN - MSE Training Curve"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "final_mse_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# Final MAE curve.
plt.figure(
    figsize=(9, 6)
)

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

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Mean Absolute Error ($)"
)

plt.title(
    "Brisbane Final DNN - MAE Training Curve"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "final_mae_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------------
# Final Actual vs Predicted Brisbane House Prices
# ---------------------------------------------------------------------------

plt.figure(
    figsize=(8, 8)
)

plt.scatter(
    y_test,
    final_predictions,
    alpha=0.5,
    s=25,
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
    "Actual Brisbane House Price ($)"
)

plt.ylabel(
    "Predicted Brisbane House Price ($)"
)

plt.title(
    "Brisbane Final DNN - Actual vs Predicted Prices"
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

final_prediction_results = pd.DataFrame(
    {
        "Actual House Price":
            y_test.reset_index(
                drop=True
            ),

        "Predicted House Price":
            final_predictions,
    }
)


final_prediction_examples = (
    final_prediction_results.sample(
        n=10,
        random_state=42,
    )
)


print(
    "\n10 example predictions from the final Brisbane model:"
)

print(
    final_prediction_examples.to_string(
        index=False,
        formatters={
            "Actual House Price":
                lambda value: (
                    f"${value:,.2f}"
                ),

            "Predicted House Price":
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


# ---------------------------------------------------------------------------
# Permutation Feature Importance
# ---------------------------------------------------------------------------

# Measure each feature's importance by randomly shuffling one test feature at
# a time and observing the resulting increase in test prediction error.
# A larger increase in MSE indicates that the trained model relies more
# strongly on that feature.

importance_rng = np.random.default_rng(
    RANDOM_SEED
)

feature_importance_results = []


for feature_index, feature_name in enumerate(
    feature_columns
):
    shuffled_test_data = (
        X_test_scaled.copy()
    )

    importance_rng.shuffle(
        shuffled_test_data[
            :,
            feature_index,
        ]
    )

    shuffled_predictions = (
        final_model.predict(
            shuffled_test_data,
            verbose=0,
        ).reshape(-1)
        * TARGET_SCALE
    )

    shuffled_mse = mean_squared_error(
        y_test,
        shuffled_predictions,
    )

    mse_increase = (
        shuffled_mse
        - final_mse
    )

    feature_importance_results.append(
        {
            "Feature":
                feature_name,

            "Shuffled MSE":
                shuffled_mse,

            "MSE Increase":
                mse_increase,
        }
    )


feature_importance_df = pd.DataFrame(
    feature_importance_results
).sort_values(
    "MSE Increase",
    ascending=False,
)


print(
    "\nPermutation feature importance:"
)

print(
    feature_importance_df.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:,.2f}"
        ),
    )
)


feature_importance_df.to_csv(
    OUTPUT_DIR
    / "feature_importance.csv",
    index=False,
)


# Visualise the relative importance of each feature.
plt.figure(
    figsize=(10, 7)
)

plt.barh(
    feature_importance_df[
        "Feature"
    ][::-1],
    feature_importance_df[
        "MSE Increase"
    ][::-1],
)

plt.xlabel(
    "Increase in Test MSE After Feature Shuffling"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Brisbane Final DNN - Permutation Feature Importance"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------------
# Interactive House Price Prediction
# ---------------------------------------------------------------------------

def predict_user_house_price():
    """Collect property features and predict a Brisbane house price."""

    print(
        "\n--- Brisbane House Price Predictor ---"
    )

    print(
        "\nEnter the property information below."
    )

    print(
        "For property type: "
        "0 = Apartment, "
        "1 = Townhouse, "
        "2 = Detached House"
    )


    try:
        user_values = [
            float(
                input(
                    "x1 - East/West distance from CBD (km): "
                )
            ),

            float(
                input(
                    "x2 - North/South distance from CBD (km): "
                )
            ),

            float(
                input(
                    "x3 - Property type (0, 1, or 2): "
                )
            ),

            float(
                input(
                    "x4 - Internal floor area (m²): "
                )
            ),

            float(
                input(
                    "x5 - Number of bathrooms: "
                )
            ),

            float(
                input(
                    "x6 - Number of car spaces: "
                )
            ),

            float(
                input(
                    "x7 - Public transport accessibility (0-10): "
                )
            ),

            float(
                input(
                    "x8 - Flood-risk score (0-10): "
                )
            ),

            float(
                input(
                    "x9 - Shopping accessibility (0-10): "
                )
            ),

            float(
                input(
                    "x10 - Annual body-corporate fee ($): "
                )
            ),

            float(
                input(
                    "x11 - Solar capacity (kW): "
                )
            ),

            float(
                input(
                    "x12 - Private outdoor area (m²): "
                )
            ),

            float(
                input(
                    "x13 - Walkability score (0-10): "
                )
            ),
        ]

    except ValueError:
        print(
            "\nInvalid input. "
            "Please enter numeric values only."
        )

        return


    # Preserve the same feature order used during model training.
    user_property = pd.DataFrame(
        [
            user_values
        ],
        columns=feature_columns,
    )


    # Apply the scaler fitted only on the training dataset.
    user_property_scaled = (
        nn_scaler.transform(
            user_property
        )
    )


    # Convert the scaled network output back into dollars.
    predicted_price = (
        final_model.predict(
            user_property_scaled,
            verbose=0,
        ).reshape(-1)[0]
        * TARGET_SCALE
    )


    print(
        "\nPredicted Brisbane house price:"
    )

    print(
        f"${predicted_price:,.2f}"
    )


# Ask whether the user wants to make an interactive prediction.
prediction_choice = input(
    "\nWould you like to predict a Brisbane house price? (y/n): "
).strip().lower()


if prediction_choice == "y":
    predict_user_house_price()

else:
    print(
        "\nInteractive prediction skipped."
    )