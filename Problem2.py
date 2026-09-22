"""
COIT29225 Neural Networks & Deep Learning
Assessment 2 - Problem 2

Olive Oil Neural Network Classification.

This program:
- Loads and inspects the supplied olive oil dataset.
- Corrects the malformed CSV header using explicit column names.
- Extracts the eight required fatty-acid features.
- Creates two olive-oil classes from locality information.
- Generates 50 additional Organic Extra Virgin samples using a
  Gaussian Mixture Model (GMM).
- Creates the complete augmented classification dataset.
- Performs Principal Component Analysis (PCA).
- Visualises the first two principal components for the two classes.
- Visualises the oleic fatty-acid distribution by class.
- Splits the augmented dataset into training, validation, and test sets.
- Trains and evaluates a baseline deep neural network classifier.
- Performs controlled neural-network parameter experiments.
- Selects the final classifier using validation performance.
- Evaluates the selected classifier on the test dataset.
- Reports training accuracy, test accuracy, precision, recall,
  F1-score, and support.
- Generates report-ready figures and CSV output files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = Path(
    "data/olive_original.csv"
)

OUTPUT_DIR = Path(
    "outputs/problem2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 42

NUMBER_OF_SYNTHETIC_SAMPLES = 50


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

# The supplied CSV contains one fewer header field than the number of values
# contained in each data row. Explicit column names are therefore assigned
# so that the values are read into the correct fields.

column_names = [
    "sample_id",
    "locality",
    "region_code",
    "area_code",
    "palmitic",
    "palmitoleic",
    "stearic",
    "oleic",
    "linoleic",
    "linolenic",
    "arachidic",
    "eicosenoic",
]


olive_data = pd.read_csv(
    DATA_FILE,
    skiprows=1,
    header=None,
    names=column_names,
)


# ---------------------------------------------------------------------------
# Initial Dataset Inspection
# ---------------------------------------------------------------------------

print(
    "First five rows of the olive oil dataset:"
)

print(
    olive_data.head()
)


print(
    "\nDataset shape:"
)

print(
    olive_data.shape
)


print(
    "\nColumn names:"
)

print(
    olive_data.columns.tolist()
)


print(
    "\nData types:"
)

print(
    olive_data.dtypes
)


print(
    "\nMissing values:"
)

print(
    olive_data.isnull().sum()
)


print(
    "\nDuplicate rows:"
)

print(
    olive_data.duplicated().sum()
)


print(
    "\nLocality distribution:"
)

print(
    olive_data[
        "locality"
    ].value_counts()
)


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

feature_columns = [
    "palmitic",
    "palmitoleic",
    "stearic",
    "oleic",
    "linoleic",
    "linolenic",
    "arachidic",
    "eicosenoic",
]


# ---------------------------------------------------------------------------
# Target-Class Creation
# ---------------------------------------------------------------------------

# Calabria and Sicily:
#   Organic Extra Virgin
#
# All remaining localities:
#   Extra Virgin

organic_localities = [
    "Calabria",
    "Sicily",
]


olive_data[
    "oil_class"
] = (
    olive_data[
        "locality"
    ]
    .isin(
        organic_localities
    )
    .map(
        {
            False: "Extra Virgin",
            True: "Organic Extra Virgin",
        }
    )
)


# ---------------------------------------------------------------------------
# Classification Dataset
# ---------------------------------------------------------------------------

classification_data = olive_data[
    feature_columns
    + [
        "oil_class"
    ]
].copy()


print(
    "\nClassification dataset shape:"
)

print(
    classification_data.shape
)


print(
    "\nClassification dataset:"
)

print(
    classification_data.head()
)


print(
    "\nClass distribution:"
)

print(
    classification_data[
        "oil_class"
    ].value_counts()
)


print(
    "\nClass percentages:"
)

print(
    classification_data[
        "oil_class"
    ]
    .value_counts(
        normalize=True
    )
    .mul(
        100
    )
    .round(
        2
    )
)


print(
    "\nFatty-acid descriptive statistics:"
)

print(
    classification_data[
        feature_columns
    ].describe()
)


# ---------------------------------------------------------------------------
# Gaussian Mixture Model Minority-Class Augmentation
# ---------------------------------------------------------------------------

# The Organic Extra Virgin class is the minority class.
# The assignment requires exactly 50 additional samples to be generated
# using a Gaussian Mixture Model.

organic_data = classification_data[
    classification_data[
        "oil_class"
    ]
    == "Organic Extra Virgin"
].copy()


organic_features = organic_data[
    feature_columns
].copy()


# Standardise before fitting the GMM because the fatty-acid variables have
# different numerical ranges.

gmm_scaler = StandardScaler()

organic_features_scaled = (
    gmm_scaler.fit_transform(
        organic_features
    )
)


# Two Gaussian components are used because the minority class contains
# observations originating from Calabria and Sicily.

gmm_model = GaussianMixture(
    n_components=2,
    covariance_type="full",
    random_state=RANDOM_SEED,
)

gmm_model.fit(
    organic_features_scaled
)


# Generate exactly 50 synthetic minority-class observations.

synthetic_scaled, _ = (
    gmm_model.sample(
        NUMBER_OF_SYNTHETIC_SAMPLES
    )
)


synthetic_features = (
    gmm_scaler.inverse_transform(
        synthetic_scaled
    )
)


# ---------------------------------------------------------------------------
# Synthetic Sample Range Control
# ---------------------------------------------------------------------------

# Gaussian distributions can generate observations outside the original
# observed range. Restrict each generated feature to the range found in the
# original Organic Extra Virgin data.

organic_minimums = (
    organic_features
    .min()
    .to_numpy()
)

organic_maximums = (
    organic_features
    .max()
    .to_numpy()
)


synthetic_features = np.clip(
    synthetic_features,
    organic_minimums,
    organic_maximums,
)


# The supplied dataset stores fatty-acid measurements as integers.

synthetic_features = (
    np.rint(
        synthetic_features
    )
    .astype(
        int
    )
)


# ---------------------------------------------------------------------------
# Synthetic Dataset
# ---------------------------------------------------------------------------

synthetic_data = pd.DataFrame(
    synthetic_features,
    columns=feature_columns,
)


synthetic_data[
    "oil_class"
] = (
    "Organic Extra Virgin"
)


print(
    "\nSynthetic Organic Extra Virgin samples generated:"
)

print(
    synthetic_data.shape
)


print(
    "\nFirst five synthetic samples:"
)

print(
    synthetic_data.head()
)


# ---------------------------------------------------------------------------
# Complete Augmented Dataset
# ---------------------------------------------------------------------------

# The assignment requires the 50 generated observations to be added to the
# dataset for training and testing purposes. The augmented dataset therefore
# becomes the dataset subsequently split for neural-network experiments.

augmented_data = pd.concat(
    [
        classification_data,
        synthetic_data,
    ],
    ignore_index=True,
)


print(
    "\nAugmented classification dataset shape:"
)

print(
    augmented_data.shape
)


print(
    "\nAugmented class distribution:"
)

print(
    augmented_data[
        "oil_class"
    ].value_counts()
)


print(
    "\nAugmented class percentages:"
)

print(
    augmented_data[
        "oil_class"
    ]
    .value_counts(
        normalize=True
    )
    .mul(
        100
    )
    .round(
        2
    )
)


print(
    "\nMissing values after GMM augmentation:"
)

print(
    augmented_data.isnull().sum()
)


print(
    "\nSynthetic fatty-acid descriptive statistics:"
)

print(
    synthetic_data[
        feature_columns
    ].describe()
)


augmented_data.to_csv(
    OUTPUT_DIR
    / "olive_classification_augmented.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# PCA Analysis
# ---------------------------------------------------------------------------

# Standardise all eight fatty-acid measurements before PCA.

pca_scaler = StandardScaler()

pca_features_scaled = (
    pca_scaler.fit_transform(
        augmented_data[
            feature_columns
        ]
    )
)


pca_model = PCA(
    n_components=2
)


pca_components = (
    pca_model.fit_transform(
        pca_features_scaled
    )
)


pca_results = pd.DataFrame(
    {
        "PC1":
            pca_components[
                :,
                0
            ],

        "PC2":
            pca_components[
                :,
                1
            ],

        "oil_class":
            augmented_data[
                "oil_class"
            ]
            .reset_index(
                drop=True
            ),
    }
)


pc1_variance = (
    pca_model
    .explained_variance_ratio_[
        0
    ]
)

pc2_variance = (
    pca_model
    .explained_variance_ratio_[
        1
    ]
)

combined_variance = (
    pc1_variance
    + pc2_variance
)


print(
    "\nPCA explained variance ratio:"
)

print(
    f"PC1: "
    f"{pc1_variance:.4f}"
)

print(
    f"PC2: "
    f"{pc2_variance:.4f}"
)

print(
    f"Combined PC1 + PC2: "
    f"{combined_variance:.4f}"
)


# ---------------------------------------------------------------------------
# PCA Visualisation
# ---------------------------------------------------------------------------

plt.figure(
    figsize=(
        9,
        7,
    )
)


for oil_class in [
    "Extra Virgin",
    "Organic Extra Virgin",
]:
    class_mask = (
        pca_results[
            "oil_class"
        ]
        == oil_class
    )

    plt.scatter(
        pca_results.loc[
            class_mask,
            "PC1",
        ],
        pca_results.loc[
            class_mask,
            "PC2",
        ],
        alpha=0.65,
        s=35,
        label=oil_class,
    )


plt.xlabel(
    f"Principal Component 1 "
    f"({pc1_variance * 100:.2f}% variance)"
)

plt.ylabel(
    f"Principal Component 2 "
    f"({pc2_variance * 100:.2f}% variance)"
)

plt.title(
    "Olive Oil PCA - First Two Principal Components"
)

plt.legend(
    title="Oil Class"
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "olive_pca_two_classes.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------------
# Oleic Distribution by Class
# ---------------------------------------------------------------------------

extra_virgin_oleic = (
    augmented_data.loc[
        augmented_data[
            "oil_class"
        ]
        == "Extra Virgin",
        "oleic",
    ]
)


organic_oleic = (
    augmented_data.loc[
        augmented_data[
            "oil_class"
        ]
        == "Organic Extra Virgin",
        "oleic",
    ]
)


# Use common histogram boundaries so both classes are directly comparable.

oleic_bins = np.linspace(
    augmented_data[
        "oleic"
    ].min(),
    augmented_data[
        "oleic"
    ].max(),
    21,
)


plt.figure(
    figsize=(
        9,
        6,
    )
)


plt.hist(
    extra_virgin_oleic,
    bins=oleic_bins,
    alpha=0.50,
    density=True,
    label="Extra Virgin",
)


plt.hist(
    organic_oleic,
    bins=oleic_bins,
    alpha=0.50,
    density=True,
    label="Organic Extra Virgin",
)


# Add outline curves to make the class distributions easier to compare.

plt.hist(
    extra_virgin_oleic,
    bins=oleic_bins,
    density=True,
    histtype="step",
    linewidth=2,
)

plt.hist(
    organic_oleic,
    bins=oleic_bins,
    density=True,
    histtype="step",
    linewidth=2,
)


plt.xlabel(
    "Oleic Fatty-Acid Measurement"
)

plt.ylabel(
    "Density"
)

plt.title(
    "Oleic Distribution by Olive Oil Class"
)

plt.legend(
    title="Oil Class"
)

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "oleic_distribution_by_class.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "\nGenerated data-analysis outputs:"
)

print(
    OUTPUT_DIR
    / "olive_classification_augmented.csv"
)

print(
    OUTPUT_DIR
    / "olive_pca_two_classes.png"
)

print(
    OUTPUT_DIR
    / "oleic_distribution_by_class.png"
)


# ---------------------------------------------------------------------------
# Neural Network Data Preparation
# ---------------------------------------------------------------------------

# Binary target encoding:
#
# 0 = Extra Virgin
# 1 = Organic Extra Virgin

class_mapping = {
    "Extra Virgin": 0,
    "Organic Extra Virgin": 1,
}


classification_model_data = (
    augmented_data.copy()
)


classification_model_data[
    "target"
] = (
    classification_model_data[
        "oil_class"
    ]
    .map(
        class_mapping
    )
)


X_all = classification_model_data[
    feature_columns
]

y_all = classification_model_data[
    "target"
]


# ---------------------------------------------------------------------------
# Train / Validation / Test Split
# ---------------------------------------------------------------------------

# Reserve 20% of the complete augmented dataset for testing.

X_train_validation, X_test, y_train_validation, y_test = (
    train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_all,
    )
)


# Use 20% of the remaining data as validation data.
#
# Approximate total proportions:
#
# 64% training
# 16% validation
# 20% testing

X_train_augmented, X_validation, y_train_augmented, y_validation = (
    train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_train_validation,
    )
)


print(
    "\nNeural network augmented-data split:"
)

print(
    f"Training samples: "
    f"{len(X_train_augmented)}"
)

print(
    f"Validation samples: "
    f"{len(X_validation)}"
)

print(
    f"Testing samples: "
    f"{len(X_test)}"
)


print(
    "\nTraining class distribution:"
)

print(
    y_train_augmented
    .value_counts()
    .sort_index()
)


print(
    "\nValidation class distribution:"
)

print(
    y_validation
    .value_counts()
    .sort_index()
)


print(
    "\nTest class distribution:"
)

print(
    y_test
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------------------------

# Fit the neural-network scaler using only the training partition.

nn_scaler = StandardScaler()


X_train_scaled = (
    nn_scaler.fit_transform(
        X_train_augmented
    )
)


X_validation_scaled = (
    nn_scaler.transform(
        X_validation
    )
)


X_test_scaled = (
    nn_scaler.transform(
        X_test
    )
)


print(
    "\nNumber of neural network input features:"
)

print(
    X_train_scaled.shape[1]
)


# ---------------------------------------------------------------------------
# Neural Network Utility Functions
# ---------------------------------------------------------------------------

def build_olive_classifier(
    hidden_layers,
    learning_rate,
    optimizer_name,
):
    """Build and compile a configurable olive oil classifier."""

    tf.keras.backend.clear_session()

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


    for neurons in hidden_layers:
        model.add(
            Dense(
                neurons,
                activation="relu",
            )
        )


    model.add(
        Dense(
            1,
            activation="sigmoid",
        )
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
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ],
    )


    return model


# ---------------------------------------------------------------------------
# Baseline Neural Network Classifier
# ---------------------------------------------------------------------------

baseline_config = {
    "hidden_layers": [
        32,
        16,
    ],
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.001,
    "optimizer_name": "adam",
}


baseline_classifier = build_olive_classifier(
    hidden_layers=baseline_config[
        "hidden_layers"
    ],
    learning_rate=baseline_config[
        "learning_rate"
    ],
    optimizer_name=baseline_config[
        "optimizer_name"
    ],
)


print(
    "\nBaseline olive oil classifier architecture:"
)

baseline_classifier.summary()


print(
    "\nTraining baseline olive oil classifier..."
)


baseline_history = baseline_classifier.fit(
    X_train_scaled,
    y_train_augmented,
    validation_data=(
        X_validation_scaled,
        y_validation,
    ),
    epochs=baseline_config[
        "epochs"
    ],
    batch_size=baseline_config[
        "batch_size"
    ],
    verbose=1,
)


# ---------------------------------------------------------------------------
# Baseline Evaluation
# ---------------------------------------------------------------------------

baseline_training_probability = (
    baseline_classifier.predict(
        X_train_scaled,
        verbose=0,
    )
    .reshape(-1)
)


baseline_test_probability = (
    baseline_classifier.predict(
        X_test_scaled,
        verbose=0,
    )
    .reshape(-1)
)


baseline_training_predictions = (
    baseline_training_probability
    >= 0.5
).astype(
    int
)


baseline_test_predictions = (
    baseline_test_probability
    >= 0.5
).astype(
    int
)


baseline_training_accuracy = accuracy_score(
    y_train_augmented,
    baseline_training_predictions,
)


baseline_test_accuracy = accuracy_score(
    y_test,
    baseline_test_predictions,
)


print(
    "\nBaseline olive oil classifier results:"
)

print(
    f"Training accuracy: "
    f"{baseline_training_accuracy:.4f}"
)

print(
    f"Test accuracy: "
    f"{baseline_test_accuracy:.4f}"
)


# ---------------------------------------------------------------------------
# Baseline Classification Report
# ---------------------------------------------------------------------------

class_names = [
    "Extra Virgin",
    "Organic Extra Virgin",
]


baseline_report_text = classification_report(
    y_test,
    baseline_test_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0,
)


print(
    "\nBaseline classification report:"
)

print(
    baseline_report_text
)


baseline_report_dict = classification_report(
    y_test,
    baseline_test_predictions,
    target_names=class_names,
    digits=4,
    output_dict=True,
    zero_division=0,
)


baseline_report_df = pd.DataFrame(
    baseline_report_dict
).transpose()


baseline_report_df.to_csv(
    OUTPUT_DIR
    / "baseline_classification_report.csv",
    index_label="Class",
)


# ---------------------------------------------------------------------------
# Baseline Training Curves
# ---------------------------------------------------------------------------

baseline_epochs = range(
    1,
    len(
        baseline_history.history[
            "accuracy"
        ]
    )
    + 1,
)


plt.figure(
    figsize=(
        9,
        6,
    )
)


plt.plot(
    baseline_epochs,
    baseline_history.history[
        "accuracy"
    ],
    label="Training Accuracy",
)


plt.plot(
    baseline_epochs,
    baseline_history.history[
        "val_accuracy"
    ],
    label="Validation Accuracy",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Olive Oil Baseline DNN - Accuracy Curve"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "baseline_accuracy_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


plt.figure(
    figsize=(
        9,
        6,
    )
)


plt.plot(
    baseline_epochs,
    baseline_history.history[
        "loss"
    ],
    label="Training Loss",
)


plt.plot(
    baseline_epochs,
    baseline_history.history[
        "val_loss"
    ],
    label="Validation Loss",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Binary Cross-Entropy Loss"
)

plt.title(
    "Olive Oil Baseline DNN - Loss Curve"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "baseline_loss_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "\nGenerated baseline classifier outputs:"
)

print(
    OUTPUT_DIR
    / "baseline_accuracy_curve.png"
)

print(
    OUTPUT_DIR
    / "baseline_loss_curve.png"
)

print(
    OUTPUT_DIR
    / "baseline_classification_report.csv"
)


# ---------------------------------------------------------------------------
# Controlled Experiment Function
# ---------------------------------------------------------------------------

def run_olive_experiment(
    name,
    hidden_layers,
    batch_size,
    epochs,
    learning_rate,
    optimizer_name,
):
    """Train one controlled neural-network experiment."""

    print(
        f"\nRunning experiment: "
        f"{name}"
    )


    model = build_olive_classifier(
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
    )


    history = model.fit(
        X_train_scaled,
        y_train_augmented,
        validation_data=(
            X_validation_scaled,
            y_validation,
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )


    training_accuracy = (
        history.history[
            "accuracy"
        ][-1]
    )


    validation_losses = np.array(
        history.history[
            "val_loss"
        ]
    )


    validation_accuracies = np.array(
        history.history[
            "val_accuracy"
        ]
    )


    best_epoch_index = int(
        np.argmin(
            validation_losses
        )
    )


    best_epoch = (
        best_epoch_index
        + 1
    )


    minimum_validation_loss = (
        validation_losses[
            best_epoch_index
        ]
    )


    validation_accuracy_at_best_loss = (
        validation_accuracies[
            best_epoch_index
        ]
    )


    maximum_validation_accuracy = (
        validation_accuracies.max()
    )


    print(
        f"{name}: "
        f"Training Accuracy="
        f"{training_accuracy:.4f}, "
        f"Minimum Validation Loss="
        f"{minimum_validation_loss:.4f}, "
        f"Best Epoch="
        f"{best_epoch}, "
        f"Validation Accuracy at Best Loss="
        f"{validation_accuracy_at_best_loss:.4f}, "
        f"Maximum Validation Accuracy="
        f"{maximum_validation_accuracy:.4f}"
    )


    return {
        "Experiment":
            name,

        "Hidden Layers":
            len(
                hidden_layers
            ),

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

        "Training Accuracy":
            training_accuracy,

        "Minimum Validation Loss":
            minimum_validation_loss,

        "Best Epoch":
            best_epoch,

        "Validation Accuracy at Best Loss":
            validation_accuracy_at_best_loss,

        "Maximum Validation Accuracy":
            maximum_validation_accuracy,
    }


# ---------------------------------------------------------------------------
# Baseline Experiment Record
# ---------------------------------------------------------------------------

baseline_validation_losses = np.array(
    baseline_history.history[
        "val_loss"
    ]
)


baseline_validation_accuracies = np.array(
    baseline_history.history[
        "val_accuracy"
    ]
)


baseline_best_epoch_index = int(
    np.argmin(
        baseline_validation_losses
    )
)


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

        "Training Accuracy":
            baseline_history.history[
                "accuracy"
            ][-1],

        "Minimum Validation Loss":
            baseline_validation_losses[
                baseline_best_epoch_index
            ],

        "Best Epoch":
            baseline_best_epoch_index
            + 1,

        "Validation Accuracy at Best Loss":
            baseline_validation_accuracies[
                baseline_best_epoch_index
            ],

        "Maximum Validation Accuracy":
            baseline_validation_accuracies.max(),
    }
]


# ---------------------------------------------------------------------------
# Controlled Experiment Configurations
# ---------------------------------------------------------------------------

experiment_configs = [
    {
        "name":
            "1 Hidden Layer",

        "hidden_layers":
            [
                32,
            ],

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
            [
                50,
                32,
                16,
            ],

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
            [
                50,
                40,
                30,
                20,
            ],

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
            [
                50,
                40,
                30,
                20,
                10,
            ],

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
            [
                32,
                16,
            ],

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
            [
                32,
                16,
            ],

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
            [
                32,
                16,
            ],

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
            [
                32,
                16,
            ],

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
            [
                32,
                16,
            ],

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
            [
                32,
                16,
            ],

        "batch_size":
            64,

        "epochs":
            100,

        "learning_rate":
            0.001,

        "optimizer_name":
            "sgd",
    },

    {
        "name":
            "10 Neurons Per Layer",

        "hidden_layers":
            [
                10,
                10,
            ],

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
            [
                50,
                50,
            ],

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


# ---------------------------------------------------------------------------
# Configuration Lookup
# ---------------------------------------------------------------------------

# This allows the final selected model to be built dynamically rather than
# hard-coding a particular experiment before the new experiment results are
# known.

experiment_config_lookup = {
    "Baseline": {
        "hidden_layers":
            [
                32,
                16,
            ],

        "batch_size":
            64,

        "epochs":
            100,

        "learning_rate":
            0.001,

        "optimizer_name":
            "adam",
    }
}


for config in experiment_configs:
    experiment_config_lookup[
        config[
            "name"
        ]
    ] = config


# ---------------------------------------------------------------------------
# Run Controlled Experiments
# ---------------------------------------------------------------------------

for config in experiment_configs:
    result = run_olive_experiment(
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
    "\nOlive oil neural network experiment results:"
)


print(
    experiment_results_df.to_string(
        index=False,
        float_format=lambda value: (
            f"{value:.4f}"
        ),
    )
)


experiment_results_df.to_csv(
    OUTPUT_DIR
    / "olive_dnn_experiment_results.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# Validation-Based Final Model Selection
# ---------------------------------------------------------------------------

# Select the final configuration using validation performance only.

best_experiment_index = (
    experiment_results_df[
        "Minimum Validation Loss"
    ]
    .idxmin()
)


best_experiment = (
    experiment_results_df.loc[
        best_experiment_index
    ]
)


best_experiment_name = (
    best_experiment[
        "Experiment"
    ]
)


best_config = (
    experiment_config_lookup[
        best_experiment_name
    ]
)


best_epoch = int(
    best_experiment[
        "Best Epoch"
    ]
)


print(
    "\nBest experiment based on minimum validation loss:"
)

print(
    best_experiment.to_string()
)


print(
    "\nSelected final configuration:"
)

print(
    f"Experiment: "
    f"{best_experiment_name}"
)

print(
    f"Hidden layers: "
    f"{best_config['hidden_layers']}"
)

print(
    f"Batch size: "
    f"{best_config['batch_size']}"
)

print(
    f"Learning rate: "
    f"{best_config['learning_rate']}"
)

print(
    f"Optimizer: "
    f"{best_config['optimizer_name'].upper()}"
)

print(
    f"Selected epochs: "
    f"{best_epoch}"
)


print(
    "\nExperiment results saved to:"
)

print(
    OUTPUT_DIR
    / "olive_dnn_experiment_results.csv"
)


# ---------------------------------------------------------------------------
# Final Selected Neural Network Classifier
# ---------------------------------------------------------------------------

print(
    "\n--- Final Selected Olive Oil Neural Network Classifier ---"
)


final_classifier = build_olive_classifier(
    hidden_layers=best_config[
        "hidden_layers"
    ],
    learning_rate=best_config[
        "learning_rate"
    ],
    optimizer_name=best_config[
        "optimizer_name"
    ],
)


print(
    "\nFinal olive oil classifier architecture:"
)

final_classifier.summary()


# ---------------------------------------------------------------------------
# Final Classifier Training
# ---------------------------------------------------------------------------

final_history = final_classifier.fit(
    X_train_scaled,
    y_train_augmented,
    validation_data=(
        X_validation_scaled,
        y_validation,
    ),
    epochs=best_epoch,
    batch_size=best_config[
        "batch_size"
    ],
    verbose=1,
)


# ---------------------------------------------------------------------------
# Final Classifier Evaluation
# ---------------------------------------------------------------------------

final_training_probability = (
    final_classifier.predict(
        X_train_scaled,
        verbose=0,
    )
    .reshape(-1)
)


final_test_probability = (
    final_classifier.predict(
        X_test_scaled,
        verbose=0,
    )
    .reshape(-1)
)


final_training_predictions = (
    final_training_probability
    >= 0.5
).astype(
    int
)


final_test_predictions = (
    final_test_probability
    >= 0.5
).astype(
    int
)


final_training_accuracy = accuracy_score(
    y_train_augmented,
    final_training_predictions,
)


final_test_accuracy = accuracy_score(
    y_test,
    final_test_predictions,
)


print(
    "\nFinal olive oil classifier results:"
)

print(
    f"Training accuracy: "
    f"{final_training_accuracy:.4f}"
)

print(
    f"Test accuracy: "
    f"{final_test_accuracy:.4f}"
)


# ---------------------------------------------------------------------------
# Final Classification Report
# ---------------------------------------------------------------------------

final_report_text = classification_report(
    y_test,
    final_test_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0,
)


print(
    "\nFinal classification report:"
)

print(
    final_report_text
)


final_report_dict = classification_report(
    y_test,
    final_test_predictions,
    target_names=class_names,
    digits=4,
    output_dict=True,
    zero_division=0,
)


final_report_df = pd.DataFrame(
    final_report_dict
).transpose()


final_report_df.to_csv(
    OUTPUT_DIR
    / "final_classification_report.csv",
    index_label="Class",
)


# ---------------------------------------------------------------------------
# Final Accuracy Curve
# ---------------------------------------------------------------------------

final_epochs = range(
    1,
    len(
        final_history.history[
            "accuracy"
        ]
    )
    + 1,
)


plt.figure(
    figsize=(
        9,
        6,
    )
)


plt.plot(
    final_epochs,
    final_history.history[
        "accuracy"
    ],
    label="Training Accuracy",
)


plt.plot(
    final_epochs,
    final_history.history[
        "val_accuracy"
    ],
    label="Validation Accuracy",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Olive Oil Final DNN - Accuracy Curve"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "final_accuracy_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------------
# Final Loss Curve
# ---------------------------------------------------------------------------

plt.figure(
    figsize=(
        9,
        6,
    )
)


plt.plot(
    final_epochs,
    final_history.history[
        "loss"
    ],
    label="Training Loss",
)


plt.plot(
    final_epochs,
    final_history.history[
        "val_loss"
    ],
    label="Validation Loss",
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Binary Cross-Entropy Loss"
)

plt.title(
    "Olive Oil Final DNN - Loss Curve"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "final_loss_curve.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------------
# Final Prediction Examples
# ---------------------------------------------------------------------------

final_prediction_results = pd.DataFrame(
    {
        "Actual Label":
            y_test.reset_index(
                drop=True
            ),

        "Predicted Label":
            final_test_predictions,

        "Organic Probability":
            final_test_probability,
    }
)


label_names = {
    0: "Extra Virgin",
    1: "Organic Extra Virgin",
}


# Select five examples from each actual class.

extra_examples = (
    final_prediction_results[
        final_prediction_results[
            "Actual Label"
        ]
        == 0
    ]
    .sample(
        n=5,
        random_state=RANDOM_SEED,
    )
)


organic_examples = (
    final_prediction_results[
        final_prediction_results[
            "Actual Label"
        ]
        == 1
    ]
    .sample(
        n=5,
        random_state=RANDOM_SEED,
    )
)


final_prediction_examples = pd.concat(
    [
        extra_examples,
        organic_examples,
    ],
    ignore_index=True,
)


final_prediction_examples[
    "Actual Label"
] = (
    final_prediction_examples[
        "Actual Label"
    ]
    .map(
        label_names
    )
)


final_prediction_examples[
    "Predicted Label"
] = (
    final_prediction_examples[
        "Predicted Label"
    ]
    .map(
        label_names
    )
)


print(
    "\n10 example predictions from the final olive oil classifier:"
)


print(
    final_prediction_examples.to_string(
        index=False,
        formatters={
            "Organic Probability":
                lambda value: (
                    f"{value:.4f}"
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
# Final Output Summary
# ---------------------------------------------------------------------------

print(
    "\nGenerated final Problem 2 outputs:"
)

print(
    OUTPUT_DIR
    / "olive_classification_augmented.csv"
)

print(
    OUTPUT_DIR
    / "olive_pca_two_classes.png"
)

print(
    OUTPUT_DIR
    / "oleic_distribution_by_class.png"
)

print(
    OUTPUT_DIR
    / "baseline_accuracy_curve.png"
)

print(
    OUTPUT_DIR
    / "baseline_loss_curve.png"
)

print(
    OUTPUT_DIR
    / "baseline_classification_report.csv"
)

print(
    OUTPUT_DIR
    / "olive_dnn_experiment_results.csv"
)

print(
    OUTPUT_DIR
    / "final_accuracy_curve.png"
)

print(
    OUTPUT_DIR
    / "final_loss_curve.png"
)

print(
    OUTPUT_DIR
    / "final_classification_report.csv"
)

print(
    OUTPUT_DIR
    / "final_prediction_examples.csv"
)