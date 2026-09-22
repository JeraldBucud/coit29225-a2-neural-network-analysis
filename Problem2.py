"""
COIT29225 Neural Networks & Deep Learning
Assessment 2 - Problem 2

Olive Oil Neural Network Classification.

This program:
- Loads and inspects the supplied olive oil dataset.
- Corrects the malformed CSV header by assigning explicit column names.
- Extracts the eight required fatty-acid features.
- Creates the two-class olive oil target from locality information.
- Checks the dataset for missing and duplicate records.
- Examines the class imbalance before synthetic sample generation.
- Generates 50 additional Organic Extra Virgin samples using a
  Gaussian Mixture Model (GMM).
- Creates an augmented olive oil classification dataset.
- Performs Principal Component Analysis (PCA).
- Visualises the first two principal components by olive oil class.
- Visualises the oleic fatty-acid distribution for the two classes.
- Saves generated datasets and report-ready visualisations.
- Prepares the analysis for later neural network classification.
"""

import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


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

# The supplied CSV header contains one fewer field than the data rows.
# Explicit column names are assigned so Pandas does not shift the values
# into incorrect columns.

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
    ]
    .value_counts()
)


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

# The assignment requires the eight fatty-acid variables to be used as
# predictor features.

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

# According to the assignment specification:
#
# Calabria and Sicily:
#   Organic Extra Virgin Olive Oil
#
# All other localities:
#   Extra Virgin Olive Oil

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
            False:
                "Extra Virgin",

            True:
                "Organic Extra Virgin",
        }
    )
)


# ---------------------------------------------------------------------------
# Final Classification Dataset
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
    ]
    .value_counts()
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
    ]
    .describe()
)


# ---------------------------------------------------------------------------
# Gaussian Mixture Model Minority-Class Augmentation
# ---------------------------------------------------------------------------

# The Organic Extra Virgin class contains samples from Calabria and Sicily
# and is considerably smaller than the Extra Virgin class.
#
# The assignment requires 50 additional minority-class samples to be
# generated using a Gaussian Mixture Model.

organic_data = classification_data[
    classification_data[
        "oil_class"
    ]
    == "Organic Extra Virgin"
].copy()


organic_features = organic_data[
    feature_columns
]


# ---------------------------------------------------------------------------
# Standardise Minority-Class Features
# ---------------------------------------------------------------------------

# The eight fatty-acid variables operate on different numerical scales.
# Standardisation prevents variables with larger values from dominating
# the Gaussian mixture fitting process.

gmm_scaler = StandardScaler()

organic_features_scaled = (
    gmm_scaler.fit_transform(
        organic_features
    )
)


# ---------------------------------------------------------------------------
# Fit Gaussian Mixture Model
# ---------------------------------------------------------------------------

# Two mixture components are used because the Organic Extra Virgin class
# originates from two localities: Calabria and Sicily.

gmm_model = GaussianMixture(
    n_components=2,
    covariance_type="full",
    random_state=RANDOM_SEED,
)

gmm_model.fit(
    organic_features_scaled
)


# ---------------------------------------------------------------------------
# Generate Synthetic Minority-Class Samples
# ---------------------------------------------------------------------------

synthetic_scaled, synthetic_components = (
    gmm_model.sample(
        NUMBER_OF_SYNTHETIC_SAMPLES
    )
)


# Convert generated observations from standardised values back to the
# original fatty-acid measurement scale.

synthetic_features = (
    gmm_scaler.inverse_transform(
        synthetic_scaled
    )
)


# ---------------------------------------------------------------------------
# Synthetic Sample Range Control
# ---------------------------------------------------------------------------

# Gaussian distributions can occasionally generate observations outside the
# range seen in the original minority class.
#
# Each synthetic feature is therefore restricted to the observed minimum
# and maximum values of the Organic Extra Virgin class.

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


# The supplied olive dataset stores fatty-acid measurements as integers.
# Round the generated observations so they follow the same representation.

synthetic_features = (
    np.rint(
        synthetic_features
    )
    .astype(
        int
    )
)


# ---------------------------------------------------------------------------
# Create Synthetic Classification Records
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
# Create Augmented Dataset
# ---------------------------------------------------------------------------

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
    ]
    .value_counts()
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


# ---------------------------------------------------------------------------
# Verify Augmented Dataset
# ---------------------------------------------------------------------------

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
    ]
    .describe()
)


# Save the augmented dataset for inspection and report evidence.

augmented_data.to_csv(
    OUTPUT_DIR
    / "olive_classification_augmented.csv",
    index=False,
)


# ---------------------------------------------------------------------------
# PCA Analysis
# ---------------------------------------------------------------------------

# Standardise all eight fatty-acid variables before PCA so variables with
# larger measurement ranges do not dominate the principal components.

pca_scaler = StandardScaler()

pca_features_scaled = (
    pca_scaler.fit_transform(
        augmented_data[
            feature_columns
        ]
    )
)


# Retain the first two principal components for the required 2D plot.

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


# ---------------------------------------------------------------------------
# PCA Explained Variance
# ---------------------------------------------------------------------------

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


# Use common histogram boundaries for both classes so their oleic
# distributions can be compared directly.

oleic_bins = np.linspace(
    augmented_data[
        "oleic"
    ]
    .min(),
    augmented_data[
        "oleic"
    ]
    .max(),
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
    alpha=0.5,
    density=True,
    label="Extra Virgin",
)


plt.hist(
    organic_oleic,
    bins=oleic_bins,
    alpha=0.5,
    density=True,
    label="Organic Extra Virgin",
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


# ---------------------------------------------------------------------------
# Output Summary
# ---------------------------------------------------------------------------

print(
    "\nGenerated Problem 2 outputs:"
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
# Leakage-Safe Neural Network Data Preparation
# ---------------------------------------------------------------------------

# Encode the two classes numerically:
#
# 0 = Extra Virgin
# 1 = Organic Extra Virgin

class_mapping = {
    "Extra Virgin": 0,
    "Organic Extra Virgin": 1,
}

classification_model_data = (
    classification_data.copy()
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


X_original = classification_model_data[
    feature_columns
]

y_original = classification_model_data[
    "target"
]


# ---------------------------------------------------------------------------
# Original Train / Validation / Test Split
# ---------------------------------------------------------------------------

# First reserve 20% of the original observations as the untouched test set.

X_train_validation, X_test, y_train_validation, y_test = (
    train_test_split(
        X_original,
        y_original,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_original,
    )
)


# Split the remaining original observations into training and validation sets.
#
# 20% of the remaining 80% becomes validation data, producing approximately:
#
# 64% training
# 16% validation
# 20% test

X_train_original, X_validation, y_train_original, y_validation = (
    train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_train_validation,
    )
)


print(
    "\nNeural network original-data split:"
)

print(
    f"Training samples before GMM augmentation: "
    f"{len(X_train_original)}"
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
    "\nOriginal training class distribution:"
)

print(
    y_train_original.value_counts().sort_index()
)


print(
    "\nValidation class distribution:"
)

print(
    y_validation.value_counts().sort_index()
)


print(
    "\nTest class distribution:"
)

print(
    y_test.value_counts().sort_index()
)


# ---------------------------------------------------------------------------
# Training-Only GMM Augmentation
# ---------------------------------------------------------------------------

# Extract only Organic Extra Virgin observations from the original training
# partition. The validation and test observations are not used when fitting
# this GMM.

organic_training_features = X_train_original[
    y_train_original == 1
].copy()


training_gmm_scaler = StandardScaler()

organic_training_scaled = (
    training_gmm_scaler.fit_transform(
        organic_training_features
    )
)


training_gmm_model = GaussianMixture(
    n_components=2,
    covariance_type="full",
    random_state=RANDOM_SEED,
)

training_gmm_model.fit(
    organic_training_scaled
)


synthetic_training_scaled, _ = (
    training_gmm_model.sample(
        NUMBER_OF_SYNTHETIC_SAMPLES
    )
)


synthetic_training_features = (
    training_gmm_scaler.inverse_transform(
        synthetic_training_scaled
    )
)


# Limit generated values to the range observed in the Organic training data.

training_organic_minimums = (
    organic_training_features
    .min()
    .to_numpy()
)

training_organic_maximums = (
    organic_training_features
    .max()
    .to_numpy()
)


synthetic_training_features = np.clip(
    synthetic_training_features,
    training_organic_minimums,
    training_organic_maximums,
)


synthetic_training_features = (
    np.rint(
        synthetic_training_features
    )
    .astype(
        int
    )
)


synthetic_training_df = pd.DataFrame(
    synthetic_training_features,
    columns=feature_columns,
)


synthetic_training_target = pd.Series(
    np.ones(
        NUMBER_OF_SYNTHETIC_SAMPLES,
        dtype=int,
    ),
    name="target",
)


# ---------------------------------------------------------------------------
# Augmented Training Dataset
# ---------------------------------------------------------------------------

X_train_augmented = pd.concat(
    [
        X_train_original.reset_index(
            drop=True
        ),
        synthetic_training_df,
    ],
    ignore_index=True,
)


y_train_augmented = pd.concat(
    [
        y_train_original.reset_index(
            drop=True
        ),
        synthetic_training_target,
    ],
    ignore_index=True,
)


# Shuffle the augmented training set reproducibly.

training_shuffle = np.random.default_rng(
    RANDOM_SEED
).permutation(
    len(
        X_train_augmented
    )
)


X_train_augmented = (
    X_train_augmented
    .iloc[
        training_shuffle
    ]
    .reset_index(
        drop=True
    )
)

y_train_augmented = (
    y_train_augmented
    .iloc[
        training_shuffle
    ]
    .reset_index(
        drop=True
    )
)


print(
    "\nTraining samples after GMM augmentation:"
)

print(
    len(
        X_train_augmented
    )
)


print(
    "\nAugmented training class distribution:"
)

print(
    y_train_augmented
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------------------------
# Neural Network Feature Scaling
# ---------------------------------------------------------------------------

# Fit the neural-network scaler only on the augmented training set.

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
# Baseline Neural Network Classifier
# ---------------------------------------------------------------------------

tf.keras.utils.set_random_seed(
    RANDOM_SEED
)

baseline_classifier = Sequential(
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
        Dense(
            1,
            activation="sigmoid",
        ),
    ]
)


baseline_classifier.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy"
    ],
)


print(
    "\nBaseline olive oil classifier architecture:"
)

baseline_classifier.summary()


# ---------------------------------------------------------------------------
# Baseline Classifier Training
# ---------------------------------------------------------------------------

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
    epochs=100,
    batch_size=64,
    verbose=1,
)


# ---------------------------------------------------------------------------
# Baseline Classifier Evaluation
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
)


print(
    "\nBaseline classification report:"
)

print(
    baseline_report_text
)


# Create a structured version of the classification report so the results
# can be saved as a CSV file for the technical report.

baseline_report_dict = classification_report(
    y_test,
    baseline_test_predictions,
    target_names=class_names,
    digits=4,
    output_dict=True,
)


baseline_report_df = pd.DataFrame(
    baseline_report_dict
).transpose()


baseline_report_df.to_csv(
    OUTPUT_DIR
    / "baseline_classification_report.csv"
)

# ---------------------------------------------------------------------------
# Baseline Accuracy Training Curve
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


# ---------------------------------------------------------------------------
# Baseline Loss Training Curve
# ---------------------------------------------------------------------------

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