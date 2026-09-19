"""
COIT29225 Neural Networks & Deep Learning
Assessment 2 - Problem 1, Part 2

Custom Brisbane House Price Dataset

This section generates a synthetic Brisbane property dataset for subsequent
K-means analysis and deep neural network regression experiments.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_FILE = Path("data/BrisbaneHousePrice.csv")

NUMBER_OF_PROPERTIES = 2000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


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

print("Brisbane House Price dataset created successfully.")

print("\nDataset shape:")
print(brisbane_data.shape)

print("\nFirst five rows:")
print(brisbane_data.head())

print("\nMissing values:")
print(brisbane_data.isnull().sum())

print("\nDescriptive statistics:")
print(brisbane_data.describe())

print(
    "\nDataset saved to:"
)
print(OUTPUT_FILE)


# ---------------------------------------------------------------------------
# Dataset Quality Checks
# ---------------------------------------------------------------------------

print("\nProperty type distribution:")
print(
    brisbane_data["x3"]
    .value_counts()
    .sort_index()
)

print("\nHouse price summary by property type:")
print(
    brisbane_data.groupby("x3")["house_price"]
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

print("\nPrice clipping checks:")

minimum_price_count = (
    brisbane_data["house_price"] == 250_000
).sum()

maximum_price_count = (
    brisbane_data["house_price"] == 3_000_000
).sum()

print(
    f"Properties at $250,000 minimum: "
    f"{minimum_price_count}"
)

print(
    f"Properties at $3,000,000 maximum: "
    f"{maximum_price_count}"
)