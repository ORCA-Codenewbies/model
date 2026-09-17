from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ERA5_PATH = Path(
    "Datasets/processed/ERA5/"
    "era5_monthly_indian_ocean_1960_2026.nc"
)

CYCLONE_PATH = Path(
    "Datasets/processed/IMD/Cyclone_Best_Track/"
    "imd_cyclone_best_track_2020_2025.csv"
)

OUT_DIR = Path("Datasets/processed/training/weather_risk")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Label definition
# ---------------------------------------------------------------------
# Real IMD grades -> safety classes.
GRADE_TO_CLASS = {
    "D": "CAUTION",
    "DD": "CAUTION",
    "CS": "DANGEROUS",
    "SCS": "DANGEROUS",
    "VSCS": "DANGEROUS",
    "ESCS": "DANGEROUS",
    "SUCS": "DANGEROUS",
}

CLASS_TO_ID = {
    "NORMAL": 0,
    "CAUTION": 1,
    "DANGEROUS": 2,
}


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance.
    Inputs may be scalars or NumPy arrays.
    """
    r = 6371.0088

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    return 2.0 * r * np.arcsin(np.sqrt(a))


def severity_rank(grade):
    return 1 if GRADE_TO_CLASS[grade] == "CAUTION" else 2


print("Loading ERA5...")
ds = xr.open_dataset(ERA5_PATH)

# Restrict to the labelled period only.
ds = ds.sel(
    valid_time=slice("2020-01-01", "2025-12-01")
)

era_time = pd.to_datetime(ds.valid_time.values)

print(
    f"ERA5 labelled-period months: "
    f"{era_time.min().date()} -> {era_time.max().date()}"
)

print("Loading IMD cyclone tracks...")
cyclone = pd.read_csv(
    CYCLONE_PATH,
    parse_dates=["datetime_utc"],
)

# Keep only observations that overlap ERA5 domain.
cyclone = cyclone[
    cyclone["latitude"].between(
        float(ds.latitude.min()), float(ds.latitude.max())
    )
    & cyclone["longitude"].between(
        float(ds.longitude.min()), float(ds.longitude.max())
    )
].copy()

cyclone["month"] = cyclone["datetime_utc"].dt.to_period("M")
cyclone["risk_class"] = cyclone["grade"].map(GRADE_TO_CLASS)
cyclone = cyclone.dropna(subset=["risk_class"])

print(f"Track observations inside domain: {len(cyclone)}")
print(f"Labelled months with tracks: {cyclone.month.nunique()}")

lat = ds.latitude.values
lon = ds.longitude.values
lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

# Build spatial grid.
grid = pd.DataFrame({
    "latitude": lat2d.ravel(),
    "longitude": lon2d.ravel(),
})

# ---------------------------------------------------------------------
# Construct monthly spatial labels.
#
# We use a 300 km influence radius around the observed track.
# This is a documented hazard-footprint rule, not a model feature.
#
# If multiple storms affect a cell/month, retain the highest observed
# severity class.
# ---------------------------------------------------------------------
INFLUENCE_RADIUS_KM = 300.0

label_rows = []

for month, tracks in cyclone.groupby("month"):
    month_label = np.zeros(len(grid), dtype=np.int8)

    for _, row in tracks.iterrows():
        distance = haversine_km(
            grid["latitude"].to_numpy(),
            grid["longitude"].to_numpy(),
            row["latitude"],
            row["longitude"],
        )

        affected = distance <= INFLUENCE_RADIUS_KM
        rank = severity_rank(row["grade"])

        month_label[affected] = np.maximum(
            month_label[affected],
            rank,
        )

    temp = grid.copy()
    temp["month"] = month
    temp["target_id"] = month_label

    label_rows.append(temp)

# Add NORMAL months as genuine non-event grid/month observations.
months = pd.PeriodIndex(
    pd.to_datetime(era_time),
    freq="M",
)

labelled_months = {
    x for x in cyclone["month"].dropna().unique()
}

all_rows = []

# We build the complete 2020-2025 grid-month population.
for month in months:
    month = pd.Period(month, freq="M")

    if month in labelled_months:
        part = next(
            x for x in label_rows
            if x["month"].iloc[0] == month
        )
    else:
        part = grid.copy()
        part["month"] = month
        part["target_id"] = 0

    all_rows.append(part)

labels = pd.concat(all_rows, ignore_index=True)

# ---------------------------------------------------------------------
# Extract monthly ERA5 features.
# ---------------------------------------------------------------------
feature_names = [
    "u10",
    "v10",
    "wind_speed_10m",
    "t2m",
    "msl",
    "sst",
    "tp",
]

features = (
    ds[feature_names]
    .to_dataframe()
    .reset_index()
)

features["month"] = (
    pd.to_datetime(features["valid_time"])
    .dt.to_period("M")
)

features = features.drop(columns=["valid_time"])

# Grid/month join.
data = labels.merge(
    features,
    on=["month", "latitude", "longitude"],
    how="left",
    validate="one_to_one",
)

# Seasonal features.
data["month_number"] = data["month"].dt.month.astype(np.int8)
data["sin_month"] = np.sin(
    2 * np.pi * data["month_number"] / 12.0
)
data["cos_month"] = np.cos(
    2 * np.pi * data["month_number"] / 12.0
)

# Feature NaNs are allowed to remain for later training preprocessing.
# We do not impute the original ERA5 ocean SST values here.

data["target"] = data["target_id"].astype(np.int8)
data["target_name"] = data["target"].map({
    0: "NORMAL",
    1: "CAUTION",
    2: "DANGEROUS",
})

data = data.sort_values(
    ["month", "latitude", "longitude"]
).reset_index(drop=True)

# Chronological split.
data["year"] = data["month"].dt.year

data["split"] = np.select(
    [
        data["year"].between(2020, 2023),
        data["year"].eq(2024),
        data["year"].eq(2025),
    ],
    [
        "train",
        "validation",
        "test",
    ],
    default="excluded",
)

data = data[data["split"] != "excluded"].copy()

# Save full labelled table.
data.to_parquet(
    OUT_DIR / "weather_risk_dataset.parquet",
    index=False,
)

# Save metadata.
metadata = {
    "task": "Weather Risk Classification",
    "label_source": "IMD cyclone best-track observed grades",
    "feature_source": "ERA5 monthly averaged single-level data",
    "domain": {
        "south": 5.0,
        "north": 25.0,
        "west": 65.0,
        "east": 95.0,
    },
    "label_period": "2020-2025",
    "influence_radius_km": INFLUENCE_RADIUS_KM,
    "class_mapping": GRADE_TO_CLASS,
    "class_ids": CLASS_TO_ID,
    "split": {
        "train": "2020-2023",
        "validation": "2024",
        "test": "2025",
    },
    "features": [
        "u10",
        "v10",
        "wind_speed_10m",
        "t2m",
        "msl",
        "sst",
        "tp",
        "sin_month",
        "cos_month",
    ],
    "no_synthetic_labels": True,
    "no_cyclone_grade_as_feature": True,
    "no_track_distance_as_feature": True,
}

pd.Series(metadata).to_json(
    OUT_DIR / "metadata.json",
    indent=2,
)

# Audit.
print("\n=== FINAL WEATHER DATASET ===")
print("rows:", len(data))
print("features:", feature_names + ["sin_month", "cos_month"])

print("\nClass counts:")
print(
    data.groupby(["split", "target_name"])
        .size()
        .unstack(fill_value=0)
        .to_string()
)

print("\nMissingness:")
print(
    data[
        feature_names
    ].isna()
     .mean()
     .mul(100)
     .sort_values(ascending=False)
     .to_string()
)

print("\nSaved:")
print(OUT_DIR / "weather_risk_dataset.parquet")
print(OUT_DIR / "metadata.json")

ds.close()
