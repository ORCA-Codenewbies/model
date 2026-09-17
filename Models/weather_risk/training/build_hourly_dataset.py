from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
import xarray as xr


IMD_PATH = Path(
    "Datasets/processed/IMD/Cyclone_Best_Track/"
    "imd_cyclone_best_track_2020_2025.csv"
)

ERA5_ROOT = Path(
    "Datasets/raw/ERA5/hourly_storm_windows"
)

OUT_DIR = Path(
    "Datasets/processed/training/weather_risk"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT = OUT_DIR / "weather_risk_hourly_features.parquet"
METADATA = OUT_DIR / "weather_risk_hourly_metadata.json"

RADIUS_KM = 300.0

# Keep all real positive samples.
# Retain at most this many NORMAL samples per positive sample.
NORMAL_TO_POSITIVE = 3

DANGEROUS_GRADES = {
    "CS",
    "SCS",
    "VSCS",
    "ESCS",
    "SUCS",
}

CAUTION_GRADES = {
    "D",
    "DD",
}


def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    radius = 6371.0088

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return 2.0 * radius * np.arcsin(
        np.sqrt(a)
    )


def stable_normal_sample(
    frame,
    target_size,
):
    if len(frame) <= target_size:
        return frame

    # Stable hash-based selection so reruns produce the same sample.
    key = (
        frame["timestamp"]
        .astype(str)
        + "|"
        + frame["latitude"].astype(str)
        + "|"
        + frame["longitude"].astype(str)
    )

    hashes = key.map(
        lambda x: int(
            hashlib.sha256(
                x.encode("utf-8")
            ).hexdigest()[:16],
            16,
        )
    )

    frame = frame.copy()
    frame["_hash"] = hashes

    frame = (
        frame.sort_values("_hash")
        .head(target_size)
        .drop(columns="_hash")
    )

    return frame


print("=" * 90)
print("ORCA WEATHER RISK — HOURLY DATASET BUILDER")
print("=" * 90)

# ---------------------------------------------------------------------
# IMD observations.
# ---------------------------------------------------------------------
imd = pd.read_csv(
    IMD_PATH,
    parse_dates=["datetime_utc"],
)

imd["datetime_utc"] = (
    imd["datetime_utc"]
    .dt.tz_localize(None)
)

# We need observations that can influence the ERA5 domain.
# This buffer comfortably covers the 300 km radius.
imd = imd[
    imd["latitude"].between(2, 28)
    & imd["longitude"].between(62, 98)
].copy()

print("\nIMD observations in influence buffer:", len(imd))

# ---------------------------------------------------------------------
# Load ERA5 files into a timestamp lookup.
# ---------------------------------------------------------------------
instant_files = sorted(
    ERA5_ROOT.glob(
        "window_*/era5_hourly_*/"
        "data_stream-oper_stepType-instant.nc"
    )
)

accum_files = sorted(
    ERA5_ROOT.glob(
        "window_*/era5_hourly_*/"
        "data_stream-oper_stepType-accum.nc"
    )
)

if len(instant_files) != 62:
    raise RuntimeError(
        f"Expected 62 instant files, found {len(instant_files)}"
    )

if len(accum_files) != 62:
    raise RuntimeError(
        f"Expected 62 accum files, found {len(accum_files)}"
    )

print("ERA5 instant files:", len(instant_files))
print("ERA5 accumulated files:", len(accum_files))

# Read all hourly fields.
instant_parts = []
accum_parts = []

for path in instant_files:
    ds = xr.open_dataset(
        path,
        engine="netcdf4",
    )

    ds = ds[
        [
            "u10",
            "v10",
            "t2m",
            "msl",
        ]
    ].sortby("latitude")

    instant_parts.append(ds)

for path in accum_files:
    ds = xr.open_dataset(
        path,
        engine="netcdf4",
    )

    ds = ds[
        ["tp"]
    ].sortby("latitude")

    accum_parts.append(ds)

instant = xr.concat(
    instant_parts,
    dim="valid_time",
).sortby("valid_time")

accum = xr.concat(
    accum_parts,
    dim="valid_time",
).sortby("valid_time")

# Remove duplicate timestamps created by overlapping windows.
_, unique_idx = np.unique(
    instant.valid_time.values,
    return_index=True,
)

instant = instant.isel(
    valid_time=np.sort(unique_idx)
)

_, unique_idx = np.unique(
    accum.valid_time.values,
    return_index=True,
)

accum = accum.isel(
    valid_time=np.sort(unique_idx)
)

print(
    "\nUnique ERA5 hourly timestamps:",
    instant.sizes["valid_time"],
)

# ---------------------------------------------------------------------
# Exact IMD ↔ ERA5 timestamp intersection.
# ---------------------------------------------------------------------
era5_times = pd.DatetimeIndex(
    instant.valid_time.values
)

imd = imd[
    imd["datetime_utc"].isin(era5_times)
].copy()

print(
    "IMD timestamps with exact ERA5 match:",
    imd["datetime_utc"].nunique(),
)

# ---------------------------------------------------------------------
# ERA5 coordinate grid.
# ---------------------------------------------------------------------
lat = instant.latitude.values
lon = instant.longitude.values

lon2d, lat2d = np.meshgrid(
    lon,
    lat,
)

lat_flat = lat2d.ravel()
lon_flat = lon2d.ravel()

grid_count = len(lat_flat)

print(
    "ERA5 grid cells per timestamp:",
    grid_count,
)

# ---------------------------------------------------------------------
# Build samples at exact IMD timestamps.
# ---------------------------------------------------------------------
records = []

for idx, timestamp in enumerate(
    sorted(imd["datetime_utc"].unique()),
    start=1,
):
    timestamp = pd.Timestamp(timestamp)

    obs = imd[
        imd["datetime_utc"] == timestamp
    ]

    current = instant.sel(
        valid_time=timestamp
    )

    precip = accum.sel(
        valid_time=timestamp
    )["tp"].values.ravel()

    u10 = current["u10"].values.ravel()
    v10 = current["v10"].values.ravel()
    t2m = current["t2m"].values.ravel()
    msl = current["msl"].values.ravel()

    wind_speed = np.sqrt(
        u10 ** 2 + v10 ** 2
    )

    # Target masks.
    caution = np.zeros(
        grid_count,
        dtype=bool,
    )

    dangerous = np.zeros(
        grid_count,
        dtype=bool,
    )

    for row in obs.itertuples():
        distance = haversine_km(
            lat_flat,
            lon_flat,
            row.latitude,
            row.longitude,
        )

        if row.grade in DANGEROUS_GRADES:
            dangerous |= (
                distance <= RADIUS_KM
            )
        elif row.grade in CAUTION_GRADES:
            caution |= (
                distance <= RADIUS_KM
            )

    target = np.zeros(
        grid_count,
        dtype=np.int8,
    )

    target[caution] = 1
    target[dangerous] = 2

    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "latitude": lat_flat,
            "longitude": lon_flat,
            "u10": u10,
            "v10": v10,
            "wind_speed_10m": wind_speed,
            "t2m": t2m,
            "msl": msl,
            "tp": precip,
            "target": target,
        }
    )

    # Only exact, real ERA5/IMD samples are retained.
    frame["hour"] = timestamp.hour
    frame["month"] = timestamp.month

    frame["sin_hour"] = np.sin(
        2.0 * np.pi * timestamp.hour / 24.0
    )

    frame["cos_hour"] = np.cos(
        2.0 * np.pi * timestamp.hour / 24.0
    )

    frame["sin_month"] = np.sin(
        2.0 * np.pi * timestamp.month / 12.0
    )

    frame["cos_month"] = np.cos(
        2.0 * np.pi * timestamp.month / 12.0
    )

    frame["split"] = np.select(
        [
            frame["timestamp"].dt.year <= 2023,
            frame["timestamp"].dt.year == 2024,
            frame["timestamp"].dt.year == 2025,
        ],
        [
            "train",
            "validation",
            "test",
        ],
        default="excluded",
    )

    records.append(frame)

    if idx % 50 == 0:
        print(
            f"Processed {idx} / "
            f"{imd['datetime_utc'].nunique()} timestamps"
        )

dataset = pd.concat(
    records,
    ignore_index=True,
)

# ---------------------------------------------------------------------
# Keep every positive observation.
# Sample only genuine NORMAL cells.
# ---------------------------------------------------------------------
positive = dataset[
    dataset["target"] > 0
].copy()

normal = dataset[
    dataset["target"] == 0
].copy()

print(
    "\nRaw samples before NORMAL sampling:",
    len(dataset),
)

print(
    "Positive samples:",
    len(positive),
)

# Sample separately within each chronological split.
sampled_normal_parts = []

for split in [
    "train",
    "validation",
    "test",
]:
    split_positive = positive[
        positive["split"] == split
    ]

    split_normal = normal[
        normal["split"] == split
    ]

    target_normal_count = min(
        len(split_normal),
        len(split_positive)
        * NORMAL_TO_POSITIVE,
    )

    sampled = stable_normal_sample(
        split_normal,
        target_normal_count,
    )

    sampled_normal_parts.append(
        sampled
    )

sampled_normal = pd.concat(
    sampled_normal_parts,
    ignore_index=True,
)

dataset = pd.concat(
    [
        positive,
        sampled_normal,
    ],
    ignore_index=True,
)

dataset = dataset.sort_values(
    ["timestamp", "latitude", "longitude"]
).reset_index(drop=True)

# ---------------------------------------------------------------------
# Target names.
# ---------------------------------------------------------------------
dataset["target_name"] = dataset[
    "target"
].map(
    {
        0: "NORMAL",
        1: "CAUTION",
        2: "DANGEROUS",
    }
)

# ---------------------------------------------------------------------
# Save.
# ---------------------------------------------------------------------
dataset.to_parquet(
    OUTPUT,
    index=False,
)

metadata = {
    "model": "weather_risk_xgboost",
    "dataset": "hourly_storm_aligned",
    "source": {
        "era5": "ERA5 single-level hourly reanalysis",
        "imd": "IMD cyclone best-track 2020-2025",
    },
    "spatial_domain": {
        "latitude": [5.0, 25.0],
        "longitude": [65.0, 95.0],
        "resolution_degrees": 0.25,
    },
    "target": {
        "NORMAL": "No qualifying IMD system within 300 km",
        "CAUTION": "D/DD within 300 km",
        "DANGEROUS": (
            "CS/SCS/VSCS/ESCS/SUCS within 300 km"
        ),
        "radius_km": RADIUS_KM,
        "note": (
            "The 300 km influence footprint is a "
            "project-defined operational rule."
        ),
    },
    "sampling": {
        "all_positive_samples_retained": True,
        "normal_to_positive_ratio": NORMAL_TO_POSITIVE,
        "synthetic_data": False,
        "negative_labels_synthetic": False,
    },
    "split": {
        "train": "2020-2023",
        "validation": "2024",
        "test": "2025",
    },
    "features": [
        "latitude",
        "longitude",
        "u10",
        "v10",
        "wind_speed_10m",
        "t2m",
        "msl",
        "tp",
        "hour",
        "month",
        "sin_hour",
        "cos_hour",
        "sin_month",
        "cos_month",
    ],
    "rows": int(len(dataset)),
    "class_counts": {
        name: int(
            (dataset["target_name"] == name).sum()
        )
        for name in [
            "NORMAL",
            "CAUTION",
            "DANGEROUS",
        ]
    },
}

with open(
    METADATA,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )

print("\n" + "=" * 90)
print("HOURLY DATASET COMPLETE")
print("=" * 90)

print("Rows:", len(dataset))
print("\nClass distribution:")
print(
    dataset["target_name"]
    .value_counts()
    .to_string()
)

print("\nSplit distribution:")
print(
    dataset.groupby(
        ["split", "target_name"]
    ).size().to_string()
)

print("\nSaved:")
print(OUTPUT)
print(METADATA)
