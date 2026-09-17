from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path("Datasets/raw/ERA5/hourly_storm_windows")
INPUT = Path(
    "Datasets/processed/training/weather_risk/"
    "weather_risk_hourly_features.parquet"
)
OUTPUT = Path(
    "Datasets/processed/training/weather_risk/"
    "weather_risk_hourly_advanced_features.parquet"
)

INSTANT_FILES = sorted(
    ROOT.glob(
        "window_*/era5_hourly_*/"
        "data_stream-oper_stepType-instant.nc"
    )
)

ACCUM_FILES = sorted(
    ROOT.glob(
        "window_*/era5_hourly_*/"
        "data_stream-oper_stepType-accum.nc"
    )
)


def gradient_magnitude(field, lat, lon):
    dlat = np.gradient(field, axis=-2)
    dlon = np.gradient(field, axis=-1)

    # Grid spacing is approximately 0.25 degree.
    # Use a local physical-distance approximation.
    lat_rad = np.radians(lat)

    dy_km = 111.32 * 0.25
    dx_km = (
        111.32
        * 0.25
        * np.cos(lat_rad)
    )

    dx_km = np.maximum(dx_km, 1.0)

    dy = dlat / dy_km
    dx = dlon / dx_km[:, None]

    return np.sqrt(
        dx**2 + dy**2
    )


print("=" * 90)
print("ORCA WEATHER RISK — ADVANCED HOURLY FEATURES")
print("=" * 90)

df = pd.read_parquet(INPUT)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

print(
    "Input rows:",
    len(df)
)

# ---------------------------------------------------------------------
# Load unique ERA5 fields.
# ---------------------------------------------------------------------
instant_parts = []

for path in INSTANT_FILES:
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
    ]

    instant_parts.append(ds)

instant = xr.concat(
    instant_parts,
    dim="valid_time",
)

instant = (
    instant
    .sortby("valid_time")
    .drop_duplicates("valid_time")
)

accum_parts = []

for path in ACCUM_FILES:
    ds = xr.open_dataset(
        path,
        engine="netcdf4",
    )

    accum_parts.append(
        ds[["tp"]]
    )

accum = xr.concat(
    accum_parts,
    dim="valid_time",
)

accum = (
    accum
    .sortby("valid_time")
    .drop_duplicates("valid_time")
)

print(
    "ERA5 unique hours:",
    instant.sizes["valid_time"]
)

lat = instant.latitude.values
lon = instant.longitude.values

# ---------------------------------------------------------------------
# Work hour-by-hour and compute local physical gradients.
# ---------------------------------------------------------------------
rows = []

timestamps = pd.DatetimeIndex(
    df["timestamp"].drop_duplicates()
)

for n, timestamp in enumerate(
    timestamps,
    start=1,
):
    ts = pd.Timestamp(timestamp)

    era = instant.sel(
        valid_time=ts
    )

    u = era["u10"].values
    v = era["v10"].values
    t = era["t2m"].values
    p = era["msl"].values

    speed = np.sqrt(
        u**2 + v**2
    )

    wind_grad = gradient_magnitude(
        speed,
        lat,
        lon,
    )

    pressure_grad = gradient_magnitude(
        p,
        lat,
        lon,
    )

    temp_grad = gradient_magnitude(
        t,
        lat,
        lon,
    )

    # Local spatial anomaly against a 3x3 neighborhood.
    def local_anomaly(field):
        padded = np.pad(
            field,
            1,
            mode="edge",
        )

        neighborhood_sum = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )

        neighborhood_mean = (
            neighborhood_sum / 8.0
        )

        return (
            field
            - neighborhood_mean
        )

    pressure_local_anomaly = (
        local_anomaly(p)
    )

    wind_local_anomaly = (
        local_anomaly(speed)
    )

    # Short-term temporal changes.
    prev_time = (
        ts - pd.Timedelta(hours=3)
    )

    if prev_time in instant.valid_time.values:
        prev = instant.sel(
            valid_time=prev_time
        )

        prev_u = prev["u10"].values
        prev_v = prev["v10"].values
        prev_speed = np.sqrt(
            prev_u**2 + prev_v**2
        )
        prev_p = prev["msl"].values

        wind_delta_3h = (
            speed - prev_speed
        )

        pressure_delta_3h = (
            p - prev_p
        )

    else:
        wind_delta_3h = np.full_like(
            speed,
            np.nan,
        )
        pressure_delta_3h = np.full_like(
            p,
            np.nan,
        )

    rows.append(
        pd.DataFrame(
            {
                "timestamp": ts,
                "latitude": np.repeat(
                    lat,
                    len(lon),
                ),
                "longitude": np.tile(
                    lon,
                    len(lat),
                ),
                "wind_gradient": wind_grad.ravel(),
                "pressure_gradient": (
                    pressure_grad.ravel()
                ),
                "temperature_gradient": (
                    temp_grad.ravel()
                ),
                "pressure_local_anomaly": (
                    pressure_local_anomaly.ravel()
                ),
                "wind_local_anomaly": (
                    wind_local_anomaly.ravel()
                ),
                "wind_delta_3h": (
                    wind_delta_3h.ravel()
                ),
                "pressure_delta_3h": (
                    pressure_delta_3h.ravel()
                ),
            }
        )
    )

    if n % 50 == 0:
        print(
            f"Processed {n} / "
            f"{len(timestamps)} timestamps"
        )

advanced = pd.concat(
    rows,
    ignore_index=True,
)

# ---------------------------------------------------------------------
# Join features to the existing labeled dataset.
# ---------------------------------------------------------------------
df = df.merge(
    advanced,
    on=[
        "timestamp",
        "latitude",
        "longitude",
    ],
    how="left",
    validate="one_to_one",
)

# Make sure temporal deltas are NaN only when unavailable.
print(
    "\nMissing advanced features:"
)

feature_cols = [
    "wind_gradient",
    "pressure_gradient",
    "temperature_gradient",
    "pressure_local_anomaly",
    "wind_local_anomaly",
    "wind_delta_3h",
    "pressure_delta_3h",
]

for col in feature_cols:
    print(
        f"  {col}: "
        f"{df[col].isna().mean() * 100:.4f}%"
    )

df.to_parquet(
    OUTPUT,
    index=False,
)

print("\n" + "=" * 90)
print("ADVANCED FEATURE DATASET COMPLETE")
print("=" * 90)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)

print(
    "Saved:",
    OUTPUT
)
