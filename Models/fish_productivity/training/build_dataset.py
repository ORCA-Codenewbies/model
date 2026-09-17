from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(".")

CATCH = ROOT / "Datasets/processed/IOTC/iotc_catch_clean.parquet"
EFFORT = ROOT / "Datasets/processed/IOTC/iotc_effort_clean.parquet"
ERA5 = ROOT / "Datasets/processed/ERA5/era5_monthly_indian_ocean_1960_2026.nc"

OUT = ROOT / "Datasets/processed/training/fish_productivity"
OUT.mkdir(parents=True, exist_ok=True)

LOOKBACK = 24
TRAIN_END = pd.Timestamp("2016-12-01")
VAL_END = pd.Timestamp("2020-12-01")


def load_iotc():
    catch = pd.read_parquet(CATCH)
    effort = pd.read_parquet(EFFORT)

    # Homogeneous, real IOTC series already validated for productivity.
    catch = catch[
        (catch["FISHERY"] == "Longline | Deep-freezing") &
        (catch["GEAR"] == "Longline (deep-freezing)") &
        (catch["CATCH_UNIT_CODE"] == "MT")
    ].copy()

    effort = effort[
        (effort["FISHERY"] == "Longline | Deep-freezing") &
        (effort["GEAR"] == "Longline (deep-freezing)") &
        (effort["EFFORT_UNIT_CODE"] == "HOOKS")
    ].copy()

    catch_m = (
        catch.groupby(["YEAR", "MONTH_START"], as_index=False)["CATCH"]
        .sum()
        .rename(columns={"CATCH": "catch_mt"})
    )

    effort_m = (
        effort.groupby(["YEAR", "MONTH_START"], as_index=False)["EFFORT"]
        .sum()
        .rename(columns={"EFFORT": "effort_hooks"})
    )

    df = catch_m.merge(
        effort_m,
        on=["YEAR", "MONTH_START"],
        how="inner",
    )

    df["date"] = pd.to_datetime(
        dict(
            year=df["YEAR"],
            month=df["MONTH_START"],
            day=1,
        )
    )

    df = df.sort_values("date").reset_index(drop=True)

    df["cpue"] = (
        df["catch_mt"] / df["effort_hooks"]
    )

    if not np.isfinite(df["cpue"]).all():
        raise ValueError("Non-finite CPUE found.")

    return df[["date", "catch_mt", "effort_hooks", "cpue"]]


def load_era5():
    ds = xr.open_dataset(ERA5)

    # Regional monthly environmental statistics.
    # This is deliberately spatially aggregated because the IOTC
    # productivity series is currently aggregated rather than point-located.
    variables = [
        "u10",
        "v10",
        "t2m",
        "msl",
        "sst",
        "tp",
        "wind_speed_10m",
    ]

    frames = []

    for var in variables:
        da = ds[var]

        # Area mean over the validated Indian Ocean domain.
        regional = da.mean(
            dim=["latitude", "longitude"],
            skipna=True,
        )

        frames.append(
            regional.to_dataframe(name=var).reset_index()
        )

    env = frames[0][["valid_time", variables[0]]].copy()

    for frame, var in zip(frames[1:], variables[1:]):
        env = env.merge(
            frame[["valid_time", var]],
            on="valid_time",
            how="inner",
        )

    env["date"] = pd.to_datetime(env["valid_time"]).dt.to_period("M").dt.to_timestamp()

    env = (
        env.groupby("date", as_index=False)[variables]
        .mean()
        .sort_values("date")
    )

    ds.close()

    return env


def engineer(df):
    out = df.copy()

    out["log_catch"] = np.log1p(out["catch_mt"])
    out["log_effort"] = np.log1p(out["effort_hooks"])
    out["log_cpue"] = np.log(out["cpue"])

    for lag in [1, 3, 6, 12, 18, 24]:
        out[f"cpue_lag{lag}"] = out["cpue"].shift(lag)

    for window in [3, 6, 12]:
        out[f"cpue_roll{window}"] = (
            out["cpue"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    out["cpue_yoy_change"] = (
        out["cpue"] / out["cpue"].shift(12) - 1.0
    )

    month = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    out["year_index"] = (
        out["date"].dt.year - out["date"].dt.year.min()
    )

    # Forecast target: next month's CPUE.
    out["target_cpue"] = out["cpue"].shift(-1)

    return out


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)

    scale[scale < 1e-12] = 1.0

    return (
        (train_x - mean) / scale,
        (val_x - mean) / scale,
        (test_x - mean) / scale,
        mean,
        scale,
    )


def main():
    print("Loading IOTC...")
    iotc = load_iotc()

    print("IOTC:", iotc["date"].min(), "->", iotc["date"].max(), len(iotc))

    print("Loading ERA5...")
    era5 = load_era5()

    print("ERA5:", era5["date"].min(), "->", era5["date"].max(), len(era5))

    df = iotc.merge(
        era5,
        on="date",
        how="inner",
    )

    df = df.sort_values("date").reset_index(drop=True)

    print("Merged:", df["date"].min(), "->", df["date"].max(), len(df))

    df = engineer(df)

    feature_columns = [
        "log_catch",
        "log_effort",
        "log_cpue",
        "cpue_lag1",
        "cpue_lag3",
        "cpue_lag6",
        "cpue_lag12",
        "cpue_lag18",
        "cpue_lag24",
        "cpue_roll3",
        "cpue_roll6",
        "cpue_roll12",
        "cpue_yoy_change",
        "month_sin",
        "month_cos",
        "year_index",
        "u10",
        "v10",
        "t2m",
        "msl",
        "sst",
        "tp",
        "wind_speed_10m",
    ]

    # Require complete sequences and targets.
    required = feature_columns + ["target_cpue"]
    df = df.dropna(subset=required).reset_index(drop=True)

    dates = df["date"].to_numpy()

    # IMPORTANT:
    # Split by target date before fitting scaler to avoid leakage.
    train_mask = df["date"] <= TRAIN_END
    val_mask = (df["date"] > TRAIN_END) & (df["date"] <= VAL_END)
    test_mask = df["date"] > VAL_END

    train_rows = df[train_mask].copy()
    val_rows = df[val_mask].copy()
    test_rows = df[test_mask].copy()

    print("Rows:")
    print("  train:", len(train_rows))
    print("  val:  ", len(val_rows))
    print("  test: ", len(test_rows))

    X_raw = df[feature_columns].to_numpy(dtype=np.float32)
    y_raw = df["target_cpue"].to_numpy(dtype=np.float32)

    date_array = df["date"].to_numpy()

    X_train_seq = []
    X_val_seq = []
    X_test_seq = []
    y_train = []
    y_val = []
    y_test = []

    dates_train = []
    dates_val = []
    dates_test = []

    train_end_idx = df.index[df["date"] <= TRAIN_END].max()
    val_end_idx = df.index[df["date"] <= VAL_END].max()

    # Fit scaler ONLY from rows in the training period.
    scaler_base = X_raw[: train_end_idx + 1]

    mean = scaler_base.mean(axis=0)
    scale = scaler_base.std(axis=0)
    scale[scale < 1e-12] = 1.0

    X_scaled = (X_raw - mean) / scale

    y_train_rows = y_raw[: train_end_idx + 1]
    y_mean = y_train_rows.mean()
    y_scale = y_train_rows.std()
    if y_scale < 1e-12:
        y_scale = 1.0

    y_scaled = (y_raw - y_mean) / y_scale

    for end_idx in range(LOOKBACK, len(df)):
        start_idx = end_idx - LOOKBACK

        target_date = pd.Timestamp(date_array[end_idx])

        seq = X_scaled[start_idx:end_idx]
        target = y_scaled[end_idx]

        if target_date <= TRAIN_END:
            X_train_seq.append(seq)
            y_train.append(target)
            dates_train.append(target_date)

        elif target_date <= VAL_END:
            X_val_seq.append(seq)
            y_val.append(target)
            dates_val.append(target_date)

        else:
            X_test_seq.append(seq)
            y_test.append(target)
            dates_test.append(target_date)

    X_train_seq = np.asarray(X_train_seq, dtype=np.float32)
    X_val_seq = np.asarray(X_val_seq, dtype=np.float32)
    X_test_seq = np.asarray(X_test_seq, dtype=np.float32)

    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    print("Final sequence shapes:")
    print("  X_train:", X_train_seq.shape)
    print("  X_val:  ", X_val_seq.shape)
    print("  X_test: ", X_test_seq.shape)

    np.save(OUT / "X_train.npy", X_train_seq)
    np.save(OUT / "X_validation.npy", X_val_seq)
    np.save(OUT / "X_test.npy", X_test_seq)

    np.save(OUT / "y_train.npy", y_train)
    np.save(OUT / "y_validation.npy", y_val)
    np.save(OUT / "y_test.npy", y_test)

    np.save(OUT / "train_dates.npy", np.array(dates_train, dtype="datetime64[ns]"))
    np.save(OUT / "validation_dates.npy", np.array(dates_val, dtype="datetime64[ns]"))
    np.save(OUT / "test_dates.npy", np.array(dates_test, dtype="datetime64[ns]"))

    metadata = {
        "model_type": "LSTM",
        "lookback_months": LOOKBACK,
        "feature_columns": feature_columns,
        "target_column": "target_cpue",
        "target_definition": "next-month CPUE = monthly catch in MT / monthly effort in hooks",
        "train_end": str(TRAIN_END.date()),
        "validation_end": str(VAL_END.date()),
        "source_catch": str(CATCH),
        "source_effort": str(EFFORT),
        "source_environment": str(ERA5),
        "environment_aggregation": "monthly regional mean over ERA5 domain 5-25N, 65-95E",
        "feature_scaler": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        },
        "target_scaler": {
            "mean": [float(y_mean)],
            "scale": [float(y_scale)],
        },
    }

    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nDataset build complete.")


if __name__ == "__main__":
    main()
