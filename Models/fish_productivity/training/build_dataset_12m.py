from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(".")

CATCH = ROOT / "Datasets/processed/IOTC/iotc_catch_clean.parquet"
EFFORT = ROOT / "Datasets/processed/IOTC/iotc_effort_clean.parquet"
ERA5 = ROOT / "Datasets/processed/ERA5/era5_monthly_indian_ocean_1960_2026.nc"

OUT = ROOT / "Datasets/processed/training/fish_productivity_12m_env"
OUT.mkdir(parents=True, exist_ok=True)

LOOKBACK = 12
TRAIN_END = pd.Timestamp("2016-12-01")
VAL_END = pd.Timestamp("2020-12-01")


def load_iotc():
    catch = pd.read_parquet(CATCH)
    effort = pd.read_parquet(EFFORT)

    catch = catch[
        (catch["FISHERY"] == "Longline | Deep-freezing") &
        (catch["GEAR"] == "Longline (deep-freezing)") &
        (catch["CATCH_UNIT_CODE"] == "MT")
    ]

    effort = effort[
        (effort["FISHERY"] == "Longline | Deep-freezing") &
        (effort["GEAR"] == "Longline (deep-freezing)") &
        (effort["EFFORT_UNIT_CODE"] == "HOOKS")
    ]

    catch_m = (
        catch.groupby(["YEAR", "MONTH_START"])["CATCH"]
        .sum()
        .reset_index()
        .rename(columns={"CATCH": "catch_mt"})
    )

    effort_m = (
        effort.groupby(["YEAR", "MONTH_START"])["EFFORT"]
        .sum()
        .reset_index()
        .rename(columns={"EFFORT": "effort_hooks"})
    )

    df = catch_m.merge(
        effort_m,
        on=["YEAR", "MONTH_START"],
        how="inner",
    )

    df["date"] = pd.to_datetime({
        "year": df["YEAR"],
        "month": df["MONTH_START"],
        "day": 1,
    })

    df = df.sort_values("date").reset_index(drop=True)

    df["cpue"] = df["catch_mt"] / df["effort_hooks"]

    return df[["date", "catch_mt", "effort_hooks", "cpue"]]


def load_era5():
    ds = xr.open_dataset(ERA5)

    variables = [
        "t2m",
        "msl",
        "sst",
        "tp",
        "wind_speed_10m",
    ]

    result = None

    for var in variables:
        da = ds[var].mean(
            dim=["latitude", "longitude"],
            skipna=True,
        )

        frame = (
            da.to_dataframe(name=var)
            .reset_index()
        )

        frame["date"] = (
            pd.to_datetime(frame["valid_time"])
            .dt.to_period("M")
            .dt.to_timestamp()
        )

        frame = (
            frame.groupby("date")[var]
            .mean()
            .reset_index()
        )

        if result is None:
            result = frame
        else:
            result = result.merge(
                frame,
                on="date",
                how="inner",
            )

    ds.close()

    return result.sort_values("date")


def engineer(df):
    x = df.copy()

    x["log_catch"] = np.log1p(x["catch_mt"])
    x["log_effort"] = np.log1p(x["effort_hooks"])
    x["log_cpue"] = np.log(x["cpue"])

    x["cpue_lag1"] = x["cpue"].shift(1)
    x["cpue_lag3"] = x["cpue"].shift(3)

    x["cpue_roll3"] = (
        x["cpue"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    x["cpue_roll12"] = (
        x["cpue"]
        .shift(1)
        .rolling(12)
        .mean()
    )

    x["cpue_yoy_change"] = (
        x["cpue"] / x["cpue"].shift(12) - 1.0
    )

    month = x["date"].dt.month

    x["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )

    x["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    x["target_cpue"] = x["cpue"].shift(-1)

    return x


def main():
    iotc = load_iotc()
    era5 = load_era5()

    df = iotc.merge(
        era5,
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    df = engineer(df)

    features = [
        "log_catch",
        "log_effort",
        "log_cpue",
        "cpue_lag1",
        "cpue_lag3",
        "cpue_roll3",
        "cpue_roll12",
        "cpue_yoy_change",
        "month_sin",
        "month_cos",
        "wind_speed_10m",
        "t2m",
        "msl",
        "sst",
        "tp",
    ]

    df = df.dropna(
        subset=features + ["target_cpue"]
    ).reset_index(drop=True)

    X_raw = df[features].to_numpy(np.float32)
    y_raw = df["target_cpue"].to_numpy(np.float32)

    dates = df["date"].to_numpy()

    train_rows = df["date"] <= TRAIN_END

    train_end_idx = np.where(train_rows)[0].max()

    feature_mean = X_raw[:train_end_idx + 1].mean(axis=0)
    feature_scale = X_raw[:train_end_idx + 1].std(axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0

    X_scaled = (
        X_raw - feature_mean
    ) / feature_scale

    target_mean = y_raw[:train_end_idx + 1].mean()
    target_scale = y_raw[:train_end_idx + 1].std()
    if target_scale < 1e-12:
        target_scale = 1.0

    y_scaled = (
        y_raw - target_mean
    ) / target_scale

    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []

    dates_train, dates_val, dates_test = [], [], []

    for end in range(LOOKBACK, len(df)):

        start = end - LOOKBACK
        target_date = pd.Timestamp(dates[end])

        sequence = X_scaled[start:end]
        target = y_scaled[end]

        if target_date <= TRAIN_END:
            X_train.append(sequence)
            y_train.append(target)
            dates_train.append(target_date)

        elif target_date <= VAL_END:
            X_val.append(sequence)
            y_val.append(target)
            dates_val.append(target_date)

        else:
            X_test.append(sequence)
            y_test.append(target)
            dates_test.append(target_date)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    print("Dataset:", df["date"].min(), "->", df["date"].max())
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)

    np.save(OUT / "X_train.npy", X_train)
    np.save(OUT / "X_validation.npy", X_val)
    np.save(OUT / "X_test.npy", X_test)

    np.save(OUT / "y_train.npy", y_train)
    np.save(OUT / "y_validation.npy", y_val)
    np.save(OUT / "y_test.npy", y_test)

    np.save(
        OUT / "train_dates.npy",
        np.asarray(dates_train, dtype="datetime64[ns]")
    )
    np.save(
        OUT / "validation_dates.npy",
        np.asarray(dates_val, dtype="datetime64[ns]")
    )
    np.save(
        OUT / "test_dates.npy",
        np.asarray(dates_test, dtype="datetime64[ns]")
    )

    metadata = {
        "model_type": "LSTM",
        "lookback_months": LOOKBACK,
        "feature_columns": features,
        "target_column": "target_cpue",
        "target_definition": (
            "next-month CPUE = monthly catch in MT / "
            "monthly effort in hooks"
        ),
        "train_end": str(TRAIN_END.date()),
        "validation_end": str(VAL_END.date()),
        "source_catch": str(CATCH),
        "source_effort": str(EFFORT),
        "source_environment": str(ERA5),
        "environment_aggregation": (
            "monthly regional mean over ERA5 "
            "domain 5-25N, 65-95E"
        ),
        "feature_scaler": {
            "mean": feature_mean.tolist(),
            "scale": feature_scale.tolist(),
        },
        "target_scaler": {
            "mean": [float(target_mean)],
            "scale": [float(target_scale)],
        },
    }

    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


if __name__ == "__main__":
    main()
