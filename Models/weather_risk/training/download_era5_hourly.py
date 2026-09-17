from pathlib import Path
from calendar import monthrange
import cdsapi
import pandas as pd


CYCLONE_FILE = Path(
    "Datasets/processed/IMD/Cyclone_Best_Track/"
    "imd_cyclone_best_track_2020_2025.csv"
)

OUT_ROOT = Path(
    "Datasets/raw/ERA5/hourly_storm_windows"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


def build_windows():
    df = pd.read_csv(
        CYCLONE_FILE,
        parse_dates=["datetime_utc"],
    )

    df = df[
        df["latitude"].between(5, 25)
        & df["longitude"].between(65, 95)
    ].copy()

    dates = (
        df["datetime_utc"]
        .dt.floor("D")
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    windows = []

    start = prev = dates.iloc[0]

    for current in dates.iloc[1:]:
        gap = (current - prev).days

        if gap <= 2:
            prev = current
        else:
            windows.append((start, prev))
            start = prev = current

    windows.append((start, prev))

    # Same 2-day temporal buffer used in the acquisition plan.
    windows = [
        (
            start - pd.Timedelta(days=2),
            end + pd.Timedelta(days=2),
        )
        for start, end in windows
    ]

    # Merge overlaps.
    merged = []

    for start, end in windows:
        if not merged:
            merged.append([start, end])
            continue

        if start <= merged[-1][1] + pd.Timedelta(days=1):
            merged[-1][1] = max(
                merged[-1][1],
                end,
            )
        else:
            merged.append([start, end])

    return merged


def split_by_month(start, end):
    chunks = []

    cursor = start.normalize()

    while cursor <= end.normalize():
        month_start = cursor
        month_end = (
            cursor + pd.offsets.MonthEnd(0)
        ).normalize()

        chunk_start = max(
            start.normalize(),
            month_start,
        )

        chunk_end = min(
            end.normalize(),
            month_end,
        )

        days = list(
            range(
                chunk_start.day,
                chunk_end.day + 1,
            )
        )

        chunks.append(
            {
                "year": chunk_start.year,
                "month": chunk_start.month,
                "days": days,
                "start": chunk_start,
                "end": chunk_end,
            }
        )

        cursor = (
            month_end
            + pd.Timedelta(days=1)
        )

    return chunks


def request_chunk(
    client,
    chunk,
    window_id,
):
    year = chunk["year"]
    month = chunk["month"]
    days = chunk["days"]

    label = (
        f"{year:04d}-"
        f"{month:02d}-"
        f"{days[0]:02d}_"
        f"{days[-1]:02d}"
    )

    window_dir = (
        OUT_ROOT
        / f"window_{window_id:02d}"
    )

    window_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive = (
        window_dir
        / f"era5_hourly_{label}.zip"
    )

    if archive.exists():
        print(
            f"[SKIP] {archive}"
        )
        return

    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
            "total_precipitation",
        ],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [
            f"{day:02d}"
            for day in days
        ],
        "time": [
            f"{hour:02d}:00"
            for hour in range(24)
        ],
        "area": [25, 65, 5, 95],
        "format": "netcdf",
    }

    print(
        f"\n[DOWNLOAD] Window {window_id:02d}: "
        f"{chunk['start'].date()} -> "
        f"{chunk['end'].date()}"
    )

    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        str(archive),
    )

    print(
        f"[SAVED] {archive}"
    )

    # Verify it really is the ZIP returned by CDS.
    with archive.open("rb") as f:
        magic = f.read(4)

    if magic != b"PK\x03\x04":
        raise RuntimeError(
            f"Unexpected response format: {archive}"
        )

    # Extract without deleting the original CDS archive.
    import zipfile

    extract_dir = (
        window_dir
        / archive.stem
    )

    extract_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(archive) as z:
        z.extractall(extract_dir)

    members = sorted(
        extract_dir.glob("*.nc")
    )

    if not members:
        raise RuntimeError(
            f"No NetCDF files found in {archive}"
        )

    print(
        "[EXTRACTED]",
        ", ".join(
            p.name for p in members
        ),
    )


def main():
    import sys

    windows = build_windows()

    print("=" * 80)
    print("ORCA — ERA5 HOURLY STORM DATA DOWNLOADER")
    print("=" * 80)

    print(
        f"\nStorm windows: {len(windows)}"
    )

    chunks = []

    for window_id, (start, end) in enumerate(
        windows,
        start=1,
    ):
        for chunk in split_by_month(
            start,
            end,
        ):
            chunks.append(
                (
                    window_id,
                    chunk,
                )
            )

    print(
        f"Calendar-month requests: {len(chunks)}"
    )

    print("\nPlanned requests:")

    for window_id, chunk in chunks:
        print(
            f"  Window {window_id:02d}: "
            f"{chunk['start'].date()} -> "
            f"{chunk['end'].date()}"
        )

    if "--download" not in sys.argv:
        print(
            "\nNo downloads were started."
        )
        return

    print(
        "\nStarting CDS downloads..."
    )

    client = cdsapi.Client()

    for window_id, chunk in chunks:
        request_chunk(
            client,
            chunk,
            window_id,
        )

    print(
        "\nAll planned CDS requests completed."
    )


if __name__ == "__main__":
    main()
