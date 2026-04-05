import argparse
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict

import numpy as np
import polars as pl
import pyarrow.parquet as pq

SHADY_CITIZEN_THRESHOLD = 0.3

RACES = [
    "Nord",
    "Imperial",
    "Altmer",
    "Dunmer",
    "Bosmer",
    "Khajiit",
    "Argonian",
    "Redguard",
    "Breton",
    "Orsimer",
]
HOLDS = [
    "Whiterun",
    "The Rift",
    "Haafingar",
    "Eastmarch",
    "The Reach",
    "Hjaalmarch",
    "The Pale",
    "Winterhold",
    "Falkreath",
]
FACTIONS = [
    "None",
    "Companions",
    "Thieves Guild",
    "Dark Brotherhood",
    "College of Winterhold",
    "Imperial Legion",
    "Stormcloaks",
]
ACTIVITIES = [
    "Patrolling the roads",
    "Complaining about sweetrolls",
    "Tending the forge",
    "Running from Draugr",
    "Drinking mead",
    "Practicing magic",
    "Sleeping",
    "Farming",
    "Fighting Bandits",
    "Hiding from Dragons",
]


def generate_chunk(start_id: int, chunk_size: int, is_wide: bool = False) -> pl.DataFrame:
    """Generates a 1-million row chunk of Skyrim census data using fast Numpy arrays."""

    # Base Column Generation
    df = pl.DataFrame(
        {
            "npc_id": np.arange(start_id, start_id + chunk_size, dtype=np.int32),
            "level": np.random.randint(1, 82, size=chunk_size, dtype=np.int32),
            "race": np.random.choice(RACES, size=chunk_size),
            "home_hold": np.random.choice(HOLDS, size=chunk_size),
            "faction": np.random.choice(FACTIONS, size=chunk_size),
            "gold_inventory": np.round(np.random.uniform(0.0, 5000.0, size=chunk_size), 2),
            # 70% of citizens are law-abiding (Null bounty). We generate floats, then mask with None
            "bounty_amount": np.where(
                np.random.rand(chunk_size) > SHADY_CITIZEN_THRESHOLD,
                np.round(np.random.uniform(10.0, 10000.0, size=chunk_size), 2),
                np.nan,
            ),
            "is_essential": np.random.choice([True, False], size=chunk_size, p=[0.05, 0.95]),
            "threat_level": np.random.randint(1, 6, size=chunk_size, dtype=np.int32),
            "current_activity": np.random.choice(ACTIVITIES, size=chunk_size),
        }
    )

    # Starting from Skyrim's legendary launch date: 11/11/11
    base_date = date(2011, 11, 11)
    # 5475 days gives us 15 years (2011-2026) of distributed data to profile
    random_days = np.random.randint(0, 5475, size=chunk_size)
    df = df.with_columns(
        pl.Series("last_encountered", [base_date + timedelta(days=int(d)) for d in random_days])
    )

    # Wide Dataset Expansion (100+ columns)
    if is_wide:
        wide_columns = []
        base_columns = [column for column in df.columns if column != "npc_id"]
        for i in range(1, 10):
            for col in base_columns:
                wide_columns.append(pl.col(col).alias(f"{col}_clone_{i}"))
        df = df.with_columns(wide_columns)

    return df


class DatasetConfig(TypedDict):
    chunks: int
    is_wide: bool
    formats: list[str]


def build_dataset(dataset_type: str) -> None:
    chunk_size = 1_000_000

    config: dict[str, DatasetConfig] = {
        "standard": {"chunks": 12, "is_wide": False, "formats": ["csv", "parquet"]},  # 12M rows
        "deep": {"chunks": 120, "is_wide": False, "formats": ["csv", "parquet"]},  # 120M rows
        "wide": {
            "chunks": 12,
            "is_wide": True,
            "formats": ["csv", "parquet"],
        },  # 12M rows, ~100 cols
        "flex": {
            "chunks": 960,
            "is_wide": False,
            "formats": ["parquet"],
        },
    }

    settings = config[dataset_type]
    total_chunks = settings["chunks"]
    output_dir = Path("benchmarks/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🗡️  Forging the '{dataset_type.upper()}' Dataset...")
    print(f"Target: {total_chunks * chunk_size:,} rows | Formats: {settings['formats']}")
    print(f"Output Directory: ./{output_dir}/")

    start_time = time.time()
    parquet_writer = None

    for i in range(total_chunks):
        chunk_start_time = time.time()
        start_id = (i * chunk_size) + 1

        # 1. Generate Data
        df = generate_chunk(start_id, chunk_size, settings["is_wide"])

        # 2. Write CSV (if requested)
        if "csv" in settings["formats"]:
            csv_path = output_dir / f"benchmark_{dataset_type}.csv"
            # Open in binary write ("wb") for the first chunk, binary append ("ab") for the rest
            mode = "wb" if i == 0 else "ab"
            with open(csv_path, mode) as f:
                df.write_csv(f, include_header=(i == 0))

        # 3. Write Parquet (if requested)
        if "parquet" in settings["formats"]:
            parquet_path = output_dir / f"benchmark_{dataset_type}.parquet"
            arrow_table = df.to_arrow()
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(
                    parquet_path, arrow_table.schema, compression="snappy"
                )
            parquet_writer.write_table(arrow_table)

        chunk_time = time.time() - chunk_start_time
        print(f"  -> Forged chunk {i + 1}/{total_chunks} in {chunk_time:.2f}s")

    if parquet_writer:
        parquet_writer.close()

    total_time = time.time() - start_time
    print(f"✅ Dataset generation complete in {total_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Netra Benchmarking Datasets")
    parser.add_argument(
        "--type",
        type=str,
        choices=["standard", "deep", "wide", "flex"],
        required=True,
        help="Type of dataset to generate",
    )
    args = parser.parse_args()
    build_dataset(args.type)
