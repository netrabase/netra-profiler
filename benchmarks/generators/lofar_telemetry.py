"""
Deterministic Synthetic Telemetry Generator for LOFAR 2.0 Station Health (Profiler Suite).

Synthesizes high-velocity radio telescope telemetry containing all primitive and complex
Polars types (Float64, Int32, Int64, Boolean, Datetime, String, Categorical, Struct, List)
and deterministic statistical anomalies to stress-test all 5 passes of Netra Profiler.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from benchmarks.generators.base import BaseDatasetGenerator, GeneratorConfig

# Lookup Tables for LOFAR parameters
CORE_STATIONS = [f"CS{i:03d}" for i in range(1, 25)]
REMOTE_STATIONS = [f"RS{i:03d}" for i in range(101, 125)]
INTL_STATIONS = [
    "DE601",
    "DE602",
    "DE603",
    "DE604",
    "DE605",
    "FR606",
    "UK608",
    "SE607",
    "PL610",
    "PL611",
    "PL612",
    "IE613",
]
VALID_STATIONS = np.array(CORE_STATIONS + REMOTE_STATIONS + INTL_STATIONS)

HARDWARE_STATES = ["OK", "DOWN", "SHORT", "FLAT", "SPURIOUS", "OSCILLATING"]
HARDWARE_WEIGHTS = [0.950, 0.010, 0.010, 0.010, 0.010, 0.010]

GPS_STATES = ["LOCKED", "DRIFTING", "LOST_SYNC"]
GPS_WEIGHTS = [0.980, 0.015, 0.005]

SUBBAND_CHANNELS = np.array(["104", "208", "312", "416", "511"])

# Explicit Enum dtypes ensure identical Arrow dictionary schemas across chunks
ANTENNA_ENUM = pl.Enum(["LBA", "HBA"])
HARDWARE_ENUM = pl.Enum(HARDWARE_STATES)
GPS_ENUM = pl.Enum(GPS_STATES)


def generate_chunk(start_index: int, number_of_rows: int, seed: int) -> pl.DataFrame:
    """
    Generates a single vectorized telemetry batch engineered to exercise every
    aggregation, type-flattening path, and diagnostic alert rule in Netra Profiler.
    """
    rng = np.random.default_rng(seed)
    row_indices = start_index + np.arange(number_of_rows, dtype=np.int64)

    # 1. Exact Primary Key (Triggers ALL_DISTINCT / ALL_DISTINCT_APPROX)
    packet_id = row_indices

    # 2. Corrupted Primary Key (~99.95% unique, triggers DUPLICATE_KEYS)
    sequence_id = row_indices.copy()
    dup_mask = rng.uniform(0.0, 1.0, size=number_of_rows) < 0.0005
    sequence_id[dup_mask] = start_index

    # 3. Temporal Vector with ~0.25% Nulls
    base_epoch = np.datetime64("2026-03-01T00:00:00", "ms")
    timestamps = base_epoch + (row_indices * 100).astype("timedelta64[ms]")

    # 4. Station & Antenna Metadata
    stations = rng.choice(VALID_STATIONS, size=number_of_rows)
    rcu_ids = rng.integers(0, 192, size=number_of_rows, dtype=np.int32)
    antenna_types = rng.choice(["LBA", "HBA"], size=number_of_rows, p=[0.5, 0.5])

    # 5. Bimodal Frequency Distribution (LBA: 10-90 MHz | HBA: 110-240 MHz)
    lba_freqs = rng.uniform(10.0, 90.0, size=number_of_rows)
    hba_freqs = rng.uniform(110.0, 240.0, size=number_of_rows)
    frequencies = np.where(antenna_types == "LBA", lba_freqs, hba_freqs)

    # 6. Categoricals
    hardware_status = rng.choice(HARDWARE_STATES, size=number_of_rows, p=HARDWARE_WEIGHTS)
    gps_sync_status = rng.choice(GPS_STATES, size=number_of_rows, p=GPS_WEIGHTS)

    # 7. Continuous Signals & Collinear Sensor Pair (Triggers HIGH_CORRELATION & OUTLIERS_DETECTED)
    signal_power = rng.normal(loc=-75.0, scale=12.0, size=number_of_rows)
    ambient_temp = rng.normal(loc=12.5, scale=6.5, size=number_of_rows)
    # Board temp is strongly collinear with ambient temp (r > 0.99)
    board_temp = ambient_temp * 1.85 + 15.0 + rng.normal(loc=0.0, scale=0.3, size=number_of_rows)

    # Inject extreme thermal spikes (~0.1% -> triggers OUTLIERS_DETECTED via Tukey IQR)
    temp_anomaly_mask = rng.uniform(0.0, 1.0, size=number_of_rows) < 0.001
    if temp_anomaly_mask.any():
        spikes = rng.choice([-273.15, 650.0], size=int(temp_anomaly_mask.sum()))
        ambient_temp[temp_anomaly_mask] = spikes
        board_temp[temp_anomaly_mask] = spikes * 1.85 + 15.0

    # Inject NaNs into Float64 to test NaN-to-Null coercion in Pass 1 & Pass 2
    nan_mask = rng.uniform(0.0, 1.0, size=number_of_rows) < 0.001
    signal_power[nan_mask] = np.nan

    # 8. Heavily Skewed Distribution (Pareto/Exponential -> triggers SKEWED |skew| > 2.0)
    packet_drop_rate = (rng.pareto(a=1.5, size=number_of_rows) * 0.002).clip(0.0, 1.0)

    # 9. Zero-Inflated Numeric (~65% zeros -> triggers ZERO_INFLATED)
    solar_flare_index = np.where(
        rng.uniform(0.0, 1.0, size=number_of_rows) < 0.65,
        0.0,
        rng.exponential(scale=15.0, size=number_of_rows),
    )

    # 10. High Nulls Column (~60% Null -> triggers HIGH_NULLS)
    calibration_offset = rng.normal(loc=0.0, scale=1.0, size=number_of_rows)
    high_null_mask = rng.uniform(0.0, 1.0, size=number_of_rows) < 0.60
    calibration_offset[high_null_mask] = np.nan

    # 11. Empty Column (~98% Null -> triggers EMPTY_COLUMN)
    decommissioned_sensor = np.full(number_of_rows, np.nan, dtype=np.float64)
    surviving_mask = rng.uniform(0.0, 1.0, size=number_of_rows) < 0.01
    decommissioned_sensor[surviving_mask] = rng.normal(
        loc=42.0, scale=2.0, size=int(surviving_mask.sum())
    ).round(2)

    # 12. High Cardinality String (>10,000 unique values -> triggers HIGH_CARDINALITY)
    # Drawn from 50,000 unique target hashes
    hash_ids = rng.integers(0, 50_000, size=number_of_rows)
    subband_hash = np.char.add("SB_HASH_", hash_ids.astype(str))

    # 13. Possible Numeric String (Triggers POSSIBLE_NUMERIC in Top-K check)
    subband_channel_str = rng.choice(SUBBAND_CHANNELS, size=number_of_rows)

    # 14. Blank / Whitespace String (~15% whitespace -> triggers BLANK_STRINGS)
    # & String Length Anomaly (~0.05% 800-char payload -> triggers INCONSISTENT_STRING_LENGTH)
    operator_notes = np.full(number_of_rows, "NOMINAL_OP", dtype=object)
    blank_mask = rng.uniform(0.0, 1.0, size=number_of_rows) < 0.15
    operator_notes[blank_mask] = "   "
    long_mask = (~blank_mask) & (rng.uniform(0.0, 1.0, size=number_of_rows) < 0.0005)
    operator_notes[long_mask] = "STACK_TRACE_" + ("X" * 800)

    # 15. Constant Columns (Triggers CONSTANT alert & tests min == max histogram fallback)
    firmware_version = ["LOFAR2.0-rc4"] * number_of_rows
    schema_version_num = np.full(number_of_rows, 2, dtype=np.int32)

    # 16. Boolean RFI Flag
    rfi_detected = (hardware_status == "SPURIOUS") | (
        rng.uniform(0.0, 1.0, size=number_of_rows) < 0.01
    )

    # Missing timestamp injection on LOST_SYNC
    sync_failure_mask = (gps_sync_status == "LOST_SYNC") & (
        rng.uniform(0.0, 1.0, size=number_of_rows) < 0.5
    )
    if sync_failure_mask.any():
        timestamps[sync_failure_mask] = np.datetime64("NaT", "ms")

    # 17. Complex Nested Types (Tests preprocess_complex_types: Structs & Fixed-Size Arrays/Lists)
    az_el_struct = pl.DataFrame(
        {
            "azimuth_deg": rng.uniform(0.0, 360.0, size=number_of_rows).round(2),
            "elevation_deg": rng.uniform(15.0, 90.0, size=number_of_rows).round(2),
        }
    ).to_struct("pointing_coords")

    return pl.DataFrame(
        {
            "packet_id": pl.Series("packet_id", packet_id, dtype=pl.Int64),
            "sequence_id": pl.Series("sequence_id", sequence_id, dtype=pl.Int64),
            "timestamp_utc": pl.Series("timestamp_utc", timestamps, dtype=pl.Datetime("ms")),
            "station_id": pl.Series("station_id", stations, dtype=pl.String),
            "rcu_id": pl.Series("rcu_id", rcu_ids, dtype=pl.Int32),
            "antenna_type": pl.Series("antenna_type", antenna_types, dtype=ANTENNA_ENUM),
            "frequency_mhz": pl.Series("frequency_mhz", np.round(frequencies, 3), dtype=pl.Float64),
            "hardware_status": pl.Series("hardware_status", hardware_status, dtype=HARDWARE_ENUM),
            "signal_power_db": pl.Series(
                "signal_power_db", np.round(signal_power, 2), dtype=pl.Float64
            ),
            "rfi_flag": pl.Series("rfi_flag", rfi_detected, dtype=pl.Boolean),
            "ambient_temp_c": pl.Series(
                "ambient_temp_c", np.round(ambient_temp, 2), dtype=pl.Float64
            ),
            "board_temp_c": pl.Series("board_temp_c", np.round(board_temp, 2), dtype=pl.Float64),
            "gps_sync_status": pl.Series("gps_sync_status", gps_sync_status, dtype=GPS_ENUM),
            "packet_drop_rate": pl.Series(
                "packet_drop_rate", np.round(packet_drop_rate, 5), dtype=pl.Float64
            ),
            "solar_flare_index": pl.Series(
                "solar_flare_index", np.round(solar_flare_index, 2), dtype=pl.Float64
            ),
            "calibration_offset": pl.Series(
                "calibration_offset", np.round(calibration_offset, 3), dtype=pl.Float64
            ),
            "decommissioned_sensor": pl.Series(
                "decommissioned_sensor", decommissioned_sensor, dtype=pl.Float64
            ),
            "subband_hash": pl.Series("subband_hash", subband_hash, dtype=pl.String),
            "subband_channel_str": pl.Series(
                "subband_channel_str", subband_channel_str, dtype=pl.String
            ),
            "operator_notes": pl.Series("operator_notes", operator_notes, dtype=pl.String),
            "firmware_version": pl.Series("firmware_version", firmware_version, dtype=pl.String),
            "schema_version_num": pl.Series(
                "schema_version_num", schema_version_num, dtype=pl.Int32
            ),
            "pointing_coords": az_el_struct,
        }
    )


def generate_lofar_dataset(
    output_path: Path,
    total_rows: int = 10_000_000,
    chunk_size: int = 2_000_000,
    seed: int = 42,
) -> Path:
    """
    Generates a scaled LOFAR dataset using Polars out-of-core streaming chunks.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = GeneratorConfig(
        total_rows=total_rows,
        chunk_size=chunk_size,
        seed=seed,
        output_path=output_path,
    )

    total_chunks = max(1, (total_rows + chunk_size - 1) // chunk_size)
    print(
        f"\n[+] Synthesizing LOFAR Telemetry (Profiler Suite): {total_rows:,} rows across {total_chunks} chunks"
    )
    print(f"[+] Output Target: {output_path}")

    start_time = time.perf_counter()
    writer = None

    try:
        rows_remaining = total_rows
        current_idx = 0

        for chunk_idx in range(total_chunks):
            current_batch_size = min(rows_remaining, chunk_size)
            chunk_seed = config.derive_chunk_seed(chunk_idx)

            t0 = time.perf_counter()
            df_chunk = generate_chunk(current_idx, current_batch_size, chunk_seed)
            arrow_table = df_chunk.to_arrow()

            if writer is None:
                writer = pq.ParquetWriter(output_path, arrow_table.schema, compression="snappy")

            writer.write_table(arrow_table)
            chunk_time = time.perf_counter() - t0

            current_idx += current_batch_size
            rows_remaining -= current_batch_size

            print(
                f"    -> Chunk {chunk_idx + 1:02d}/{total_chunks} "
                f"({current_batch_size:,} rows) written in {chunk_time:.2f}s"
            )
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    velocity = (total_rows / elapsed) / 1_000_000

    print("\n[✔] LOFAR Profiler Dataset Generation Complete!")
    print(f"    - Total Rows:    {total_rows:,}")
    print(f"    - File Size:     {file_size_mb:.2f} MB")
    print(f"    - Elapsed Time:  {elapsed:.2f}s")
    print(f"    - Velocity:      {velocity:.2f}M rows/sec\n")

    return output_path


class LofarTelemetryGenerator(BaseDatasetGenerator):
    """Object-oriented wrapper implementing BaseDatasetGenerator for LOFAR telemetry."""

    def generate(self, destination: Path) -> Path:
        return generate_lofar_dataset(
            output_path=destination,
            total_rows=self.config.total_rows,
            chunk_size=self.config.chunk_size,
            seed=self.config.seed,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic LOFAR telemetry datasets for Netra Profiler benchmarking."
    )
    parser.add_argument("--rows", "-n", type=int, default=10_000_000)
    parser.add_argument("--chunk-size", "-c", type=int, default=2_000_000)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("benchmarks/data/synthetic/lofar_10M.parquet"),
    )
    parser.add_argument("--seed", "-s", type=int, default=42)

    args = parser.parse_args()
    generate_lofar_dataset(
        output_path=args.output,
        total_rows=args.rows,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
