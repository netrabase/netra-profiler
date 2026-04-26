import os
import sys
import time
import traceback

WORKER_ARGUMENTS = 4
MASSIVE_MULTIPLIER = 1


def _get_peak_ram_usage_in_mb() -> float:
    if sys.platform == "win32":
        import psutil  # noqa: PLC0415

        process = psutil.Process(os.getpid())
        return process.memory_info().peak_wset / (1024 * 1024)
    else:
        import resource  # noqa: PLC0415

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return rusage.ru_maxrss / (1024 * 1024)
        else:
            return rusage.ru_maxrss / 1024


def main():
    if len(sys.argv) != WORKER_ARGUMENTS:
        print("ERROR: Invalid arguments. Usage: worker.py <tool> <data_dir> <mode>")
        sys.exit(1)

    tool = sys.argv[1]
    data_dir = sys.argv[2]
    mode = sys.argv[3]

    # Target all parquet files in the directory
    glob_pattern = os.path.join(data_dir, "*.parquet")

    start_time = time.time()

    try:
        if tool == "netra":
            import glob  # noqa: PLC0415

            import polars as pl  # noqa: PLC0415

            from netra_profiler import Profiler  # noqa: PLC0415

            # We handle schema evolution by scanning the files into LazyFrames
            # individually, and letting Polars merge them using pl.concat()
            files = glob.glob(glob_pattern)
            if not files:
                raise FileNotFoundError(f"No parquet files found in {data_dir}")

            # Create the base lazy frames
            lazy_frames = [pl.scan_parquet(f) for f in files]

            # If we are running the massive scale, duplicate the lazy frames 20x.
            # 5 GB of files * 20 = 100 GB of streaming data.
            if "massive" in data_dir:
                lazy_frames = lazy_frames * MASSIVE_MULTIPLIER

            # Concatenate all of them into a single continuous execution graph
            df = pl.concat(lazy_frames, how="diagonal_relaxed")

            is_low_memory = mode == "low_memory"
            profiler = Profiler(df, low_memory=is_low_memory)
            _ = profiler.run(bins=20, top_k=10)

        elif tool == "ydata":
            import glob  # noqa: PLC0415

            import pandas as pd  # noqa: PLC0415
            from ydata_profiling import ProfileReport  # noqa: PLC0415

            files = glob.glob(glob_pattern)
            if not files:
                raise FileNotFoundError(f"No parquet files found in {data_dir}")

            df = pd.concat([pd.read_parquet(f) for f in files])
            is_minimal = mode == "minimal"

            profile = ProfileReport(df, minimal=is_minimal)
            _ = profile.get_description()

        else:
            print(f"ERROR: Unknown tool '{tool}'")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

    execution_time = time.time() - start_time
    peak_ram = _get_peak_ram_usage_in_mb()

    # The ONLY output this script should ever produce
    print(f"{execution_time:.4f},{peak_ram:.2f}")


if __name__ == "__main__":
    main()
