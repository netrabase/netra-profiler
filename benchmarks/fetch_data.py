import argparse
import asyncio
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import yaml
from tqdm.asyncio import tqdm

MAX_CONCURRENT_DOWNLOADS = 10


def load_config(config_path: Path) -> dict[str, Any]:
    """Loads the benchmark dataset definitions."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_dataset_urls(dataset_config: dict[str, Any], scale: str) -> list[str]:
    """Generates the list of specific Parquet URLs based on the requested scale."""
    base_url = dataset_config["base_url"]
    years = dataset_config["years"]
    limit = dataset_config["scales"].get(scale, 1)

    urls = []
    for year in years:
        for month in range(1, 13):
            urls.append(base_url.format(year=year, month=month))

    # If limit is 0, return all. Otherwise, slice the list.
    return urls if limit == 0 else urls[:limit]


async def download_file(
    client: httpx.AsyncClient, url: str, destination_dir: Path, semaphore: asyncio.Semaphore
) -> None:
    """Asynchronously streams a file directly to disk using a bounded semaphore."""
    filename = url.split("/")[-1]
    destination_path = destination_dir / filename

    if destination_path.exists():
        return  # Skip already downloaded files

    async with semaphore:
        try:
            # We use stream() to avoid loading the massive file into RAM
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                async with aiofiles.open(destination_path, "wb") as f:
                    with tqdm(
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        desc=filename,
                        leave=False,
                    ) as progress_bar:
                        async for chunk in response.aiter_bytes(
                            chunk_size=1024 * 1024
                        ):  # 1MB chunks
                            await f.write(chunk)
                            progress_bar.update(len(chunk))
        except Exception as e:
            print(f"\n[ERROR] Failed to download {filename}: {e}")


async def fetch_dataset(dataset_name: str, scale: str, output_dir: Path) -> None:
    """Orchestrates the asynchronous download pool."""
    config_path = Path(__file__).parent / "datasets.yaml"
    config = load_config(config_path)

    if dataset_name not in config:
        raise ValueError(f"Dataset '{dataset_name}' not found in datasets.yaml")

    dataset_config = config[dataset_name]

    urls = generate_dataset_urls(dataset_config, scale)

    target_dir = output_dir / dataset_name / scale
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching '{dataset_config['name']}' at [{scale.upper()}] scale...")
    print(f"Target: {len(urls)} files -> {target_dir.absolute()}\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    # Timeout settings
    # 60-second wait time for Read, Write, and Pool operations
    # 60-second handshake timeout (connect param)
    timeout = httpx.Timeout(60.0, connect=60.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [download_file(client, url, target_dir, semaphore) for url in urls]

        # Run all downloads concurrently and show a master progress bar
        for f in tqdm.as_completed(tasks, total=len(urls), desc="Overall Progress"):
            await f

    print("\nDataset fetch complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Netra Benchmark Data Fetcher")
    parser.add_argument("--dataset", type=str, default="taxi", help="Dataset name in yaml")
    parser.add_argument(
        "--scale", type=str, choices=["small", "medium", "massive"], default="small"
    )
    parser.add_argument(
        "--destination", type=str, default="./benchmarks/data", help="Output directory"
    )

    args = parser.parse_args()

    # Trigger the async event loop
    asyncio.run(fetch_dataset(args.dataset, args.scale, Path(args.destination)))
