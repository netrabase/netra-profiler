"""
Abstract base classes, configuration models, and utilities for benchmark dataset generators.
Identical across Netra Validator and Netra Profiler for ecosystem consistency.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


def parse_scale(scale_str: str | int) -> int:
    """
    Parses human-readable scale strings into row counts.

    Examples:
        '10K'   -> 10_000
        '10M'   -> 10_000_000
        '1.5M'  -> 1_500_000
        '1B'    -> 1_000_000_000
        100000  -> 100_000
    """
    if isinstance(scale_str, int):
        return scale_str

    clean_str = scale_str.strip().upper()
    match = re.match(r"^([\d.]+)\s*([KMB]?)$", clean_str)
    if not match:
        raise ValueError(
            f"Invalid scale format: '{scale_str}'. Expected format like '500K', '10M', '1B'."
        )

    value_string, unit = match.groups()
    multiplier = 1
    if unit == "K":
        multiplier = 1_000
    elif unit == "M":
        multiplier = 1_000_000
    elif unit == "B":
        multiplier = 1_000_000_000

    return int(float(value_string) * multiplier)


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration contract for deterministic data generation."""

    total_rows: int
    chunk_size: int = 2_000_000
    seed: int = 42
    output_path: Path | None = None

    def derive_chunk_seed(self, chunk_index: int) -> int:
        """Derives a deterministic, isolated seed for each streaming chunk."""
        return (self.seed * 100_003 + chunk_index * 37) & 0xFFFFFFFF


class BaseDatasetGenerator(ABC):
    """Abstract protocol for synthetic and real-world benchmark data generators."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, destination: Path) -> Path:
        """
        Synthesizes or downloads the dataset and writes it to the destination path.

        Returns:
            The resolved Path to the finalized dataset file.
        """
        ...
