"""
Golden Baseline Accuracy Oracle for Netra Profiler Benchmarks.

Compares a newly generated NetraProfile against a saved golden baseline profile
to guarantee that query-plan optimizations and streaming conversions introduce
zero statistical regressions.
"""

from __future__ import annotations

import math
from typing import Any


def _floats_equal(a: float, b: float, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def _compare_histograms(
    col_name: str,
    golden_hist: Any,
    candidate_hist: Any,
    rel_tol: float,
) -> list[str]:
    """Compares bin counts, breakpoint edges, and bin labels for a column histogram."""

    if not isinstance(golden_hist, list) or not isinstance(candidate_hist, list):
        return [
            f"[{col_name}.histogram] Type mismatch: "
            f"golden={type(golden_hist).__name__} vs candidate={type(candidate_hist).__name__}"
        ]
    if len(golden_hist) != len(candidate_hist):
        return [
            f"[{col_name}.histogram] Bin count mismatch: "
            f"{len(golden_hist)} vs {len(candidate_hist)}"
        ]

    diffs: list[str] = []
    for idx, (g_bin, c_bin) in enumerate(zip(golden_hist, candidate_hist, strict=True)):
        if not isinstance(g_bin, dict) or not isinstance(c_bin, dict):
            if g_bin != c_bin:
                diffs.append(f"[{col_name}.histogram[{idx}]] Bin mismatch: {g_bin} vs {c_bin}")
            continue

        if g_bin.get("count") != c_bin.get("count"):
            diffs.append(
                f"[{col_name}.histogram[{idx}]] Count mismatch: "
                f"{g_bin.get('count')} vs {c_bin.get('count')}"
            )

        g_bp, c_bp = g_bin.get("breakpoint"), c_bin.get("breakpoint")
        if isinstance(g_bp, (int, float)) and not isinstance(g_bp, bool):
            if (
                not isinstance(c_bp, (int, float))
                or isinstance(c_bp, bool)
                or not _floats_equal(float(g_bp), float(c_bp), rel_tol=rel_tol)
            ):
                diffs.append(f"[{col_name}.histogram[{idx}]] Breakpoint drift: {g_bp} vs {c_bp}")
        elif g_bp != c_bp:
            diffs.append(f"[{col_name}.histogram[{idx}]] Breakpoint mismatch: {g_bp} vs {c_bp}")

        if g_bin.get("bin") != c_bin.get("bin"):
            diffs.append(
                f"[{col_name}.histogram[{idx}]] Label mismatch: "
                f"{g_bin.get('bin')} vs {c_bin.get('bin')}"
            )

    return diffs


def _compare_correlations(
    golden_corr: dict[str, Any],
    candidate_corr: dict[str, Any],
    rel_tol: float,
) -> list[str]:
    """Compares sampling metadata and pairwise Pearson/Spearman correlation edges."""

    diffs: list[str] = []

    if golden_corr.get("sampling_method") != candidate_corr.get("sampling_method"):
        diffs.append(
            f"[correlations.sampling_method] Mismatch: "
            f"golden={golden_corr.get('sampling_method')} vs "
            f"candidate={candidate_corr.get('sampling_method')}"
        )

    for method in ("pearson", "spearman"):
        g_edges: dict[tuple[str, str], float] = {
            (str(e["column_a"]), str(e["column_b"])): float(e["score"])
            for e in golden_corr.get(method, [])
        }
        c_edges: dict[tuple[str, str], float] = {
            (str(e["column_a"]), str(e["column_b"])): float(e["score"])
            for e in candidate_corr.get(method, [])
        }
        if set(g_edges.keys()) != set(c_edges.keys()):
            diffs.append(f"[correlations.{method}] Edge set mismatch.")
        else:
            for pair, g_score in g_edges.items():
                c_score = c_edges[pair]
                if not _floats_equal(g_score, c_score, rel_tol=rel_tol):
                    diffs.append(
                        f"[correlations.{method} {pair[0]}<->{pair[1]}] "
                        f"Score drift: golden={g_score:.6f} vs candidate={c_score:.6f}"
                    )

    return diffs


def compare_profiles(
    golden: dict[str, Any],
    candidate: dict[str, Any],
    rel_tol: float = 1e-6,
) -> list[str]:
    """
    Compares dataset counts, column metrics, histograms, top_k,
    correlations, and alerts between two NetraProfile payloads.

    Returns:
        A list of human-readable discrepancy strings (empty if 100% accurate).
    """

    diffs: list[str] = []

    # 1. Dataset Level
    if golden.get("dataset") != candidate.get("dataset"):
        diffs.append(
            f"Dataset mismatch: expected {golden.get('dataset')}, got {candidate.get('dataset')}"
        )

    # 2. Column Level
    golden_cols: dict[str, dict[str, Any]] = golden.get("columns", {})
    candidate_columns: dict[str, dict[str, Any]] = candidate.get("columns", {})

    if set(golden_cols.keys()) != set(candidate_columns.keys()):
        diffs.append(
            f"Column keys mismatch: missing={set(golden_cols) - set(candidate_columns)}, "
            f"extra={set(candidate_columns) - set(golden_cols)}"
        )

    for col_name, g_metrics in golden_cols.items():
        if col_name not in candidate_columns:
            continue
        c_metrics = candidate_columns[col_name]

        for key, g_val in g_metrics.items():
            c_val = c_metrics.get(key)

            if isinstance(g_val, float):
                if (
                    not isinstance(c_val, (int, float))
                    or isinstance(c_val, bool)
                    or not _floats_equal(g_val, float(c_val), rel_tol=rel_tol)
                ):
                    diffs.append(
                        f"[{col_name}.{key}] Float drift: golden={g_val} vs candidate={c_val}"
                    )
            elif key == "histogram":
                diffs.extend(_compare_histograms(col_name, g_val, c_val, rel_tol=rel_tol))
            elif g_val != c_val:
                diffs.append(
                    f"[{col_name}.{key}] Value mismatch: golden={g_val} vs candidate={c_val}"
                )

    # 3. Correlations
    diffs.extend(
        _compare_correlations(
            golden.get("correlations", {}),
            candidate.get("correlations", {}),
            rel_tol=rel_tol,
        )
    )

    # 4. Diagnostic Alerts
    g_alerts = {(a["column_name"], a["type"], a["level"]) for a in golden.get("alerts", [])}
    c_alerts = {(a["column_name"], a["type"], a["level"]) for a in candidate.get("alerts", [])}
    if g_alerts != c_alerts:
        diffs.append(
            f"[alerts] Alert set mismatch: "
            f"missing={g_alerts - c_alerts}, extra={c_alerts - g_alerts}"
        )

    return diffs
