"""Analyze and compare ligand strain evaluation results across models.

Loads JSON result files produced by run_ligand_strain.py and computes:
  - Per-system strain energy (eV and kcal/mol)
  - Mean, median, std of strain distribution
  - Failure rate (systems with no valid gas-phase optimisation)
  - MAE / Pearson correlation vs DFT reference (if provided)
  - Summary table (CSV) and comparison plots (PNG)

Usage:
    python eval/analyze_results.py \
        --results results/esen_v1.json results/esen_v2.json results/mace.json \
        --labels eSEN-v1 eSEN-v2 MACE \
        --output-dir results/analysis \
        [--reference results/dft_reference.json]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EV_TO_KCAL = 23.0605


def compute_strain(system_result: dict) -> float | None:
    """Return strain energy in eV, or None if optimisation failed for all conformers."""
    bioactive_energy = system_result["bioactive"]["energy"]

    final_energies = []
    for conf in system_result["gas_phase"].values():
        e = conf.get("final", {}).get("energy")
        if e is not None:
            final_energies.append(e)

    if not final_energies:
        return None

    return bioactive_energy - min(final_energies)


def load_results(path: str) -> dict[str, float | None]:
    """Load a results JSON and return {system_id: strain_eV}."""
    with open(path) as f:
        data = json.load(f)
    return {sid: compute_strain(res) for sid, res in data.items()}


def convergence_rate(path: str) -> float:
    """Fraction of systems where at least one conformer converged."""
    with open(path) as f:
        data = json.load(f)
    converged = sum(
        1
        for res in data.values()
        if any(
            conf.get("final", {}).get("energy") is not None
            for conf in res["gas_phase"].values()
        )
    )
    return converged / len(data) if data else 0.0


def build_dataframe(result_files: list[str], labels: list[str]) -> pd.DataFrame:
    all_strains: dict[str, dict[str, float | None]] = {}
    for path, label in zip(result_files, labels):
        all_strains[label] = load_results(path)

    system_ids = sorted(
        set(sid for strains in all_strains.values() for sid in strains)
    )
    rows = []
    for sid in system_ids:
        row: dict = {"system_id": sid}
        for label in labels:
            ev = all_strains[label].get(sid)
            row[f"{label}_eV"] = ev
            row[f"{label}_kcal"] = ev * EV_TO_KCAL if ev is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        col = f"{label}_kcal"
        valid = df[col].dropna()
        rows.append(
            {
                "model": label,
                "n_systems": len(df),
                "n_valid": len(valid),
                "failure_rate_%": round(100 * (1 - len(valid) / len(df)), 1),
                "mean_strain_kcal": round(valid.mean(), 2) if len(valid) else None,
                "median_strain_kcal": round(valid.median(), 2) if len(valid) else None,
                "std_strain_kcal": round(valid.std(), 2) if len(valid) else None,
            }
        )
    summary = pd.DataFrame(rows)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    return summary


def add_reference_metrics(
    df: pd.DataFrame, labels: list[str], reference_path: str
) -> pd.DataFrame:
    ref = load_results(reference_path)
    ref_col = pd.Series(
        {sid: v * EV_TO_KCAL for sid, v in ref.items() if v is not None},
        name="reference_kcal",
    )
    df = df.join(ref_col, on="system_id")

    rows = []
    for label in labels:
        col = f"{label}_kcal"
        both = df[[col, "reference_kcal"]].dropna()
        if len(both) < 2:
            rows.append({"model": label, "MAE": None, "pearson_r": None, "n_paired": len(both)})
            continue
        mae = (both[col] - both["reference_kcal"]).abs().mean()
        r = both[col].corr(both["reference_kcal"])
        rows.append({"model": label, "MAE": round(mae, 2), "pearson_r": round(r, 4), "n_paired": len(both)})

    ref_df = pd.DataFrame(rows)
    print("\n=== vs Reference ===")
    print(ref_df.to_string(index=False))
    return ref_df


def plot_distributions(df: pd.DataFrame, labels: list[str], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), sharey=False)
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels):
        col = f"{label}_kcal"
        valid = df[col].dropna()
        ax.hist(valid, bins=30, edgecolor="white", linewidth=0.5)
        ax.set_title(label)
        ax.set_xlabel("Strain energy (kcal/mol)")
        ax.set_ylabel("Count")
        ax.axvline(valid.median(), color="red", linestyle="--", linewidth=1, label=f"median={valid.median():.1f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = output_dir / "strain_distributions.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def plot_comparison(df: pd.DataFrame, labels: list[str], output_dir: Path) -> None:
    if len(labels) < 2:
        return

    n = len(labels)
    fig, axes = plt.subplots(1, n - 1, figsize=(5 * (n - 1), 4))
    if n == 2:
        axes = [axes]

    ref_label = labels[0]
    for ax, label in zip(axes, labels[1:]):
        both = df[[f"{ref_label}_kcal", f"{label}_kcal"]].dropna()
        ax.scatter(both[f"{ref_label}_kcal"], both[f"{label}_kcal"], s=8, alpha=0.5)
        lims = [
            min(both[f"{ref_label}_kcal"].min(), both[f"{label}_kcal"].min()),
            max(both[f"{ref_label}_kcal"].max(), both[f"{label}_kcal"].max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=0.8)
        ax.set_xlabel(f"{ref_label} (kcal/mol)")
        ax.set_ylabel(f"{label} (kcal/mol)")
        r = both[f"{ref_label}_kcal"].corr(both[f"{label}_kcal"])
        ax.set_title(f"{ref_label} vs {label}  r={r:.3f}")

    plt.tight_layout()
    out = output_dir / "model_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze ligand strain evaluation results.")
    parser.add_argument("--results", nargs="+", required=True, help="Paths to result JSON files")
    parser.add_argument(
        "--labels", nargs="+", required=True, help="Labels for each model (same order as --results)"
    )
    parser.add_argument("--output-dir", default="results/analysis", help="Directory for outputs")
    parser.add_argument(
        "--reference", default=None, help="Optional: path to DFT reference JSON for MAE/correlation"
    )
    args = parser.parse_args()

    if len(args.results) != len(args.labels):
        raise ValueError("--results and --labels must have the same number of entries")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.results, args.labels)
    summary = print_summary(df, args.labels)

    if args.reference:
        ref_metrics = add_reference_metrics(df, args.labels, args.reference)
        ref_metrics.to_csv(output_dir / "reference_metrics.csv", index=False)

    df.to_csv(output_dir / "strain_energies.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(f"\nCSV saved to {output_dir}/")

    try:
        plot_distributions(df, args.labels, output_dir)
        plot_comparison(df, args.labels, output_dir)
    except Exception as e:
        print(f"Warning: plotting failed ({e}). CSVs are still available.")


if __name__ == "__main__":
    main()
