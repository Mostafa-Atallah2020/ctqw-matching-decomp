#!/usr/bin/env python3
"""
Combined Trotterization 2-Norm Error Plot: Greedy vs Compression-Aware vs Pauli.

Produces one figure per graph size showing three time values
(t = 0.1, 0.5, 1.0) for each of the three methods, with +/- std bands
aggregated over all timestamp runs. Styled for npj Quantum Information:
sans-serif, single-column width (~89 mm), inward ticks, no grid.

Usage:
  python plot_trotterization_combined.py \
      --greedy outputs/trotterization_error/counting_16v \
      --comp-aware outputs_comp_aware/trotterization_error/counting_16v \
      --output .../assets/trotterization_error_16v.pdf \
      [--times 0.1 0.5 1.0] [--title "N = 16"]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# npj Quantum Information style (11 pt body, matching document)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 1.2,
    'lines.markersize': 3.5,
    'legend.frameon': False,
})

SHADE_ALPHA = 0.15

COLORS = {
    "greedy":            "#1f77b4",  # blue
    "compression_aware": "#2ca02c",  # green
    "pauli":             "#d62728",  # red
}
MARKERS = {"greedy": "o", "compression_aware": "s", "pauli": "^"}
LINESTYLES_BY_TIME = ["-", "--", ":", "-."]


def load_runs(data_dir):
    """Load raw_results.csv from all timestamp subfolders (or self)."""
    data_dir = Path(data_dir)
    dfs = []
    for sub in sorted(data_dir.iterdir() if data_dir.is_dir() else []):
        if sub.is_dir() and (sub / "raw_results.csv").exists():
            dfs.append(pd.read_csv(sub / "raw_results.csv"))
    if not dfs and (data_dir / "raw_results.csv").exists():
        dfs.append(pd.read_csv(data_dir / "raw_results.csv"))
    if not dfs:
        raise FileNotFoundError(f"No raw_results.csv under {data_dir}")
    return pd.concat(dfs, ignore_index=True)


def aggregate(df):
    """Aggregate by (time, trotter_steps): mean +/- std."""
    rows = []
    for t in sorted(df["time"].unique()):
        for n_steps in sorted(df["trotter_steps"].unique()):
            mask = (df["time"] == t) & (df["trotter_steps"] == n_steps)
            subset = df[mask]
            if subset.empty:
                continue
            m = subset["matching_diff"].dropna().values
            p = subset["pauli_diff"].dropna().values
            rows.append({
                "time": t, "trotter_steps": n_steps,
                "matching_mean": np.mean(m) if len(m) else np.nan,
                "matching_std":  np.std(m)  if len(m) else np.nan,
                "pauli_mean":    np.mean(p) if len(p) else np.nan,
                "pauli_std":     np.std(p)  if len(p) else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", required=True)
    parser.add_argument("--comp-aware", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--times", nargs="+", type=float,
                        default=[0.1, 0.5, 1.0])
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    g = aggregate(load_runs(args.greedy))
    c = aggregate(load_runs(args.comp_aware))

    fig, ax = plt.subplots(figsize=(3.6, 2.9))

    for ti, t in enumerate(args.times):
        ls = LINESTYLES_BY_TIME[ti % len(LINESTYLES_BY_TIME)]

        gs = g[g["time"] == t].sort_values("trotter_steps")
        cs = c[c["time"] == t].sort_values("trotter_steps")

        if gs.empty or cs.empty:
            continue

        x_g = gs["trotter_steps"].values
        x_c = cs["trotter_steps"].values

        ax.plot(x_g, gs["matching_mean"], marker=MARKERS["greedy"],
                linestyle=ls, color=COLORS["greedy"])
        ax.fill_between(x_g,
                        gs["matching_mean"] - gs["matching_std"],
                        gs["matching_mean"] + gs["matching_std"],
                        color=COLORS["greedy"], alpha=SHADE_ALPHA, linewidth=0)

        ax.plot(x_c, cs["matching_mean"], marker=MARKERS["compression_aware"],
                linestyle=ls, color=COLORS["compression_aware"])
        ax.fill_between(x_c,
                        cs["matching_mean"] - cs["matching_std"],
                        cs["matching_mean"] + cs["matching_std"],
                        color=COLORS["compression_aware"], alpha=SHADE_ALPHA, linewidth=0)

        ax.plot(x_g, gs["pauli_mean"], marker=MARKERS["pauli"],
                linestyle=ls, color=COLORS["pauli"])
        ax.fill_between(x_g,
                        gs["pauli_mean"] - gs["pauli_std"],
                        gs["pauli_mean"] + gs["pauli_std"],
                        color=COLORS["pauli"], alpha=SHADE_ALPHA, linewidth=0)

    method_handles = [
        Line2D([0], [0], color=COLORS["greedy"], marker="o", linestyle="-",
               label="Greedy"),
        Line2D([0], [0], color=COLORS["compression_aware"], marker="s",
               linestyle="-", label="Comp-aware"),
        Line2D([0], [0], color=COLORS["pauli"], marker="^", linestyle="-",
               label="Pauli"),
    ]
    time_handles = [
        Line2D([0], [0], color="gray",
               linestyle=LINESTYLES_BY_TIME[ti % len(LINESTYLES_BY_TIME)],
               label=rf"$t = {t}$")
        for ti, t in enumerate(args.times)
    ]

    # Two legends stacked vertically below the plot (below x-axis label)
    leg1 = ax.legend(handles=method_handles,
                     loc="upper center", bbox_to_anchor=(0.5, -0.25),
                     title="Method", title_fontsize=7,
                     ncol=len(method_handles),
                     frameon=False,
                     borderaxespad=0.)
    ax.add_artist(leg1)
    ax.legend(handles=time_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.45),
              title="Time", title_fontsize=7,
              ncol=len(time_handles),
              frameon=False,
              borderaxespad=0.)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trotter steps")
    ax.set_ylabel(r"$\| U_{\mathrm{exact}} - U_{\mathrm{Trotter}} \|_2$")

    # Linear-spaced tick labels on log axis (1, 2, 5, 10, 20)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    xticks = [1, 2, 5, 10, 20]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in xticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if args.title:
        ax.set_title(args.title)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.3)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
