"""Plot one test engine in four panels. Requires: pandas, numpy, matplotlib.

Example:
python plot_hi_comparison.py --unit 12 --kgld kgld.csv \
    --proposal-c proposal_c.csv --residual-a residual_a.csv \
    --residual-b residual_b.csv --output figures/hi_unit12

The CSV files must contain split, unit, cycle, hi, hi_gt. Markdown artifacts
and HTML character references shown in the supplied example are tolerated.
"""

import argparse
import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


STYLES = [
    ("KGLD", "#D55E00", "kgld"),
    ("Hybrid AE", "#0072B2", "proposal_c"),
    ("Residual-AE", "#009E73", "residual_a"),
    ("Residual-Regression", "#CC79A7", "residual_b"),
]


def clean(value):
    return html.unescape(str(value)).replace("*", "").strip().lstrip("_")


def read_engine(path, unit, split):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [clean(c) for c in df.columns]
    required = ["split", "unit", "cycle", "hi", "hi_gt"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df.columns.duplicated().any():
        raise ValueError(f"{path}: duplicate column names after cleaning")
    df["split"] = df["split"].map(clean).str.lower()
    df["unit"] = pd.to_numeric(df["unit"].map(clean), errors="coerce")
    df = df.loc[(df["split"] == split.lower()) & (df["unit"] == unit)].copy()
    if df.empty:
        raise ValueError(f"{path}: no records for split={split}, unit={unit}")
    for col in ["cycle", "hi", "hi_gt"]:
        df[col] = pd.to_numeric(df[col].map(clean), errors="raise")
    if not np.isfinite(df[["cycle", "hi", "hi_gt"]].to_numpy()).all():
        raise ValueError(f"{path}: non-finite cycle, hi, or hi_gt")
    if df["cycle"].duplicated().any():
        raise ValueError(f"{path}: repeated cycles for this engine")
    return df.sort_values("cycle").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", type=int, default=12)
    parser.add_argument("--split", default="test")
    for _, _, key in STYLES:
        parser.add_argument("--" + key.replace("_", "-"), required=True)
    parser.add_argument("--output", default="figures/hi_comparison",
                        help="Output prefix without extension")
    args = parser.parse_args()
    datasets = [read_engine(getattr(args, key), args.unit, args.split)
                for _, _, key in STYLES]

    # A fair comparison uses the same flights and the same reference HI.
    reference = datasets[0]
    for (name, _, _), df in zip(STYLES[1:], datasets[1:]):
        if not np.array_equal(df["cycle"].to_numpy(), reference["cycle"].to_numpy()):
            raise ValueError(f"{name}: cycles differ from KGLD; align CSV records first")
        if not np.allclose(df["hi_gt"], reference["hi_gt"], rtol=1e-6, atol=1e-8):
            raise ValueError(f"{name}: reference HI differs from KGLD")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8), sharex=True, sharey=True)
    values = np.concatenate([df[["hi", "hi_gt"]].to_numpy().ravel() for df in datasets])
    low, high = min(0., values.min()), max(1., values.max())
    pad = .04 * max(high - low, .1)

    for i, (ax, (name, color, _), df) in enumerate(zip(axes.flat, STYLES, datasets)):
        # Use exported flight-level HI directly: no renormalization or smoothing.
        pred, = ax.plot(df["cycle"], df["hi"], color=color, lw=1.5,
                        linestyle="-", label=name, zorder=3)
        ref, = ax.plot(df["cycle"], df["hi_gt"], color="#333333", lw=1.8,
                       linestyle="--", dashes=(4, 2.5), label="Reference HI", zorder=4)
        ax.set_title(f"({chr(97+i)}) {name}", pad=7)
        ax.legend(handles=[ref, pred], loc="lower left", frameon=False, handlelength=2.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", linewidth=.55)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=3, width=.65)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_ylim(low - pad, high + pad)
        ax.margins(x=.01)
        if i >= 2:
            ax.set_xlabel("Flight cycle")
        if i % 2 == 0:
            ax.set_ylabel("Health index")

    fig.suptitle(f"Unit {args.unit}", fontsize=10, y=.985)
    fig.subplots_adjust(left=.085, right=.985, bottom=.10, top=.90,
                        hspace=.31, wspace=.17)
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        output = Path(str(prefix) + "." + extension)
        fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
        print(f"Saved: {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
