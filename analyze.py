from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import beta, kstest, pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent

RESULTS_PATH = (
    BASE_DIR
    / "results"
    / "hatexplain_laya_results.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EPS = 1e-6
N_BINS = 50


def fit_beta(values):
    values = np.clip(
        np.asarray(values, dtype=float),
        EPS,
        1 - EPS,
    )

    a, b, _, _ = beta.fit(
        values,
        floc=0,
        fscale=1,
    )

    return a, b


def expected_calibration_error(
    y,
    p,
    n_bins=10,
):
    bins = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    total = len(y)
    ece = 0.0

    for lo, hi in zip(
        bins[:-1],
        bins[1:],
    ):
        if hi == 1:
            mask = (
                (p >= lo)
                & (p <= hi)
            )
        else:
            mask = (
                (p >= lo)
                & (p < hi)
            )

        n = mask.sum()

        if n == 0:
            continue

        observed = y[mask].mean()
        predicted = p[mask].mean()

        ece += (
            n / total
        ) * abs(
            observed - predicted
        )

    return ece


df = pd.read_csv(
    RESULTS_PATH
)

y = df[
    "ground_truth"
].to_numpy(dtype=int)

p = df[
    "laya_probability"
].to_numpy(dtype=float)

pred = (
    p >= 0.5
).astype(int)

human_p = df[
    "human_toxic_probability"
].to_numpy(dtype=float)


print()
print("=" * 65)
print("GENERAL PERFORMANCE")
print("=" * 65)

print(
    f"N:          {len(df)}"
)

print(
    f"Accuracy:   "
    f"{accuracy_score(y, pred):.4f}"
)

print(
    f"ROC AUC:    "
    f"{roc_auc_score(y, p):.4f}"
)

print(
    f"Brier:      "
    f"{brier_score_loss(y, p):.4f}"
)

print(
    f"Log loss:   "
    f"{log_loss(y, np.clip(p, EPS, 1-EPS)):.4f}"
)

print(
    f"ECE:        "
    f"{expected_calibration_error(y, p):.4f}"
)


# --------------------------------------------------
# Human agreement correlation
# --------------------------------------------------

pearson = pearsonr(
    human_p,
    p,
)

spearman = spearmanr(
    human_p,
    p,
)

print()
print("=" * 65)
print("LAYA VS HUMAN AGREEMENT")
print("=" * 65)

print(
    f"Pearson r:  "
    f"{pearson.statistic:.4f}"
)

print(
    f"Spearman ρ: "
    f"{spearman.statistic:.4f}"
)


# --------------------------------------------------
# Beta distributions
# --------------------------------------------------

toxic_values = p[y == 1]
normal_values = p[y == 0]

a_tox, b_tox = fit_beta(
    toxic_values
)

a_norm, b_norm = fit_beta(
    normal_values
)

ks_tox = kstest(
    np.clip(
        toxic_values,
        EPS,
        1 - EPS,
    ),
    "beta",
    args=(
        a_tox,
        b_tox,
        0,
        1,
    ),
)

ks_norm = kstest(
    np.clip(
        normal_values,
        EPS,
        1 - EPS,
    ),
    "beta",
    args=(
        a_norm,
        b_norm,
        0,
        1,
    ),
)


print()
print("=" * 65)
print("BETA FIT")
print("=" * 65)

print()
print("TOXIC")
print(
    f"alpha = {a_tox:.6f}"
)
print(
    f"beta  = {b_tox:.6f}"
)
print(
    f"mean  = "
    f"{toxic_values.mean():.6f}"
)
print(
    f"KS    = "
    f"{ks_tox.statistic:.6f}"
)
print(
    f"KS p  = "
    f"{ks_tox.pvalue:.6f}"
)

print()
print("NON-TOXIC")
print(
    f"alpha = {a_norm:.6f}"
)
print(
    f"beta  = {b_norm:.6f}"
)
print(
    f"mean  = "
    f"{normal_values.mean():.6f}"
)
print(
    f"KS    = "
    f"{ks_norm.statistic:.6f}"
)
print(
    f"KS p  = "
    f"{ks_norm.pvalue:.6f}"
)


# --------------------------------------------------
# Thesis-style plot
# --------------------------------------------------

x = np.linspace(
    0.001,
    0.999,
    1000,
)

fig, ax = plt.subplots(
    figsize=(11, 7),
)

ax.hist(
    toxic_values,
    bins=N_BINS,
    range=(0, 1),
    density=True,
    alpha=0.55,
    label="Toxic",
)

ax.hist(
    normal_values,
    bins=N_BINS,
    range=(0, 1),
    density=True,
    alpha=0.55,
    label="Non-toxic",
)

ax.plot(
    x,
    beta.pdf(
        x,
        a_tox,
        b_tox,
    ),
    linewidth=2.3,
    label=(
        rf"Beta Toxic "
        rf"($\alpha={a_tox:.2f}$, "
        rf"$\beta={b_tox:.2f}$)"
    ),
)

ax.plot(
    x,
    beta.pdf(
        x,
        a_norm,
        b_norm,
    ),
    linewidth=2.3,
    label=(
        rf"Beta Non-toxic "
        rf"($\alpha={a_norm:.2f}$, "
        rf"$\beta={b_norm:.2f}$)"
    ),
)

ax.set_xlim(
    0,
    1,
)

ax.set_xlabel(
    "Output probability"
)

ax.set_ylabel(
    "Density of samples"
)

ax.set_title(
    "Laya output distributions on HateXplain"
)

ax.legend()

fig.tight_layout()

plot_path = (
    OUTPUT_DIR
    / "hatexplain_beta_fit.png"
)

fig.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print()
print(
    f"Plot saved to:\n{plot_path}"
)