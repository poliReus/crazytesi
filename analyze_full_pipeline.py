from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    beta,
    kstest,
    pearsonr,
    spearmanr,
)

from scipy.integrate import quad

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_PATH = (
    BASE_DIR
    / "results"
    / "hatexplain_balanced_laya_results.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "balanced_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.txt"
)

METRICS_CSV = (
    OUTPUT_DIR
    / "metrics.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EPS = 1e-6

CALIBRATION_BINS = 20

HISTOGRAM_BINS = 60


# ============================================================
# BETA FIT
# ============================================================

def fit_beta(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.clip(
        values,
        EPS,
        1 - EPS,
    )

    alpha, beta_param, _, _ = (
        beta.fit(
            values,
            floc=0,
            fscale=1,
        )
    )

    return (
        alpha,
        beta_param,
    )


# ============================================================
# CALIBRATION ERROR
# ============================================================

def calibration_error(
    y,
    p,
    n_bins=20,
):

    y = np.asarray(y)
    p = np.asarray(p)

    bins = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    error = 0.0

    rows = []

    for i, (
        lower,
        upper,
    ) in enumerate(
        zip(
            bins[:-1],
            bins[1:],
        )
    ):

        if upper == 1:

            mask = (
                (p >= lower)
                & (p <= upper)
            )

        else:

            mask = (
                (p >= lower)
                & (p < upper)
            )

        n = mask.sum()

        if n == 0:
            continue

        mean_probability = (
            p[mask].mean()
        )

        empirical_frequency = (
            y[mask].mean()
        )

        weight = (
            n / len(y)
        )

        bin_error = abs(
            empirical_frequency
            - mean_probability
        )

        error += (
            weight
            * bin_error
        )

        rows.append(
            {
                "bin": i,
                "lower": lower,
                "upper": upper,
                "count": n,
                "mean_probability":
                    mean_probability,
                "empirical_frequency":
                    empirical_frequency,
                "absolute_error":
                    bin_error,
            }
        )

    return (
        error,
        pd.DataFrame(rows),
    )


# ============================================================
# APPROXIMATE KL DIVERGENCE
# ============================================================

def empirical_beta_kl(
    values,
    alpha,
    beta_param,
    bins=100,
):

    values = np.asarray(
        values
    )

    empirical, edges = (
        np.histogram(
            values,
            bins=bins,
            range=(0, 1),
            density=False,
        )
    )

    empirical = (
        empirical
        / empirical.sum()
    )

    theoretical = []

    for left, right in zip(
        edges[:-1],
        edges[1:],
    ):

        probability = (
            beta.cdf(
                right,
                alpha,
                beta_param,
            )
            -
            beta.cdf(
                left,
                alpha,
                beta_param,
            )
        )

        theoretical.append(
            probability
        )

    theoretical = np.asarray(
        theoretical
    )

    empirical = np.clip(
        empirical,
        EPS,
        None,
    )

    theoretical = np.clip(
        theoretical,
        EPS,
        None,
    )

    empirical /= (
        empirical.sum()
    )

    theoretical /= (
        theoretical.sum()
    )

    kl = np.sum(
        empirical
        * np.log(
            empirical
            / theoretical
        )
    )

    return kl


# ============================================================
# SUBJECTIVE LOGIC
# ============================================================

def subjective_logic(
    alpha,
    beta_param,
):

    denominator = (
        2 * alpha
        + 2 * beta_param
    )

    belief = (
        2 * alpha - 1
    ) / denominator

    disbelief = (
        2 * beta_param - 1
    ) / denominator

    uncertainty = (
        1
        / (
            alpha
            + beta_param
        )
    )

    return {
        "belief": belief,
        "disbelief": disbelief,
        "uncertainty": uncertainty,
        "proper_opinion": (
            alpha >= 1
            and beta_param >= 1
        ),
    }


# ============================================================
# METRICS FROM BETA DISTRIBUTIONS
# ============================================================

def beta_metrics(
    alpha_pos,
    beta_pos,
    alpha_neg,
    beta_neg,
    positive_prior,
    threshold=0.5,
):

    negative_prior = (
        1
        - positive_prior
    )

    # P(predicted YES | true YES)
    tpr = (
        1
        -
        beta.cdf(
            threshold,
            alpha_pos,
            beta_pos,
        )
    )

    # P(predicted NO | true NO)
    tnr = (
        beta.cdf(
            threshold,
            alpha_neg,
            beta_neg,
        )
    )

    fpr = 1 - tnr
    fnr = 1 - tpr

    accuracy = (
        positive_prior * tpr
        +
        negative_prior * tnr
    )

    # Precision estimated using Bayes rule
    predicted_positive = (
        positive_prior * tpr
        +
        negative_prior * fpr
    )

    if predicted_positive > 0:

        precision = (
            positive_prior * tpr
            / predicted_positive
        )

    else:

        precision = np.nan

    # ROC AUC:
    # P(score_positive > score_negative)

    def integrand(x):

        f_pos = beta.pdf(
            x,
            alpha_pos,
            beta_pos,
        )

        f_neg_cdf = beta.cdf(
            x,
            alpha_neg,
            beta_neg,
        )

        return (
            f_pos
            * f_neg_cdf
        )

    roc_auc, _ = quad(
        integrand,
        0,
        1,
        limit=300,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": tpr,
        "specificity": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
    }


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    RESULTS_PATH
)

y = df[
    "ground_truth"
].to_numpy(
    dtype=int
)

p = df[
    "laya_probability"
].to_numpy(
    dtype=float
)

pred = (
    p >= 0.5
).astype(int)

human_p = df[
    "human_toxic_probability"
].to_numpy(
    dtype=float
)


# ============================================================
# EMPIRICAL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y,
    pred,
)

precision = precision_score(
    y,
    pred,
)

recall = recall_score(
    y,
    pred,
)

tn, fp, fn, tp = (
    confusion_matrix(
        y,
        pred,
    )
    .ravel()
)

specificity = (
    tn
    / (
        tn + fp
    )
)

roc_auc = roc_auc_score(
    y,
    p,
)

brier = brier_score_loss(
    y,
    p,
)

logloss = log_loss(
    y,
    np.clip(
        p,
        EPS,
        1 - EPS,
    ),
)

cal_error, calibration_df = (
    calibration_error(
        y,
        p,
        CALIBRATION_BINS,
    )
)

calibration_df.to_csv(
    OUTPUT_DIR
    / "calibration_bins.csv",
    index=False,
)


# ============================================================
# HUMAN AGREEMENT
# ============================================================

pearson = pearsonr(
    human_p,
    p,
)

spearman = spearmanr(
    human_p,
    p,
)


# ============================================================
# SPLIT CLASSES
# ============================================================

toxic_values = p[
    y == 1
]

normal_values = p[
    y == 0
]


# ============================================================
# BETA FIT
# ============================================================

alpha_toxic, beta_toxic = (
    fit_beta(
        toxic_values
    )
)

alpha_normal, beta_normal = (
    fit_beta(
        normal_values
    )
)


# ============================================================
# KS
# ============================================================

ks_toxic = kstest(
    np.clip(
        toxic_values,
        EPS,
        1 - EPS,
    ),
    "beta",
    args=(
        alpha_toxic,
        beta_toxic,
        0,
        1,
    ),
)

ks_normal = kstest(
    np.clip(
        normal_values,
        EPS,
        1 - EPS,
    ),
    "beta",
    args=(
        alpha_normal,
        beta_normal,
        0,
        1,
    ),
)


# ============================================================
# KL FIT ERROR
# ============================================================

kl_toxic = empirical_beta_kl(
    toxic_values,
    alpha_toxic,
    beta_toxic,
)

kl_normal = empirical_beta_kl(
    normal_values,
    alpha_normal,
    beta_normal,
)


# ============================================================
# SUBJECTIVE LOGIC
# ============================================================

sl_toxic = subjective_logic(
    alpha_toxic,
    beta_toxic,
)

sl_normal = subjective_logic(
    alpha_normal,
    beta_normal,
)


# ============================================================
# RECONSTRUCT CLASSICAL METRICS FROM BETA
# ============================================================

positive_prior = (
    y.mean()
)

beta_reconstruction = (
    beta_metrics(
        alpha_toxic,
        beta_toxic,
        alpha_normal,
        beta_normal,
        positive_prior,
    )
)


# ============================================================
# PRINT SUMMARY
# ============================================================

lines = []


def out(text=""):

    print(text)
    lines.append(
        str(text)
    )


out(
    "=" * 70
)

out(
    "HATEXPLAIN + LAYA — FULL ANALYSIS"
)

out(
    "=" * 70
)

out()

out(
    f"N samples:           {len(df)}"
)

out(
    f"Toxic:               {(y == 1).sum()}"
)

out(
    f"Non-toxic:           {(y == 0).sum()}"
)

out(
    f"Toxic prior:         {positive_prior:.6f}"
)


out()

out(
    "=" * 70
)

out(
    "EMPIRICAL PERFORMANCE"
)

out(
    "=" * 70
)

out(
    f"Accuracy:            {accuracy:.6f}"
)

out(
    f"Precision:           {precision:.6f}"
)

out(
    f"Recall:              {recall:.6f}"
)

out(
    f"Specificity:         {specificity:.6f}"
)

out(
    f"ROC AUC:             {roc_auc:.6f}"
)

out(
    f"Brier score:         {brier:.6f}"
)

out(
    f"Log loss:            {logloss:.6f}"
)

out(
    f"Calibration error:   {cal_error:.6f}"
)


out()

out(
    "=" * 70
)

out(
    "LAYA VS HUMAN AGREEMENT"
)

out(
    "=" * 70
)

out(
    f"Pearson r:           {pearson.statistic:.6f}"
)

out(
    f"Pearson p:           {pearson.pvalue:.6e}"
)

out(
    f"Spearman rho:        {spearman.statistic:.6f}"
)

out(
    f"Spearman p:          {spearman.pvalue:.6e}"
)


out()

out(
    "=" * 70
)

out(
    "BETA — TOXIC"
)

out(
    "=" * 70
)

out(
    f"Alpha:               {alpha_toxic:.6f}"
)

out(
    f"Beta:                {beta_toxic:.6f}"
)

out(
    f"Empirical mean:      {toxic_values.mean():.6f}"
)

out(
    f"Beta mean:           "
    f"{alpha_toxic / (alpha_toxic + beta_toxic):.6f}"
)

out(
    f"KS statistic:        {ks_toxic.statistic:.6f}"
)

out(
    f"KS p-value:          {ks_toxic.pvalue:.6e}"
)

out(
    f"KL fitting error:    {kl_toxic:.6f}"
)


out()

out(
    "=" * 70
)

out(
    "BETA — NON-TOXIC"
)

out(
    "=" * 70
)

out(
    f"Alpha:               {alpha_normal:.6f}"
)

out(
    f"Beta:                {beta_normal:.6f}"
)

out(
    f"Empirical mean:      {normal_values.mean():.6f}"
)

out(
    f"Beta mean:           "
    f"{alpha_normal / (alpha_normal + beta_normal):.6f}"
)

out(
    f"KS statistic:        {ks_normal.statistic:.6f}"
)

out(
    f"KS p-value:          {ks_normal.pvalue:.6e}"
)

out(
    f"KL fitting error:    {kl_normal:.6f}"
)


out()

out(
    "=" * 70
)

out(
    "SUBJECTIVE LOGIC — TOXIC"
)

out(
    "=" * 70
)

out(
    f"Belief:              {sl_toxic['belief']:.6f}"
)

out(
    f"Disbelief:           {sl_toxic['disbelief']:.6f}"
)

out(
    f"Uncertainty:         {sl_toxic['uncertainty']:.6f}"
)

out(
    f"Proper opinion:      {sl_toxic['proper_opinion']}"
)


out()

out(
    "=" * 70
)

out(
    "SUBJECTIVE LOGIC — NON-TOXIC"
)

out(
    "=" * 70
)

out(
    f"Belief:              {sl_normal['belief']:.6f}"
)

out(
    f"Disbelief:           {sl_normal['disbelief']:.6f}"
)

out(
    f"Uncertainty:         {sl_normal['uncertainty']:.6f}"
)

out(
    f"Proper opinion:      {sl_normal['proper_opinion']}"
)


out()

out(
    "=" * 70
)

out(
    "CLASSICAL METRICS RECONSTRUCTED FROM BETA"
)

out(
    "=" * 70
)

for key, value in (
    beta_reconstruction.items()
):

    out(
        f"{key:20s}: "
        f"{value:.6f}"
    )


out()

out(
    "=" * 70
)

out(
    "BETA RECONSTRUCTION ERRORS"
)

out(
    "=" * 70
)

out(
    f"Accuracy error:      "
    f"{abs(beta_reconstruction['accuracy'] - accuracy):.6f}"
)

out(
    f"ROC AUC error:       "
    f"{abs(beta_reconstruction['roc_auc'] - roc_auc):.6f}"
)

out(
    f"Recall error:        "
    f"{abs(beta_reconstruction['recall'] - recall):.6f}"
)

out(
    f"Specificity error:   "
    f"{abs(beta_reconstruction['specificity'] - specificity):.6f}"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(lines)
    )


# ============================================================
# SAVE METRICS CSV
# ============================================================

metrics = {
    "n": len(df),

    "accuracy_empirical":
        accuracy,

    "accuracy_beta":
        beta_reconstruction[
            "accuracy"
        ],

    "roc_auc_empirical":
        roc_auc,

    "roc_auc_beta":
        beta_reconstruction[
            "roc_auc"
        ],

    "calibration_error":
        cal_error,

    "brier":
        brier,

    "alpha_toxic":
        alpha_toxic,

    "beta_toxic":
        beta_toxic,

    "alpha_normal":
        alpha_normal,

    "beta_normal":
        beta_normal,

    "kl_toxic":
        kl_toxic,

    "kl_normal":
        kl_normal,

    "belief_toxic":
        sl_toxic[
            "belief"
        ],

    "disbelief_toxic":
        sl_toxic[
            "disbelief"
        ],

    "uncertainty_toxic":
        sl_toxic[
            "uncertainty"
        ],

    "belief_normal":
        sl_normal[
            "belief"
        ],

    "disbelief_normal":
        sl_normal[
            "disbelief"
        ],

    "uncertainty_normal":
        sl_normal[
            "uncertainty"
        ],

    "human_pearson":
        pearson.statistic,

    "human_spearman":
        spearman.statistic,
}

pd.DataFrame(
    [metrics]
).to_csv(
    METRICS_CSV,
    index=False,
)


# ============================================================
# FIGURE 1 — THESIS-STYLE BETA FIT
# ============================================================

x = np.linspace(
    0.001,
    0.999,
    1000,
)

fig, ax = plt.subplots(
    figsize=(12, 7),
)

ax.hist(
    toxic_values,
    bins=HISTOGRAM_BINS,
    range=(0, 1),
    density=True,
    alpha=0.55,
    label="Toxic",
)

ax.hist(
    normal_values,
    bins=HISTOGRAM_BINS,
    range=(0, 1),
    density=True,
    alpha=0.55,
    label="Non-toxic",
)

ax.plot(
    x,
    beta.pdf(
        x,
        alpha_toxic,
        beta_toxic,
    ),
    linewidth=2.4,
    label=(
        rf"Beta Toxic "
        rf"($\alpha={alpha_toxic:.2f}$, "
        rf"$\beta={beta_toxic:.2f}$)"
    ),
)

ax.plot(
    x,
    beta.pdf(
        x,
        alpha_normal,
        beta_normal,
    ),
    linewidth=2.4,
    label=(
        rf"Beta Non-toxic "
        rf"($\alpha={alpha_normal:.2f}$, "
        rf"$\beta={beta_normal:.2f}$)"
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

fig.savefig(
    OUTPUT_DIR
    / "beta_fit_thesis_style.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2 — RELIABILITY DIAGRAM
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7),
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)

ax.plot(
    calibration_df[
        "mean_probability"
    ],
    calibration_df[
        "empirical_frequency"
    ],
    marker="o",
    label="Laya",
)

ax.set_xlim(
    0,
    1,
)

ax.set_ylim(
    0,
    1,
)

ax.set_xlabel(
    "Predicted probability"
)

ax.set_ylabel(
    "Observed toxic frequency"
)

ax.set_title(
    "Reliability diagram"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "reliability_diagram.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3 — HUMAN CONSENSUS VS LAYA
# ============================================================

grouped = (
    df.groupby(
        "human_toxic_probability"
    )[
        "laya_probability"
    ]
    .agg(
        [
            "mean",
            "std",
            "count",
        ]
    )
    .reset_index()
)

fig, ax = plt.subplots(
    figsize=(8, 6),
)

ax.errorbar(
    grouped[
        "human_toxic_probability"
    ],
    grouped[
        "mean"
    ],
    yerr=grouped[
        "std"
    ],
    marker="o",
    capsize=4,
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

ax.set_xlim(
    0,
    1,
)

ax.set_ylim(
    0,
    1,
)

ax.set_xlabel(
    "Human toxic vote proportion"
)

ax.set_ylabel(
    "Mean Laya probability"
)

ax.set_title(
    "Laya probability vs human agreement"
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "human_agreement_vs_laya.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print(
    f"Analysis saved in:\n"
    f"{OUTPUT_DIR}"
)