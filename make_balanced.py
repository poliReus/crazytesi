from pathlib import Path

import pandas as pd


SEED = 20261002

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = (
    BASE_DIR
    / "results"
    / "hatexplain_laya_results_full.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "results"
    / "hatexplain_balanced_laya_results.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    toxic = df[df["ground_truth"] == 1].copy()
    non_toxic = df[df["ground_truth"] == 0].copy()

    n_toxic = len(toxic)
    n_non_toxic = len(non_toxic)

    print("=" * 60)
    print("ORIGINAL DATASET")
    print("=" * 60)
    print(f"Toxic:      {n_toxic}")
    print(f"Non-toxic:  {n_non_toxic}")

    n_per_class = min(
        n_toxic,
        n_non_toxic,
    )

    print()
    print(
        f"Balanced size per class: "
        f"{n_per_class}"
    )

    toxic_sample = toxic.sample(
        n=n_per_class,
        random_state=SEED,
        replace=False,
    )

    non_toxic_sample = non_toxic.sample(
        n=n_per_class,
        random_state=SEED,
        replace=False,
    )

    balanced = pd.concat(
        [
            toxic_sample,
            non_toxic_sample,
        ],
        ignore_index=True,
    )

    balanced = balanced.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    balanced.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("BALANCED DATASET")
    print("=" * 60)

    print(
        balanced[
            "ground_truth"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        f"Total: {len(balanced)}"
    )

    print()
    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()