import json
import random
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


SEED = 20261001

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_PATH = DATA_DIR / "hatexplain_dataset.json"
OUTPUT_PATH = DATA_DIR / "hatexplain_full.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "hate-alert/HateXplain/master/Data/dataset.json"
)

random.seed(SEED)


def download_dataset():
    if RAW_PATH.exists():
        print(f"Dataset already exists: {RAW_PATH}")
        return

    print("Downloading HateXplain...")

    urlretrieve(
        DATA_URL,
        RAW_PATH,
    )

    print(f"Saved to: {RAW_PATH}")


def binary_vote(label):
    label = label.lower().strip()

    if label in {
        "hatespeech",
        "offensive",
    }:
        return 1

    if label == "normal":
        return 0

    raise ValueError(
        f"Unknown label: {label}"
    )


def majority_label(votes):
    return int(
        sum(votes) > len(votes) / 2
    )


def main():
    download_dataset()

    print()
    print("Loading dataset...")

    with open(
        RAW_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    rows = []

    skipped_no_annotators = 0
    skipped_empty_text = 0

    for post_id, item in data.items():

        tokens = item.get(
            "post_tokens",
            []
        )

        if not tokens:
            skipped_empty_text += 1
            continue

        text = " ".join(tokens).strip()

        if not text:
            skipped_empty_text += 1
            continue

        annotators = item.get(
            "annotators",
            []
        )

        if not annotators:
            skipped_no_annotators += 1
            continue

        labels = [
            a["label"]
            for a in annotators
        ]

        votes = [
            binary_vote(label)
            for label in labels
        ]

        toxic_votes = sum(votes)
        total_votes = len(votes)

        human_toxic_probability = (
            toxic_votes
            / total_votes
        )

        ground_truth = majority_label(
            votes
        )

        rows.append(
            {
                "post_id":
                    post_id,

                "text":
                    text,

                "ground_truth":
                    ground_truth,

                "toxic_votes":
                    toxic_votes,

                "total_votes":
                    total_votes,

                "human_toxic_probability":
                    human_toxic_probability,

                "annotator_labels":
                    "|".join(labels),
            }
        )

    df = pd.DataFrame(rows)

    print()
    print(
        f"Valid examples loaded: "
        f"{len(df)}"
    )

    print(
        f"Skipped without annotators: "
        f"{skipped_no_annotators}"
    )

    print(
        f"Skipped empty texts: "
        f"{skipped_empty_text}"
    )

    # Shuffle reproducibly.
    final = df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    final.insert(
        0,
        "scenario_id",
        [
            f"HX{i:05d}"
            for i in range(
                1,
                len(final) + 1
            )
        ],
    )

    final.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("FULL HATEXPLAIN DATASET")
    print("=" * 60)

    print()
    print(
        f"Saved to:\n{OUTPUT_PATH}"
    )

    print()
    print(
        f"Total examples: "
        f"{len(final)}"
    )

    print()
    print("Class counts:")
    print(
        final[
            "ground_truth"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("Class percentages:")
    print(
        (
            final[
                "ground_truth"
            ]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        ).round(2)
    )

    print()
    print(
        "Human toxic probability "
        "distribution:"
    )

    print(
        final[
            "human_toxic_probability"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Annotator-count distribution:"
    )

    print(
        final[
            "total_votes"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()