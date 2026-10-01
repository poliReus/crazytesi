import json
import random
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


SEED = 20261001
N_PER_CLASS = 1000

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_PATH = DATA_DIR / "hatexplain_dataset.json"
OUTPUT_PATH = DATA_DIR / "hatexplain_2000.csv"

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
    urlretrieve(DATA_URL, RAW_PATH)
    print(f"Saved to: {RAW_PATH}")


def binary_vote(label):
    label = label.lower()

    if label in {"hatespeech", "offensive"}:
        return 1

    if label == "normal":
        return 0

    raise ValueError(f"Unknown label: {label}")


def majority_label(votes):
    return int(sum(votes) > len(votes) / 2)


def main():
    download_dataset()

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for post_id, item in data.items():
        tokens = item["post_tokens"]
        text = " ".join(tokens)

        annotators = item.get("annotators", [])

        if not annotators:
            continue

        votes = [
            binary_vote(a["label"])
            for a in annotators
        ]

        toxic_votes = sum(votes)
        total_votes = len(votes)
        human_toxic_probability = toxic_votes / total_votes

        ground_truth = majority_label(votes)

        rows.append(
            {
                "post_id": post_id,
                "text": text,
                "ground_truth": ground_truth,
                "toxic_votes": toxic_votes,
                "total_votes": total_votes,
                "human_toxic_probability": human_toxic_probability,
                "annotator_labels": "|".join(
                    a["label"] for a in annotators
                ),
            }
        )

    df = pd.DataFrame(rows)

    toxic = df[df["ground_truth"] == 1]
    non_toxic = df[df["ground_truth"] == 0]

    if len(toxic) < N_PER_CLASS or len(non_toxic) < N_PER_CLASS:
        raise RuntimeError(
            f"Not enough examples: "
            f"toxic={len(toxic)}, "
            f"non_toxic={len(non_toxic)}"
        )

    toxic_sample = toxic.sample(
        n=N_PER_CLASS,
        random_state=SEED,
    )

    non_toxic_sample = non_toxic.sample(
        n=N_PER_CLASS,
        random_state=SEED,
    )

    final = pd.concat(
        [toxic_sample, non_toxic_sample],
        ignore_index=True,
    )

    final = final.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    final.insert(
        0,
        "scenario_id",
        [
            f"HX{i:04d}"
            for i in range(1, len(final) + 1)
        ],
    )

    final.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("Class counts:")
    print(final["ground_truth"].value_counts().sort_index())

    print()
    print("Human toxic probability:")
    print(
        final["human_toxic_probability"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()