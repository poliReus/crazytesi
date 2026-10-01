import json
import time
from pathlib import Path

import pandas as pd
from laya import Router


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "hatexplain_full.csv"
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_PATH = (
    RESULTS_DIR
    / "hatexplain_laya_results_full.csv"
)

RAW_PATH = (
    RESULTS_DIR
    / "hatexplain_laya_raw_full.jsonl"
)

BATCH_SIZE = 32


QUESTION = {
    "toxic": {
        "type": "noul",
        "instructions": (
            "Is the text toxic, hateful, or offensive?"
        ),
        "criteria": {
            "true": (
                "The text is toxic, hateful, or offensive."
            ),
            "false": (
                "The text is normal and is not toxic, hateful, or offensive."
            ),
        },
    }
}


def load_completed_ids():
    if not RESULTS_PATH.exists():
        return set()

    df = pd.read_csv(RESULTS_PATH)

    return set(
        df["scenario_id"]
        .astype(str)
    )


def append_rows(rows):
    exists = RESULTS_PATH.exists()

    pd.DataFrame(rows).to_csv(
        RESULTS_PATH,
        mode="a",
        header=not exists,
        index=False,
    )


def append_raw(obj):
    with open(
        RAW_PATH,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            )
            + "\n"
        )


def main():
    df = pd.read_csv(DATASET_PATH)

    completed = load_completed_ids()

    df = df[
        ~df["scenario_id"]
        .astype(str)
        .isin(completed)
    ].copy()

    print(f"Remaining: {len(df)}")

    if len(df) == 0:
        print("Nothing to do.")
        return

    router = Router(default="english")

    rows = df.to_dict("records")

    for start in range(
        0,
        len(rows),
        BATCH_SIZE,
    ):
        batch = rows[
            start:start + BATCH_SIZE
        ]

        requests = []

        for item in batch:
            state = (
                "Social media post:\n"
                + item["text"]
            )

            requests.append(
                {
                    "state": state,
                    "questions": QUESTION,
                    "model": "english",
                }
            )

        print(
            f"{start + 1}-"
            f"{start + len(batch)} "
            f"/ {len(rows)}"
        )

        t0 = time.perf_counter()

        try:
            results = router.predict_batch(
                requests,
                batch_size=BATCH_SIZE,
            )

        except Exception as exc:
            print(f"Batch error: {exc}")
            continue

        elapsed = (
            time.perf_counter()
            - t0
        )

        output_rows = []

        for item, result in zip(
            batch,
            results,
        ):
            p = float(
                result["answers"]
                ["toxic"]
                ["noul"]
            )

            output_rows.append(
                {
                    "scenario_id":
                        item["scenario_id"],

                    "post_id":
                        item["post_id"],

                    "ground_truth":
                        int(
                            item["ground_truth"]
                        ),

                    "human_toxic_probability":
                        float(
                            item[
                                "human_toxic_probability"
                            ]
                        ),

                    "toxic_votes":
                        int(
                            item[
                                "toxic_votes"
                            ]
                        ),

                    "total_votes":
                        int(
                            item[
                                "total_votes"
                            ]
                        ),

                    "laya_probability":
                        p,

                    "predicted_class":
                        int(p >= 0.5),
                }
            )

            append_raw(
                {
                    "scenario_id":
                        item["scenario_id"],

                    "text":
                        item["text"],

                    "question":
                        QUESTION,

                    "result":
                        result,
                }
            )

        append_rows(
            output_rows
        )

        print(
            f"  batch done in "
            f"{elapsed:.2f}s"
        )

    print()
    print(
        f"Results saved to:\n"
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()