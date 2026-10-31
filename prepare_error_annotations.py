
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
SAMPLE_PER_TYPE = 30
RANDOM_SEED = 42

COLUMNS = [
    "error_id",
    "error_type",
    "text1",
    "text2",
    "probability",
    "category",
    "notes",
]


def main():
    samples = []

    for error_type, filename in [
        ("false_positive", "false_positives.csv"),
        ("false_negative", "false_negatives.csv"),
    ]:
        df = pd.read_csv(RESULTS_DIR / filename)

        sample = df.sample(
            n=min(SAMPLE_PER_TYPE, len(df)),
            random_state=RANDOM_SEED,
        ).copy()

        sample["error_id"] = [
            f"{error_type}_{index:03d}"
            for index in sample.index
        ]

        sample["error_type"] = error_type
        sample["category"] = ""
        sample["notes"] = ""

        samples.append(sample[COLUMNS])

    annotations = pd.concat(
        samples,
        ignore_index=True,
    )

    output_path = RESULTS_DIR / "error_annotation_sample.csv"

    annotations.to_csv(output_path, index=False)

    print(f"Saved {len(annotations)} errors to {output_path}")
    print(annotations["error_type"].value_counts())


if __name__ == "__main__":
    main()
