
from pathlib import Path

import pandas as pd


ANNOTATION_FILE = Path("results/error_annotation_sample.csv")

LABEL_ASSESSMENTS = {
    "1": "clear_model_error",
    "2": "ambiguous",
    "3": "possible_label_error",
}

SEMANTIC_CATEGORIES = {
    "1": "numerical_identifier_difference",
    "2": "entity_referent_difference",
    "3": "negation_logical_difference",
    "4": "scope_intent_difference",
    "5": "missed_paraphrase",
    "6": "implicit_semantic_equivalence",
    "7": "other",
}


def choose_option(options, heading):
    print(f"\n{heading}")

    for key, value in options.items():
        print(f"  {key}. {value.replace('_', ' ')}")

    while True:
        choice = input("Select number: ").strip()

        if choice in options:
            return options[choice]

        print("Invalid selection. Try again.")


def main():
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Missing annotation file: {ANNOTATION_FILE}"
        )

    df = pd.read_csv(
        ANNOTATION_FILE,
        keep_default_na=False,
    )

    if "label_assessment" not in df.columns:
        df["label_assessment"] = ""

    if "semantic_category" not in df.columns:
        df["semantic_category"] = ""

    if "notes" not in df.columns:
        df["notes"] = ""

    total = len(df)

    print("\nDUPLICATE QUESTION ERROR ANNOTATION")
    print("=" * 45)
    print(f"Total examples: {total}")
    print("Enter q at any prompt to quit safely.")

    for index, row in df.iterrows():
        if (
            row["label_assessment"]
            and row["semantic_category"]
        ):
            continue

        print("\n" + "=" * 70)
        print(f"Example {index + 1}/{total}")
        print(f"ID: {row['error_id']}")
        print(f"Type: {row['error_type']}")
        print(f"Model score: {float(row['probability']):.4f}")
        print(f"\nQuestion 1: {row['text1']}")
        print(f"Question 2: {row['text2']}")

        print("\nLabel assessment:")
        for key, value in LABEL_ASSESSMENTS.items():
            print(f"  {key}. {value.replace('_', ' ')}")

        choice = input("Select number (or q): ").strip()

        if choice.lower() == "q":
            break

        if choice not in LABEL_ASSESSMENTS:
            print("Invalid choice. Example skipped.")
            continue

        label_assessment = LABEL_ASSESSMENTS[choice]

        print("\nSemantic phenomenon:")
        for key, value in SEMANTIC_CATEGORIES.items():
            print(f"  {key}. {value.replace('_', ' ')}")

        choice = input("Select number (or q): ").strip()

        if choice.lower() == "q":
            break

        if choice not in SEMANTIC_CATEGORIES:
            print("Invalid choice. Example skipped.")
            continue

        semantic_category = SEMANTIC_CATEGORIES[choice]

        notes = input("Optional notes (Enter to skip): ").strip()

        if notes.lower() == "q":
            break

        df.at[index, "label_assessment"] = label_assessment
        df.at[index, "semantic_category"] = semantic_category
        df.at[index, "notes"] = notes

        df.to_csv(ANNOTATION_FILE, index=False)

        completed = (
            (df["label_assessment"] != "")
            & (df["semantic_category"] != "")
        ).sum()

        print(f"\nSaved. Progress: {completed}/{total}")

    print("\nAnnotation session ended.")
    print(
        "Completed:",
        (
            (df["label_assessment"] != "")
            & (df["semantic_category"] != "")
        ).sum(),
        "/",
        total,
    )


if __name__ == "__main__":
    main()
