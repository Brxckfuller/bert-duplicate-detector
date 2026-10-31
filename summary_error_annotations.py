from pathlib import Path

import pandas as pd


INPUT_PATH = Path("results/error_annotation_sample.csv")
OUTPUT_PATH = Path("results/error_annotation_summary.csv")

VALID_ASSESSMENTS = {
    "clear_model_error",
    "ambiguous",
    "possible_label_error",
}

VALID_CATEGORIES = {
    "numerical_identifier_difference",
    "entity_referent_difference",
    "negation_logical_difference",
    "scope_intent_difference",
    "missed_paraphrase",
    "implicit_semantic_equivalence",
    "other",
}


def main():
    df = pd.read_csv(INPUT_PATH, keep_default_na=False)

    required = {
        "error_id",
        "label_assessment",
        "semantic_category",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if len(df) != 60:
        raise ValueError(f"Expected 60 annotations, found {len(df)}")

    if df["error_id"].duplicated().any():
        raise ValueError("Duplicate error IDs found")

    if not df["label_assessment"].isin(VALID_ASSESSMENTS).all():
        raise ValueError("Invalid or missing label assessments")

    if not df["semantic_category"].isin(VALID_CATEGORIES).all():
        raise ValueError("Invalid or missing semantic categories")

    df["error_type"] = df["error_id"].str.extract(
        r"^(false_positive|false_negative)_"
    )[0]

    if df["error_type"].isna().any():
        raise ValueError("Invalid error ID format")

    counts = df["error_type"].value_counts()

    if counts.get("false_positive", 0) != 30:
        raise ValueError("Expected 30 false positives")

    if counts.get("false_negative", 0) != 30:
        raise ValueError("Expected 30 false negatives")

    print("\nANNOTATION VALIDATION")
    print("=" * 55)
    print(f"Total annotated examples: {len(df)}")
    print(f"False positives: {counts['false_positive']}")
    print(f"False negatives: {counts['false_negative']}")
    print("Validation: PASSED")

    print("\nLABEL ASSESSMENT BY ERROR TYPE")
    print("=" * 55)

    assessment_table = pd.crosstab(
        df["error_type"],
        df["label_assessment"],
    )
    print(assessment_table.to_string())

    print("\nSEMANTIC CATEGORY BY ERROR TYPE")
    print("=" * 55)

    category_table = pd.crosstab(
        df["error_type"],
        df["semantic_category"],
    )
    print(category_table.to_string())

    summary = (
        df.groupby(
            ["error_type", "label_assessment", "semantic_category"]
        )
        .size()
        .reset_index(name="count")
    )

    summary["percentage_within_error_type"] = (
        summary["count"]
        / summary.groupby("error_type")["count"].transform("sum")
        * 100
    ).round(2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSummary saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()