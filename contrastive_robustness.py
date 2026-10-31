import json
from pathlib import Path

from predict import DuplicateQuestionDetector


OUTPUT_PATH = Path("results/contrastive_robustness.json")

# Each group has:
# - A reference question
# - A semantically equivalent question (expected label 1)
# - A meaning-changing question (expected label 0)
#
# These are hand-constructed diagnostic examples.
# Their labels are provisional and should be reviewed.

CONTRASTIVE_GROUPS = [
    {
        "category": "numerical",
        "reference": "How can I finish a course in two months?",
        "equivalent": "How do I complete a course within eight weeks?",
        "contrast": "How can I finish a course in one year?",
    },
    {
        "category": "numerical",
        "reference": "How much does a laptop with 16 GB of RAM cost?",
        "equivalent": "What is the price of a laptop with 16 GB RAM?",
        "contrast": "How much does a laptop with 32 GB of RAM cost?",
    },
    {
        "category": "numerical",
        "reference": "How do I get from Heathrow Terminal 3 to Paddington?",
        "equivalent": "What is the best way to travel from Heathrow's third terminal to Paddington?",
        "contrast": "How do I get from Heathrow Terminal 5 to Paddington?",
    },
    {
        "category": "numerical",
        "reference": "What happened in the year 1990?",
        "equivalent": "What events took place during 1990?",
        "contrast": "What happened in the year 2000?",
    },
    {
        "category": "negation",
        "reference": "Why is remote working beneficial for employees?",
        "equivalent": "What are the benefits of working remotely?",
        "contrast": "Why is remote working harmful for employees?",
    },
    {
        "category": "negation",
        "reference": "Why should students attend university?",
        "equivalent": "What are the reasons students should go to university?",
        "contrast": "Why should students avoid university?",
    },
    {
        "category": "negation",
        "reference": "Why is this medication safe?",
        "equivalent": "What makes this medication safe to use?",
        "contrast": "Why is this medication unsafe?",
    },
    {
        "category": "negation",
        "reference": "Why should people trust artificial intelligence?",
        "equivalent": "What reasons are there to trust AI?",
        "contrast": "Why should people distrust artificial intelligence?",
    },
    {
        "category": "entity",
        "reference": "What is the population of Melbourne?",
        "equivalent": "How many people live in Melbourne?",
        "contrast": "What is the population of Sydney?",
    },
    {
        "category": "entity",
        "reference": "How do I install Python on macOS?",
        "equivalent": "What are the steps for installing Python on a Mac?",
        "contrast": "How do I install Python on Windows?",
    },
    {
        "category": "entity",
        "reference": "What causes Alzheimer's disease?",
        "equivalent": "What are the causes of Alzheimer's disease?",
        "contrast": "What causes Parkinson's disease?",
    },
    {
        "category": "entity",
        "reference": "How does supervised learning work?",
        "equivalent": "Can you explain how supervised machine learning works?",
        "contrast": "How does reinforcement learning work?",
    },
    {
        "category": "paraphrase",
        "reference": "How are long-distance relationships maintained?",
        "equivalent": "How can people make a long-distance relationship work?",
        "contrast": "Why do long-distance relationships end?",
    },
    {
        "category": "paraphrase",
        "reference": "What causes earthquakes?",
        "equivalent": "Why do earthquakes occur?",
        "contrast": "How are earthquakes measured?",
    },
    {
        "category": "paraphrase",
        "reference": "How can I improve my programming skills?",
        "equivalent": "What can I do to become better at programming?",
        "contrast": "Which programming language is the most popular?",
    },
    {
        "category": "paraphrase",
        "reference": "Why is regular exercise important?",
        "equivalent": "What are the benefits of exercising regularly?",
        "contrast": "What exercises should beginners avoid?",
    },
]


def evaluate_group(detector, group):
    positive = detector.predict(
        group["reference"],
        group["equivalent"],
    )

    negative = detector.predict(
        group["reference"],
        group["contrast"],
    )

    positive_score = positive["duplicate_probability"]
    negative_score = negative["duplicate_probability"]

    positive_correct = bool(positive["is_duplicate"])
    negative_correct = not bool(negative["is_duplicate"])

    return {
        "category": group["category"],
        "positive_score": positive_score,
        "negative_score": negative_score,
        "score_margin": round(positive_score - negative_score, 4),
        "positive_correct": positive_correct,
        "negative_correct": negative_correct,
        "both_correct": positive_correct and negative_correct,
        "correct_ordering": positive_score > negative_score,
    }


def main():
    detector = DuplicateQuestionDetector()

    results = []

    print("\nCONTRASTIVE ROBUSTNESS EVALUATION")
    print("=" * 75)

    for index, group in enumerate(CONTRASTIVE_GROUPS, start=1):
        result = evaluate_group(detector, group)
        result["group_id"] = index
        results.append(result)

        status = "PASS" if result["both_correct"] else "FAIL"

        print(
            f"[{status}] Group {index:02d} | "
            f"{result['category']:<12} | "
            f"Equivalent: {result['positive_score']:.4f} | "
            f"Contrast: {result['negative_score']:.4f} | "
            f"Margin: {result['score_margin']:+.4f}"
        )

    print("\nCATEGORY SUMMARY")
    print("=" * 75)

    categories = sorted(set(r["category"] for r in results))

    summary = {}

    for category in categories:
        group_results = [
            r for r in results if r["category"] == category
        ]

        n = len(group_results)
        both_correct = sum(r["both_correct"] for r in group_results)
        correct_ordering = sum(
            r["correct_ordering"] for r in group_results
        )

        positive_correct = sum(
            r["positive_correct"] for r in group_results
        )

        negative_correct = sum(
            r["negative_correct"] for r in group_results
        )

        summary[category] = {
            "groups": n,
            "equivalent_accuracy": positive_correct / n,
            "contrast_accuracy": negative_correct / n,
            "both_correct_rate": both_correct / n,
            "correct_ordering_rate": correct_ordering / n,
        }

        print(
            f"{category:<12} | "
            f"Equivalent: {positive_correct}/{n} | "
            f"Contrast: {negative_correct}/{n} | "
            f"Both correct: {both_correct}/{n} | "
            f"Ordering: {correct_ordering}/{n}"
        )

    total_groups = len(results)
    total_correct_pairs = sum(
        int(r["positive_correct"]) + int(r["negative_correct"])
        for r in results
    )

    overall = {
        "total_groups": total_groups,
        "total_pairs": total_groups * 2,
        "correct_pairs": total_correct_pairs,
        "pair_accuracy": total_correct_pairs / (total_groups * 2),
        "both_correct_groups": sum(r["both_correct"] for r in results),
        "correctly_ordered_groups": sum(
            r["correct_ordering"] for r in results
        ),
    }

    print("\nOVERALL SUMMARY")
    print("=" * 75)
    print(
        f"Pair accuracy: "
        f"{overall['correct_pairs']}/{overall['total_pairs']} "
        f"({overall['pair_accuracy']:.1%})"
    )
    print(
        f"Both correct: "
        f"{overall['both_correct_groups']}/{total_groups}"
    )
    print(
        f"Correct score ordering: "
        f"{overall['correctly_ordered_groups']}/{total_groups}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as file:
        json.dump(
            {
                "evaluation_type": "hand_constructed_contrastive_diagnostic",
                "threshold": detector.threshold,
                "overall": overall,
                "category_summary": summary,
                "group_results": results,
            },
            file,
            indent=2,
        )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()