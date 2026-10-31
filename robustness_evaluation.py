import json
from collections import defaultdict
from pathlib import Path

from predict import DuplicateQuestionDetector


OUTPUT_DIR = Path("results")
OUTPUT_PATH = OUTPUT_DIR / "robustness_evaluation.json"


# Diagnostic question pairs constructed specifically to test
# different semantic behaviours.
#
# Labels:
# 1 = duplicate
# 0 = not duplicate
#
# These are manually specified expected labels, not QQP labels.

ROBUSTNESS_CASES = [
    # Numerical sensitivity
    {
        "category": "numerical_difference",
        "question1": "How can I complete my course in two months?",
        "question2": "How can I complete my course in one year?",
        "label": 0,
    },
    {
        "category": "numerical_difference",
        "question1": "How much does a 16 GB laptop cost?",
        "question2": "How much does a 32 GB laptop cost?",
        "label": 0,
    },
    {
        "category": "numerical_difference",
        "question1": "What happened in the year 1990?",
        "question2": "What happened in the year 2000?",
        "label": 0,
    },
    {
        "category": "numerical_difference",
        "question1": "How do I travel from Terminal 3 to the city?",
        "question2": "How do I travel from Terminal 5 to the city?",
        "label": 0,
    },

    # Negation and logical differences
    {
        "category": "negation",
        "question1": "Why is this medication safe?",
        "question2": "Why is this medication unsafe?",
        "label": 0,
    },
    {
        "category": "negation",
        "question1": "Can dogs eat chocolate?",
        "question2": "Why can't dogs eat chocolate?",
        "label": 0,
    },
    {
        "category": "negation",
        "question1": "Is remote work beneficial for employees?",
        "question2": "Is remote work harmful for employees?",
        "label": 0,
    },
    {
        "category": "negation",
        "question1": "Why should students attend university?",
        "question2": "Why should students avoid university?",
        "label": 0,
    },

    # Entity differences
    {
        "category": "entity_difference",
        "question1": "What is the population of Melbourne?",
        "question2": "What is the population of Sydney?",
        "label": 0,
    },
    {
        "category": "entity_difference",
        "question1": "How do I install Python on macOS?",
        "question2": "How do I install Python on Windows?",
        "label": 0,
    },
    {
        "category": "entity_difference",
        "question1": "What causes Alzheimer's disease?",
        "question2": "What causes Parkinson's disease?",
        "label": 0,
    },
    {
        "category": "entity_difference",
        "question1": "How does supervised learning work?",
        "question2": "How does reinforcement learning work?",
        "label": 0,
    },

    # Paraphrase recognition
    {
        "category": "paraphrase",
        "question1": "How do neural networks learn?",
        "question2": "How are neural networks trained?",
        "label": 1,
    },
    {
        "category": "paraphrase",
        "question1": "What causes earthquakes?",
        "question2": "Why do earthquakes occur?",
        "label": 1,
    },
    {
        "category": "paraphrase",
        "question1": "How can I improve my programming skills?",
        "question2": "What can I do to become better at programming?",
        "label": 1,
    },
    {
        "category": "paraphrase",
        "question1": "What is the purpose of regular exercise?",
        "question2": "Why is exercising regularly important?",
        "label": 1,
    },
]


def evaluate_robustness(detector, cases):
    results = []
    category_results = defaultdict(list)

    for index, case in enumerate(cases, start=1):
        prediction = detector.predict(
            case["question1"],
            case["question2"],
        )

        predicted_label = int(prediction["is_duplicate"])
        correct = predicted_label == case["label"]

        result = {
            "case_id": index,
            "category": case["category"],
            "expected_label": case["label"],
            "predicted_label": predicted_label,
            "duplicate_score": prediction["duplicate_probability"],
            "correct": correct,
        }

        results.append(result)
        category_results[case["category"]].append(correct)

        status = "PASS" if correct else "FAIL"

        print(
            f"[{status}] "
            f"Case {index:02d} | "
            f"{case['category']:<22} | "
            f"Score: {prediction['duplicate_probability']:.4f}"
        )

    summary = {}

    for category, outcomes in category_results.items():
        correct_count = sum(outcomes)
        total = len(outcomes)

        summary[category] = {
            "correct": correct_count,
            "total": total,
            "accuracy": correct_count / total,
        }

    return results, summary


def main():
    detector = DuplicateQuestionDetector()

    print("\nCONTROLLED ROBUSTNESS EVALUATION")
    print("=" * 65)

    results, summary = evaluate_robustness(
        detector,
        ROBUSTNESS_CASES,
    )

    print("\nCATEGORY RESULTS")
    print("=" * 65)

    for category, metrics in summary.items():
        print(
            f"{category:<24} "
            f"{metrics['correct']}/{metrics['total']} correct "
            f"({metrics['accuracy']:.1%})"
        )

    total_correct = sum(result["correct"] for result in results)
    total_cases = len(results)

    print("\nOVERALL RESULTS")
    print("=" * 65)
    print(
        f"{total_correct}/{total_cases} correct "
        f"({total_correct / total_cases:.1%})"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as file:
        json.dump(
            {
                "evaluation_type": "hand_constructed_diagnostic",
                "threshold": detector.threshold,
                "total_cases": total_cases,
                "correct": total_correct,
                "accuracy": total_correct / total_cases,
                "category_summary": summary,
                "case_results": results,
            },
            file,
            indent=2,
        )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()