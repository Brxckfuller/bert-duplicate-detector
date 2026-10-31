
# BERT Duplicate Question Detector

A reproducible NLP and machine learning pipeline for identifying semantically duplicate questions using fine-tuned DistilBERT, benchmarked against classical TF-IDF and logistic regression models.

The project covers leakage-aware data preparation, transformer fine-tuning, validation-based threshold optimisation, held-out evaluation, component-aware statistical uncertainty analysis, error analysis, local inference and automated testing.

## Key Findings

- **DistilBERT outperformed the strongest classical baseline on ranking metrics.** Compared with enhanced TF-IDF, average precision improved by 0.1251 (95% CI: [0.0695, 0.1827]) and ROC-AUC improved by 0.0836 (95% CI: [0.0515, 0.1163]). Both paired component-bootstrap confidence intervals exclude zero.
- **The F1 improvement was not statistically established.** DistilBERT achieved an observed F1 improvement of 0.0362 over enhanced TF-IDF, but the 95% confidence interval of [-0.0012, 0.0740] includes zero.
- **Error analysis identified sensitivity to meaning-changing numerical differences.** The model produced 110 false positives and 62 false negatives. Numerical substitutions are an observed failure example, not yet an established dominant error category.

## Overview

Duplicate question detection is a binary classification task: determining whether two questions express the same underlying intent despite differences in wording or sentence structure.

This project compares lexical similarity methods with transformer-based semantic classification using the Quora Question Pairs (QQP) dataset.

**Models:**
- Majority-class and all-positive reference baselines
- TF-IDF cosine similarity + logistic regression
- Enhanced TF-IDF features + logistic regression
- Fine-tuned `distilbert-base-uncased`

**Stack:** Python, PyTorch, Hugging Face Transformers, scikit-learn, SciPy, pandas, NetworkX, Matplotlib, pytest and GitHub Actions.

## Dataset and Methodology

The experiment uses 5,000 question pairs from Hugging Face's `SetFit/qqp` dataset, comprising 1,838 duplicates (36.76%) and 3,162 non-duplicates (63.24%).

### Question-Disjoint Splitting

Random splitting can introduce leakage when the same question appears in multiple dataset partitions.

To address this, questions were represented as graph nodes and question pairs as edges. Connected components were used to construct question-disjoint splits.

| Split | Pairs | Duplicate pairs | Duplicate rate |
|---|---:|---:|---:|
| Training | 3,500 | 1,250 | 35.71% |
| Validation | 750 | 282 | 37.60% |
| Test | 750 | 306 | 40.80% |
| **Total** | **5,000** | **1,838** | **36.76%** |

An automated audit confirmed **zero shared exact question strings** between the three splits.

All models were evaluated on the same held-out test set. Classification thresholds were selected using validation data rather than the test set.

This splitting procedure reduces exact-question overlap but does not eliminate every form of semantic similarity or dataset bias. The test set also has a higher duplicate prevalence than the full sample, which should be considered when interpreting precision, F1 and average precision.

### Test-Set Component Structure

An additional audit of the 750 test pairs found:

| Statistic | Value |
|---|---:|
| Connected components | 722 |
| Single-pair components | 698 |
| Components containing multiple pairs | 24 |
| Pairs belonging to multi-pair components | 52 (6.93%) |
| Largest component | 4 pairs |

These components form the resampling units for the component-aware bootstrap described below.

## Models and Training

### Classical Baselines

Four reference approaches were evaluated:

1. **Majority class:** Predicts the most frequent training label for every pair.
2. **All positive:** Predicts every question pair as a duplicate.
3. **TF-IDF cosine:** Computes cosine similarity between separately vectorised questions and uses logistic regression for classification.
4. **Enhanced TF-IDF:** Combines cosine similarity, absolute TF-IDF differences, elementwise TF-IDF products and lexical overlap features (Jaccard similarity and containment) in a logistic regression classifier.

TF-IDF vocabulary fitting and classifier training used training data only. Classification thresholds were selected by maximising validation F1.

### DistilBERT

DistilBERT was fine-tuned for binary question-pair classification using the following configuration:

| Parameter | Value |
|---|---|
| Model | `distilbert-base-uncased` |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Maximum sequence length | 128 |
| Optimizer | AdamW |
| Seed | 42 |
| Acceleration | Apple Silicon MPS |

The best checkpoint was selected using validation F1 at the default 0.5 threshold. A subsequent validation-based search selected an inference threshold of `0.15`.

The corresponding thresholds for the cosine and enhanced TF-IDF baselines were `0.25` and `0.30`, respectively.

The relatively low DistilBERT threshold reflects the operating point selected by the validation F1 search. It should not, by itself, be interpreted as evidence of poor probability calibration or a particular underlying cause.

## Results

Performance on **750 held-out question pairs**:

| Metric | Majority class | All positive | TF-IDF cosine | Enhanced TF-IDF | DistilBERT |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.5920 | 0.4080 | 0.6400 | 0.7040 | **0.7707** |
| Precision | 0.0000* | 0.4080 | 0.5352 | 0.5950 | **0.6893** |
| Recall | 0.0000 | **1.0000** | 0.8954 | 0.8595 | 0.7974 |
| F1 | 0.0000 | 0.5795 | 0.6699 | 0.7032 | **0.7394** |

*The majority-class model predicts no positive examples, so positive-class precision is mathematically undefined (0/0). The evaluation convention reports it as 0.0000.*

DistilBERT achieved an observed F1 improvement of **3.62 percentage points** over enhanced TF-IDF and **6.95 percentage points** over the cosine baseline. The statistical uncertainty associated with these differences is examined below.

### Threshold-Independent Evaluation

| Metric | TF-IDF cosine | Enhanced TF-IDF | DistilBERT |
|---|---:|---:|---:|
| Average precision | 0.5130 | 0.6525 | **0.7776** |
| ROC-AUC | 0.6933 | 0.7741 | **0.8576** |

These metrics assess ranking and discrimination without relying on the selected classification thresholds. DistilBERT achieved higher observed values on both measures.

### Precision-Recall Curves

![Precision-recall curves comparing TF-IDF cosine, enhanced TF-IDF and DistilBERT](results/plots/precision_recall_curves.png)

The dashed horizontal line represents the positive-class prevalence of 0.408 in the test set. The extreme low-recall region is based on very few top-ranked predictions and should not be overinterpreted.

### ROC Curves

![ROC curves comparing TF-IDF cosine, enhanced TF-IDF and DistilBERT](results/plots/roc_curves.png)

The diagonal dashed line represents random ranking performance.

### Component-Aware Statistical Evaluation

To quantify uncertainty while accounting for exact-question overlap within the test set, the evaluation uses a **paired connected-component bootstrap**.

Rather than resampling individual question pairs independently, the procedure resamples all 722 test-set connected components with replacement. Every selected component contributes all of its question pairs, preserving the observed within-component structure.

The procedure uses:
- 5,000 bootstrap replicates
- A fixed random seed of 42
- The same sampled components for all models in each replicate
- Percentile-based 95% confidence intervals
- Paired differences calculated within each bootstrap replicate

Because connected components contain different numbers of pairs, bootstrap replicate sizes can vary slightly.

#### Individual Model Confidence Intervals

| Model | F1 (95% CI) | Average precision (95% CI) | ROC-AUC (95% CI) |
|---|---|---|---|
| TF-IDF cosine | 0.6699 [0.6311, 0.7052] | 0.5130 [0.4617, 0.5731] | 0.6933 [0.6550, 0.7301] |
| Enhanced TF-IDF | 0.7032 [0.6640, 0.7397] | 0.6525 [0.5933, 0.7111] | 0.7741 [0.7409, 0.8061] |
| **DistilBERT** | **0.7394 [0.6995, 0.7768]** | **0.7776 [0.7248, 0.8298]** | **0.8576 [0.8309, 0.8829]** |

#### Paired Model Comparisons

| Comparison | Metric | Observed difference | 95% paired component-bootstrap CI |
|---|---|---:|---|
| DistilBERT - enhanced TF-IDF | F1 | +0.0362 | [-0.0012, 0.0740] |
| DistilBERT - enhanced TF-IDF | Average precision | +0.1251 | **[0.0695, 0.1827]** |
| DistilBERT - enhanced TF-IDF | ROC-AUC | +0.0836 | **[0.0515, 0.1163]** |
| DistilBERT - TF-IDF cosine | F1 | +0.0695 | **[0.0299, 0.1096]** |
| DistilBERT - TF-IDF cosine | Average precision | +0.2646 | **[0.2041, 0.3209]** |
| DistilBERT - TF-IDF cosine | ROC-AUC | +0.1643 | **[0.1241, 0.2047]** |

**Interpretation:** The confidence intervals for DistilBERT's average precision and ROC-AUC improvements over enhanced TF-IDF exclude zero. The corresponding F1 interval includes zero, so a positive F1 difference is not established at the 95% confidence level by this procedure.

These findings support improved ranking performance on the evaluated sample, while distinguishing that result from uncertainty in threshold-dependent classification performance.

#### McNemar's Test and Operating-Point Effects

The conventional exact McNemar test for classification correctness produced:

- **DistilBERT vs enhanced TF-IDF:** p = 0.000379
- **DistilBERT vs TF-IDF cosine:** p = 2.58e-10

McNemar's test examines disagreement in whether individual predictions are correct. It is related to accuracy, not F1, and the conventional pair-level test used here does not adjust for connected-component clustering.

Consequently, McNemar's result and the F1 bootstrap interval are not contradictory. DistilBERT has higher accuracy than enhanced TF-IDF (0.7707 vs 0.7040), whereas the validation-F1-optimised operating points favour recall to different degrees. The enhanced TF-IDF classifier has higher recall but lower precision, narrowing the F1 difference despite its lower accuracy.

The bootstrap results are conditional on the fitted models and observed test sample. Component resampling accounts for exact-question connectivity within the test set, but does not eliminate all possible semantic dependence, measure variation from retraining or establish generalisation beyond QQP-style data.

### Reproducible Evaluation Artifacts

The repository includes:

- `results/baseline_comparison.json` — baseline metrics and selected thresholds
- `results/statistical_evaluation.json` — component-aware confidence intervals and paired comparisons
- `results/evaluation_predictions.csv` — text-free aligned prediction records for all three score-based models
- `results/plots/precision_recall_curves.png` — precision-recall comparison
- `results/plots/roc_curves.png` — ROC comparison

The text-free prediction file contains pair indices, component identifiers, true labels, model scores and binary predictions. It enables the statistical analysis to be reproduced without retraining the models or redistributing the question text.

## Error Analysis

DistilBERT produced:

- **110 false positives**
- **62 false negatives**

One false positive involved questions about increasing IQ to 100 versus 180, illustrating difficulty recognising a meaning-changing numerical difference despite strong lexical similarity.

This example identifies a potential robustness weakness, but a systematic categorisation of errors is still needed to determine how frequently numerical substitutions, entity changes, negation, ambiguity and possible label noise contribute to failures.

Model scores are uncalibrated softmax outputs and should not be interpreted as reliable probabilities.

Question-level predictions and error-analysis records are retained locally rather than committed to the repository.

## Installation and Usage

Python 3.12 was used during development.

### 1. Clone and install

```bash
git clone https://github.com/Brxckfuller/bert-duplicate-detector.git
cd bert-duplicate-detector

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Prepare and audit the dataset

```bash
python prepare_data.py
python audit_data.py
```

### 3. Train and evaluate

```bash
python baseline.py
python baseline_comparison.py
python train_bert.py
python evaluate_bert.py
python record_checkpoint.py
```

The trained checkpoint is saved locally under `models/distilbert-best/`. Training metadata and the selected inference threshold are recorded alongside it.

### 4. Run statistical evaluation

First, generate the question-level DistilBERT prediction records:

```bash
python error_analysis.py
```

Then run:

```bash
python statistical_evaluation.py
```

This verifies test-example alignment, reconstructs connected components, calculates paired component-bootstrap confidence intervals, performs conventional McNemar tests, exports text-free prediction records and generates precision-recall and ROC plots.

### 5. Run inference

```bash
python predict.py --q1 "How do neural networks learn?" --q2 "How are neural networks trained?"
```

Example output:

```text
Duplicate score: 0.5299
Threshold: 0.15
Prediction: Duplicate
```

### 6. Run tests

```bash
python -m pytest -q
```

The current local test suite contains **20 passing tests**, including checks for connected-component assignment, bootstrap cluster preservation, confidence intervals, classification metrics and McNemar's exact test.

Unit tests also run automatically through GitHub Actions on pushes and pull requests to `main`. The integration test requires a locally trained checkpoint.

## Reproducibility

The repository includes dataset preparation, baseline training, transformer fine-tuning, evaluation, statistical analysis, inference and testing code, along with pinned dependencies and checkpoint configuration tracking.

Aggregate evaluation metrics, plots and text-free prediction records are committed so the statistical comparisons can be inspected and reproduced without retraining.

Dataset files, question-level prediction records and trained weights are excluded from Git. Reproducing the complete training experiment requires downloading the dataset and training the model locally.

Exact results may vary with hardware, dependency versions and random-number generation.

## Limitations

- **Dataset size:** Only 5,000 question pairs were used, limiting statistical precision and representativeness.
- **Class balance:** Duplicate prevalence varies across the training, validation and test partitions.
- **Generalisation:** Evaluation is limited to QQP-style questions.
- **Semantic ambiguity:** Some question pairs have subjective or unclear duplicate labels.
- **Robustness:** Numerical differences and subtle changes in meaning can produce incorrect classifications; error categories have not yet been systematically quantified.
- **Statistical uncertainty:** The paired component-bootstrap interval for the F1 improvement over enhanced TF-IDF includes zero.
- **Dependence:** Component-aware resampling preserves exact-question overlap within the test set but does not account for every possible source of dependence.
- **Training variability:** Multiple training seeds and alternative data samples have not been evaluated.
- **Threshold selection:** The selected 0.15 DistilBERT threshold has not yet been accompanied by a full validation-F1-versus-threshold analysis.
- **Calibration:** Model confidence scores have not been calibrated.
- **Deployment:** Inference is local; no hosted API or production service has been implemented.

## Author

**Brock Fuller**

Master of Artificial Intelligence student at RMIT University

[GitHub Profile](https://github.com/Brxckfuller)
