
# BERT Duplicate Question Detector

An end-to-end NLP and machine learning project for identifying semantically duplicate questions using fine-tuned DistilBERT, benchmarked against classical TF-IDF and logistic regression models.

The project covers leakage-aware data preparation, transformer fine-tuning, validation-based threshold optimisation, held-out evaluation, statistical uncertainty analysis, error analysis, local inference and automated testing.

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

The experiment uses 5,000 question pairs from Hugging Face's `SetFit/qqp` dataset, comprising 1,838 duplicates and 3,162 non-duplicates.

### Question-Disjoint Splitting

Random splitting can introduce leakage when the same question appears in multiple dataset partitions.

To address this, questions were represented as graph nodes and question pairs as edges. Connected components were used to construct question-disjoint splits.

| Split | Pairs |
|---|---:|
| Training | 3,500 |
| Validation | 750 |
| Test | 750 |

An automated audit confirmed **zero shared exact question strings** between the three splits.

All models were evaluated on the same held-out test set. Classification thresholds were selected using validation data rather than the test set.

This splitting procedure reduces exact-question overlap but does not eliminate every form of semantic similarity or dataset bias.

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

## Results

Performance on **750 held-out question pairs**:

| Metric | Majority class | All positive | TF-IDF cosine | Enhanced TF-IDF | DistilBERT |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.5920 | 0.4080 | 0.6400 | 0.7040 | **0.7707** |
| Precision | 0.0000 | 0.4080 | 0.5352 | 0.5950 | **0.6893** |
| Recall | 0.0000 | **1.0000** | 0.8954 | 0.8595 | 0.7974 |
| F1 | 0.0000 | 0.5795 | 0.6699 | 0.7032 | **0.7394** |

DistilBERT achieved an observed F1 improvement of **3.62 percentage points** over the enhanced TF-IDF baseline and **6.95 percentage points** over the cosine baseline.

### Threshold-Independent Evaluation

| Metric | TF-IDF cosine | Enhanced TF-IDF | DistilBERT |
|---|---:|---:|---:|
| Average precision | 0.5130 | 0.6525 | **0.7776** |
| ROC-AUC | 0.6933 | 0.7741 | **0.8576** |

These results indicate stronger observed ranking and discrimination performance for DistilBERT across classification thresholds.

### Precision-Recall Curves

![Precision-recall curves comparing TF-IDF cosine, enhanced TF-IDF and DistilBERT](results/plots/precision_recall_curves.png)

### ROC Curves

![ROC curves comparing TF-IDF cosine, enhanced TF-IDF and DistilBERT](results/plots/roc_curves.png)

### Statistical Evaluation

To quantify sampling uncertainty, 5,000 paired bootstrap resamples were drawn from the held-out test set using a fixed random seed.

| Model | F1 | 95% Bootstrap CI |
|---|---:|---|
| TF-IDF cosine | 0.6699 | [0.6330, 0.7063] |
| Enhanced TF-IDF | 0.7032 | [0.6649, 0.7399] |
| DistilBERT | **0.7394** | [0.7013, 0.7767] |

For DistilBERT compared with enhanced TF-IDF:

- **Observed F1 difference:** +0.0362
- **95% paired bootstrap CI for F1 difference:** [-0.0012, 0.0749]
- **McNemar's exact test for classification correctness:** p = 0.000379

Although DistilBERT achieved higher observed F1, the paired bootstrap confidence interval includes zero. The analysis therefore does not establish a positive F1 difference at the 95% confidence level.

McNemar's test provides evidence of a difference in classification correctness, but it tests accuracy-related outcomes rather than F1 directly.

These uncertainty estimates are conditional on the trained models and test sample. They do not capture variation from retraining with different random seeds or from constructing alternative datasets.

Aggregate results are available in `results/baseline_comparison.json` and `results/statistical_evaluation.json`.

## Error Analysis

DistilBERT produced:

- **110 false positives**
- **62 false negatives**

One false positive involved questions about increasing IQ to 100 versus 180, illustrating difficulty recognising meaning-changing numerical differences despite strong lexical similarity.

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

This verifies test-example alignment, calculates confidence intervals and comparison statistics, and generates the precision-recall and ROC plots.

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

Unit tests also run automatically through GitHub Actions on pushes and pull requests to `main`. The integration test requires a locally trained checkpoint.

## Reproducibility

The repository includes dataset preparation, baseline training, transformer fine-tuning, evaluation, statistical analysis, inference and testing code, along with pinned dependencies and checkpoint configuration tracking.

Aggregate evaluation metrics and plots are committed to the repository so results can be inspected without retraining the models.

Dataset files, question-level prediction records and trained weights are excluded from Git. Reproducing the experiment requires downloading the dataset and training the model locally.

Exact results may vary with hardware, dependency versions and random-number generation.

## Limitations

- **Dataset size:** Only 5,000 question pairs were used, limiting statistical precision and representativeness.
- **Generalisation:** Evaluation is limited to QQP-style questions.
- **Semantic ambiguity:** Some question pairs have subjective or unclear duplicate labels.
- **Robustness:** Numerical differences and subtle changes in meaning can produce incorrect classifications.
- **Statistical uncertainty:** The paired bootstrap interval for the F1 improvement over enhanced TF-IDF includes zero.
- **Training variability:** Multiple training seeds and alternative data samples have not been evaluated.
- **Calibration:** Model confidence scores have not been calibrated.
- **Deployment:** Inference is local; no hosted API or production service has been implemented.

## Author

**Brock Fuller**

Master of Artificial Intelligence student at RMIT University

Developed and maintained by [Brock Fuller](https://github.com/Brxckfuller).
