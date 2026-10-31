
# BERT Duplicate Question Detector

An end-to-end NLP project for identifying semantically duplicate questions using fine-tuned DistilBERT, benchmarked against a TF-IDF and logistic regression baseline.

The project covers data preparation, leakage-aware splitting, transformer fine-tuning, threshold optimisation, held-out evaluation, error analysis, local inference and automated testing.

## Overview

Duplicate question detection is a binary classification task: determining whether two questions express the same underlying intent despite differences in wording or sentence structure.

This project compares traditional text classification with transformer-based semantic classification using the Quora Question Pairs (QQP) dataset.

**Models:**
- TF-IDF + logistic regression
- Fine-tuned `distilbert-base-uncased`

**Stack:** Python, PyTorch, Hugging Face Transformers, scikit-learn, pandas, NetworkX, pytest and GitHub Actions.

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

An automated audit confirmed **zero shared question strings** between the three splits.

Both models were evaluated on the same held-out test set. Classification thresholds were selected using validation data rather than the test set.

## Model Training

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

The best checkpoint was selected using validation F1 at the default 0.5 threshold. A subsequent validation-based search selected an inference threshold of `0.15`, compared with `0.25` for the TF-IDF baseline.

## Results

Performance on 750 held-out question pairs:

| Metric | TF-IDF + LR | DistilBERT |
|---|---:|---:|
| Accuracy | 0.6400 | **0.7707** |
| Precision | 0.5352 | **0.6893** |
| Recall | **0.8954** | 0.7974 |
| F1 | 0.6699 | **0.7394** |

DistilBERT improved F1 by **6.95 percentage points**, achieving higher accuracy and precision while sacrificing some recall.

### Error Analysis

DistilBERT produced 110 false positives and 62 false negatives.

One false positive involved questions about increasing IQ to 100 versus 180, illustrating difficulty recognising meaning-changing numerical differences despite strong lexical similarity.

Model scores are uncalibrated softmax outputs and should not be interpreted as reliable probabilities.

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
python train_bert.py
python evaluate_bert.py
python record_checkpoint.py
```

The trained checkpoint is saved locally under `models/distilbert-best/`. Training metadata and the selected inference threshold are recorded alongside it.

### 4. Run inference

```bash
python predict.py --q1 "How do neural networks learn?" --q2 "How are neural networks trained?"
```

Example output:

```text
Duplicate score: 0.5299
Threshold: 0.15
Prediction: Duplicate
```

### 5. Run error analysis and tests

```bash
python error_analysis.py
python -m pytest -q
```

Unit tests also run automatically through GitHub Actions on pushes and pull requests to `main`. The integration test requires a locally trained checkpoint.

## Reproducibility

The repository includes dataset preparation, training, evaluation, inference and testing code, along with pinned dependencies and checkpoint configuration tracking.

Dataset files and trained weights are excluded from Git. Reproducing the experiment requires downloading the dataset and training the model locally.

Exact results may vary with hardware, dependency versions and random-number generation.

## Limitations

- **Dataset size:** Only 5,000 question pairs were used.
- **Generalisation:** Evaluation is limited to QQP-style questions.
- **Semantic ambiguity:** Some question pairs have subjective or unclear duplicate labels.
- **Robustness:** Numerical differences and subtle changes in meaning can produce incorrect classifications.
- **Calibration:** Model confidence scores have not been calibrated.
- **Deployment:** Inference is local; no hosted API or production service has been implemented.


## Author

**Brock Fuller**

Master of Artificial Intelligence student at RMIT University

Developed and maintained by [Brock Fuller](https://github.com/Brxckfuller).


