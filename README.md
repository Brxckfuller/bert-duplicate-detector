# BERT Duplicate Question Detector

A natural language processing project that investigates whether BERT can accurately identify semantically equivalent questions, even when they differ in wording or sentence structure. It involves fine-tuning DistilBERT on the Quora Question Pairs dataset and evaluating its performance against a traditional TF-IDF and logistic regression baseline using accuracy, precision, recall and F1 score.

## Project Objectives

- Explore and analyse the Quora Question Pairs dataset.
- Establish a TF-IDF and logistic regression baseline.
- Fine-tune DistilBERT for binary question-pair classification.
- Compare model performance using accuracy, precision, recall and F1.
- Build a local application for duplicate question detection.

## Dataset

Quora Question Pairs (QQP), accessed through Hugging Face.

Initial exploratory dataset: 5,000 question pairs.

- Non-duplicate pairs: 3,162
- Duplicate pairs: 1,838
- Missing values: 0


