
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.metrics.pairwise import paired_cosine_distances


# Load final question-disjoint partitions
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")

# Fit TF-IDF vocabulary using training data only
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
)

vectorizer.fit(
    train_df["text1"].tolist()
    + train_df["text2"].tolist()
)


def get_similarity_features(data):
    q1 = vectorizer.transform(data["text1"])
    q2 = vectorizer.transform(data["text2"])

    similarity = 1 - paired_cosine_distances(q1, q2)
    return similarity.reshape(-1, 1)


X_train = get_similarity_features(train_df)
X_val = get_similarity_features(val_df)
X_test = get_similarity_features(test_df)

y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

# Train baseline classifier
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Select classification threshold using validation F1
val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = [i / 100 for i in range(10, 91, 5)]

best_threshold = max(
    thresholds,
    key=lambda threshold: f1_score(
        y_val,
        (val_probabilities >= threshold).astype(int),
        zero_division=0,
    ),
)

print("Selected threshold:", best_threshold)

# Evaluate once on held-out test set
test_probabilities = model.predict_proba(X_test)[:, 1]
predictions = (test_probabilities >= best_threshold).astype(int)

print("\nTF-IDF BASELINE RESULTS")
print("----------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1 Score:", round(f1_score(y_test, predictions), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))
