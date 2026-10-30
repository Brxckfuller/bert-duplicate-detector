
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import paired_cosine_distances


# 1. Load dataset
dataset = load_dataset("SetFit/qqp", split="train[:5000]")
df = dataset.to_pandas()

# 2. Split data BEFORE fitting the vectorizer
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

# 3. Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
)

train_questions = (
    train_df["text1"].tolist()
    + train_df["text2"].tolist()
)

vectorizer.fit(train_questions)


# 4. Calculate cosine similarity for each pair
def get_similarity_features(data):
    q1 = vectorizer.transform(data["text1"])
    q2 = vectorizer.transform(data["text2"])

    similarity = 1 - paired_cosine_distances(q1, q2)

    return similarity.reshape(-1, 1)


X_train = get_similarity_features(train_df)
X_test = get_similarity_features(test_df)

y_train = train_df["label"]
y_test = test_df["label"]

# 5. Train logistic regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)

print("\nBASELINE RESULTS")
print("----------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1 Score:", round(f1_score(y_test, predictions), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))


# 7. Test on new questions
def predict_duplicate(question1, question2):
    pair = {
        "text1": [question1],
        "text2": [question2],
    }

    features = get_similarity_features(pair)
    probability = model.predict_proba(features)[0][1]

    print("\nQuestion 1:", question1)
    print("Question 2:", question2)
    print("Duplicate probability:", round(probability, 4))
    print(
        "Prediction:",
        "Duplicate" if probability >= 0.5 else "Not duplicate"
    )


predict_duplicate(
    "How do neural networks learn?",
    "How does a neural network get trained?",
)

predict_duplicate(
    "What is machine learning?",
    "How do I cook pasta?",
)
