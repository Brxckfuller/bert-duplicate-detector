
import pandas as pd

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")


def get_questions(df):
    return set(
        pd.concat([df["text1"], df["text2"]])
        .str.strip()
        .str.lower()
    )


train_questions = get_questions(train)
val_questions = get_questions(val)
test_questions = get_questions(test)

print("DATASET OVERLAP AUDIT")
print("---------------------------")

print(
    "Train / Validation overlap:",
    len(train_questions & val_questions)
)

print(
    "Train / Test overlap:",
    len(train_questions & test_questions)
)

print(
    "Validation / Test overlap:",
    len(val_questions & test_questions)
)
