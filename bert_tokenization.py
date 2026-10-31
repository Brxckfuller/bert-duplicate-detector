
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

# Load the pretrained DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

question1 = "How do neural networks learn?"
question2 = "How does a neural network get trained?"

# Tokenise both questions together
encoded = tokenizer(
    question1,
    question2,
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="pt",
)

# Convert token IDs back to readable tokens
tokens = tokenizer.convert_ids_to_tokens(
    encoded["input_ids"][0]
)

print("\nTOKENS:")
print(tokens)

print("\nINPUT IDS:")
print(encoded["input_ids"])

print("\nATTENTION MASK:")
print(encoded["attention_mask"])

print("\nINPUT SHAPE:")
print(encoded["input_ids"].shape)

print("\nNUMBER OF REAL TOKENS:")
print(encoded["attention_mask"].sum().item())
