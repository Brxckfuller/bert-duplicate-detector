
import json

from config import MODEL_DIR, DUPLICATE_THRESHOLD
from train_bert import TRAINING_CONFIG

with open("results/training_history.json") as file:
    history = json.load(file)

best_epoch = max(history, key=lambda row: row["val_f1"])

metadata = {
    **TRAINING_CONFIG,
    "best_epoch": best_epoch["epoch"],
    "best_validation_f1": best_epoch["val_f1"],
    "inference_threshold": DUPLICATE_THRESHOLD,
    "threshold_selection": "validation_f1",
}

output_path = MODEL_DIR / "training_config.json"

with open(output_path, "w") as file:
    json.dump(metadata, file, indent=2)

print(f"Saved checkpoint metadata to {output_path}")
print(json.dumps(metadata, indent=2))
