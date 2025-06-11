# Save a sample from the digits dataset as JSON
import json
from sklearn.datasets import load_digits

digits = load_digits()
sample = digits.data[85]  # Take the first sample

# Prepare as a dict with keys matching your DataFrame columns (0-63)
sample_dict = {f"pixel_{i}": float(val) for i, val in enumerate(sample)}

with open("test_sample.json", "w") as f:
    json.dump(sample_dict, f)