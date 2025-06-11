import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import numpy as np
import json
import sys
import logging
import matplotlib.pyplot as plt

# Configure logging for Docker Desktop viewing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

if len(sys.argv) != 2:
    logging.error("Usage: python server.py <input_json_file>")
    sys.exit(1)

json_path = sys.argv[1]
logging.info(f"Loading input JSON from: {json_path}")
with open(json_path, "r") as f:
    data = json.load(f)

# Extract pixel values in order
input_values = [data[f"pixel_{i}"] for i in range(64)]

# Load pipeline components
logging.info("Loading model pipeline from digits_pca_svm.pkl")
model_dict = joblib.load("digits_pca_svm.pkl")
scaler = model_dict["scaler"]
pca = model_dict["pca"]
svm = model_dict["svm"]

# Prepare input
X = np.array(input_values).reshape(1, -1)
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

# Predict
y_pred = svm.predict(X_pca)
logging.info(f"Predicted Number: {y_pred[0]}")

# Save the 8x8 image as a PNG
img = np.array(input_values).reshape(8, 8)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.savefig("output/output.png", bbox_inches='tight', pad_inches=0)
plt.close()