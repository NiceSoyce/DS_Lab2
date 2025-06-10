import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

# Load digits dataset
digits = load_digits()

# Plot first 10 images with labels
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, img, label in zip(axes.flatten(), digits.images, digits.target):
    ax.imshow(img, cmap='gray_r')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('sample_digits.png')
plt.close()

# Compute class distribution
counts = np.bincount(digits.target, minlength=len(digits.target_names))

# Print distribution
for digit, count in enumerate(counts):
    print(f"{digit}: {count}")

# Plot distribution
plt.figure(figsize=(6, 4))
plt.bar(np.arange(len(counts)), counts, color='skyblue')
plt.xlabel('Digit class')
plt.ylabel('Count')
plt.title('Class distribution in digits dataset')
plt.xticks(np.arange(len(counts)))
plt.tight_layout()
plt.savefig('class_distribution.png')

