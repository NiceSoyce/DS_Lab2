# -*- coding: utf-8 -*-
"""Run PCA analysis on the Decathlon dataset.
This script reproduces the steps from the first part of the
`robert_thibault_lab2.ipynb` notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Load the data
df = pd.read_csv("decathlon.csv")

# Quantitative variables
quant_vars = [
    '100m', 'Long.jump', 'Shot.put', 'High.jump', '400m', '110m.hurdle',
    'Discus', 'Pole.vault', 'Javeline', '1500m', 'Points'
]
X = df[quant_vars]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Compute covariance matrix and its eigenvalues/vectors
A = np.cov(X_scaled, rowvar=False)
print("Matrice de covariance A :\n", A)

eigvals, eigvecs = np.linalg.eig(A)
print("\nValeurs propres :\n", eigvals)
print("\nVecteurs propres (colonnes) :\n", eigvecs)

for i in range(len(eigvals)):
    print(f"\nComposante principale {i+1} :")
    print(f"  Valeur propre : {eigvals[i]}")
    print(f"  Vecteur propre : {eigvecs[:, i]}")

# Sort eigenvalues and eigenvectors in descending order
idx_sorted = np.argsort(eigvals)[::-1]
eigvals_sorted = eigvals[idx_sorted]
eigvecs_sorted = eigvecs[:, idx_sorted]
print("\nValeurs propres triées (ordre décroissant) :\n", eigvals_sorted)

# Scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigvals_sorted) + 1), eigvals_sorted, marker='o')
plt.title("Scree plot")
plt.xlabel("Numéro de la composante principale")
plt.ylabel("Valeur propre (Eigenvalue)")
plt.xticks(range(1, len(eigvals_sorted) + 1))
plt.grid(True)
plt.show()

# Inertia explained by each principal axis
explained_var = pca.explained_variance_ratio_ * 100
cumulative_var = np.cumsum(explained_var)
for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var), 1):
    print(f"Axe {i}: {var:.2f}% (cumulé: {cum_var:.2f}%)")

# Correlation circle for the first two PCs
plt.figure(figsize=(7, 7))
circle = plt.Circle((0, 0), 1, color='gray', fill=False)
plt.gca().add_artist(circle)
for i, var in enumerate(quant_vars):
    x = eigvecs_sorted[i, 0]
    y = eigvecs_sorted[i, 1]
    plt.arrow(0, 0, x, y, head_width=0.04, head_length=0.04, fc='b', ec='b')
    plt.text(x * 1.08, y * 1.08, var, fontsize=12)
plt.xlabel("Axe 1 (PC1)")
plt.ylabel("Axe 2 (PC2)")
plt.title("Cercle de corrélation (axes 1 et 2)")
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.grid()
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.show()

# Projection of individuals on the first two PCs
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
for i, name in enumerate(df["Athlets"]):
    plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=8, alpha=0.6)
plt.xlabel("Axe 1 (PC1)")
plt.ylabel("Axe 2 (PC2)")
plt.title("Projection des individus sur le plan factoriel (PC1 vs PC2)")
plt.grid(True)
plt.show()

# Contribution des individus aux axes 1 et 2
contrib_PC1 = (X_pca[:, 0] ** 2) / np.sum(X_pca[:, 0] ** 2)
contrib_PC2 = (X_pca[:, 1] ** 2) / np.sum(X_pca[:, 1] ** 2)

top_PC1 = np.argsort(contrib_PC1)[::-1][:5]
top_PC2 = np.argsort(contrib_PC2)[::-1][:5]

print("Individus qui contribuent le plus à l'axe 1 (PC1) :")
for idx in top_PC1:
    print(f"{df['Athlets'].iloc[idx]} ({contrib_PC1[idx]*100:.2f}%)")

print("\nIndividus qui contribuent le plus à l'axe 2 (PC2) :")
for idx in top_PC2:
    print(f"{df['Athlets'].iloc[idx]} ({contrib_PC2[idx]*100:.2f}%)")

# Projection of individuals on the first three PCs
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.7)
for i, name in enumerate(df["Athlets"]):
    ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], name, fontsize=8, alpha=0.6)
ax.set_xlabel("Axe 1 (PC1)")
ax.set_ylabel("Axe 2 (PC2)")
ax.set_zlabel("Axe 3 (PC3)")
ax.set_title("Projection des individus sur les 3 premiers axes principaux (ACP)")
plt.show()

contrib_PC3 = (X_pca[:, 2] ** 2) / np.sum(X_pca[:, 2] ** 2)
top_PC3 = np.argsort(contrib_PC3)[::-1][:5]
print("Individus qui contribuent le plus à l'axe 3 (PC3) :")
for idx in top_PC3:
    print(f"{df['Athlets'].iloc[idx]} ({contrib_PC3[idx]*100:.2f}%)")

# Projection 3D with color based on PC4
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
pc4_values = X_pca[:, 3]
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=pc4_values, cmap='viridis', alpha=0.8)
for i, name in enumerate(df["Athlets"]):
    ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], name, fontsize=8, alpha=0.6)
ax.set_xlabel("Axe 1 (PC1)")
ax.set_ylabel("Axe 2 (PC2)")
ax.set_zlabel("Axe 3 (PC3)")
ax.set_title("Projection 3D (PC1, PC2, PC3) - Couleur = PC4")
cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Valeur sur la 4e composante principale (PC4)")
plt.show()

contrib_PC4 = (X_pca[:, 3] ** 2) / np.sum(X_pca[:, 3] ** 2)
top_PC4 = np.argsort(contrib_PC4)[::-1][:5]
print("Individus qui contribuent le plus à l'axe 4 (PC4) :")
for idx in top_PC4:
    print(f"{df['Athlets'].iloc[idx]} ({contrib_PC4[idx]*100:.2f}%)")

# Quality of representation (cos²) on the first two axes
cos2_vars = eigvecs_sorted[:, 0] ** 2 + eigvecs_sorted[:, 1] ** 2
print("Qualité de représentation des variables (cos², axes 1 et 2) :")
for var, cos2 in zip(quant_vars, cos2_vars):
    print(f"{var}: {cos2:.2f}")

cos2_ind = (X_pca[:, 0] ** 2 + X_pca[:, 1] ** 2) / np.sum(X_pca ** 2, axis=1)
print("\nIndividus les mieux représentés (cos² élevé, axes 1 et 2) :")
top_ind = np.argsort(cos2_ind)[::-1][:5]
for idx in top_ind:
    print(f"{df['Athlets'].iloc[idx]} (cos²={cos2_ind[idx]:.2f})")

print("\nIndividus les moins bien représentés (cos² faible, axes 1 et 2) :")
bottom_ind = np.argsort(cos2_ind)[:5]
for idx in bottom_ind:
    print(f"{df['Athlets'].iloc[idx]} (cos²={cos2_ind[idx]:.2f})")

