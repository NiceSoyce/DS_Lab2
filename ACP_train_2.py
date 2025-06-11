import numpy as np
import joblib
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Charger les données
digits = load_digits()
X = digits.data
y = digits.target

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# ACP pour expliquer au moins 90% de la variance
pca = PCA()
pca.fit(X_train)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
k = np.argmax(cumulative_variance >= 0.90) + 1

pca_k = PCA(n_components=k)
X_train_pca = pca_k.fit_transform(X_train)

# Définition et entraînement du modèle SVM
svm_rbf_pca = SVC(kernel='rbf')
svm_rbf_pca.fit(X_train_pca, y_train)

# Argument parsing for output file path
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='digits_pca_svm.pkl', help='Output file path')
args = parser.parse_args()

# Sauvegarde du pipeline (scaler, pca, svm) dans un fichier .pkl
joblib.dump({
    "scaler": scaler,
    "pca": pca_k,
    "svm": svm_rbf_pca
}, args.output)
