import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits


def main():
    # Load the digits dataset
    digits = load_digits()

    # Display some images with labels
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5),
                             subplot_kw={'xticks': [], 'yticks': []})
    for ax, image, label in zip(axes.flat, digits.images, digits.target):
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(label)
    plt.show()

    # Display the distribution of classes
    class_distribution = np.bincount(digits.target)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(class_distribution)), class_distribution)
    plt.xticks(range(len(class_distribution)))
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.title('Distribution of Digits')
    plt.show()

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(digits.data)
    y = digits.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

    print("Shape of training data (X_train):", X_train.shape)
    print("Shape of testing data (X_test):", X_test.shape)
    print("Shape of training labels (y_train):", y_train.shape)
    print("Shape of testing labels (y_test):", y_test.shape)

    # Train an RBF SVM classifier on original data
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)

    # Evaluate the classifier on the test set
    y_pred = svm_rbf.predict(X_test)

    # Print the classification report
    print("Classification Report for RBF SVM on original data:")
    print(classification_report(y_test, y_pred))

    # Apply PCA on the training data
    pca = PCA()
    pca.fit(X_train)

    # Compute cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Choose k to explain at least 90% of the variance
    k = np.argmax(cumulative_variance >= 0.90) + 1
    print(f"Nombre de composantes pour expliquer au moins 90% de la variance : {k}")

    # Transform the training and test data
    pca_k = PCA(n_components=k)
    X_train_pca = pca_k.fit_transform(X_train)
    X_test_pca = pca_k.transform(X_test)

    print("Shape de X_train_pca :", X_train_pca.shape)
    print("Shape de X_test_pca :", X_test_pca.shape)

    # Retrain the SVM classifier on the PCA-reduced data
    svm_rbf_pca = SVC(kernel='rbf')
    svm_rbf_pca.fit(X_train_pca, y_train)

    # Predict on the reduced test set
    y_pred_pca = svm_rbf_pca.predict(X_test_pca)

    # Display the classification report
    print("Classification Report for RBF SVM after PCA:")
    print(classification_report(y_test, y_pred_pca))

    # Reduce the data to 2 principal components for visualization
    pca_2 = PCA(n_components=2)
    X_train_pca2 = pca_2.fit_transform(X_train)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_train_pca2[:, 0], X_train_pca2[:, 1], c=y_train,
                          cmap='tab10', alpha=0.7)
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.title('Projection des données sur les 2 premières composantes principales')
    plt.colorbar(scatter, label='Classe réelle')
    plt.show()


if __name__ == "__main__":
    main()
