import numpy as np
from typing import Any

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

from src.data import create_cropped_dataset, load_dataset


def train_and_evaluate(images: np.ndarray, labels: np.ndarray, classifier: Any,
                       use_pca: bool = False, pca_components: int = None) -> None:
    """
    Train and evaluate the machine learning classifier on the given dataset.

    Parameters:
        images (np.ndarray): The input feature data (images).
        labels (np.ndarray): The target labels corresponding to the input images.
        classifier (Any): The machine learning classifier to be trained and evaluated.
        use_pca (bool, optional): Whether to use PCA for dimensionality reduction. Defaults to False.
        pca_components (int, optional): The number of components for PCA. Defaults to None.

    This function splits the dataset into training and testing sets, applies PCA if specified,
    fits the classifier to the training data, and then evaluates the classifier on the testing data.
    It prints the classification report showing the performance of the classifier on each class.
    """
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

    if use_pca:
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        explained_variance_ratio = pca.explained_variance_ratio_.sum()
        print(f"PCA: Number of components = {pca_components}, Explained Variance Ratio = {explained_variance_ratio:.2f}")

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f"\n--- {classifier.__class__.__name__} ---")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    RAW_DATA_PATH = "raw data"
    DATASET_PATH = "dataset"
    IMG_SIZE = (64, 64)

    # Create the cropped dataset
    create_cropped_dataset(RAW_DATA_PATH, DATASET_PATH, IMG_SIZE)

    # Load the dataset
    images, labels = load_dataset(RAW_DATA_PATH, DATASET_PATH)

    # Define different n components for PCA
    pca_components_list = [1024, 512, 256]

    # Define classifiers
    svm_classifier = SVC(kernel='linear', random_state=42)
    random_forest_classifier = RandomForestClassifier(random_state=42)
    knn_classifier = KNeighborsClassifier()

    # Train and evaluate with different PCA
    print("\n--- Without PCA ---")
    train_and_evaluate(images, labels, svm_classifier)
    train_and_evaluate(images, labels, random_forest_classifier)
    train_and_evaluate(images, labels, knn_classifier)

    # Train and evaluate with different PCA components for each classifier
    for pca_components in pca_components_list:
        print(f"\n--- PCA Components: {pca_components} ---")
        train_and_evaluate(images, labels, svm_classifier, use_pca=True, pca_components=pca_components)
        train_and_evaluate(images, labels, random_forest_classifier, use_pca=True, pca_components=pca_components)
        train_and_evaluate(images, labels, knn_classifier, use_pca=True, pca_components=pca_components)
