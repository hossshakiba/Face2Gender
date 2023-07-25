import numpy as np
from skimage.feature import hog, local_binary_pattern


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extract Histogram of Oriented Gradients (HOG) features from the input image.

    Parameters:
        image (np.ndarray): The input grayscale image.

    Returns:
        np.ndarray: The HOG features of the image.
    """
    hog_features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return hog_features

def extract_lbp_features(image: np.ndarray) -> np.ndarray:
    """
    Extract Local Binary Pattern (LBP) features from the input image.

    Parameters:
        image (np.ndarray): The input grayscale image.

    Returns:
        np.ndarray: The LBP histogram features of the image.
    """
    lbp = local_binary_pattern(image, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    return hist