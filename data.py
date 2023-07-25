import os

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

from utils import extract_hog_features, extract_lbp_features


def detect_faces(image_path: str) -> list:
    """
    Detect faces in the input image using the MTCNN algorithm.

    Parameters:
        image_path (str): The path to the input image.

    Returns:
        list: A list of cropped face images detected in the input image.

    This function uses the MTCNN (Multi-Task Cascaded Convolutional Networks) algorithm
    to detect faces in the input image. The function loads the image, detects the faces,
    and crops the face regions to generate a list of cropped face images.
    """
    detector = MTCNN()
    img = cv2.imread(image_path)
    faces = detector.detect_faces(img)  # Detect faces in the image
    cropped_faces = []
    
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cropped_face = img[y:y1, x:x1]
        cropped_faces.append(cropped_face)
    
    return cropped_faces

def create_cropped_dataset(raw_data_path: str, dataset_path: str, img_size: tuple) -> None:
    """
    Create a cropped dataset of face images.

    Parameters:
    raw_data_path (str): The path to the directory containing the raw image data.
    dataset_path (str): The path to the directory where the cropped dataset will be saved.
    img_size (tuple): The target size of the cropped face images in the format (width, height).
    
    This function reads images from a directory, detects faces in each image,
    and creates a new dataset of cropped face images with the specified size.
    """
    images = os.listdir(raw_data_path)
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    
    for img in images:
        old_img_path = os.path.join(raw_data_path, img)
        new_img_path = os.path.join(dataset_path, img)
        detected_faces = detect_faces(old_img_path)
        for detected_face in detected_faces:
            # Check if the image is at least 64x64 in size
            if detected_face.shape[0] < 64 or detected_face.shape[1] < 64:
                # Skip this image if it's too small
                continue
            resized_image = cv2.resize(detected_face, img_size)
            cv2.imwrite(new_img_path, resized_image)

def load_dataset(dataset_path: str, feautre_extraction_method: str) -> tuple:
    """
    Load the cropped face dataset and extract features for machine learning.

    Parameters:
        dataset_path (str): The path to the directory containing the cropped face dataset.
        feautre_extraction_method (str): The feature extraction method to use ('hog' or 'lbp').

    Returns:
        tuple: A tuple containing the feature matrix (X) and the label array (y) for machine learning.

    This function loads the cropped face dataset from the specified directory and extracts features
    using the specified feature extraction method ('hog' for HOG features or 'lbp' for LBP features).

    The function reads each image in the dataset, extracts the specified features from the face images,
    and stores the feature matrix (X) and label array (y) for use in machine learning tasks.
    """
    images = os.listdir(dataset_path)
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    
    X = []
    y = []

    for img in images:
        img_path = os.path.join(dataset_path, img)
        # Extract label from the filename of the image
        label = int(img.split('_')[1])
        # Convert image format from RGB to GRAY-SCALE
        face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if feautre_extraction_method == 'hog':
            # Extract HOG features
            extracted_features = extract_hog_features(face_img)
        elif feautre_extraction_method == 'lbp':
            # Extract LBP features
            extracted_features = extract_lbp_features(face_img)

        X.append(extracted_features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y