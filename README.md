# Face2Gender: Intelligent Gender Classification from Facial Images

This project aims to develop a gender classification system using facial images based on state-of-the-art machine learning techniques. 
The goal is to automatically predict the gender of individuals from facial features, which has various practical applications in human-computer interaction, marketing, and security systems.
This project was undertaken as a course project for the "Pattern Recognition" course, under the guidance of Professor [Hassan Khotanlou](https://scholar.google.com/citations?user=5YX31NgAAAAJ&hl=en). The course provided valuable insights into the principles and applications of pattern recognition algorithms, which formed the foundation for the development of the gender classification system using machine learning. 

> **NOTE:** For a detailed explanation of the project, including methodologies, experimental setup, and results, please refer to the full report provided in [report.pdf](ProjectReport.pdf).

# Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Extraction](#feature-extraction)
5. [Dimensionality Reduction](#dimensionality-reduction)
6. [Classification Models](#classification-models)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [Conclusion](#conclusion)

## Introduction

Gender classification from facial images is a significant and challenging task in computer vision and machine learning. 
This project proposes a comprehensive approach that leverages machine learning algorithms and feature extraction techniques to tackle this problem effectively. 


## Dataset

The dataset used in this project is the UTKFace dataset, which consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover a wide variation in pose, facial expression, illumination, occlusion, and resolution. The dataset provides aligned and cropped face images along with corresponding landmarks (68 points). The images are labeled by age, gender, and ethnicity, making it suitable for gender classification tasks.

## Data Preprocessing

In the data preprocessing step, we converted the images to grayscale and resized them to a standardized 64x64 resolution. Additionally, we removed faces that were smaller than 64x64 pixels. The application of Multi-Task Cascaded Convolutional Networks (MTCNN) allowed us to efficiently detect and extract facial regions from the wild images, eliminating extraneous background information and irrelevant features unrelated to gender classification.

## Feature Extraction

To represent facial features in a machine-readable format, we employed two feature extraction techniques: Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP). HOG extracts shape and texture information from facial images, while LBP captures local patterns in the image. The extracted features were used as input for our machine learning models.

## Dimensionality Reduction

In order to reduce the dimensionality of the feature space and enhance the efficiency of our models, Principal Component Analysis (PCA) was applied. By choosing an optimal number of principal components, we achieved a significant reduction in training time and feature space dimension from 64x64 (4096 features) to a smaller set of features. The application of PCA with 256 components resulted in better results in most cases.

## Classification Models

We experimented with three popular classification models for gender classification: Support Vector Machine (SVM), Random Forest, and K-Nearest Neighbors (KNN). We employed the SVC version of SVM from scikit-learn, which is suitable for binary classification tasks. The performance of each model was evaluated using various metrics, including accuracy, sensitivity, specificity, and F1-score.

## Evaluation Metrics

During the evaluation of our models, we employed weighted averaging for accuracy, sensitivity, and F1-score to assess their performance. Weighted averaging accounts for class imbalance and provides a more comprehensive evaluation of the model's overall effectiveness.

## Results

The performance of each classifier was compared with and without PCA using features extracted from both HOG and LBP methods. 
The results showed that PCA with 256 components consistently outperformed other settings with 80.88% accuracy. 
The models achieved high performance in other evaluation metrics, demonstrating the effectiveness of our approach in gender classification from facial images.

## Conclusion

In conclusion, this project successfully developed a gender classification system based on facial images using machine learning algorithms and feature extraction techniques. The application of PCA significantly reduced training time and the dimensionality of the feature space, resulting in improved performance for most classifiers. The project demonstrates the potential of using facial features for gender classification and opens up avenues for further research and applications in computer vision and machine learning.

Feel free to reach out for any questions or collaboration opportunities! I appreciate your interest in this project ❤️
