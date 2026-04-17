# Optical Character Recognition (OCR) for Character Classification

A comprehensive Machine Learning and Computer Vision project designed to segment, preprocess, and recognize individual characters from combined images (e.g., CAPTCHAs). This project explores both traditional Machine Learning pipelines (with HOG feature extraction) and Deep Learning approaches (Convolutional Neural Networks).

## 🚀 Project Overview
The objective of this project is to accurately classify lowercase English alphabets extracted from noisy combined image sequences. The pipeline tackles real-world computer vision challenges, including noise reduction, feature engineering, and hyperparameter tuning to maximize classification accuracy.

## 🛠️ Technologies & Libraries Used
* **Languages:** Python
* **Computer Vision & Image Processing:** OpenCV (`cv2`), Scikit-Image (`skimage`), Pillow (`PIL`)
* **Machine Learning:** Scikit-Learn (`sklearn`)
* **Deep Learning:** Keras, TensorFlow
* **Data Visualization:** Matplotlib, Seaborn, Pandas, NumPy

## 🧠 Approach & Methodology

### 1. Image Segmentation
* Combined images containing 5 letters are mathematically split into equal parts to isolate individual characters, resulting in a dataset of single-character images.

### 2. Image Preprocessing Pipeline
To improve dataset quality and isolate the character from background noise, the following preprocessing steps are applied:
* **Resizing:** Standardizing image dimensions to 64x128.
* **Grayscaling:** Dimensionality reduction.
* **Histogram Equalization:** Enhancing contrast.
* **Morphological Operations:** Dilation to fill holes and thicken character lines.
* **Denoising & Filtering:** Non-Local Means (NLMeans) denoising and Median Filtering to smoothen edges.

### 3. Feature Extraction
* **Histogram of Oriented Gradients (HOG):** Extracted gradient and edge directions using 9 orientations, 8x8 pixels per cell, and 1x1 cells per block to feed into traditional ML models.

### 4. Classification Models
Three traditional ML classifiers were trained and evaluated:
* **Logistic Regression**
* **Random Forest Classifier**
* **Support Vector Machine (SVM)** (Yielded the best initial accuracy)

**Hyperparameter Tuning:**
Applied `GridSearchCV` on the SVM model to optimize hyperparameters. 
* **Best Configuration:** Polynomial Kernel (`kernel='poly'`), `C=10`, `degree=2`, `coef0=0.0`. 
* *Insight:* A higher penalty parameter (`C=10`) generated a smaller margin but fit the complex decision boundaries better, significantly improving accuracy.

### 5. Deep Learning Alternative (CNN)
Implemented a Convolutional Neural Network (CNN) architecture using Keras for automated feature extraction and classification.
* **Architecture:** 2x Conv2D layers (with ReLU) -> 2x MaxPooling2D -> Flatten -> Dense (64 units) -> Softmax Output.

## 📊 Evaluation & Insights
The models were evaluated using Accuracy, Precision, Recall, and F1-score. A Confusion Matrix was generated using Seaborn to visualize misclassifications.
* **Observations:** The model occasionally struggles with visually similar characters due to noise constraints (e.g., predicting 'h' as 'b', 'x' as 'f' or 'i', and 't' as 'c' or 'l').

## 👨‍💻 Author
**Bhavya Khandelwal**
