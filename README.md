# Chest X-ray Pneumonia Detection using Deep Learning

## Project Overview

This project develops a deep learning model to classify Chest X-ray images for the automated detection of Pneumonia. Leveraging transfer learning with a pre-trained Convolutional Neural Network (CNN), the model achieves high performance in distinguishing between Normal and Pneumonia cases, crucial for aiding medical diagnosis.

## Problem Statement

Pneumonia is a serious lung infection that can be challenging to diagnose quickly and accurately. Automated analysis of Chest X-ray images using AI can potentially support radiologists and healthcare professionals by providing a faster and more consistent screening tool, especially in resource-constrained environments or for large-scale screening programs.

## Dataset

* **Name:** Chest X-Ray Images (Pneumonia) Dataset
* **Source:** Available on Kaggle.
* **Content:** Contains a large collection of Chest X-ray images categorized as 'Normal' or 'Pneumonia'.
* **Size:** The dataset is split into Training, Validation, and Test sets (Total ~5856 images).
* **Challenge:** The dataset exhibits a significant class imbalance, with considerably more images belonging to the Pneumonia class than the Normal class. This requires careful evaluation using appropriate metrics.

## Methodology

The project follows a standard machine learning workflow adapted for image data:

1.  **Data Preparation:**
    * Downloaded and organized the dataset into training, validation, and test directories.
    * Utilized TensorFlow/Keras `ImageDataGenerator` for efficient batch loading of images directly from directories.
    * Implemented **Data Augmentation** techniques (including rotations, shifts, shear, zoom, and horizontal flips) on the training set to increase dataset size and variability, helping the model generalize better.
    * **Preprocessing** steps included resizing all images to 150x150 pixels and scaling pixel values from [0, 255] to [0, 1].
2.  **Model Architecture:**
    * Built a model using a **Convolutional Neural Network (CNN)** architecture.
    * Applied **Transfer Learning** by starting with the weights of **MobileNetV2**, a powerful CNN pre-trained on the vast ImageNet dataset. The original top classification layer was removed.
    * Added custom **Dense layers** on top of the pre-trained base to learn features specific to Chest X-rays and perform binary classification.
3.  **Training:**
    * Compiled the model using the **Adam optimizer** and **Binary Crossentropy loss function**.
    * Tracked performance during training using **Accuracy** and **AUC (Area Under the ROC Curve)** metrics.
    * Implemented key **Callbacks** for robust training management:
        * **Model Checkpointing:** Saved the model weights whenever a new best performance (monitored using validation AUC) was achieved.
        * **Early Stopping:** Automatically stopped training if validation performance didn't improve for a set number of epochs, preventing overfitting.
    * **(Optional/Advanced Exploration):** Explored techniques like fine-tuning layers of the pre-trained base model and using class weights during a second training phase, evaluating their impact on performance (though the initial approach yielded the best test results in this case).
4.  **Evaluation:**
    * Evaluated the final model on the unseen **Test Set** to obtain an unbiased measure of its performance.
    * Used comprehensive metrics including **Test Loss, Test Accuracy, Test AUC, Confusion Matrix, and Classification Report (Precision, Recall, F1-score)** to understand performance across both classes.

## Results

The model achieved strong performance on the held-out test set:

* **Test AUC (Area Under the ROC Curve):** **0.9528**
* **Test Accuracy:** **0.8766 (~88%)**
* Test Loss: 0.2970

The high **Test AUC (0.95+)** is a key indicator of the model's excellent ability to distinguish between Normal and Pneumonia cases, which is particularly important given the class imbalance in the dataset.

## Evaluation Metrics Explained

* **Accuracy:** The proportion of total correct predictions. While intuitive, it can be misleading in imbalanced datasets.
* **AUC (Area Under the ROC Curve):** Measures the model's capability to discriminate between positive and negative classes across various classification thresholds. A score closer to 1.0 indicates better discrimination. It's a robust metric for imbalanced classification.
* **Confusion Matrix:** A table summarizing classification results, showing True Positives, True Negatives, False Positives, and False Negatives.
* **Precision (for a class):** Out of all instances predicted as this class, how many were correct?
* **Recall (Sensitivity) (for a class):** Out of all actual instances of this class, how many were correctly identified?

## Code

The full project code and step-by-step implementation details are available in the `Medical_Image_Analysis_CheXpert.ipynb` notebook in this repository.

## Dependencies

Key Python libraries and frameworks used in this project:

* `tensorflow` / `keras`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `Pillow` (PIL)
* `os`, `random`, `math`

## Visualizations

Key visualizations from the project:


* **Confusion Matrix on Test Set:**
  
    ![Confusion Matrix Heatmap](https://github.com/user-attachments/assets/a4d59f2d-99e6-498f-8781-9fe46a961ccb)

* **Distribution of Predicted Probabilities by True Class:**
  
    ![Predicted Probability Distribution](https://github.com/user-attachments/assets/426f95d9-de3d-4bbd-9d11-6fe1c3c1f054)

* **Visualizing Predictions on Random Images:**
  
    ![Visualizing Predictions on Random Images](https://github.com/user-attachments/assets/89d576cb-9a68-4f4a-be33-983cbbabc3d2)

---
