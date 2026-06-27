# Brain Tumor Segmentation using N-1 Net

This repository contains a PyTorch implementation of a custom deep learning architecture, **NMinusOneNet**, designed for the semantic segmentation of brain tumors from medical images.

## 🧠 Project Overview
The project trains a segmentation model to accurately identify and mask brain tumors. The custom `NMinusOneNet` architecture is inspired by U-Net but introduces unique N-1 downsampling routing in the decoder and an attention mechanism at the bottleneck to improve spatial feature reconstruction.

## 📊 Dataset
The dataset consists of `.jpg` image and mask pairs split into training, validation, and testing sets:
* **Training:** 485 samples
* **Validation:** 40 samples
* **Testing:** 41 samples

**Data Augmentation:** To prevent overfitting, the training pipeline applies random horizontal flips (50%), vertical flips (50%), and random rotations (-20 to +20 degrees).

## 🏗️ Model Architecture (`NMinusOneNet`)
* **Encoder:** 5 blocks of Convolution + BatchNorm + ReLU with MaxPooling. 
* **Bottleneck:** Fully connected convolutions coupled with a Softmax-based attention mechanism.
* **Decoder:** Utilizes **N-1 routing**, meaning upsampled features are concatenated not only with their symmetric encoder counterpart but also with pooled diagonal features from the previous encoder level.
* **Output:** A 1x1 convolution with a Sigmoid activation to produce a binary tumor mask.

## ⚙️ Training Details
* **Loss Function:** A custom weighted loss combining Binary Cross-Entropy (20%) and Jaccard/IoU Loss (80%).
* **Optimizer:** Adam optimizer (`lr=0.001`, `weight_decay=1e-4`).
* **Scheduler:** `ReduceLROnPlateau` to reduce the learning rate when validation loss plateaus.
* **Epochs:** Trained for 200 epochs.

## 📈 Results
The model achieves strong performance, reaching an **Average Intersection over Union (IoU) of ~86.6%** on the hold-out test set. 

The notebook also includes visualization code using OpenCV to overlay the predicted segmentation masks directly onto the original MRI images for qualitative assessment, as well as code to generate ROC (Receiver Operating Characteristic) and PR (Precision-Recall) curves.

## 🛠️ Requirements
* `torch`, `torchvision`
* `numpy`, `pandas`
* `opencv-python` (`cv2`)
* `Pillow`
* `matplotlib`
* `scikit-learn`
* `tqdm`