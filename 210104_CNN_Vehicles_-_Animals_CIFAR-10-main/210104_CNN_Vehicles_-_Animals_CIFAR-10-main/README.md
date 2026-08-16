# Vehicles & Animals Image Classification with CIFAR-10 (CNN)

This project presents a complete **Convolutional Neural Network (CNN)** based image classification pipeline implemented using **PyTorch** from scratch.  
The model is trained on the **CIFAR-10 dataset** and evaluated on both:
- the standard CIFAR-10 test set, and  
- **real-world smartphone images** to analyze generalization performance.

The notebook is fully automated and can be executed using **Run All** in Google Colab.

---

## Dataset

### Standard Dataset
- **CIFAR-10** (10 classes, 60,000 images, 32×32 RGB):
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
  - Training set: 50,000 images | Test set: 10,000 images

### Custom Dataset (Real-world Images)
- Real-world images captured using a smartphone camera
- Preprocessed to 32×32 pixels with CIFAR-10 normalization
- Used to evaluate model generalization under domain shift conditions

---

## Project Structure

210113_CNN_Vehicles_-_Animals_CIFAR-10/<br>
├── dataset/ <br>
├── model/<br>
│ └── 210113.pth <br>
├── 210113.ipynb <br>
└── README.md<br>

---

## Model Architecture

The CNN is implemented **from scratch** using `torch.nn.Module` and follows an encoder-classifier design pattern.

### Architecture Overview

| Layer | Type            | Configuration                             | Output Shape  |
|:-----:|-----------------|-------------------------------------------|:-------------:|
| 1     | Convolutional   | 32 filters, 3×3, padding=1, ReLU         | 32 × 32 × 32  |
| 2     | Max Pooling     | 2×2, stride=2                             | 32 × 16 × 16  |
| 3     | Convolutional   | 64 filters, 3×3, padding=1, ReLU         | 64 × 16 × 16  |
| 4     | Max Pooling     | 2×2, stride=2                             | 64 × 8 × 8    |
| 5     | Convolutional   | 128 filters, 3×3, padding=1, ReLU        | 128 × 8 × 8   |
| 6     | Max Pooling     | 2×2, stride=2                             | 128 × 4 × 4   |
| 7     | Flatten         | 128 × 4 × 4 = 2048 units                 | 2048          |
| 8     | Fully Connected | 2048 → 256, ReLU, Dropout (p=0.3)        | 256           |
| 9     | Output          | 256 → 10, softmax at inference            | 10            |


<img width="1495" height="606" alt="cnn architecture" src="https://github.com/user-attachments/assets/fbf244a3-6175-4b76-a7b9-7b7aaa4ea38d" />


### Key Components
- **Feature Extractor:** 3 convolutional blocks (Conv → ReLU → MaxPool)
- **Classifier Head:** Fully connected layer with dropout regularization
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001, β₁=0.9, β₂=0.999)
- **Regularization:** Dropout (p=0.3) to reduce overfitting

### Design Notes
- All conv layers use `padding=1` to **preserve spatial dimensions** after each convolution
- Spatial reduction is handled exclusively by **MaxPool2d(2)** layers
- Three pooling stages reduce 32×32 input down to 4×4 feature maps
- Total learnable parameters concentrated in the FC layer (2048 → 256 → 10)

---

## Training

- Dataset automatically downloaded using `torchvision.datasets.CIFAR10`
- Images preprocessed using `torchvision.transforms` (Resize → ToTensor → Normalize)
- **Normalization:** mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
- **Train/Validation split:** 90% train (45,000) / 10% val (5,000)
- Training and validation performance tracked across all epochs
- Best model checkpoint saved based on highest validation accuracy

---

## Training & Validation Results

### Training Log (Per Epoch)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1     | 1.4306     | 0.4792    | 1.1085   | 0.5956  |
| 2     | 1.0077     | 0.6428    | 0.8827   | 0.6870  |
| 3     | 0.8386     | 0.7045    | 0.8156   | 0.7148  |
| 4     | 0.7162     | 0.7489    | 0.7539   | 0.7350  |
| 5     | 0.6225     | 0.7805    | 0.7380   | 0.7418  |
| 6     | 0.5576     | 0.8042    | 0.7646   | 0.7430  |
| 7     | 0.4828     | 0.8289    | 0.7666   | 0.7486  |
| 8     | 0.4248     | 0.8466    | 0.7717   | 0.7542  |
| 9     | 0.3719     | 0.8661    | 0.8016   | 0.7432  |
| 10    | 0.3329     | 0.8799    | 0.7982   | 0.7548  |

### Observations
- Training loss decreases monotonically across all 10 epochs
- Training accuracy reaches **~88%** by the final epoch
- Validation accuracy stabilizes around **75–76%**
- Validation loss begins increasing after epoch 5–6, indicating **overfitting**
- Best generalization performance observed around **Epoch 7–8**
- Gap between train acc (~88%) and val acc (~75%) confirms moderate overfitting

---

## Training Curves

### Loss vs Epoch
<img width="567" height="455" alt="download" src="https://github.com/user-attachments/assets/ec1bb977-5139-4471-ac7b-abf41897ed5b" />

### Accuracy vs Epoch
<img width="576" height="455" alt="download" src="https://github.com/user-attachments/assets/faf9d9be-4f5d-4689-9bed-927a3fd6b817" />

### Observations
- Training loss decreases steadily; validation loss diverges after epoch 5–6
- Training accuracy improves consistently throughout
- Validation accuracy plateaus, showing the model's generalization limit
- Overfitting is an expected behavior given no data augmentation is applied

---

## Evaluation on CIFAR-10 Test Set

### Confusion Matrix
<img width="853" height="766" alt="download" src="https://github.com/user-attachments/assets/4086501d-accd-4c7f-ab71-9d5c13681613" />

### Key Observations
- **Strong performance** on vehicle classes (airplane, automobile, ship, truck)
- **Notable confusion** between visually similar animal pairs: cat↔dog, deer↔horse
- Low resolution (32×32) limits discriminative feature availability
- Prediction errors cluster among semantically and visually related categories

---

## Real-World Smartphone Image Predictions

<img width="950" height="665" alt="download" src="https://github.com/user-attachments/assets/c3f0448c-194f-496b-afb6-470a513408e7" />

### Sample Predictions
- **Truck** — Confidence: 87.7%
- **Airplane** — Confidence: 76.1%
- **Cat** — Confidence: 68.0%

### Observations
- Vehicle images classified with **high confidence** due to distinct structural features
- Animal classes show **lower confidence** due to inter-class visual similarity
- Confidence reduced by **domain shift** between CIFAR-10 training images and real-world photos

---

## Visual Error Analysis

<img width="970" height="397" alt="download" src="https://github.com/user-attachments/assets/19dd1337-e107-4797-b8ac-f93e0e3d32ba" />

These examples highlight the core challenge: at 32×32 resolution, fine-grained details such as facial features, fur texture, or vehicle insignia are largely lost. The model relies on coarser shape and color cues that may be shared across multiple classes, leading to systematic confusion among visually similar categories.

---

## How to Run (Google Colab)

1. Open the notebook in Google Colab:  
   Click here to open [`210113.ipynb`](https://colab.research.google.com/drive/1H0BBNrzxFN-SPrRPr2kSWs63XaEaWGUg?usp=sharing) in Colab

2. Select **Runtime → Run all**

All training, evaluation, and visualizations will execute automatically.  
No manual cloning or file uploads are required.

---

## Author
**Nafiullah Khan Siam**  
**Student ID: 210113**
