# CodeAlpha_HandwrittenCharacterRecognition

Machine Learning Internship — Task 3: Handwritten Character Recognition
(CodeAlpha)

## Overview
This project trains a Convolutional Neural Network (CNN) to recognize
handwritten digits (0–9) using the MNIST dataset.

## Project Structure
```
CodeAlpha_HandwrittenCharacterRecognition/
├── models/
│   └── mnist_cnn_model.h5       # saved trained model (after running train_model.py)
├── notebooks_or_scripts/
│   ├── explore_data.py          # validates dataset, plots class distribution + samples
│   ├── train_model.py           # builds and trains the CNN, saves the model
│   └── evaluate_model.py        # loads model, evaluates on test set, confusion matrix
├── results/
│   ├── class_distribution.png
│   ├── sample_digits.png
│   ├── training_curves.png
│   └── confusion_matrix.png
├── mnist_train.csv              # 60,000 labeled training images
├── mnist_test.csv               # 10,000 labeled test images
└── README.md
```

## Dataset
- Source: MNIST (28x28 grayscale handwritten digit images)
- Format: CSV, each row = 1 label + 784 pixel values (0-255)
- Classes: 10 (digits 0-9)
- **Note:** `mnist_train.csv` and `mnist_test.csv` are not included in this repo
  (file size exceeds GitHub's limit). Download them from:
  https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
  and place both files in the project root before running the scripts.

## How to Run

1. Install dependencies:
   ```
   pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
   ```

2. From inside `notebooks_or_scripts/`, run in order:
   ```
   python explore_data.py     # validates data, saves class_distribution.png + sample_digits.png
   python train_model.py      # trains the CNN, saves models/mnist_cnn_model.h5 + training_curves.png
   python evaluate_model.py   # evaluates on test set, saves confusion_matrix.png
   ```

## Model Architecture
- Conv2D(32) -> MaxPooling -> Conv2D(64) -> MaxPooling
- Flatten -> Dense(128) -> Dropout(0.3) -> Dense(10, softmax)

## Results
Achieved **99.18% accuracy** on the held-out test set (10,000 images), with
0.98-1.00 precision/recall/F1 across all 10 digit classes. See
`Handwritten_Character_Recognition_Report.docx` for the full report.

## Notes
- Extendable to full EMNIST (digits + letters) or sequence models (CRNN)
  for word/sentence-level recognition, as noted in the task description.
