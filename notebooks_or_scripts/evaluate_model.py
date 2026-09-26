import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

TEST_FILE = "../mnist_test.csv"
MODEL_FILE = "../models/mnist_cnn_model.h5"
if not os.path.exists(TEST_FILE):
    print("dataset not found at", TEST_FILE)
    print("make sure mnist_test.csv is placed one folder above this script")
    exit()

if not os.path.exists(MODEL_FILE):
    print("model file not found at", MODEL_FILE)
    print("run train_model.py first to create it")
    exit()
RESULTS_FOLDER = "../results"


def load_data(file_path):
    data = pd.read_csv(file_path)
    labels = data["label"].values
    pixels = data.drop("label", axis=1).values

    images = pixels.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    labels_onehot = to_categorical(labels, num_classes=10)

    return images, labels_onehot, labels


def plot_confusion_matrix(y_true, y_pred, save_path):
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    X_test, y_test, y_test_labels = load_data(TEST_FILE)
    print(f"Testing on {X_test.shape[0]} images")

    model = load_model(MODEL_FILE)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Loss: {loss:.4f}")

    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)  # highest probability digit

    print("\nPer-digit report:")
    print(classification_report(y_test_labels, predicted_labels))

    save_path = os.path.join(RESULTS_FOLDER, "confusion_matrix.png")
    plot_confusion_matrix(y_test_labels, predicted_labels, save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
