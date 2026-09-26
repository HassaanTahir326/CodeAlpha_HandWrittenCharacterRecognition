import os
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_FILE = "../mnist_train.csv"
TEST_FILE = "../mnist_test.csv"
RESULTS_FOLDER = "../results"


def check_data(data, name):
    print(f"\n{name} data: {data.shape[0]} rows, {data.shape[1]} columns")

    if "label" not in data.columns:
        raise ValueError(f"{name}: no label column")

    pixel_cols = [col for col in data.columns if col != "label"]
    if len(pixel_cols) != 784:  # 28x28 image
        raise ValueError(f"{name}: expected 784 pixel columns")

    if data.isnull().values.any():
        raise ValueError(f"{name}: missing values found")

    print(f"{name} data looks good")
    return pixel_cols


def plot_digit_counts(data, save_path):
    plt.figure(figsize=(8, 5))
    data["label"].value_counts().sort_index().plot(kind="bar")
    plt.title("Digit counts")
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_sample_images(data, pixel_cols, save_path):
    plt.figure(figsize=(8, 8))
    for i in range(16):
        row = data.iloc[i]
        image = row[pixel_cols].values.reshape(28, 28)  # 784 numbers -> image
        plt.subplot(4, 4, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {row['label']}")
        plt.axis("off")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    pixel_cols = check_data(train_data, "Train")
    check_data(test_data, "Test")

    plot_digit_counts(train_data, os.path.join(RESULTS_FOLDER, "class_distribution.png"))
    plot_sample_images(train_data, pixel_cols, os.path.join(RESULTS_FOLDER, "sample_digits.png"))

    print("\nDone. Run train_model.py next")


if __name__ == "__main__":
    main()
