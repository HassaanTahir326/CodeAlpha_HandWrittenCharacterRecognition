import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

TRAIN_FILE = "../mnist_train.csv"
if not os.path.exists(TRAIN_FILE):
    print("dataset not found at", TRAIN_FILE)
    print("make sure mnist_train.csv is placed one folder above this script")
    exit()
MODEL_FOLDER = "../models"
RESULTS_FOLDER = "../results"
EPOCHS = 15
BATCH_SIZE = 128


def load_data(file_path):
    data = pd.read_csv(file_path)
    labels = data["label"].values
    pixels = data.drop("label", axis=1).values

    images = pixels.reshape(-1, 28, 28, 1).astype("float32") / 255.0  # reshape and scale
    labels = to_categorical(labels, num_classes=10)  # 5 -> [0,0,0,0,0,1,0,0,0,0]

    return images, labels


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),  # finds edges/curves
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),  # finds bigger shapes
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),  # helps avoid overfitting
        Dense(10, activation="softmax")  # 10 digits (0-9)
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_history(history, save_path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    X_train, y_train = load_data(TRAIN_FILE)
    print(f"Training on {X_train.shape[0]} images")

    model = build_model()
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    model_path = os.path.join(MODEL_FOLDER, "mnist_cnn_model.h5")
    model.save(model_path)
    print(f"Model saved: {model_path}")

    graph_path = os.path.join(RESULTS_FOLDER, "training_curves.png")
    plot_history(history, graph_path)
    print(f"Graph saved: {graph_path}")

    print("\nDone. Run evaluate_model.py next")


if __name__ == "__main__":
    main()
