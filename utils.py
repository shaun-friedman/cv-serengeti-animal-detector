"""
Utility functions for the Serengeti Animal Detector project.

This module contains:
- Path constants for Google Colab/Drive environments
- Data download and label preprocessing helpers
- TensorFlow dataset builders
- Visualization helpers for bounding boxes and predictions
"""

import os
import shutil
import subprocess

import cv2
import kagglehub
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# Path Constants
# =============================================================================

GOOGLE_DRIVE_SERENGETI_PATH = "/content/drive/MyDrive/serengeti"
DATASET_PATH = "/content/dataset"
LABELS_URL = "https://drive.google.com/uc?export=download&id=1F6PzJw6WqUUP_-a7r_xNpkKCjZ00tIWc"
DATASET_BASE_PATH = "/content/dataset/Set1/"
UNLABELED_PATH_BASE = "/content/dataset/Set1/1.58-Roe_Deer/SEQ767"

# =============================================================================
# Configuration Constants
# =============================================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# =============================================================================
# Data Download Helpers
# =============================================================================


def download_dataset(dataset_id: str = "silviamatoke/serengeti-dataset") -> str:
    """
    Download dataset from Kaggle using kagglehub.

    Args:
        dataset_id: Kaggle dataset identifier (default: silviamatoke/serengeti-dataset)

    Returns:
        Path to the downloaded dataset (kagglehub cache location)
    """
    path = kagglehub.dataset_download(dataset_id)
    print("Path to dataset files:", path)
    return path


# =============================================================================
# Label Preprocessing Helpers
# =============================================================================


def load_labels(output_path: str = "labels.csv") -> pd.DataFrame:
    """
    Download labels CSV from Google Drive and load it.

    Args:
        output_path: Local path to save the downloaded CSV file

    Returns:
        DataFrame with label data
    """
    # Download from Google Drive using wget
    subprocess.run(
        ["wget", "-q", LABELS_URL, "-O", output_path],
        check=True
    )
    print(f"✓ Downloaded labels.csv to {output_path}")
    
    labels_df = pd.read_csv(output_path)
    return labels_df


def prepare_label_data(
    labels_df: pd.DataFrame,
    path_base: str = DATASET_BASE_PATH,
    min_animals: int = 1,
) -> tuple:
    """
    Prepare label data for training by filtering positives and encoding labels.

    Args:
        labels_df: DataFrame with label data
        path_base: Base path for image files
        min_animals: Minimum animal count to include (default: 1)

    Returns:
        Tuple of (filepaths, labels, bbox_array, label_encoder, positives_df)
    """
    # Filter images that contain animals
    positives = labels_df[labels_df["animal_count"] >= min_animals]

    # Build full file paths
    filepaths = path_base + positives["file_path"].values

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(positives["animal"].values).astype("int32")

    # Extract bounding box coordinates (using first bounding box)
    bbox_array = positives[["a1", "a2", "a3", "a4"]].values.astype("float32")

    return filepaths, labels, bbox_array, label_encoder, positives


# =============================================================================
# TensorFlow Dataset Builders
# =============================================================================


def load_and_preprocess(path, label, bbox=None):
    """
    Load and preprocess a single image with its label and bounding box.

    Args:
        path: Path to the image file
        label: Class label (integer)
        bbox: Bounding box coordinates [x1, y1, x2, y2] or None

    Returns:
        Tuple of (image, {"class_output": label, "bbox_output": bbox}, path)
    """
    image_bytes = tf.io.read_file(path)

    # Check if file is a valid JPEG
    is_jpeg = tf.image.is_jpeg(image_bytes)

    # If file is not valid, class is set to -1 and filtered out
    def invalid():
        return (
            tf.zeros(IMAGE_SIZE + (3,)),
            {"class_output": -1, "bbox_output": tf.zeros((4,))},
            path,
        )

    def valid():
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Normalize bounding box for resized image
        h = tf.cast(tf.shape(image)[0], tf.float32)
        w = tf.cast(tf.shape(image)[1], tf.float32)
        if bbox is not None:
            bbox_norm = bbox / [w, h, w, h]
        else:
            bbox_norm = tf.zeros((4,))

        bbox_norm = tf.cast(bbox_norm, tf.float32)

        return (
            image,
            {"class_output": label, "bbox_output": bbox_norm},
            path,
        )

    return tf.cond(is_jpeg, valid, invalid)


def build_tf_dataset(
    filepaths: np.ndarray,
    labels: np.ndarray,
    bbox_array: np.ndarray,
) -> tf.data.Dataset:
    """
    Build a TensorFlow dataset from file paths, labels, and bounding boxes.

    Args:
        filepaths: Array of image file paths
        labels: Array of integer class labels
        bbox_array: Array of bounding box coordinates

    Returns:
        tf.data.Dataset with (image, labels_dict, path) tuples
    """
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels, bbox_array))
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.filter(lambda image, labels, path: labels["class_output"] != -1)
    ds = ds.apply(tf.data.Dataset.ignore_errors)

    return ds


def build_unlabeled_dataset(
    unlabeled_path_base: str = UNLABELED_PATH_BASE,
    filler_label: int = 13,
) -> tf.data.Dataset:
    """
    Build a dataset from unlabeled images.

    Args:
        unlabeled_path_base: Base path containing unlabeled images
        filler_label: Label to assign to unlabeled images (default: 13)

    Returns:
        tf.data.Dataset with unlabeled images
    """
    unlabeled_files = os.listdir(unlabeled_path_base)
    unlabeled_filepaths = np.array(
        [unlabeled_path_base + "/" + file for file in unlabeled_files]
    )

    f = unlabeled_filepaths.shape[0]
    labels = np.full(f, filler_label, dtype="int32")
    bbox_array = np.zeros((f, 4), dtype="float32")

    ds_u = tf.data.Dataset.from_tensor_slices((unlabeled_filepaths, labels, bbox_array))
    ds_u = ds_u.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

    return ds_u


def split_dataset(
    ds: tf.data.Dataset,
    total_size: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    batch_size: int = BATCH_SIZE,
    shuffle_buffer: int = 500,
    seed: int = 1337,
) -> dict:
    """
    Split a dataset into train, validation, and test sets.

    Args:
        ds: Input TensorFlow dataset
        total_size: Total number of samples in the dataset
        train_frac: Fraction for training set (default: 0.8)
        val_frac: Fraction for validation set (default: 0.1)
        batch_size: Batch size for the datasets (default: BATCH_SIZE)
        shuffle_buffer: Buffer size for shuffling (default: 500)
        seed: Random seed for shuffling (default: 1337)

    Returns:
        Dictionary with keys: 'train', 'val', 'test', 'train_raw', 'val_raw', 'test_raw'
    """
    train_size = int(train_frac * total_size)
    val_size = int(val_frac * total_size)

    raw_ds = ds.shuffle(shuffle_buffer, seed=seed)

    train_raw = raw_ds.take(train_size)
    val_raw = raw_ds.skip(train_size).take(val_size)
    test_raw = raw_ds.skip(train_size + val_size)

    # Create batched and prefetched datasets (dropping the path)
    train_ds = (
        train_raw.map(lambda x, y, p: (x, y), num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_raw.map(lambda x, y, p: (x, y), num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        test_raw.map(lambda x, y, p: (x, y), num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "train_raw": train_raw,
        "val_raw": val_raw,
        "test_raw": test_raw,
    }


# =============================================================================
# Visualization Helpers
# =============================================================================


def denormalize_bbox(bbox, img_width, img_height):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        bbox: Normalized bounding box [x1, y1, x2, y2] (values 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x1, y1, x2, y2) in pixel coordinates
    """
    x1 = int(bbox[0] * img_width)
    y1 = int(bbox[1] * img_height)
    x2 = int(bbox[2] * img_width)
    y2 = int(bbox[3] * img_height)

    return x1, y1, x2, y2


def plot_prediction(
    model,
    label_encoder: LabelEncoder,
    filepath: str,
    image: np.ndarray,
    true_label: int = None,
    true_bbox: np.ndarray = None,
):
    """
    Plot an image with predicted and ground truth bounding boxes.

    Args:
        model: Trained Keras model for predictions
        label_encoder: LabelEncoder used for class names
        filepath: Path to the original image file
        image: Preprocessed image array
        true_label: Ground truth class label (optional)
        true_bbox: Ground truth bounding box (optional)
    """
    # Load original image for display
    img_bytes = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img_bytes, channels=3)

    img_batch = tf.expand_dims(image, axis=0)  # shape (1, H, W, 3)

    # Run model
    pred_class_probs, pred_bbox = model.predict(img_batch, verbose=0)
    pred_class_id = pred_class_probs.argmax(axis=1)[0]
    pred_class_name = label_encoder.inverse_transform([pred_class_id])[0]

    # Convert normalized predicted bbox → pixels
    pred_bbox = pred_bbox[0]
    x1, y1, x2, y2 = denormalize_bbox(pred_bbox, IMAGE_SIZE[0], IMAGE_SIZE[1])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img.numpy())

    # Predicted bbox in red
    rect = patches.Rectangle(
        (x1, y1),
        (x2 - x1),
        (y2 - y1),
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        label="Predicted",
    )
    ax.add_patch(rect)

    # Ground truth bbox (optional, if provided)
    if true_bbox is not None:
        tx1, ty1, tx2, ty2 = denormalize_bbox(true_bbox, IMAGE_SIZE[0], IMAGE_SIZE[1])
        rect_gt = patches.Rectangle(
            (tx1, ty1),
            (tx2 - tx1),
            (ty2 - ty1),
            linewidth=2,
            edgecolor="green",
            facecolor="none",
            label="Ground Truth",
        )
        ax.add_patch(rect_gt)

    # Build title
    if true_label is not None:
        true_label_name = label_encoder.inverse_transform([true_label])[0]
        ax.set_title(f"Prediction: {pred_class_name}, Actual: {true_label_name}")
    else:
        ax.set_title(f"Predicted: {pred_class_name}")

    ax.axis("off")
    plt.legend()
    plt.show()


def plot_learning_curves(history):
    """
    Plot training and validation learning curves.

    Args:
        history: Keras History object or history dictionary
    """
    if hasattr(history, "history"):
        history = history.history

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(history["bbox_output_mse"], label="train")
    ax[0].plot(history["val_bbox_output_mse"], label="val")
    ax[0].set_title("Bounding Box MSE")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("MSE")
    ax[0].legend()

    ax[1].plot(history["class_output_accuracy"], label="train")
    ax[1].plot(history["val_class_output_accuracy"], label="val")
    ax[1].set_title("Class Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_ds, figsize=(10, 10)):
    """
    Plot confusion matrix for model predictions on test dataset.

    Args:
        model: Trained Keras model
        test_ds: Test tf.data.Dataset
        figsize: Figure size (default: (10, 10))
    """
    from sklearn.metrics import confusion_matrix

    y_true = []
    y_pred = []

    for batch_images, batch_labels in test_ds:
        # Run model → returns (class_pred, bbox_pred)
        class_logits, _ = model.predict(batch_images, verbose=0)

        # True labels
        y_true.extend(batch_labels["class_output"].numpy())

        # Convert softmax → predicted class index
        batch_pred = np.argmax(class_logits, axis=1)
        y_pred.extend(batch_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap="crest")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    return y_true, y_pred
