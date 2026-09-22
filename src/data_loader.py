from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from src.preprocessing import normalize_character

TRAIN_ROWS = 30_000
TEST_ROWS = 10_000
NUM_CLASSES = 47


def to_upright(images: np.ndarray) -> np.ndarray:
    """Convert EMNIST's stored orientation to the orientation users write."""
    images = images.reshape(-1, 28, 28)
    return np.flip(np.transpose(images, (0, 2, 1)), axis=1)[..., np.newaxis]


def normalize_dataset(images: np.ndarray) -> np.ndarray:
    """Apply the same centering and scaling used for app inputs."""
    return np.stack(
        [normalize_character(Image.fromarray(image[..., 0])) for image in images],
        axis=0,
    )


def load_emnist(
    data_path: str | Path,
    train_rows: int = TRAIN_ROWS,
    test_rows: int = TEST_ROWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = loadmat(data_path)["dataset"][0, 0]
    train = dataset["train"][0, 0]
    test = dataset["test"][0, 0]
    train_images = normalize_dataset(to_upright(train["images"][:train_rows]))
    train_labels = train["labels"][:train_rows].reshape(-1).astype("int64")
    test_images = normalize_dataset(to_upright(test["images"][:test_rows]))
    test_labels = test["labels"][:test_rows].reshape(-1).astype("int64")
    return train_images, train_labels, test_images, test_labels


def load_mapping(mapping_path: str | Path) -> dict[int, str]:
    mapping = np.loadtxt(mapping_path, dtype=int)
    return {int(class_id): chr(int(codepoint)) for class_id, codepoint in mapping}
