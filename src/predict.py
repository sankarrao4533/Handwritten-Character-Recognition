from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from src.data_loader import load_mapping
from src.preprocessing import normalize_character

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "emnist_balanced_cnn.keras"
MAPPING_PATH = ROOT / "data" / "emnist-balanced-mapping.txt"


_model = None
_labels = None


def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
    return _model


def get_labels() -> dict[int, str]:
    global _labels
    if _labels is None:
        _labels = load_mapping(MAPPING_PATH)
    return _labels


def predict_image(image: Image.Image) -> tuple[str, float, np.ndarray]:
    tensor = normalize_character(image)
    probabilities = get_model().predict(tensor[np.newaxis, ...], verbose=0)[0]
    class_id = int(np.argmax(probabilities))
    return get_labels()[class_id], float(probabilities[class_id]), tensor[..., 0]
