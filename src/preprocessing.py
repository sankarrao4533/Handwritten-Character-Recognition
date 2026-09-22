import numpy as np
from PIL import Image, ImageOps


def normalize_character(image: Image.Image) -> np.ndarray:
    """Convert a dark-on-light character into an upright EMNIST tensor."""
    grayscale = ImageOps.grayscale(image)
    pixels = np.asarray(grayscale, dtype="float32")
    border = np.concatenate(
        [pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]]
    )

    # Keep EMNIST-style white-on-black images; invert dark-on-light uploads.
    foreground = grayscale if border.mean() < pixels.mean() else ImageOps.invert(grayscale)
    bbox = foreground.getbbox()
    canvas = Image.new("L", (28, 28), 0)

    if bbox is not None:
        cropped = foreground.crop(bbox)
        scale = min(20 / cropped.width, 20 / cropped.height)
        resized = cropped.resize(
            (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))

    return np.asarray(canvas, dtype="float32")[..., np.newaxis] / 255.0
