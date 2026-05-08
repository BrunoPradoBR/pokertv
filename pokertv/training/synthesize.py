"""
Synthesize training data by compositing card sprites onto background crops.
Produces ~5000 augmented images per class in assets/training_data/<class>/.

Prerequisites:
  - assets/sprites/<card>.png  (e.g. Ac.png, Kh.png, back.png) -- 53 files
  - assets/backgrounds/*.png or *.jpg  -- poker table crop images

Usage:
  python -m pokertv.training.synthesize
"""
from __future__ import annotations
import random
from pathlib import Path
import cv2
import numpy as np

SPRITE_DIR = Path("assets/sprites")
BACKGROUND_DIR = Path("assets/backgrounds")
OUTPUT_DIR = Path("assets/training_data")
SAMPLES_PER_CLASS = 5000
CARD_SIZE = (60, 80)


def augment(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    brightness = random.uniform(0.8, 1.2)
    brightened = np.clip(rotated.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    quality = random.randint(60, 95)
    _, enc = cv2.imencode(".jpg", brightened, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def composite(sprite: np.ndarray, background: np.ndarray) -> np.ndarray:
    bg = background.copy()
    h_bg, w_bg = bg.shape[:2]
    card_w, card_h = CARD_SIZE
    card = cv2.resize(sprite, (card_w, card_h))
    x = random.randint(0, max(0, w_bg - card_w))
    y = random.randint(0, max(0, h_bg - card_h))
    bg[y : y + card_h, x : x + card_w] = card
    return bg[y : y + card_h, x : x + card_w]


def synthesize() -> None:
    backgrounds = list(BACKGROUND_DIR.glob("*.png")) + list(BACKGROUND_DIR.glob("*.jpg"))
    if not backgrounds:
        raise FileNotFoundError(f"No background images in {BACKGROUND_DIR}")
    bg_images = [cv2.imread(str(p)) for p in backgrounds]
    bg_images = [b for b in bg_images if b is not None]

    for sprite_path in sorted(SPRITE_DIR.glob("*.png")):
        class_name = sprite_path.stem
        sprite = cv2.imread(str(sprite_path))
        if sprite is None:
            print(f"WARNING: could not read {sprite_path}, skipping")
            continue
        out_dir = OUTPUT_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(SAMPLES_PER_CLASS):
            bg = random.choice(bg_images)
            img = augment(composite(sprite, bg))
            cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), img)
        print(f"  {class_name}: {SAMPLES_PER_CLASS} samples written")


if __name__ == "__main__":
    synthesize()
