"""Generate synthetic training assets if they don't exist."""
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

SPRITE_DIR = Path("assets/sprites")
BACKGROUND_DIR = Path("assets/backgrounds")
CARD_SIZE = (60, 80)


def generate_card_sprite(card: str) -> Image.Image:
    """Generate a simple card sprite for a given card code (e.g., 'Ac', '2h')."""
    img = Image.new("RGB", CARD_SIZE, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    if card == "back":
        draw.rectangle([0, 0, CARD_SIZE[0] - 1, CARD_SIZE[1] - 1], outline=(0, 0, 0), width=2)
        draw.text((15, 32), "POKER", fill=(100, 100, 100))
    else:
        draw.rectangle([0, 0, CARD_SIZE[0] - 1, CARD_SIZE[1] - 1], outline=(0, 0, 0), width=1)
        draw.text((5, 5), card, fill=(0, 0, 0))
        draw.text((CARD_SIZE[0] - 15, CARD_SIZE[1] - 15), card, fill=(0, 0, 0))

    return img


def generate_background() -> Image.Image:
    """Generate a synthetic poker table background (green felt with variation)."""
    width, height = 512, 384
    img_array = np.random.randint(30, 60, (height, width, 3), dtype=np.uint8)
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] - 10, 0, 255)
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] + 20, 0, 255)
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] - 15, 0, 255)

    img = Image.fromarray(img_array.astype("uint8"), "RGB")
    return img


def setup_assets() -> None:
    """Generate synthetic assets if they don't exist."""
    SPRITE_DIR.mkdir(parents=True, exist_ok=True)
    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

    # Generate card sprites
    suits = ["c", "d", "h", "s"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

    for rank in ranks:
        for suit in suits:
            card = f"{rank}{suit}"
            sprite_path = SPRITE_DIR / f"{card}.png"
            if not sprite_path.exists():
                sprite = generate_card_sprite(card)
                sprite.save(sprite_path)

    # Card back
    back_path = SPRITE_DIR / "back.png"
    if not back_path.exists():
        back = generate_card_sprite("back")
        back.save(back_path)

    # Generate poker table backgrounds
    for i in range(5):
        bg_path = BACKGROUND_DIR / f"background_{i:02d}.png"
        if not bg_path.exists():
            bg = generate_background()
            bg.save(bg_path)


if __name__ == "__main__":
    setup_assets()
