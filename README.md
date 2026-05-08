# PokerTV

Captures live PokerStars Texas Hold'em tables and saves completed hand histories as JSONL and PokerStars `.txt` format files.

## How it works

A producer-consumer pipeline runs five layers in parallel:

1. **Screen capture** — grabs all monitors at up to 4 fps via `mss`
2. **Table detection** — identifies PokerStars windows using logo template matching, HSV histogram similarity, and edge density
3. **Region segmentation** — crops named regions (hole cards, community cards, pot, seats, action buttons) from each frame using a YAML layout config
4. **Card recognition** — classifies each card crop with a fine-tuned ResNet18 (53 classes: 52 cards + back)
5. **OCR + state machine** — reads pot/stack/player text with PaddleOCR, feeds a debounced FSM that emits a `Hand` object when a hand completes

Completed hands are written to a JSONL file (one JSON object per line) and optionally exported to PokerStars `.txt` format.

## Requirements

- Python 3.11+
- A trained card recognizer model at `models/card_recognizer.pt` (see [Training](#training))

## Installation

```bash
pip install -e .
```

## Usage

### Capture live hands

```bash
pokertv run
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--fps` | `2` | Capture rate (1–4) |
| `--layout` | `pokertv/layouts/pokerstars.yaml` | Region layout config |
| `--model` | `models/card_recognizer.pt` | Card recognizer model |
| `--output` | `~/pokertv/hands` | Output directory |

Hands are saved to `<output>/hands.jsonl`. Press `Ctrl+C` to stop — any in-progress hand is saved with `status: "incomplete"`.

### Export to PokerStars format

```bash
pokertv export ~/pokertv/hands/hands.jsonl
```

Writes a `.txt` file next to the JSONL file. Incomplete hands are skipped. Use `--out` to specify a different output path.

### Training

```bash
pokertv train
```

Runs three steps:

1. **Synthesize** — composites card sprites from `assets/sprites/` onto backgrounds from `assets/backgrounds/` (5,000 augmented samples per class)
2. **Train** — fine-tunes ResNet18 (frozen early layers, trainable `layer3`/`layer4`/`fc`) for 10 epochs, exports TorchScript to `models/card_recognizer.pt`
3. **Evaluate** — prints overall accuracy and a per-class confusion matrix

## Project structure

```
pokertv/
├── pokertv/
│   ├── capture.py          # Screen grab + table detection
│   ├── segmenter.py        # Region cropping from layout config
│   ├── recognizer.py       # Card CNN inference
│   ├── ocr.py              # PaddleOCR wrapper + amount parser
│   ├── state_machine.py    # Hand FSM with debouncing
│   ├── writer.py           # JSONL writer + PokerStars exporter
│   ├── pipeline.py         # Thread wiring (capture → workers → FSM → writer)
│   ├── cli.py              # Click entry points
│   ├── models.py           # Shared dataclasses
│   ├── layouts/
│   │   └── pokerstars.yaml # Region coordinates (relative, 0.0–1.0)
│   └── training/
│       ├── synthesize.py   # Training data synthesis
│       ├── train.py        # ResNet18 fine-tuning
│       └── evaluate.py     # Accuracy + confusion matrix
├── tests/                  # 41 pytest tests
├── assets/
│   ├── sprites/            # Card PNG sprites (one per class)
│   └── backgrounds/        # Background crop images
├── models/                 # Trained model output
└── templates/              # Logo + dealer button templates for detection
```

## Layout calibration

The default layout (`pokertv/layouts/pokerstars.yaml`) targets a standard PokerStars 9-max table. All coordinates are relative (0.0–1.0) to the window dimensions, so they scale with window size.

If your table theme or window size differs, adjust the coordinates in the YAML. To recalibrate the histogram-based table detector:

```bash
python -c "
from pokertv.capture import save_reference_histogram
save_reference_histogram('screenshot.png', 'templates/hist_green_felt.npy')
"
```

## Output format

### JSONL

Each line is a JSON-serialized `Hand` object:

```json
{
  "hand_id": "abc123",
  "timestamp": "2024-01-15T14:32:01",
  "table_name": "PokerStars Table",
  "game_type": "NLH",
  "stakes": "unknown",
  "seats": [{"index": 0, "name": "Hero", "stack": 10.0, "position": ""}],
  "hole_cards": ["Ah", "Kd"],
  "board": ["2c", "3d", "4h", "9s", "As"],
  "streets": [...],
  "result": {"winner": "Hero", "amount": 1.20, "shown_cards": {}},
  "status": "complete"
}
```

### PokerStars `.txt`

Standard PokerStars hand history format, compatible with hand history analysis tools.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
