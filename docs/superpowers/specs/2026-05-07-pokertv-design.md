# PokerTV Design Spec
**Date:** 2026-05-07  
**Status:** Approved  
**Scope:** v1 — PokerStars, Texas Hold'em, single-machine screen capture

---

## 1. Purpose

`pokertv` is a Python application that captures live poker client screen output and produces structured hand history files. It watches PokerStars windows, extracts game state frame-by-frame using computer vision and OCR, and saves completed hands as newline-delimited JSON (`.jsonl`) and optionally in PokerStars native `.txt` format for import into tracking software.

---

## 2. Architecture

The application is a layered producer-consumer pipeline. A dedicated capture thread feeds frames into a bounded queue. A worker pool (ThreadPoolExecutor) runs the expensive vision and OCR steps concurrently. A single-threaded state machine consumes results and accumulates `Hand` objects. A writer flushes completed hands to disk.

```
Capture Thread
  └─ mss grab @ 1–4 fps
  └─ → Frame queue (bounded, max 8 frames)

Worker Pool (ThreadPoolExecutor, GPU-pinned)
  ├─ Layer 1: Table detector  → TableDetection
  ├─ Layer 2: Segmenter       → RegionMap
  ├─ Layer 3: Card recognizer → List[CardPrediction]
  └─ Layer 4: OCR engine      → TextData
  └─ → FrameData struct

State Machine (single thread)
  └─ FSM over FrameData stream → Hand objects

Writer
  └─ Appends to session.jsonl
  └─ Exports to PokerStars .txt on demand
```

Python's `ThreadPoolExecutor` is used (not `ProcessPoolExecutor`) because PyTorch releases the GIL during GPU inference, avoiding pickling overhead and enabling shared memory. The frame queue is bounded so stale frames are dropped under load rather than accumulating in RAM.

---

## 3. Layer 1 — Screen Capture & Table Detection

**Capture:** `mss` grabs all monitor regions at a configurable interval (default: 2 fps, range: 1–4 fps). Each grab produces a numpy array per window.

**Table detection** is deterministic (no ML):

1. **Color histogram fingerprint** — HSV histogram of the table center region compared against stored reference histograms for PokerStars felt variants (green, blue). Cosine similarity > 0.80 passes.
2. **UI chrome fingerprint** — A 60×60 px crop at the known PokerStars logo position is matched via `cv2.matchTemplate` normalized cross-correlation. Score > 0.85 confirms client identity.
3. **Edge density check** — Canny edge density ratio in the felt region must fall within a stored range `[min, max]`.

**Output:** `TableDetection(window_id, client="pokerstars", confidence: float, bbox: Rect)`. Frames with `confidence < 0.7` are dropped before reaching the worker pool.

**Layout config:** `pokertv/layouts/pokerstars.yaml` holds all bounding box definitions in relative coordinates (0.0–1.0), so they scale with window resize.

---

## 4. Layer 2 — Region Segmentation

Given a confirmed `TableDetection`, the segmenter reads `pokerstars.yaml` and extracts named numpy crops from the frame.

Regions defined per layout file:
- `hole_cards` — 2 crops (hero's cards)
- `community_cards` — up to 5 crops (flop1-3, turn, river)
- `pot` — pot size text region
- `seats[0..8]` — each seat has `stack` and `name` sub-regions
- `action_buttons` — fold, call, raise text regions
- `dealer_button` — matched by template, not OCR

**Output:** `RegionMap` — a dict mapping region name to numpy array crop.

---

## 5. Layer 3 — Card Recognition

**Vocabulary:** 53 classes — 52 playing cards (`Ac`, `Kh`, `2d`, etc.) + `back`.

### Training Pipeline (`pokertv/training/`)

| File | Purpose |
|------|---------|
| `synthesize.py` | Composites card sprites onto table background crops with augmentation: rotation ±5°, brightness jitter ±20%, JPEG noise, slight blur. Generates ~5,000 samples per class. |
| `train.py` | ResNet18 fine-tune via PyTorch. Freezes early layers; trains final two residual blocks + classifier head. Exports to TorchScript. Target: >99% top-1 accuracy. |
| `evaluate.py` | Produces confusion matrix and per-class accuracy report. |

### Inference

- TorchScript model loaded once at startup, pinned to CUDA (`device="cuda:0"`).
- All card crops from a single frame are batched into one forward pass.
- **Output:** `List[CardPrediction(region_name, card: str, confidence: float)]`
- Cards with `confidence < 0.85` emit `card="unknown"` — the state machine handles this gracefully by not advancing street until resolved.

---

## 6. Layer 4 — OCR

PaddleOCR extracts text from pot, stack, name, and action button crops.

- Initialized once at startup: `use_angle_cls=False` (UI text is always upright), `use_gpu=True`.
- All text crops for a frame are submitted as a single batch call.
- Raw strings are cleaned by `parse_amount(s: str) -> float` (strips `$`, `,`, `BB`).

**Output per frame:**
- `pot: float`
- `stacks: Dict[seat_index, float]`
- `names: Dict[seat_index, str]`
- `action_labels: Dict[button_name, str]`
- `dealer_seat: int` (from template match, not OCR)

All Layer 1–4 outputs are aggregated into a `FrameData` dataclass before the state machine consumes them.

---

## 7. Layer 5 — Hand State Machine

The FSM converts a noisy stream of `FrameData` into clean `Hand` objects.

### States

```
IDLE → PREFLOP → FLOP → TURN → RIVER → SHOWDOWN → IDLE
```

### Transitions

| From | To | Trigger |
|------|----|---------|
| IDLE | PREFLOP | Hole cards appear + pot > 0 |
| PREFLOP | FLOP | 3 community cards visible |
| FLOP | TURN | 4 community cards visible |
| TURN | RIVER | 5 community cards visible |
| RIVER | SHOWDOWN | Action buttons disappear OR pot awarded |
| SHOWDOWN | IDLE | Pot resets to 0 + hole cards disappear |

### Key Behaviours

- **Debouncing:** State transitions require the same observation in 2 consecutive frames before committing. Filters OCR glitches and partial renders.
- **Action detection:** Stack size deltas between frames imply bets/calls/folds. Records `(player, action_type, amount)` per street.
- **Hand boundary detection:** New hand detected when pot resets to 0, hole cards disappear, and dealer button moves.
- **Incomplete hands:** If the app closes mid-hand, the partial hand is saved with `status: "incomplete"` and excluded from PokerStars `.txt` export.

### `Hand` Dataclass Fields

```
hand_id: str          # UUID
timestamp: datetime
table_name: str
game_type: str        # "NLH"
stakes: str           # e.g. "0.05/0.10"
seats: List[Seat]     # name, stack, position
hole_cards: List[str] # hero's cards
board: List[str]      # community cards
streets: List[Street] # each with actions + pot
result: Result        # winner, amount, hand shown
status: str           # "complete" | "incomplete"
```

---

## 8. Output Layer

### JSONL (primary)

- One file per session: `hands_2026-05-07_session1.jsonl`
- Appended line-by-line as hands complete — no batch writes.
- Each line is a complete `Hand` serialized to JSON.
- Stored in `~/pokertv/hands/` (configurable via `POKERTV_OUTPUT_DIR`).

### PokerStars Export

- `HandHistoryExporter.export(hands: List[Hand]) -> str` converts to standard PokerStars `.txt` format.
- Triggered by `pokertv export <file.jsonl>` CLI command or automatically at session end.
- Incomplete hands are skipped with a warning.

---

## 9. Project Structure

```
pokertv/
├── pokertv/
│   ├── capture.py          # Layer 1: mss grab + table detection
│   ├── segmenter.py        # Layer 2: region extraction from layout YAML
│   ├── recognizer.py       # Layer 3: card CNN inference
│   ├── ocr.py              # Layer 4: PaddleOCR wrapper + amount parser
│   ├── state_machine.py    # Layer 5: FSM + Hand/Street/Action dataclasses
│   ├── writer.py           # JSONL writer + PokerStars exporter
│   ├── pipeline.py         # Thread wiring, queue, worker pool
│   ├── models.py           # Shared dataclasses (FrameData, TableDetection, etc.)
│   ├── layouts/
│   │   └── pokerstars.yaml
│   └── training/
│       ├── synthesize.py
│       ├── train.py
│       └── evaluate.py
├── templates/              # PokerStars UI fingerprint crops (logo, dealer chip)
├── tests/
│   ├── test_capture.py
│   ├── test_segmenter.py
│   ├── test_recognizer.py
│   ├── test_ocr.py
│   ├── test_state_machine.py
│   └── test_writer.py
├── pyproject.toml
└── README.md
```

---

## 10. Dependencies

| Package | Purpose |
|---------|---------|
| `mss` | Fast cross-platform screen capture |
| `opencv-python` | Template matching, Canny, color conversion |
| `torch` + `torchvision` | ResNet18 training + TorchScript inference |
| `paddleocr` | Text extraction from UI crops |
| `pyyaml` | Layout config parsing |
| `dataclasses-json` | Hand serialization to/from JSON |
| `click` | CLI entry points |

---

## 11. CLI Entry Points

| Command | Action |
|---------|--------|
| `pokertv run` | Start live capture pipeline |
| `pokertv train` | Run training pipeline (synthesize + train + evaluate) |
| `pokertv export <file.jsonl>` | Convert session file to PokerStars `.txt` |

---

## 12. Out of Scope (v1)

- GGPoker, ACR, Pokerrrr2 support
- Omaha, Stud, Draw game types
- Real-time HUD overlay
- Cloud sync or remote API
- Villain hole card detection (only hero cards are reliably visible)
