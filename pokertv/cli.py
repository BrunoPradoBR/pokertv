from __future__ import annotations
import os
import click
import yaml


@click.group()
def main():
    """PokerTV -- live poker hand history capture for PokerStars."""


@main.command()
@click.option("--fps", default=2, show_default=True, type=click.IntRange(1, 4), help="Capture rate")
@click.option("--layout", default="pokertv/layouts/pokerstars.yaml", show_default=True)
@click.option("--model", default="models/card_recognizer.pt", show_default=True)
@click.option("--output", default=os.path.expanduser("~/pokertv/hands"), show_default=True)
def run(fps, layout, model, output):
    """Start live capture pipeline."""
    from pokertv.capture import ScreenCapture, TableDetector
    from pokertv.segmenter import Segmenter
    from pokertv.recognizer import CardRecognizer
    from pokertv.ocr import OCREngine
    from pokertv.writer import HandWriter
    from pokertv.pipeline import Pipeline

    with open(layout) as f:
        layout_config = yaml.safe_load(f)

    os.makedirs(output, exist_ok=True)
    jsonl_path = os.path.join(output, "hands.jsonl")

    Pipeline(
        capture=ScreenCapture(fps=fps),
        detector=TableDetector.from_config(layout_config),
        segmenter=Segmenter(layout=layout_config),
        recognizer=CardRecognizer.from_path(model),
        ocr=OCREngine(),
        writer=HandWriter(path=jsonl_path),
    ).start()


@main.command()
def train():
    """Synthesize training data, fine-tune ResNet18, evaluate accuracy."""
    from pokertv.training.synthesize import synthesize
    from pokertv.training.train import train as run_train
    from pokertv.training.evaluate import evaluate

    click.echo("Step 1/3: Synthesizing training data...")
    synthesize()
    click.echo("Step 2/3: Training card recognizer...")
    run_train()
    click.echo("Step 3/3: Evaluating...")
    evaluate()


@main.command("export")
@click.argument("jsonl_file")
@click.option("--out", default=None, help="Output .txt path (default: replaces .jsonl with .txt)")
def export_cmd(jsonl_file, out):
    """Convert a session JSONL file to PokerStars .txt format."""
    from pokertv.models import Hand
    from pokertv.writer import PokerStarsExporter

    with open(jsonl_file, encoding="utf-8") as f:
        hands = [Hand.from_json(line.strip()) for line in f if line.strip()]

    out_path = out or jsonl_file.replace(".jsonl", ".txt")
    with PokerStarsExporter(out_path) as exporter:
        for hand in hands:
            exporter.export(hand)

    click.echo(f"Exported {len(hands)} hands to {out_path}")
