from __future__ import annotations
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from pokertv.capture import ScreenCapture, TableDetector
from pokertv.segmenter import Segmenter
from pokertv.recognizer import CardRecognizer
from pokertv.ocr import OCREngine
from pokertv.state_machine import HandStateMachine
from pokertv.writer import HandWriter
from pokertv.models import FrameData

FRAME_QUEUE_MAX = 8
RESULT_QUEUE_MAX = 32
WORKER_THREADS = 4


class Pipeline:
    def __init__(
        self,
        capture: ScreenCapture,
        detector: TableDetector,
        segmenter: Segmenter,
        recognizer: CardRecognizer,
        ocr: OCREngine,
        writer: HandWriter,
    ):
        self._capture = capture
        self._detector = detector
        self._segmenter = segmenter
        self._recognizer = recognizer
        self._ocr = ocr
        self._state_machine = HandStateMachine()
        self._writer = writer
        self._frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        self._result_queue: queue.Queue = queue.Queue(maxsize=RESULT_QUEUE_MAX)
        self._running = False

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._capture_loop, daemon=True, name="capture").start()
        threading.Thread(target=self._state_loop, daemon=True, name="state").start()
        print("PokerTV pipeline running. Press Ctrl+C to stop.")
        try:
            with ThreadPoolExecutor(max_workers=WORKER_THREADS) as pool:
                while self._running:
                    try:
                        item = self._frame_queue.get(timeout=1.0)
                        pool.submit(self._process_frame, item)
                    except queue.Empty:
                        continue
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self._running = False
        incomplete = self._state_machine.flush_incomplete()
        if incomplete:
            self._writer.write(incomplete)
        self._writer.flush()
        print("Stopped. Hand histories saved.")

    def _capture_loop(self) -> None:
        while self._running:
            for window_id, frame in self._capture.capture_all():
                try:
                    self._frame_queue.put_nowait((window_id, frame))
                except queue.Full:
                    pass
            time.sleep(self._capture._interval)

    def _process_frame(self, item: tuple) -> None:
        window_id, frame = item
        detection = self._detector.detect(frame, window_id)
        if detection is None:
            return
        region_map = self._segmenter.segment(frame)
        cards = self._recognizer.recognize(region_map)
        text = self._ocr.extract(region_map)
        frame_data = FrameData(
            timestamp=datetime.now(),
            window_id=window_id,
            detection=detection,
            cards=cards,
            text=text,
        )
        try:
            self._result_queue.put_nowait(frame_data)
        except queue.Full:
            pass

    def _state_loop(self) -> None:
        while self._running:
            try:
                frame_data = self._result_queue.get(timeout=1.0)
                completed = self._state_machine.update(frame_data)
                if completed:
                    self._writer.write(completed)
            except queue.Empty:
                continue
