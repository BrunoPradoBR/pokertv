import numpy as np
import yaml
from pokertv.segmenter import Segmenter

MINIMAL_LAYOUT = yaml.safe_load("""
hole_cards:
  - name: hole_card1
    x: 0.4
    y: 0.7
    w: 0.1
    h: 0.1
  - name: hole_card2
    x: 0.5
    y: 0.7
    w: 0.1
    h: 0.1
community_cards:
  - name: community1
    x: 0.3
    y: 0.4
    w: 0.1
    h: 0.1
pot:
  x: 0.45
  y: 0.38
  w: 0.1
  h: 0.05
seats:
  - index: 0
    name:  {x: 0.8, y: 0.6, w: 0.1, h: 0.03}
    stack: {x: 0.8, y: 0.63, w: 0.1, h: 0.03}
action_buttons:
  fold:  {x: 0.35, y: 0.88, w: 0.09, h: 0.05}
  call:  {x: 0.45, y: 0.88, w: 0.09, h: 0.05}
  raise: {x: 0.55, y: 0.88, w: 0.09, h: 0.05}
""")


def test_segmenter_produces_expected_keys():
    seg = Segmenter(layout=MINIMAL_LAYOUT)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    region_map = seg.segment(frame)
    assert "hole_card1" in region_map
    assert "hole_card2" in region_map
    assert "community1" in region_map
    assert "pot" in region_map
    assert "seat_0_name" in region_map
    assert "seat_0_stack" in region_map
    assert "action_fold" in region_map
    assert "action_call" in region_map
    assert "action_raise" in region_map


def test_segmenter_crop_dimensions_scale_with_frame():
    seg = Segmenter(layout=MINIMAL_LAYOUT)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop = seg.segment(frame)["hole_card1"]
    # x=0.4, w=0.1 -> 0.1*1280=128 px wide; y=0.7, h=0.1 -> 0.1*720=72 px tall
    assert crop.shape[1] == 128
    assert crop.shape[0] == 72


def test_segmenter_all_crops_are_numpy_arrays():
    seg = Segmenter(layout=MINIMAL_LAYOUT)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for key, crop in seg.segment(frame).items():
        assert isinstance(crop, np.ndarray), f"{key} is not ndarray"
