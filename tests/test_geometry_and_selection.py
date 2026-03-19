from detectors import _expand_bbox
from video_processor import _overlay_style, _player_style_bbox, _select_primary_face


def test_select_primary_face_returns_first_detected():
    faces = [
        (0, 0, 20, 20, 0.9),
        (10, 10, 80, 70, 0.8),
        (5, 5, 40, 40, 0.95),
    ]
    selected = _select_primary_face(faces)
    assert selected == (0, 0, 20, 20, 0.9)


def test_player_style_bbox_is_square_and_clamped():
    bbox = (-10, -5, 200, 120)
    frame_shape = (100, 150, 3)
    x1, y1, x2, y2 = _player_style_bbox(bbox, frame_shape, scale=1.3)
    assert x1 == 0
    assert y1 == 0
    assert x2 <= 150
    assert y2 <= 100
    assert (x2 - x1) == (y2 - y1)


def test_expand_bbox_grows_around_center_and_clamps():
    x1, y1, x2, y2 = _expand_bbox(
        x1=40,
        y1=30,
        x2=60,
        y2=50,
        frame_shape=(100, 100, 3),
        scale=2.0,
    )
    assert (x1, y1, x2, y2) == (30, 20, 70, 60)


def test_overlay_style_scales_up_with_resolution():
    small = _overlay_style((360, 640, 3))
    large = _overlay_style((1080, 1920, 3))

    assert large[0] >= small[0]  # box thickness
    assert large[1] >= small[1]  # font scale
    assert large[2] >= small[2]  # text thickness
    assert large[3] >= small[3]  # label offset
