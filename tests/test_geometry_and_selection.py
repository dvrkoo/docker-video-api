from detectors import _expand_bbox
from video_processor import _clamp_face_bbox, _select_largest_face


def test_select_largest_face_uses_area():
    faces = [
        (0, 0, 20, 20, 0.9),
        (10, 10, 80, 70, 0.8),
        (5, 5, 40, 40, 0.95),
    ]
    largest = _select_largest_face(faces)
    assert largest == (10, 10, 80, 70, 0.8)


def test_clamp_face_bbox_inside_frame():
    bbox = (-10, -5, 200, 120)
    frame_shape = (100, 150, 3)
    assert _clamp_face_bbox(bbox, frame_shape) == (0, 0, 150, 100)


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
