from small_object_lab.metrics import (
    Detection,
    StabilityTracker,
    intersection_over_union,
    smallest,
)


def detection(x: int, y: int, width: int, height: int, label: str = "cup"):
    return Detection(
        label=label,
        confidence=0.8,
        x=x,
        y=y,
        width=width,
        height=height,
    )


def test_detection_reports_pixel_and_frame_ratios():
    item = detection(10, 20, 20, 10)

    result = item.as_dict(frame_width=200, frame_height=100)

    assert result["area_px"] == 200
    assert result["width_pct"] == 10.0
    assert result["height_pct"] == 10.0
    assert result["area_pct"] == 1.0


def test_iou_and_streak_require_same_label_and_overlap():
    tracker = StabilityTracker(iou_threshold=0.3)

    first = tracker.update([detection(10, 10, 20, 20)])[0]
    second = tracker.update([detection(12, 11, 20, 20)])[0]
    changed_label = tracker.update(
        [detection(12, 11, 20, 20, label="bottle")]
    )[0]

    assert intersection_over_union(first, second) > 0.3
    assert first.streak == 1
    assert second.streak == 2
    assert changed_label.streak == 1


def test_smallest_uses_bbox_area():
    large = detection(0, 0, 30, 20)
    narrow = detection(0, 0, 10, 25)

    assert smallest([large, narrow]) == narrow
