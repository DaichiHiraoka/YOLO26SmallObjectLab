import numpy as np

from small_object_lab.detector import SmallObjectLab
from small_object_lab.metrics import Detection
from small_object_lab.super_resolution import (
    FSRCNNSuperResolution,
    LargestObjectInstabilityMonitor,
    expanded_crop_region,
    remap_from_super_resolution,
    select_super_resolution_target,
)


def detection(
    *,
    confidence=0.8,
    x=20,
    y=20,
    width=40,
    height=50,
    label="cup",
):
    return Detection(
        label=label,
        confidence=confidence,
        x=x,
        y=y,
        width=width,
        height=height,
    )


def test_target_policy_selects_largest_bbox():
    small = detection(width=20, height=20)
    largest = detection(width=80, height=60, label="tv")

    assert select_super_resolution_target([small, largest]) == largest


def test_monitor_triggers_for_small_or_disappearing_largest_object():
    monitor = LargestObjectInstabilityMonitor(max_missed_frames=2)
    target = detection(width=30, height=40)

    small = monitor.decide(
        [target],
        frame_width=720,
        frame_height=720,
        trigger_confidence=0.45,
        trigger_area_pct=1.0,
    )
    first_miss = monitor.decide(
        [],
        frame_width=720,
        frame_height=720,
        trigger_confidence=0.45,
        trigger_area_pct=1.0,
    )

    assert small.target == target
    assert small.reason == "small_bbox"
    assert first_miss.target == target
    assert first_miss.reason == "detection_missing"


def test_monitor_triggers_on_confidence_drop_before_threshold():
    monitor = LargestObjectInstabilityMonitor()
    monitor.decide(
        [detection(confidence=0.9, width=120, height=120)],
        frame_width=720,
        frame_height=720,
        trigger_confidence=0.45,
        trigger_area_pct=1.0,
    )

    decision = monitor.decide(
        [detection(confidence=0.6, width=120, height=120)],
        frame_width=720,
        frame_height=720,
        trigger_confidence=0.45,
        trigger_area_pct=1.0,
    )

    assert decision.should_apply
    assert decision.reason == "confidence_drop"


def test_crop_is_clamped_and_super_resolution_box_is_remapped():
    target = detection(x=2, y=3, width=30, height=40)
    region = expanded_crop_region(target, 100, 100, padding_ratio=0.5)
    local = detection(x=16, y=20, width=60, height=80)

    mapped = remap_from_super_resolution(
        local,
        region,
        scale=2,
        frame_width=100,
        frame_height=100,
    )

    assert region.x == 0
    assert region.y == 0
    assert mapped.x == 8
    assert mapped.y == 10
    assert mapped.width == 30
    assert mapped.height == 40
    assert mapped.enhanced


def test_fsrcnn_enhancer_is_lazy_and_returns_x2_image():
    class FakeUpsampler:
        def upsample(self, image):
            return np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

    engine = FSRCNNSuperResolution()
    engine._upsampler = FakeUpsampler()
    image = np.zeros((12, 20, 3), dtype=np.uint8)

    result = engine.enhance(image)

    assert result.shape == (24, 40, 3)


def test_lab_adopts_higher_confidence_local_super_resolution_detection(tmp_path):
    class TensorLike:
        def __init__(self, values):
            self.values = np.asarray(values)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Boxes:
        def __init__(self, box, confidence):
            self.xyxy = TensorLike([box])
            self.conf = TensorLike([confidence])
            self.cls = TensorLike([0])

        def __len__(self):
            return 1

    class Result:
        def __init__(self, box, confidence):
            self.boxes = Boxes(box, confidence)
            self.names = {0: "cup"}

    class Model:
        names = {0: "cup"}

        def __init__(self):
            self.calls = 0

        def predict(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return [Result([60, 60, 100, 110], 0.3)]
            return [Result([160, 180, 240, 280], 0.8)]

    class SuperResolution:
        scale = 2
        ready = True

        def enhance(self, image):
            return np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

    lab = SmallObjectLab(device="cpu", output_root=tmp_path)
    lab.model = Model()
    lab.super_resolution = SuperResolution()

    lab._infer(np.zeros((200, 200, 3), dtype=np.uint8))

    assert lab.model.calls == 2
    assert lab.super_resolution_active
    assert lab.super_resolution_attempts == 1
    assert lab.super_resolution_adoptions == 1
    assert len(lab.latest_detections) == 1
    assert lab.latest_detections[0].enhanced
    assert lab.latest_detections[0].confidence == 0.8
