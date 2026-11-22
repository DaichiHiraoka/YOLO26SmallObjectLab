import numpy as np
import pytest

from small_object_lab.metrics import Detection
from small_object_lab.research.rediscovery import (Gallery, RediscoveryConfig, appearance, similarity,
                                                  split_detections)
from small_object_lab.research.runner import ExperimentConfig, MethodEngine


def scene(cup_at=None, patch=None):
    """White wall; a dark cup-like rectangle with a lighter rim; optional other patch."""
    image = np.full((200, 200, 3), 230, np.uint8)
    if cup_at:
        x, y = cup_at
        image[y:y + 30, x:x + 20] = (30, 30, 35)
        image[y:y + 4, x:x + 20] = (90, 90, 95)
    if patch:
        x, y, colour = patch
        image[y:y + 30, x:x + 20] = colour
    return image


def box(x, y, label="cup", confidence=0.8):
    return Detection(label, confidence, x, y, 20, 30)


class Backend:
    def synchronize(self):
        pass


def test_same_object_is_similar_and_a_different_one_or_size_is_not():
    a = appearance(scene((50, 50)), box(50, 50))
    moved = appearance(scene((120, 100)), box(120, 100))
    other = appearance(scene(patch=(120, 100, (40, 160, 40))), box(120, 100))
    assert similarity(a, moved) > 0.95
    assert similarity(a, other) < 0.75
    big = appearance(scene((50, 50)), Detection("cup", 0.8, 40, 40, 80, 100))
    assert similarity(a, big) == -1.0


def test_one_low_threshold_inference_splits_into_detections_and_hypotheses():
    raw = [box(0, 0, confidence=0.9), box(10, 0, confidence=0.05), box(20, 0, confidence=0.005),
           box(30, 0, "cell phone", 0.3), box(40, 0, "cell phone", 0.05)]
    base, hypotheses = split_detections(raw, "cup", 0.2, RediscoveryConfig())
    assert [d.x for d in base] == [0]
    assert [d.x for d in hypotheses] == [10, 30]  # low-confidence cup, other class above 0.1


def test_lost_target_is_readopted_from_another_class_only_if_it_looks_the_same():
    config = ExperimentConfig()
    engine = MethodEngine("baseline+rd", config)
    plain = MethodEngine("baseline", config)
    first = scene((50, 50))
    for e in (engine, plain):
        final, *_ = e.process(Backend(), first, [box(50, 50)], 0.0)
        assert final is not None
    assert len(engine.gallery.items) == 1
    # 2 s later (beyond retention) the detector only sees a "cell phone" where the cup now is
    later = scene((120, 100), patch=(10, 150, (40, 160, 40)))
    hypotheses = [box(120, 100, "cell phone", 0.3), box(10, 150, "cell phone", 0.4)]
    final, _, attempts, _ = engine.process(Backend(), later, [], 2.0, hypotheses)
    assert final is not None and (final.x, final.y, final.label) == (120, 100, "cup")
    assert attempts[-1]["action_id"] == "rediscover" and attempts[-1]["source_label"] == "cell phone"
    assert len(engine.gallery.items) == 1 and engine.gallery.confirmed == 1  # not learned from itself
    assert plain.process(Backend(), later, [], 2.0, hypotheses)[0] is None


def test_nothing_is_readopted_without_a_confirmed_view_or_a_lookalike():
    engine = MethodEngine("baseline+rd", ExperimentConfig())
    image = scene((120, 100))
    assert engine.process(Backend(), image, [], 0.0, [box(120, 100, "cell phone")])[0] is None
    engine.process(Backend(), scene((50, 50)), [box(50, 50)], 1.0)
    empty = scene(patch=(120, 100, (40, 160, 40)))
    final, _, attempts, _ = engine.process(Backend(), empty, [], 3.0, [box(120, 100, "cell phone")])
    assert final is None and attempts[-1]["adopted"] is False


def test_rediscovery_settings_are_validated():
    with pytest.raises(ValueError):
        RediscoveryConfig(threshold=0)
    with pytest.raises(ValueError):
        RediscoveryConfig(gallery_every=0)
    with pytest.raises(ValueError):
        MethodEngine("rule+xx", ExperimentConfig())
    gallery = Gallery(RediscoveryConfig(gallery_every=2))
    for _ in range(4):
        gallery.add(scene((50, 50)), box(50, 50))
    assert len(gallery.items) == 2
