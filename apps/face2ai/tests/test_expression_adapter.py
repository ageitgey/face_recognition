"""MediaPipe + EmotiEffLib adapter: pure helpers (no models, no extra needed) plus the engine's
unavailable path — that last test may import mediapipe when the ``expression`` extra is present."""

import logging

import numpy as np
import pytest

from face2ai_app.adapters.mediapipe_expression import (
    MediaPipeExpressionEngine,
    emotiefflib_model_path,
    landmark_bbox,
    compact_blendshapes,
    crop_with_margin,
    match_faces,
    pose_from_matrix,
    softmax_scores,
)
from face2ai_app.domain.models import EMOTIONS, FaceBox


def test_softmax_scores_maps_logits_to_labels():
    scores = softmax_scores([0, 0, 0, 0, 5, 0, 0, 0])
    assert max(scores, key=scores.get) == "Happiness" and abs(sum(scores.values()) - 1) < 1e-6
    assert tuple(scores) == EMOTIONS


def test_softmax_scores_is_numerically_stable_for_large_logits():
    scores = softmax_scores([1000, 0, 0, 0, 0, 0, 0, 0])
    assert scores["Anger"] == pytest.approx(1.0) and abs(sum(scores.values()) - 1) < 1e-6


def test_pose_from_identity_is_zero():
    yaw, pitch, roll = pose_from_matrix(np.eye(4))
    assert (round(yaw), round(pitch), round(roll)) == (0, 0, 0)


def test_pose_from_matrix_recovers_yaw():
    theta = np.radians(30)
    m = np.eye(4)
    m[:3, :3] = [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    yaw, pitch, roll = pose_from_matrix(m)
    assert (round(yaw), round(pitch), round(roll)) == (30, 0, 0)


def test_pose_from_matrix_recovers_pitch():
    theta = np.radians(20)
    m = np.eye(4)
    m[:3, :3] = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]  # Rx(20°)
    yaw, pitch, roll = pose_from_matrix(m)
    assert yaw == pytest.approx(0, abs=0.5)
    assert pitch == pytest.approx(20, abs=0.5)
    assert roll == pytest.approx(0, abs=0.5)


def test_pose_from_matrix_recovers_roll():
    theta = np.radians(20)
    m = np.eye(4)
    m[:3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]  # Rz(20°)
    yaw, pitch, roll = pose_from_matrix(m)
    assert yaw == pytest.approx(0, abs=0.5)
    assert pitch == pytest.approx(0, abs=0.5)
    assert roll == pytest.approx(20, abs=0.5)


def test_match_faces_by_iou():
    boxes = [FaceBox(top=100, right=400, bottom=300, left=200)]
    assert match_faces(boxes, [(210, 110, 390, 290)]) == [0]  # (left, top, right, bottom)
    assert match_faces(boxes, [(0, 0, 10, 10)]) == [None]


def test_match_faces_assigns_each_landmark_face_once():
    boxes = [
        FaceBox(top=100, right=400, bottom=300, left=200),
        FaceBox(top=110, right=410, bottom=310, left=210),  # near-duplicate of the first
        FaceBox(top=0, right=50, bottom=50, left=0),
    ]
    matches = match_faces(boxes, [(210, 110, 390, 290), (0, 0, 50, 50)])
    assert matches == [0, None, 1]
    assert match_faces([], [(0, 0, 10, 10)]) == []
    assert match_faces(boxes, []) == [None, None, None]


class _Pt:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_landmark_bbox_scales_to_pixels_and_degenerates_for_empty_set():
    assert landmark_bbox([_Pt(0.25, 0.5), _Pt(0.75, 0.6)], 400, 200) == (100.0, 100.0, 300.0, 120.0)
    # an empty landmark set (a malformed result) must not raise and must never match a face —
    # but it keeps its index so blendshapes/pose of the other faces stay aligned
    empty = landmark_bbox([], 400, 200)
    assert empty == (0.0, 0.0, 0.0, 0.0)
    boxes = [FaceBox(top=100, right=400, bottom=300, left=200), FaceBox(top=0, right=50, bottom=50, left=0)]
    assert match_faces(boxes, [empty, (0, 0, 50, 50)]) == [None, 1]


def test_compact_blendshapes_filters_and_rounds():
    assert compact_blendshapes([("mouthSmileLeft", 0.951), ("browDownLeft", 0.05), ("_neutral", 0.3)]) == {
        "mouthSmileLeft": 0.95
    }


def test_compact_blendshapes_clips_into_unit_interval():
    # float noise from the model (1.0000001) must not violate the 0..1 bound of the wire model
    assert compact_blendshapes([("jawOpen", 1.0000001), ("eyeBlinkLeft", 0.2)]) == {"jawOpen": 1.0, "eyeBlinkLeft": 0.2}


def test_crop_with_margin_clamps_to_image():
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = crop_with_margin(arr, FaceBox(top=0, right=200, bottom=100, left=0), margin=0.2)
    assert crop.shape == (100, 200, 3)


def test_crop_with_margin_expands_inner_box():
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    crop = crop_with_margin(arr, FaceBox(top=100, right=200, bottom=200, left=100), margin=0.2)
    assert crop.shape == (140, 140, 3)


def test_engine_is_unavailable_without_asset(tmp_path):
    engine = MediaPipeExpressionEngine(tmp_path)
    assert engine.available is False
    reason = engine.availability_reason
    assert isinstance(reason, str) and ("missing model asset" in reason or "not installed" in reason)
    assert engine.analyze(b"x", [FaceBox(top=0, right=10, bottom=10, left=0)]) == [None]


def test_engine_is_unavailable_without_emotiefflib_model_and_never_downloads(tmp_path, monkeypatch):
    """With the extra installed but emotiefflib's cached ONNX missing the engine must report why and must not
    trigger emotiefflib's own GitHub download (startup is offline by contract)."""
    pytest.importorskip("mediapipe")
    pytest.importorskip("emotiefflib")
    import urllib.request

    def _no_network(*args, **kwargs):  # pragma: no cover - reached only on a regression
        raise AssertionError(f"network access attempted: urlretrieve{args}")

    monkeypatch.setattr(urllib.request, "urlretrieve", _no_network)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))  # emotiefflib and the adapter both resolve ~ via os.path.expanduser
    assert emotiefflib_model_path() == home / ".emotiefflib" / "enet_b0_8_va_mtl.onnx"
    models = tmp_path / "models"
    models.mkdir()
    (models / "face_landmarker.task").write_bytes(b"")  # present, so the check under test is the emotiefflib one
    engine = MediaPipeExpressionEngine(models)
    assert engine.available is False
    reason = engine.availability_reason
    assert isinstance(reason, str) and "EmotiEffLib" in reason and "fetch-expression-models.sh" in reason
    assert not (home / ".emotiefflib").exists()


def test_engine_logs_first_failure_at_warning_then_debug(tmp_path, caplog):
    engine = MediaPipeExpressionEngine(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="face2ai_app.adapters.mediapipe_expression"):
        engine._log_failure("frame", RuntimeError("boom"))
        engine._log_failure("frame", RuntimeError("boom again"))
        engine._log_failure("face", ValueError("bad crop"))
    levels = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert [lvl for lvl, _ in levels] == [logging.WARNING, logging.DEBUG, logging.WARNING]
    assert "frame: RuntimeError: boom" in levels[0][1] and "face: ValueError: bad crop" in levels[2][1]
