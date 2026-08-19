"""Expression adapter: MediaPipe Face Landmarker (blendshapes + head pose) and EmotiEffLib
(8 emotions + valence/arousal). Everything here is a mood *hint*, never a fact.

Only numpy and the shared frame decoder are imported at module level; mediapipe / emotiefflib are
imported lazily so the pure helpers stay importable and testable without the ``expression`` extra.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from face2ai_app.adapters.frame_decode import decode_frame
from face2ai_app.config import DEFAULT_MAX_FRAME_PIXELS
from face2ai_app.domain.models import EMOTIONS, Expression, FaceBox

logger = logging.getLogger(__name__)

MODEL_ASSET = "face_landmarker.task"
EMOTIEFF_MODEL = "enet_b0_8_va_mtl"
# emotiefflib 1.1.1 has no local-path option: EmotiEffLibRecognizerOnnx resolves the model via
# utils.get_model_path_onnx -> os.path.expanduser("~")/.emotiefflib/<model>.onnx and would download it
# from GitHub when missing. We never let that happen at startup: the fetch script puts the file there.
EMOTIEFF_CACHE_DIR = ".emotiefflib"
MIN_IOU = 0.2
BLENDSHAPE_THRESHOLD = 0.2
CROP_MARGIN = 0.2

# (left, top, right, bottom) in pixels — the bounding box of one landmark set
LandmarkBox = tuple[float, float, float, float]


# ----------------------------------------------------------------------------- pure helpers


def emotiefflib_model_path() -> Path:
    """Where emotiefflib expects its ONNX file (same derivation as ``emotiefflib.utils.download_model``)."""
    return Path(os.path.expanduser("~")) / EMOTIEFF_CACHE_DIR / f"{EMOTIEFF_MODEL}.onnx"


def landmark_bbox(lms: Sequence, width: int, height: int) -> LandmarkBox:
    """(left, top, right, bottom) in pixels of one normalized landmark set; a degenerate (0, 0, 0, 0)
    box for an empty set so it can never match a face (IoU 0) without shifting indices."""
    if not lms:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p.x for p in lms]
    ys = [p.y for p in lms]
    return (min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height)


def _iou(box: FaceBox, lm: LandmarkBox) -> float:
    left, top, right, bottom = lm
    inter_w = min(box.right, right) - max(box.left, left)
    inter_h = min(box.bottom, bottom) - max(box.top, top)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_box = max(0, box.right - box.left) * max(0, box.bottom - box.top)
    area_lm = max(0.0, right - left) * max(0.0, bottom - top)
    union = area_box + area_lm - inter
    return inter / union if union > 0 else 0.0


def match_faces(boxes: Sequence[FaceBox], landmark_bboxes: Sequence[LandmarkBox]) -> list[int | None]:
    """Greedy best-IoU assignment of landmark faces to recognition boxes (IoU >= MIN_IOU).

    Returns one landmark index (or None) per box, same order; each landmark face is used at most once.
    """
    pairs = [
        (iou, b, l)
        for b, box in enumerate(boxes)
        for l, lm in enumerate(landmark_bboxes)
        if (iou := _iou(box, lm)) >= MIN_IOU
    ]
    pairs.sort(key=lambda p: p[0], reverse=True)
    result: list[int | None] = [None] * len(boxes)
    used_landmarks: set[int] = set()
    for _, b, l in pairs:
        if result[b] is None and l not in used_landmarks:
            result[b] = l
            used_landmarks.add(l)
    return result


def pose_from_matrix(m4x4) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from a MediaPipe facial transformation matrix (rotation in m[:3, :3])."""
    r = np.asarray(m4x4, dtype=float)[:3, :3]
    yaw = math.atan2(r[0, 2], r[2, 2])
    pitch = math.asin(float(np.clip(-r[1, 2], -1.0, 1.0)))
    roll = math.atan2(r[1, 0], r[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def softmax_scores(logits8: Sequence[float]) -> dict[str, float]:
    """Numerically stable softmax over the 8 emotion logits, keyed by EMOTIONS in order."""
    values = np.asarray(logits8, dtype=float).reshape(-1)
    if values.shape[0] != len(EMOTIONS):
        raise ValueError(f"expected {len(EMOTIONS)} logits, got {values.shape[0]}")
    shifted = np.exp(values - values.max())
    probs = shifted / shifted.sum()
    return {label: float(p) for label, p in zip(EMOTIONS, probs, strict=True)}


def compact_blendshapes(pairs: Sequence[tuple[str, float]], threshold: float = BLENDSHAPE_THRESHOLD) -> dict[str, float]:
    """Keep only named blendshapes at/above threshold (drops the `_neutral` pseudo-category), clipped to 0..1, rounded to 2 decimals."""
    return {
        name: round(min(1.0, max(0.0, float(score))), 2)
        for name, score in pairs
        if name != "_neutral" and score >= threshold
    }


def crop_with_margin(arr: np.ndarray, box: FaceBox, margin: float = CROP_MARGIN) -> np.ndarray:
    """Slice `box` out of an HxWxC array, expanded by `margin` x its size on every side, clamped to the image."""
    h, w = arr.shape[:2]
    dx = (box.right - box.left) * margin
    dy = (box.bottom - box.top) * margin
    left = max(0, int(math.floor(box.left - dx)))
    top = max(0, int(math.floor(box.top - dy)))
    right = min(w, int(math.ceil(box.right + dx)))
    bottom = min(h, int(math.ceil(box.bottom + dy)))
    if right <= left or bottom <= top:
        return arr[0:0, 0:0]
    return arr[top:bottom, left:right]


# ----------------------------------------------------------------------------- engine


class MediaPipeExpressionEngine:
    """Blendshapes + head pose from MediaPipe Face Landmarker, emotions + valence/arousal from EmotiEffLib.

    Both models are loaded in ``__init__`` (imports, file checks, session creation), CPU only, offline:
    the two assets (``face_landmarker.task`` in ``models_dir`` and emotiefflib's ``enet_b0_8_va_mtl.onnx``
    in ``~/.emotiefflib``) must be fetched up front by ``scripts/fetch-expression-models.sh``. Startup never
    touches the network; the engine is unavailable (never crashing) when the extra or an asset is missing.
    """

    def __init__(self, models_dir: Path, max_frame_pixels: int = DEFAULT_MAX_FRAME_PIXELS) -> None:
        self._max_frame_pixels = max_frame_pixels
        self._reason: str | None = None
        self._landmarker = None
        self._recognizer = None
        self._mp = None
        self._lock = threading.Lock()  # MediaPipe task instances are not thread-safe
        # First failure per scope logs at warning, later ones at debug (frame = whole image, face = one crop).
        self._warned: dict[str, bool] = {"frame": False, "face": False}
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
        except Exception as exc:  # ImportError, but native libs can fail in other ways
            self._reason = f"expression extra not installed: {type(exc).__name__}: {exc}"
            return
        asset = Path(models_dir) / MODEL_ASSET
        if not asset.exists():
            self._reason = f"missing model asset {asset} (run scripts/fetch-expression-models.sh)"
            return
        emotieff_model = emotiefflib_model_path()
        if not emotieff_model.is_file():
            # Checked before constructing the recognizer: emotiefflib would otherwise download it now.
            self._reason = f"missing EmotiEffLib model {emotieff_model} (run scripts/fetch-expression-models.sh)"
            return
        try:
            base = mp_python.BaseOptions(model_asset_path=str(asset), delegate=mp_python.BaseOptions.Delegate.CPU)
            options = vision.FaceLandmarkerOptions(
                base_options=base,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=4,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self._recognizer = EmotiEffLibRecognizer(engine="onnx", model_name=EMOTIEFF_MODEL, device="cpu")
            self._mp = mp
        except Exception as exc:
            self._landmarker = None
            self._recognizer = None
            self._reason = f"expression engine failed to initialize: {type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._landmarker is not None and self._recognizer is not None

    @property
    def availability_reason(self) -> str | None:
        return self._reason

    def analyze(self, image_bytes: bytes, boxes: list[FaceBox]) -> list[Expression | None]:
        if not self.available or not boxes:
            return [None for _ in boxes]
        try:
            # Same decoder as the recognition adapter: one owner for the pixel budget and for
            # EXIF orientation, so a frame is never analysed in two different coordinate spaces.
            arr = decode_frame(image_bytes, self._max_frame_pixels)
            with self._lock:
                result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=arr))
        except Exception as exc:
            self._log_failure("frame", exc)
            return [None for _ in boxes]

        h, w = arr.shape[:2]
        try:
            lm_boxes = [landmark_bbox(lms, w, h) for lms in result.face_landmarks]
            matches = match_faces(boxes, lm_boxes)
        except Exception as exc:
            # A malformed landmark result must not drop the emotion hint for every face: continue without
            # blendshapes/pose (idx None) instead of failing the whole frame.
            self._log_failure("frame", exc)
            matches = [None] * len(boxes)
        return [self._analyze_face(arr, result, box, idx) for box, idx in zip(boxes, matches, strict=True)]

    def _analyze_face(self, arr: np.ndarray, result, box: FaceBox, idx: int | None) -> Expression | None:
        try:
            crop = crop_with_margin(arr, box, margin=CROP_MARGIN)
            if crop.size == 0:
                return None
            with self._lock:
                _, logits = self._recognizer.predict_emotions(crop, logits=True)  # (1, 10) for enet_b0_8_va_mtl
            row = np.asarray(logits, dtype=float).reshape(-1)
            if row.shape[0] < len(EMOTIONS) + 2:
                logger.debug("expression model returned %d logits, need >= %d", row.shape[0], len(EMOTIONS) + 2)
                return None
            scores = softmax_scores(row[: len(EMOTIONS)].tolist())
            valence = float(np.clip(row[len(EMOTIONS)], -1.0, 1.0))
            arousal = float(np.clip(row[len(EMOTIONS) + 1], -1.0, 1.0))
            blend: dict[str, float] = {}
            yaw = pitch = roll = None
            if idx is not None:
                blend = compact_blendshapes([(c.category_name, c.score) for c in result.face_blendshapes[idx]])
                yaw, pitch, roll = (
                    round(v, 1) for v in pose_from_matrix(np.asarray(result.facial_transformation_matrixes[idx]))
                )
            return Expression(
                dominant=max(scores, key=scores.get),
                scores={k: round(v, 3) for k, v in scores.items()},
                valence=round(valence, 3),
                arousal=round(arousal, 3),
                blendshapes=blend,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
            )
        except Exception as exc:
            self._log_failure("face", exc)
            return None

    def _log_failure(self, scope: str, exc: Exception) -> None:
        """Fail visibly once per scope (warning), then stay quiet (debug) so a broken model does not flood the log."""
        first = not self._warned[scope]  # KeyError on an unknown scope: fail visibly
        self._warned[scope] = True
        logger.log(
            logging.WARNING if first else logging.DEBUG,
            "expression analysis skipped for %s: %s: %s%s",
            scope,
            type(exc).__name__,
            exc,
            " (further occurrences logged at debug)" if first else "",
        )
