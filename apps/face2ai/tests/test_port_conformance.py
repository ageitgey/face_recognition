"""The port contracts, executable.

Four obligations bind every recognition adapter, and they are checked against *every* adapter
present in the environment — not against one hand-picked double:

1. ``detect`` is re-entrant: several threads may call it on one process.
2. Input bounds are enforced *before* the frame is decoded.
3. Boxes and encodings live in the EXIF-upright coordinate space.
4. Only ``InvalidFrame`` and ``RecognitionUnavailable`` escape.

The ``IdentityStore`` section below does the same for that port's closed error set.

Two properties of this file are deliberate and load-bearing:

* ``ContractFakeEngine`` implements the input rules *itself* instead of importing the production
  ``decode_frame``. A double that shares the implementation it exists to cross-check proves only
  that the helper agrees with itself; it would go green the moment the helper exists, even if no
  adapter called it.
* The re-entrancy case runs its load in a **subprocess** and asserts the exit code. The defect it
  guards against aborts the interpreter (measured: SIGSEGV), which would otherwise take the whole
  pytest session down instead of failing one test.

A double satisfies every obligation by construction, so this file is only evidence when it runs
against the real adapter: the ``recognition-contract`` job in ``.github/workflows/face2ai.yml``
installs ``--extra recognition`` and sets ``FACE2AI_CONFORMANCE_REQUIRE_REAL=1``, which turns the
"adapter not installed" skip below into a failure.
"""

from __future__ import annotations

import functools
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from face2ai_app.adapters.face_recognition_engine import FaceRecognitionEngine
from face2ai_app.adapters.json_identity_store import JsonIdentityStore
from face2ai_app.adapters.null_expression import NullExpressionEngine
from face2ai_app.config import Settings
from face2ai_app.domain.errors import (
    IdentityStoreCorrupted,
    IdentityStoreUnavailable,
    InvalidFrame,
    RecognitionUnavailable,
)
from face2ai_app.domain.models import DetectedFace, FaceBox, IdentityRecord
from face2ai_app.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[3]
FACE_PHOTO = REPO_ROOT / "examples" / "obama.jpg"

# The contract's own number, asserted against config below. 4 MP is ~16x the browser's 640 px snapshot.
MAX_PIXELS = 4_000_000
# Over the budget, under Pillow's own 89 MP bomb guard, so the *adapter's* bound is what rejects it.
OVERSIZED = (2200, 2200)
FRAME_WIDTH = 320  # enough for HOG (measured: 1 face, ~50 ms/detect) and cheap enough for 96 calls

REENTRANCY_THREADS = 4
REENTRANCY_CALLS = 24

CLOSED_ERROR_SET = (InvalidFrame, RecognitionUnavailable)


# --------------------------------------------------------------------------- frames


@functools.lru_cache(maxsize=None)
def upright_frame() -> bytes:
    """A real one-face photo, browser-sized, stored with no EXIF orientation."""
    source = Image.open(FACE_PHOTO).convert("RGB")
    height = round(source.height * FRAME_WIDTH / source.width)
    buffer = io.BytesIO()
    source.resize((FRAME_WIDTH, height)).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


@functools.lru_cache(maxsize=None)
def exif_rotated_frame() -> bytes:
    """The same photo stored rotated 90° CCW with EXIF orientation 6 — what a phone hands over.

    Decoded without ``exif_transpose`` this frame holds a face lying on its side, which the HOG
    detector does not find (measured: 0 faces).
    """
    source = Image.open(io.BytesIO(upright_frame()))
    rotated = source.rotate(90, expand=True)
    exif = rotated.getexif()
    exif[0x0112] = 6  # "rotate 90° CW to display"
    buffer = io.BytesIO()
    rotated.save(buffer, format="JPEG", quality=90, exif=exif.tobytes())
    return buffer.getvalue()


@functools.lru_cache(maxsize=None)
def oversized_frame() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", OVERSIZED, (23, 42, 61)).save(buffer, format="PNG")
    return buffer.getvalue()


@functools.lru_cache(maxsize=None)
def oversized_frame_without_pixels() -> bytes:
    """The oversized PNG with most of its pixel data cut off.

    The header still declares the full size, so a bounds check that runs before decoding rejects it
    exactly like the intact file. An adapter that decodes first reports a truncation error instead.
    """
    data = oversized_frame()
    return data[: max(64, len(data) // 3)]


# --------------------------------------------------------------------------- adapters under test


class ContractFakeEngine:
    """A double built to the contract: it decodes the frame and reports what it actually saw."""

    available = True
    availability_reason = None

    def __init__(self, max_frame_pixels: int = MAX_PIXELS) -> None:
        self._max_frame_pixels = max_frame_pixels

    def detect(self, image_bytes: bytes) -> list[DetectedFace]:
        if not image_bytes:
            raise InvalidFrame("empty image payload")
        try:
            image = Image.open(io.BytesIO(image_bytes))  # lazy: header only
        except Exception as exc:
            raise InvalidFrame(f"unable to decode frame: {type(exc).__name__}: {exc}") from exc
        pixels = image.width * image.height
        if pixels > self._max_frame_pixels:
            raise InvalidFrame(
                f"frame is {image.width}x{image.height} = {pixels} pixels, "
                f"over the {self._max_frame_pixels} pixel limit"
            )
        try:
            array = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise InvalidFrame(f"unable to decode frame: {type(exc).__name__}: {exc}") from exc
        height, width = array.shape[:2]
        return [
            DetectedFace(
                box=FaceBox(top=0, right=width, bottom=height, left=0),
                encoding=[float(array[0, 0, 0]) / 255.0] * 128,
            )
        ]


ADAPTER_IDS = ["contract-fake", "face-recognition"]


def make_engine(adapter_id: str):
    """The one place that knows how to build each adapter — used by the tests and the subprocess."""
    if adapter_id == "contract-fake":
        return ContractFakeEngine()
    if adapter_id == "face-recognition":
        return FaceRecognitionEngine()  # its own default bound, so the shipped configuration is what runs
    raise KeyError(f"unknown adapter id: {adapter_id}")


def require_engine(adapter_id: str):
    engine = make_engine(adapter_id)
    if not engine.available:
        message = f"{adapter_id} adapter unavailable: {engine.availability_reason}"
        if os.getenv("FACE2AI_CONFORMANCE_REQUIRE_REAL") == "1":
            pytest.fail(message)  # CI asked for the real thing; a skip here would be a green lie
        pytest.skip(message)
    return engine


@pytest.fixture(params=ADAPTER_IDS)
def engine(request):
    return require_engine(request.param)


# --------------------------------------------------------------------------- obligation 1: re-entrancy


REENTRANCY_CHILD = r'''
"""Concurrent load on a *cold* engine, then the serial truth to compare it against.

The order is load-bearing and was measured, because the obvious order hides the defect: the crash
lives in the lazy first-call initialisation of dlib's process-global singletons. Taking the serial
baseline first warms them and the unlocked adapter then survived 5 of 5 runs at this load; taking
it afterwards, the same unlocked adapter aborted 7 of 10 runs (SIGSEGV). So: no call before the
threads start.
"""
import importlib.util, sys, threading

module_path, adapter_id, threads_n, calls_n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
spec = importlib.util.spec_from_file_location("_face2ai_conformance", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

engine = module.make_engine(adapter_id)
frame = module.upright_frame()

results, errors = [], []
lock = threading.Lock()


def worker():
    for _ in range(calls_n):
        try:
            faces = engine.detect(frame)
        except Exception as exc:
            with lock:
                errors.append("%s: %s" % (type(exc).__name__, exc))
            continue
        with lock:
            results.append([list(face.encoding) for face in faces])


workers = [threading.Thread(target=worker) for _ in range(threads_n)]
for t in workers:
    t.start()
for t in workers:
    t.join()

baseline = engine.detect(frame)  # serial truth, taken once the concurrent phase is over
counts = [len(result) for result in results]
worst = 0.0
for result in results:
    if len(result) != len(baseline):
        continue
    for encoding, expected in zip(result, baseline):
        for got, want in zip(encoding, expected.encoding):
            worst = max(worst, abs(got - want))

print("calls=%d face_counts=%s serial_baseline=%d worst_encoding_delta=%.9f exceptions=%d"
      % (len(counts), sorted(set(counts)), len(baseline), worst, len(errors)))
problems = []
if errors:
    problems.append("%d exception(s), first: %s" % (len(errors), errors[0]))
if any(count != len(baseline) for count in counts):
    problems.append("face counts %s differ from the serial baseline %d" % (sorted(set(counts)), len(baseline)))
if worst > 1e-9:
    problems.append("encoding drift %.9f > 1e-9" % worst)
if problems:
    print("CONTRACT VIOLATED: " + "; ".join(problems))
    sys.exit(1)
'''


@pytest.mark.parametrize("adapter_id", ADAPTER_IDS)
def test_detect_is_reentrant(adapter_id: str, tmp_path: Path) -> None:
    """Several threads calling detect() on one process must agree with the serial result.

    Run in a subprocess on purpose: the unlocked dlib singletons abort the interpreter, and an
    in-process load test would take the pytest session with it instead of failing this one case.

    The guard is one-sided by nature: a data race is probabilistic, and this load caught the
    unlocked adapter in 7 of 10 measured runs. It never fails a correct adapter — with the lock the
    calls are serialised, so there is nothing left to race.
    """
    require_engine(adapter_id)
    script = tmp_path / "reentrancy_child.py"
    script.write_text(REENTRANCY_CHILD, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), __file__, adapter_id, str(REENTRANCY_THREADS), str(REENTRANCY_CALLS)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    report = (
        f"exit={proc.returncode} (negative = killed by that signal)\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr (tail) ---\n{proc.stderr[-2000:]}"
    )
    assert proc.returncode == 0, f"{adapter_id} is not re-entrant:\n{report}"


# --------------------------------------------------------------------------- obligation 2: input bounds


def test_oversized_frame_is_rejected(engine) -> None:
    with pytest.raises(InvalidFrame) as excinfo:
        engine.detect(oversized_frame())
    assert "pixel" in str(excinfo.value).lower(), "the rejection must name the bound it enforced"


def test_oversized_frame_is_rejected_before_decoding(engine) -> None:
    """Same frame with its pixel data cut off: only a check that precedes the decode still says 'pixels'."""
    with pytest.raises(InvalidFrame) as excinfo:
        engine.detect(oversized_frame_without_pixels())
    assert "pixel" in str(excinfo.value).lower(), (
        "the frame was decoded before its size was checked: "
        f"the failure is a decode error, not a bounds rejection ({excinfo.value})"
    )


def test_frame_at_the_bound_is_accepted(engine) -> None:
    """The bound rejects what is over it, not what is at it."""
    buffer = io.BytesIO()
    Image.new("RGB", (2000, 2000), (23, 42, 61)).save(buffer, format="PNG")  # exactly 4_000_000 px
    engine.detect(buffer.getvalue())  # no face expected, no exception allowed


def test_max_frame_pixels_is_the_configured_bound() -> None:
    assert Settings().max_frame_pixels == MAX_PIXELS


def test_max_frame_pixels_is_configurable_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACE2AI_MAX_FRAME_PIXELS", "250000")
    assert Settings.from_env().max_frame_pixels == 250_000
    with pytest.raises(ValueError):
        Settings(max_frame_pixels=0)


# --------------------------------------------------------------------------- obligation 3: EXIF-upright


def test_frame_is_analysed_in_the_exif_upright_space(engine) -> None:
    """A phone's rotated-with-EXIF frame must yield the same faces, in the same coordinates, as the
    physically upright one — the browser draws its overlay in that space."""
    upright = engine.detect(upright_frame())
    rotated = engine.detect(exif_rotated_frame())
    assert len(upright) >= 1, "the probe photo holds one face; a conforming adapter must see it"
    assert len(rotated) == len(upright), (
        f"EXIF-rotated frame yielded {len(rotated)} face(s), upright yielded {len(upright)}"
    )
    for got, want in zip(rotated, upright):
        for edge in ("top", "right", "bottom", "left"):
            delta = abs(getattr(got.box, edge) - getattr(want.box, edge))
            assert delta <= 8, f"box.{edge} moved {delta}px between the rotated and upright frame"


# --------------------------------------------------------------------------- obligation 4: closed error set


@pytest.mark.parametrize(
    "payload_id",
    ["empty", "not-an-image", "truncated-jpeg", "oversized", "oversized-truncated"],
)
def test_only_the_closed_error_set_escapes(engine, payload_id: str) -> None:
    payloads = {
        "empty": b"",
        "not-an-image": b"this is not an image, it is a sentence" * 8,
        "truncated-jpeg": upright_frame()[:220],
        "oversized": oversized_frame(),
        "oversized-truncated": oversized_frame_without_pixels(),
    }
    try:
        engine.detect(payloads[payload_id])
    except CLOSED_ERROR_SET:
        return
    except Exception as exc:  # noqa: BLE001 - naming the leak is the point of this test
        pytest.fail(f"{type(exc).__name__} escaped detect(); the port allows only {CLOSED_ERROR_SET}: {exc}")


# --------------------------------------------------------------------------- the IdentityStore contract


def test_store_reports_a_corrupted_file_as_corrupted(tmp_path: Path) -> None:
    path = tmp_path / "identities.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(IdentityStoreCorrupted):
        JsonIdentityStore(path).list()


def test_store_reports_an_unreadable_file_as_unavailable(tmp_path: Path) -> None:
    """An OSError is not corruption: the data may be perfectly fine and unreachable."""
    path = tmp_path / "identities.json"
    path.mkdir()  # reading it raises IsADirectoryError, an OSError
    with pytest.raises(IdentityStoreUnavailable):
        JsonIdentityStore(path).list()


def test_store_reports_an_unwritable_location_as_unavailable(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the data dir should be", encoding="utf-8")
    store = JsonIdentityStore(blocker / "data" / "identities.json")
    with pytest.raises(IdentityStoreUnavailable):
        store.add(IdentityRecord.new("Ada", [0.1] * 128))


def test_unavailable_store_is_a_503(tmp_path: Path) -> None:
    path = tmp_path / "identities.json"
    path.mkdir()
    app = create_app(
        settings=Settings(data_dir=tmp_path),
        engine=ContractFakeEngine(),
        store=JsonIdentityStore(path),
        expression=NullExpressionEngine(),
    )
    with TestClient(app) as client:
        response = client.get("/api/identities")
    assert response.status_code == 503, response.text
