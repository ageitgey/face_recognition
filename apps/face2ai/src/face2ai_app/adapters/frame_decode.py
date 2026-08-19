"""One owner for turning request bytes into an analysable frame.

Every adapter that looks at pixels — recognition today, expression tomorrow — needs the same two
rules, and the repository has already shown what happens when each writes its own: the identical
unbounded ``Image.open`` with no EXIF handling appeared twice, a day apart, each blind to the other.

The rules:

* **Bounds before decoding.** ``Image.open`` is lazy: it reads the header and nothing else. The
  pixel budget is therefore checked against the declared size *before* a single pixel is decoded,
  which is the only order that helps — a 0.19 MiB PNG can declare gigabytes of pixels.
* **Upright coordinates.** ``exif_transpose`` is applied here, so every box and every encoding an
  adapter produces lives in the space the browser draws in.

The obligation this file serves is executable: ``apps/face2ai/tests/test_port_conformance.py``.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from face2ai_app.domain.errors import InvalidFrame


def decode_frame(image_bytes: bytes, max_pixels: int) -> np.ndarray:
    """Decode ``image_bytes`` into an EXIF-upright, C-contiguous RGB ``uint8`` array.

    Raises ``InvalidFrame`` — and nothing else — for empty payloads, undecodable payloads and
    frames whose declared size is over ``max_pixels``.
    """
    if not image_bytes:
        raise InvalidFrame("empty image payload")
    try:
        image = Image.open(BytesIO(image_bytes))  # lazy: header only, no pixel data touched
    except Image.DecompressionBombError as exc:
        # Pillow's own guard (~89 MP) fires inside open(), before our smaller budget can speak.
        raise InvalidFrame(f"frame exceeds the {max_pixels} pixel limit: {exc}") from exc
    except Exception as exc:
        raise InvalidFrame(f"unable to decode frame: {type(exc).__name__}: {exc}") from exc

    pixels = image.width * image.height
    if pixels > max_pixels:
        raise InvalidFrame(
            f"frame is {image.width}x{image.height} = {pixels} pixels, "
            f"over the {max_pixels} pixel limit"
        )

    try:
        upright = ImageOps.exif_transpose(image)
        return np.ascontiguousarray(np.asarray(upright.convert("RGB"), dtype=np.uint8))
    except Exception as exc:
        raise InvalidFrame(f"unable to decode frame: {type(exc).__name__}: {exc}") from exc
