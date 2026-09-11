"""Face recognition helpers (CPU-bound; call via asyncio.to_thread)."""
import io
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps

# Register HEIC/HEIF opener so iPhone photos can be decoded by PIL directly.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover
    pass

# Lazy import so the module can be imported even if dlib is still installing.
_fr = None

# Max longer-side for face detection input. Phones produce 4000+ px images
# that make HOG detection painfully slow AND less accurate on small faces
# when the frame is huge. Downscale keeps quality and boosts detection.
MAX_DETECTION_SIDE = 1600


def _lib():
    global _fr
    if _fr is None:
        import face_recognition as fr  # noqa
        _fr = fr
    return _fr


def _load_image_rgb(raw: bytes):
    """
    Open image bytes with PIL, apply EXIF orientation (crucial for iPhone/
    Android selfies which carry rotation metadata), downscale to a
    reasonable size, and return an (RGB numpy array, orig_width, orig_height).
    """
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        orig_w, orig_h = im.size
        # Normalize rotation from EXIF, then strip alpha
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        # Downscale keeping aspect ratio if too large
        long_side = max(im.size)
        if long_side > MAX_DETECTION_SIDE:
            scale = MAX_DETECTION_SIDE / long_side
            new_size = (int(im.size[0] * scale), int(im.size[1] * scale))
            im = im.resize(new_size, Image.LANCZOS)
        arr = np.asarray(im, dtype=np.uint8)
    return arr, orig_w, orig_h


def process_image_bytes(raw: bytes):
    """
    Detect faces and return (orig_width, orig_height, [locations], [encodings]).
    Runs sync CPU-bound work.
    Retries with upsampling if no face is found on first pass, to catch
    smaller / further-away faces (common on iPhone landscape shots).
    """
    fr = _lib()
    image, orig_w, orig_h = _load_image_rgb(raw)

    # First pass: fast HOG, no upsampling
    locations = fr.face_locations(image, model="hog", number_of_times_to_upsample=1)
    if not locations:
        # Retry with more upsampling — catches smaller faces
        locations = fr.face_locations(image, model="hog", number_of_times_to_upsample=2)

    encodings = fr.face_encodings(
        image, known_face_locations=locations, num_jitters=1, model="small"
    )
    return orig_w, orig_h, locations, encodings


def encode_selfie(raw: bytes):
    """Return the single best encoding for a selfie, or None if 0 faces.
    If multiple faces are found, pick the LARGEST (usually the closest one),
    which handles selfies with background people creeping in.
    """
    _, _, locations, encodings = process_image_bytes(raw)
    if len(encodings) == 0:
        return None, 0
    if len(encodings) == 1:
        return encodings[0], 1
    # Multiple faces: pick the biggest bounding box
    def area(loc):
        top, right, bottom, left = loc
        return max(0, right - left) * max(0, bottom - top)

    idx = max(range(len(locations)), key=lambda i: area(locations[i]))
    return encodings[idx], len(encodings)


def match_encodings(
    query: np.ndarray,
    stored: List[Tuple[str, List[float]]],
    threshold: float = 0.52,
):
    """
    stored: list of (photo_id, embedding-list-of-128-floats)
    Returns dict photo_id -> min distance for photos below threshold.
    """
    q = np.asarray(query, dtype=np.float32)
    photo_scores = {}
    for photo_id, emb in stored:
        e = np.asarray(emb, dtype=np.float32)
        if e.shape != (128,):
            continue
        d = float(np.linalg.norm(e - q))
        if d <= threshold:
            prev = photo_scores.get(photo_id, 1e9)
            if d < prev:
                photo_scores[photo_id] = d
    return photo_scores
