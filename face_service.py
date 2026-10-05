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
MAX_DETECTION_SIDE = 1200
SELFIE_MAX_SIDE = 800


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
        orig_w, orig_h = im.size
        # Let Pillow's JPEG decoder decode close to our target size instead of
        # fully decoding a 20-50 MP original first. This is a major speed/memory
        # win for camera JPGs and does not touch the original file.
        try:
            im.draft("RGB", (MAX_DETECTION_SIDE, MAX_DETECTION_SIDE))
        except Exception:
            pass
        im.load()
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

    # Fast first pass. Upsampling multiplies HOG work heavily, so keep the
    # normal path at zero upsampling and retry only when absolutely necessary.
    locations = fr.face_locations(image, model="hog", number_of_times_to_upsample=0)
    if not locations:
        # One fallback pass for small/far faces. This is intentionally capped
        # at 1; the previous value of 2 made large wedding photos very slow.
        locations = fr.face_locations(image, model="hog", number_of_times_to_upsample=1)

    encodings = fr.face_encodings(
        image, known_face_locations=locations, num_jitters=1, model="small"
    )
    return orig_w, orig_h, locations, encodings


def encode_selfie(raw: bytes):
    """Fast selfie encoding path tuned for a close, clear face."""
    fr = _lib()
    with Image.open(io.BytesIO(raw)) as im:
        try:
            im.draft("RGB", (SELFIE_MAX_SIDE, SELFIE_MAX_SIDE))
        except Exception:
            pass
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        long_side = max(im.size)
        if long_side > SELFIE_MAX_SIDE:
            scale = SELFIE_MAX_SIDE / long_side
            im = im.resize(
                (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))),
                Image.LANCZOS,
            )
        image = np.asarray(im, dtype=np.uint8)

    locations = fr.face_locations(image, model="hog", number_of_times_to_upsample=0)
    if not locations:
        locations = fr.face_locations(image, model="hog", number_of_times_to_upsample=1)
    if not locations:
        return None, 0

    encodings = fr.face_encodings(
        image, known_face_locations=locations, num_jitters=1, model="small"
    )
    if not encodings:
        return None, 0
    if len(encodings) == 1:
        return encodings[0], 1

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
    if q.shape != (128,) or not stored:
        return {}

    valid_ids = []
    valid_embeddings = []
    for photo_id, emb in stored:
        e = np.asarray(emb, dtype=np.float32)
        if e.shape == (128,):
            valid_ids.append(photo_id)
            valid_embeddings.append(e)

    if not valid_embeddings:
        return {}

    matrix = np.vstack(valid_embeddings)
    distances = np.linalg.norm(matrix - q, axis=1)
    photo_scores = {}
    for photo_id, distance in zip(valid_ids, distances):
        d = float(distance)
        if d <= threshold:
            prev = photo_scores.get(photo_id)
            if prev is None or d < prev:
                photo_scores[photo_id] = d
    return photo_scores
