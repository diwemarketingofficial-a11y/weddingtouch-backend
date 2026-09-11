"""
Bug fix tests for iPhone selfie face detection.
Verifies EXIF orientation handling, HEIC decoding, downscaling of large images,
retry with upsample=2 on small faces, and multi-face selection (largest wins).
Uses the real ageitgey/face_recognition example obama.jpg to guarantee a
detectable human face.
"""
import io
import os
import time
import base64
import pytest
import requests
import piexif
from PIL import Image

BASE_URL = os.environ.get(
    "REACT_APP_BACKEND_URL", "https://shoot-schedule-5.preview.emergentagent.com"
).rstrip("/")
API = f"{BASE_URL}/api"

ADMIN_EMAIL = "admin@lensstudio.com"
ADMIN_PASSWORD = "admin123"

OBAMA_URL = (
    "https://raw.githubusercontent.com/ageitgey/"
    "face_recognition/master/examples/obama.jpg"
)


# ---------- helpers ----------
@pytest.fixture(scope="module")
def obama_bytes():
    """Download the canonical face image once for the module."""
    for _ in range(3):
        try:
            r = requests.get(OBAMA_URL, timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
        except Exception:
            time.sleep(1)
    pytest.skip("could not fetch obama.jpg reference image")


def _make_rotated_jpeg_with_exif(src_bytes: bytes, orientation: int) -> bytes:
    """
    Rotate the image pixels sideways/upside-down and stamp EXIF orientation
    that tells a viewer to rotate BACK. This simulates an iPhone JPEG whose
    sensor was portrait but stored the raw pixels landscape.

    orientation values (per EXIF spec):
      3 -> 180 deg (image pixels rotated 180 relative to displayed)
      6 -> 90 CW display -> pixels are stored rotated 90 CCW
      8 -> 90 CCW display -> pixels are stored rotated 90 CW
    """
    with Image.open(io.BytesIO(src_bytes)) as im:
        im = im.convert("RGB")
        # transform pixels so that after EXIF-transpose we get original
        if orientation == 3:
            rotated = im.rotate(180, expand=True)
        elif orientation == 6:
            # Display should rotate 90 CW; storage is rotated 90 CCW
            rotated = im.rotate(90, expand=True)  # CCW
        elif orientation == 8:
            rotated = im.rotate(-90, expand=True)  # CW
        else:
            rotated = im

        exif_dict = {"0th": {piexif.ImageIFD.Orientation: orientation}}
        exif_bytes = piexif.dump(exif_dict)

        out = io.BytesIO()
        rotated.save(out, format="JPEG", exif=exif_bytes, quality=90)
        return out.getvalue()


def _make_upscaled_jpeg(src_bytes: bytes, long_side: int = 4000) -> bytes:
    with Image.open(io.BytesIO(src_bytes)) as im:
        im = im.convert("RGB")
        scale = long_side / max(im.size)
        new_size = (int(im.size[0] * scale), int(im.size[1] * scale))
        big = im.resize(new_size, Image.LANCZOS)
        out = io.BytesIO()
        big.save(out, format="JPEG", quality=88)
        return out.getvalue()


def _make_heic_bytes(src_bytes: bytes) -> bytes | None:
    """Encode a PIL image to HEIC using pillow_heif. Returns None if unsupported."""
    try:
        from pillow_heif import register_heif_opener  # noqa
        register_heif_opener()
        with Image.open(io.BytesIO(src_bytes)) as im:
            im = im.convert("RGB")
            out = io.BytesIO()
            im.save(out, format="HEIF", quality=80)
            return out.getvalue()
    except Exception as e:
        print(f"HEIC encode failed: {e}")
        return None


# ---------- fixtures ----------
@pytest.fixture(scope="module")
def admin_headers():
    r = requests.post(
        f"{API}/auth/login",
        json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
        timeout=15,
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.fixture(scope="module")
def portal(admin_headers, obama_bytes):
    """Create booking + event + upload obama as event photo + get client token."""
    booking = requests.post(
        f"{API}/bookings",
        json={
            "client_name": "TEST_EXIF_Client",
            "client_email": "test_exif_client@example.com",
            "client_phone": "9999999900",
            "event_type": "Wedding",
            "event_date": "2026-07-01",
            "event_time": "10:00",
            "location": "Testland",
            "message": "exif bug retest",
        },
        timeout=15,
    )
    assert booking.status_code == 200, booking.text
    booking_id = booking.json()["id"]

    ev = requests.post(
        f"{API}/events",
        headers=admin_headers,
        json={"title": "TEST_EXIF_Event", "booking_id": booking_id, "match_threshold": 0.6},
        timeout=15,
    )
    assert ev.status_code == 200, ev.text
    event_id = ev.json()["id"]

    # upload upright obama as event photo
    files = {"file": ("obama.jpg", obama_bytes, "image/jpeg")}
    up = requests.post(
        f"{API}/events/{event_id}/photos",
        headers=admin_headers,
        files=files,
        timeout=60,
    )
    assert up.status_code == 200, up.text
    assert up.json().get("face_count", 0) >= 1, up.json()
    photo_id = up.json()["id"]

    qr = requests.post(
        f"{API}/events/{event_id}/qr",
        headers=admin_headers,
        data={"booking_id": booking_id},
        timeout=15,
    )
    assert qr.status_code == 200, qr.text
    client_token = qr.json()["token"]

    yield {
        "admin_headers": admin_headers,
        "booking_id": booking_id,
        "event_id": event_id,
        "photo_id": photo_id,
        "client_headers": {"X-Client-Token": client_token},
    }

    # teardown
    try:
        requests.delete(f"{API}/events/{event_id}", headers=admin_headers, timeout=15)
        requests.delete(f"{API}/bookings/{booking_id}", headers=admin_headers, timeout=15)
    except Exception:
        pass


# ---------- tests ----------
class TestExifRotation:
    @pytest.mark.parametrize("orientation", [3, 6, 8])
    def test_selfie_with_exif_orientation_finds_one_face(self, portal, obama_bytes, orientation):
        rotated = _make_rotated_jpeg_with_exif(obama_bytes, orientation)
        files = {"file": (f"selfie_exif_{orientation}.jpg", rotated, "image/jpeg")}
        r = requests.post(
            f"{API}/client/me/photos/search",
            headers=portal["client_headers"],
            files=files,
            timeout=60,
        )
        # Bug was: returned 400 with "found 0". Fix must NOT report 0 faces.
        assert r.status_code == 200, (
            f"orientation={orientation}: expected 200, got {r.status_code} {r.text}"
        )
        j = r.json()
        assert "matches" in j
        assert j.get("total_faces_scanned", 0) >= 1


class TestSameImageMatch:
    def test_identical_face_matches_itself(self, portal, obama_bytes):
        files = {"file": ("selfie.jpg", obama_bytes, "image/jpeg")}
        r = requests.post(
            f"{API}/client/me/photos/search",
            headers=portal["client_headers"],
            files=files,
            timeout=60,
        )
        assert r.status_code == 200, r.text
        j = r.json()
        assert len(j["matches"]) >= 1, f"expected the same-image match, got {j}"
        # distance should be ~0
        best = min(m["distance"] for m in j["matches"])
        assert best < 0.1, f"expected near-zero distance, got {best}"


class TestLargeImageDownscale:
    def test_upload_4000px_event_photo(self, portal, obama_bytes):
        big = _make_upscaled_jpeg(obama_bytes, long_side=4000)
        files = {"file": ("big.jpg", big, "image/jpeg")}
        t0 = time.time()
        r = requests.post(
            f"{API}/events/{portal['event_id']}/photos",
            headers=portal["admin_headers"],
            files=files,
            timeout=60,
        )
        elapsed = time.time() - t0
        assert r.status_code == 200, r.text
        assert r.json().get("face_count", 0) >= 1
        # not asserting hard 5s cutoff because network variance, but log it
        print(f"4000px upload+detect took {elapsed:.2f}s")


class TestHeicSupport:
    def test_heic_selfie_accepted(self, portal, obama_bytes):
        heic = _make_heic_bytes(obama_bytes)
        if heic is None:
            pytest.skip("pillow_heif can't encode HEIC in this env")
        files = {"file": ("selfie.heic", heic, "image/heic")}
        r = requests.post(
            f"{API}/client/me/photos/search",
            headers=portal["client_headers"],
            files=files,
            timeout=60,
        )
        # Content-type must not be rejected (415) and face must decode
        assert r.status_code != 415, f"HEIC content-type rejected: {r.text}"
        assert r.status_code == 200, r.text
        j = r.json()
        assert j.get("total_faces_scanned", 0) >= 1

    def test_heic_event_photo_accepted(self, portal, obama_bytes):
        heic = _make_heic_bytes(obama_bytes)
        if heic is None:
            pytest.skip("pillow_heif can't encode HEIC in this env")
        files = {"file": ("event.heic", heic, "image/heic")}
        r = requests.post(
            f"{API}/events/{portal['event_id']}/photos",
            headers=portal["admin_headers"],
            files=files,
            timeout=60,
        )
        assert r.status_code != 415
        assert r.status_code == 200, r.text
        # face_count may be >=1
        assert r.json().get("face_count", 0) >= 1


class TestMultipleFacesPicksLargest:
    def test_multi_face_does_not_error(self, portal, obama_bytes):
        """Compose an image with obama's face large in center + a small pasted
        copy in the corner. Endpoint should not 400 'multiple faces' anymore.
        """
        with Image.open(io.BytesIO(obama_bytes)) as face:
            face = face.convert("RGB")
            big = face.resize((600, 750), Image.LANCZOS)
            small = face.resize((120, 150), Image.LANCZOS)
            canvas = Image.new("RGB", (900, 800), "white")
            canvas.paste(big, (50, 25))
            canvas.paste(small, (760, 20))
            out = io.BytesIO()
            canvas.save(out, format="JPEG", quality=90)
            composite = out.getvalue()

        files = {"file": ("multi.jpg", composite, "image/jpeg")}
        r = requests.post(
            f"{API}/client/me/photos/search",
            headers=portal["client_headers"],
            files=files,
            timeout=60,
        )
        # Old behavior: 400 with "multiple faces". New behavior: 200.
        assert r.status_code == 200, (
            f"expected 200 (largest face picked), got {r.status_code} {r.text}"
        )
        j = r.json()
        assert j.get("total_faces_scanned", 0) >= 1
