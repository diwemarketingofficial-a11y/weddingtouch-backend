"""
Wedding Touch backend integration tests.
Covers: auth, bookings (event_time), events, photos, QR, client portal,
selfie search, album selection, admin regression, auth boundaries.
"""
import io
import os
import base64
import pytest
import requests
from PIL import Image, ImageDraw

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://shoot-schedule-5.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"

ADMIN_EMAIL = "admin@lensstudio.com"
ADMIN_PASSWORD = "admin123"


# ---------- helpers ----------
def _make_face_png() -> bytes:
    """Generate a synthetic image the hog detector will recognize as a face."""
    img = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(img)
    # face oval
    d.ellipse((80, 60, 320, 360), fill=(240, 200, 170), outline=(0, 0, 0), width=3)
    # eyes
    d.ellipse((140, 160, 180, 200), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    d.ellipse((150, 170, 170, 190), fill=(0, 0, 0))
    d.ellipse((220, 160, 260, 200), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    d.ellipse((230, 170, 250, 190), fill=(0, 0, 0))
    # nose + mouth
    d.polygon([(200, 210), (185, 260), (215, 260)], fill=(200, 150, 130))
    d.arc((160, 260, 240, 310), start=0, end=180, fill=(120, 40, 40), width=4)
    # eyebrows
    d.line((135, 150, 185, 145), fill=(60, 40, 20), width=4)
    d.line((215, 145, 265, 150), fill=(60, 40, 20), width=4)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_blank_png() -> bytes:
    img = Image.new("RGB", (400, 400), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- fixtures ----------
@pytest.fixture(scope="session")
def admin_token():
    r = requests.post(f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
    return r.json()["token"]


@pytest.fixture(scope="session")
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="session")
def face_png():
    return _make_face_png()


@pytest.fixture(scope="session")
def created_booking():
    """Public booking creation with event_time."""
    payload = {
        "client_name": "TEST_Client_QR",
        "client_email": "test_client_qr@example.com",
        "client_phone": "9999999999",
        "event_type": "Wedding",
        "event_date": "2026-06-15",
        "event_time": "17:30",
        "location": "Kolkata",
        "message": "Test booking",
    }
    r = requests.post(f"{API}/bookings", json=payload)
    assert r.status_code == 200, f"booking create failed: {r.status_code} {r.text}"
    b = r.json()
    assert b["event_time"] == "17:30"
    assert "id" in b
    return b


@pytest.fixture(scope="session")
def created_event(admin_headers, created_booking, face_png):
    r = requests.post(
        f"{API}/events",
        headers=admin_headers,
        json={"title": "TEST_Event_AI", "booking_id": created_booking["id"], "match_threshold": 0.6},
    )
    assert r.status_code == 200, f"event create failed: {r.status_code} {r.text}"
    ev = r.json()
    # Pre-upload a face photo so client-portal-scoped worker sees it under loadscope
    files = {"file": ("seed_face.png", face_png, "image/png")}
    up = requests.post(f"{API}/events/{ev['id']}/photos", headers=admin_headers, files=files)
    assert up.status_code == 200, f"seed photo upload failed: {up.status_code} {up.text}"
    return ev


# ---------- auth ----------
class TestAuth:
    def test_login_success(self, admin_token):
        assert admin_token and isinstance(admin_token, str)

    def test_login_bad_pw(self):
        r = requests.post(f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong"})
        assert r.status_code == 401

    def test_me(self, admin_headers):
        r = requests.get(f"{API}/auth/me", headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["email"] == ADMIN_EMAIL
        assert r.json()["role"] == "admin"

    def test_me_unauth(self):
        r = requests.get(f"{API}/auth/me")
        assert r.status_code == 401


# ---------- bookings + event_time ----------
class TestBookings:
    def test_booking_persisted_with_time(self, admin_headers, created_booking):
        r = requests.get(f"{API}/bookings/{created_booking['id']}", headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["event_time"] == "17:30"

    def test_list_bookings_requires_auth(self):
        r = requests.get(f"{API}/bookings")
        assert r.status_code == 401

    def test_patch_event_time(self, admin_headers, created_booking):
        r = requests.patch(
            f"{API}/bookings/{created_booking['id']}",
            headers=admin_headers,
            json={"event_time": "19:00"},
        )
        assert r.status_code == 200
        assert r.json()["event_time"] == "19:00"
        # verify GET returns updated value
        r2 = requests.get(f"{API}/bookings/{created_booking['id']}", headers=admin_headers)
        assert r2.json()["event_time"] == "19:00"


# ---------- events ----------
class TestEvents:
    def test_events_admin_required(self):
        assert requests.get(f"{API}/events").status_code == 401
        assert requests.post(f"{API}/events", json={"title": "x"}).status_code == 401

    def test_list_events_shows_client_info(self, admin_headers, created_event, created_booking):
        r = requests.get(f"{API}/events", headers=admin_headers)
        assert r.status_code == 200
        found = next((e for e in r.json() if e["id"] == created_event["id"]), None)
        assert found is not None
        assert "photo_count" in found
        assert found.get("client_name") == created_booking["client_name"]

    def test_get_event(self, admin_headers, created_event):
        r = requests.get(f"{API}/events/{created_event['id']}", headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["title"] == "TEST_Event_AI"


# ---------- photo upload + face detection ----------
class TestEventPhotos:
    def test_upload_photo_detects_face(self, admin_headers, created_event, face_png):
        files = {"file": ("face.png", face_png, "image/png")}
        r = requests.post(
            f"{API}/events/{created_event['id']}/photos",
            headers=admin_headers,
            files=files,
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["face_count"] >= 1, f"expected at least 1 face, got {data['face_count']}"
        pytest.shared_photo_id = data["id"]

    def test_reject_bad_content_type(self, admin_headers, created_event):
        files = {"file": ("t.txt", b"hello", "text/plain")}
        r = requests.post(
            f"{API}/events/{created_event['id']}/photos",
            headers=admin_headers, files=files,
        )
        assert r.status_code == 415

    def test_list_photos(self, admin_headers, created_event):
        r = requests.get(f"{API}/events/{created_event['id']}/photos", headers=admin_headers)
        assert r.status_code == 200
        assert isinstance(r.json(), list)
        assert len(r.json()) >= 1


# ---------- QR & client portal ----------
class TestClientPortal:
    def test_generate_qr(self, admin_headers, created_event, created_booking):
        r = requests.post(
            f"{API}/events/{created_event['id']}/qr",
            headers=admin_headers,
            data={"booking_id": created_booking["id"]},
        )
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["qr"].startswith("data:image/png;base64,")
        assert len(j["qr"]) > 100
        assert "token" in j and j["token"]
        assert "/client/scan?token=" in j["url"]
        pytest.shared_client_token = j["token"]

    def test_client_scan(self):
        token = pytest.shared_client_token
        r = requests.post(f"{API}/client/scan", data={"token": token})
        assert r.status_code == 200
        assert r.json()["event"]["title"] == "TEST_Event_AI"

    def test_client_me_no_auth(self):
        assert requests.get(f"{API}/client/me").status_code == 401

    def test_client_me_with_header(self):
        token = pytest.shared_client_token
        r = requests.get(f"{API}/client/me", headers={"X-Client-Token": token})
        assert r.status_code == 200
        assert r.json()["event"]["title"] == "TEST_Event_AI"

    def test_client_photos_list(self):
        token = pytest.shared_client_token
        r = requests.get(f"{API}/client/me/photos", headers={"X-Client-Token": token})
        assert r.status_code == 200
        assert len(r.json()) >= 1

    def test_selfie_bad_content_type(self):
        token = pytest.shared_client_token
        files = {"file": ("t.txt", b"x", "text/plain")}
        r = requests.post(
            f"{API}/client/me/photos/search",
            headers={"X-Client-Token": token}, files=files,
        )
        assert r.status_code == 415

    def test_selfie_no_face(self):
        token = pytest.shared_client_token
        files = {"file": ("blank.png", _make_blank_png(), "image/png")}
        r = requests.post(
            f"{API}/client/me/photos/search",
            headers={"X-Client-Token": token}, files=files,
        )
        assert r.status_code == 400
        assert "exactly one face" in r.json().get("detail", "").lower()

    def test_selfie_match(self, face_png):
        token = pytest.shared_client_token
        files = {"file": ("selfie.png", face_png, "image/png")}
        r = requests.post(
            f"{API}/client/me/photos/search",
            headers={"X-Client-Token": token}, files=files,
        )
        assert r.status_code == 200, r.text
        j = r.json()
        assert "matches" in j and "threshold" in j
        assert j["total_faces_scanned"] >= 1
        assert len(j["matches"]) >= 1, "expected identical selfie to match its own uploaded photo"


# ---------- Album selection ----------
class TestAlbum:
    def test_get_empty_album(self):
        token = pytest.shared_client_token
        r = requests.get(f"{API}/client/me/album", headers={"X-Client-Token": token})
        assert r.status_code == 200
        # initially no selection
        assert r.json().get("photo_ids", []) == [] or "photo_ids" in r.json()

    def test_save_album(self, admin_headers, created_event):
        token = pytest.shared_client_token
        # get an event photo id
        ph = requests.get(f"{API}/events/{created_event['id']}/photos", headers=admin_headers).json()
        assert ph
        pids = [ph[0]["id"]]
        r = requests.post(
            f"{API}/client/me/album",
            headers={"X-Client-Token": token},
            json={"photo_ids": pids},
        )
        assert r.status_code == 200
        assert r.json()["photo_ids"] == pids
        assert r.json()["submitted"] is True

    def test_admin_list_selections(self, admin_headers, created_event):
        r = requests.get(
            f"{API}/events/{created_event['id']}/album-selections",
            headers=admin_headers,
        )
        assert r.status_code == 200
        assert len(r.json()) >= 1
        assert r.json()[0].get("client_name")


# ---------- regression ----------
class TestRegression:
    def test_packages_public(self):
        r = requests.get(f"{API}/packages")
        assert r.status_code == 200
        assert len(r.json()) >= 1

    def test_gallery_public(self):
        r = requests.get(f"{API}/gallery")
        assert r.status_code == 200

    def test_team_requires_auth(self, admin_headers):
        assert requests.get(f"{API}/team").status_code == 401
        r = requests.get(f"{API}/team", headers=admin_headers)
        assert r.status_code == 200

    def test_stats(self, admin_headers):
        r = requests.get(f"{API}/stats", headers=admin_headers)
        assert r.status_code == 200
        assert "total_bookings" in r.json()

    def test_package_crud(self, admin_headers):
        pkg = {"name": "TEST_Pkg", "category": "Wedding", "description": "x",
               "price": 100.0, "features": ["a"], "duration": "1h"}
        r = requests.post(f"{API}/packages", headers=admin_headers, json=pkg)
        assert r.status_code == 200
        pid = r.json()["id"]
        pkg["price"] = 200.0
        r2 = requests.put(f"{API}/packages/{pid}", headers=admin_headers, json=pkg)
        assert r2.status_code == 200 and r2.json()["price"] == 200.0
        assert requests.delete(f"{API}/packages/{pid}", headers=admin_headers).status_code == 200

    def test_gallery_upload(self, admin_headers):
        img = base64.b64encode(_make_blank_png()).decode()
        r = requests.post(
            f"{API}/gallery",
            headers=admin_headers,
            json={"title": "TEST_G", "category": "wedding", "image_data": f"data:image/png;base64,{img}"},
        )
        assert r.status_code == 200
        gid = r.json()["id"]
        requests.delete(f"{API}/gallery/{gid}", headers=admin_headers)


# ---------- auth boundaries (team role can't hit admin endpoints) ----------
class TestAuthBoundaries:
    @pytest.fixture(scope="class")
    def team_headers(self, admin_headers):
        # create a team member
        payload = {
            "name": "TEST_Team",
            "email": "TEST_teammember@example.com",
            "password": "teampass123",
            "role": "team",
        }
        # cleanup if exists
        r = requests.post(f"{API}/team", headers=admin_headers, json=payload)
        if r.status_code == 400:
            # already exists; find + delete then recreate
            existing = [m for m in requests.get(f"{API}/team", headers=admin_headers).json()
                        if m.get("email") == payload["email"]]
            for m in existing:
                requests.delete(f"{API}/team/{m['id']}", headers=admin_headers)
            r = requests.post(f"{API}/team", headers=admin_headers, json=payload)
        assert r.status_code == 200, r.text
        member_id = r.json()["id"]
        login = requests.post(f"{API}/auth/login", json={
            "email": payload["email"], "password": payload["password"],
        })
        assert login.status_code == 200
        token = login.json()["token"]
        yield {"Authorization": f"Bearer {token}"}
        requests.delete(f"{API}/team/{member_id}", headers=admin_headers)

    def test_team_cannot_create_event(self, team_headers):
        r = requests.post(f"{API}/events", headers=team_headers, json={"title": "nope"})
        assert r.status_code == 403

    def test_team_cannot_create_package(self, team_headers):
        r = requests.post(f"{API}/packages", headers=team_headers,
                          json={"name": "x", "category": "y", "description": "z",
                                "price": 1.0, "features": []})
        assert r.status_code == 403


# ---------- cleanup ----------
def test_zzz_cleanup(admin_headers, created_event, created_booking):
    # delete event (cascade)
    r = requests.delete(f"{API}/events/{created_event['id']}", headers=admin_headers)
    assert r.status_code == 200
    # ensure gone
    assert requests.get(f"{API}/events/{created_event['id']}", headers=admin_headers).status_code == 404
    # delete booking
    requests.delete(f"{API}/bookings/{created_booking['id']}", headers=admin_headers)
