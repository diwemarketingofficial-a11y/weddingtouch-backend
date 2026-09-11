from dotenv import load_dotenv
from pathlib import Path
import os
import io
import uuid
import base64
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Annotated

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

import bcrypt
import jwt
import numpy as np
import qrcode

from PIL import Image
from bson import ObjectId
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr, Field, BeforeValidator, ConfigDict

from face_service import process_image_bytes, encode_selfie, match_encodings
from r2_storage import (
    upload_file as r2_upload_file,
    delete_file as r2_delete_file,
    create_download_url,
    create_upload_url,
)

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/weddingtouch")

DB_NAME = os.getenv("DB_NAME", "weddingtouch")
SECRET_KEY = os.getenv("SECRET_KEY", "fallback_secret")

# Connect to MongoDB
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]


# ---------- Helpers ----------
def obj_id_str(v):
    if isinstance(v, ObjectId):
        return str(v)
    return str(v)


PyObjectId = Annotated[str, BeforeValidator(obj_id_str)]

JWT_ALGO = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def get_jwt_secret() -> str:
    return os.environ["JWT_SECRET"]


def create_access_token(user_id: str, email: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
        "type": "access",
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALGO)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize(doc: dict) -> dict:
    if not doc:
        return doc
    doc = dict(doc)
    if "_id" in doc:
        doc["id"] = str(doc.pop("_id"))
    doc.pop("password_hash", None)
    return doc


# ---------- Auth Dependency ----------
async def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGO])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return serialize(user)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ---------- Models ----------
class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TeamMemberCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = "team"  # "team" or "admin"
    specialization: Optional[str] = None


class PackageIn(BaseModel):
    name: str
    category: str
    description: str
    price: float
    features: List[str] = []
    duration: Optional[str] = None


class BookingCreate(BaseModel):
    client_name: str
    client_email: EmailStr
    client_phone: str
    event_type: str
    event_date: str  # ISO date string
    event_time: Optional[str] = None  # HH:MM 24h
    location: Optional[str] = None
    message: Optional[str] = None
    package_id: Optional[str] = None


class BookingUpdate(BaseModel):
    status: Optional[str] = None  # inquiry, confirmed, in_progress, completed, cancelled
    assigned_to: Optional[str] = None  # user id
    total_amount: Optional[float] = None
    advance_paid: Optional[float] = None
    notes: Optional[str] = None
    event_date: Optional[str] = None
    event_time: Optional[str] = None
    location: Optional[str] = None


class GalleryImageIn(BaseModel):
    title: str
    category: str
    image_data: str  # base64 data URL


# ---------- App ----------
app = FastAPI()
api = APIRouter(prefix="/api")


# ---------- Auth Routes ----------
@api.post("/auth/login")
async def login(body: LoginIn):
    email = body.email.lower()
    user = await db.users.find_one({"email": email})
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(str(user["_id"]), user["email"], user.get("role", "team"))
    resp = JSONResponse(content={"user": serialize(user), "token": token})
    resp.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
        path="/",
    )


    return resp


@api.post("/auth/logout")
async def logout():
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie("access_token", path="/")
    return resp


@api.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return user


# ---------- Team Routes ----------
@api.get("/team")
async def list_team(user: dict = Depends(get_current_user)):
    members = await db.users.find({}).to_list(200)
    return [serialize(m) for m in members]


@api.post("/team")
async def create_team_member(body: TeamMemberCreate, admin: dict = Depends(require_admin)):
    email = body.email.lower()
    existing = await db.users.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")
    doc = {
        "name": body.name,
        "email": email,
        "password_hash": hash_password(body.password),
        "role": body.role if body.role in ("admin", "team") else "team",
        "specialization": body.specialization,
        "created_at": now_iso(),
    }
    result = await db.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize(doc)


@api.delete("/team/{member_id}")
async def delete_team_member(member_id: str, admin: dict = Depends(require_admin)):
    if member_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    result = await db.users.delete_one({"_id": ObjectId(member_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ---------- Package Routes ----------
@api.get("/packages")
async def list_packages():
    packages = await db.packages.find({}).to_list(200)
    return [serialize(p) for p in packages]


@api.post("/packages")
async def create_package(body: PackageIn, admin: dict = Depends(require_admin)):
    doc = body.model_dump()
    doc["created_at"] = now_iso()
    result = await db.packages.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize(doc)


@api.put("/packages/{package_id}")
async def update_package(package_id: str, body: PackageIn, admin: dict = Depends(require_admin)):
    await db.packages.update_one({"_id": ObjectId(package_id)}, {"$set": body.model_dump()})
    updated = await db.packages.find_one({"_id": ObjectId(package_id)})
    if not updated:
        raise HTTPException(status_code=404, detail="Not found")
    return serialize(updated)


@api.delete("/packages/{package_id}")
async def delete_package(package_id: str, admin: dict = Depends(require_admin)):
    result = await db.packages.delete_one({"_id": ObjectId(package_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ---------- Booking Routes ----------
@api.post("/bookings")
async def create_booking(body: BookingCreate):
    doc = body.model_dump()
    doc["status"] = "inquiry"
    doc["total_amount"] = 0.0
    doc["advance_paid"] = 0.0
    doc["assigned_to"] = None
    doc["notes"] = ""
    doc["created_at"] = now_iso()
    result = await db.bookings.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize(doc)


@api.get("/bookings")
async def list_bookings(user: dict = Depends(get_current_user)):
    bookings = await db.bookings.find({}).sort("created_at", -1).to_list(500)
    result = []
    for b in bookings:
        b = serialize(b)
        # attach assigned member name
        if b.get("assigned_to"):
            try:
                mem = await db.users.find_one({"_id": ObjectId(b["assigned_to"])})
                if mem:
                    b["assigned_name"] = mem.get("name")
            except Exception:
                pass
        result.append(b)
    return result


@api.get("/bookings/{booking_id}")
async def get_booking(booking_id: str, user: dict = Depends(get_current_user)):
    b = await db.bookings.find_one({"_id": ObjectId(booking_id)})
    if not b:
        raise HTTPException(status_code=404, detail="Not found")
    return serialize(b)


@api.patch("/bookings/{booking_id}")
async def update_booking(booking_id: str, body: BookingUpdate, user: dict = Depends(get_current_user)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    updates["updated_at"] = now_iso()
    await db.bookings.update_one({"_id": ObjectId(booking_id)}, {"$set": updates})
    updated = await db.bookings.find_one({"_id": ObjectId(booking_id)})
    if not updated:
        raise HTTPException(status_code=404, detail="Not found")
    return serialize(updated)


@api.delete("/bookings/{booking_id}")
async def delete_booking(booking_id: str, admin: dict = Depends(require_admin)):
    result = await db.bookings.delete_one({"_id": ObjectId(booking_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ---------- Gallery Routes ----------
@api.get("/gallery")
async def list_gallery(category: Optional[str] = None):
    query = {"category": category} if category else {}
    images = await db.gallery.find(query).sort("created_at", -1).to_list(500)
    return [serialize(i) for i in images]


@api.post("/gallery")
async def upload_gallery(body: GalleryImageIn, admin: dict = Depends(require_admin)):
    doc = body.model_dump()
    doc["created_at"] = now_iso()
    result = await db.gallery.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize(doc)


@api.delete("/gallery/{image_id}")
async def delete_gallery(image_id: str, admin: dict = Depends(require_admin)):
    result = await db.gallery.delete_one({"_id": ObjectId(image_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ---------- Stats (Admin dashboard) ----------
@api.get("/stats")
async def stats(user: dict = Depends(get_current_user)):
    total = await db.bookings.count_documents({})
    inquiry = await db.bookings.count_documents({"status": "inquiry"})
    confirmed = await db.bookings.count_documents({"status": "confirmed"})
    completed = await db.bookings.count_documents({"status": "completed"})
    # revenue: sum of advance_paid
    pipeline = [{"$group": {"_id": None, "revenue": {"$sum": "$advance_paid"}, "billed": {"$sum": "$total_amount"}}}]
    agg = await db.bookings.aggregate(pipeline).to_list(1)
    revenue = agg[0]["revenue"] if agg else 0
    billed = agg[0]["billed"] if agg else 0
    return {
        "total_bookings": total,
        "inquiries": inquiry,
        "confirmed": confirmed,
        "completed": completed,
        "revenue_collected": revenue,
        "total_billed": billed,
        "outstanding": max(0, billed - revenue),
    }


@api.get("/")
async def root():
    return {"message": "Lens Studio API"}


# ---------- Events (AI photo delivery) ----------
class EventCreate(BaseModel):
    booking_id: Optional[str] = None
    title: str
    match_threshold: float = 0.52  # face-match distance threshold


class AlbumSelectionIn(BaseModel):
    photo_ids: List[str]


def _client_token_secret() -> str:
    # Reuse main JWT secret but with a distinct token "type" claim
    return os.environ["JWT_SECRET"]


def _make_client_token(event_id: str, booking_id: str, minutes: int = 60 * 24 * 30) -> str:
    payload = {
        "type": "client",
        "event_id": event_id,
        "booking_id": booking_id,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=minutes),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, _client_token_secret(), algorithm=JWT_ALGO)


def _verify_client_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, _client_token_secret(), algorithms=[JWT_ALGO])
        if payload.get("type") != "client":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_client(request: Request) -> dict:
    token = request.cookies.get("client_token")
    if not token:
        auth = request.headers.get("X-Client-Token", "")
        if auth:
            token = auth
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated as client")
    return _verify_client_token(token)


@api.post("/events")
async def create_event(body: EventCreate, admin: dict = Depends(require_admin)):
    doc = {
        "title": body.title,
        "booking_id": body.booking_id,
        "match_threshold": body.match_threshold,
        "created_at": now_iso(),
    }
    result = await db.events.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize(doc)


@api.get("/events")
async def list_events(admin: dict = Depends(require_admin)):
    events = await db.events.find({}).sort("created_at", -1).to_list(500)
    out = []
    for e in events:
        e = serialize(e)
        # attach photo count and booking info
        e["photo_count"] = await db.event_photos.count_documents({"event_id": e["id"]})
        if e.get("booking_id"):
            try:
                b = await db.bookings.find_one({"_id": ObjectId(e["booking_id"])})
                if b:
                    e["client_name"] = b.get("client_name")
                    e["client_email"] = b.get("client_email")
            except Exception:
                pass
        out.append(e)
    return out


@api.get("/events/{event_id}")
async def get_event(event_id: str, admin: dict = Depends(require_admin)):
    e = await db.events.find_one({"_id": ObjectId(event_id)})
    if not e:
        raise HTTPException(status_code=404, detail="Event not found")
    return serialize(e)


@api.delete("/events/{event_id}")
async def delete_event(event_id: str, admin: dict = Depends(require_admin)):
    # cascade-delete related photos, encodings, selections
    await db.event_photos.delete_many({"event_id": event_id})
    await db.face_encodings.delete_many({"event_id": event_id})
    await db.album_selections.delete_many({"event_id": event_id})
    result = await db.events.delete_one({"_id": ObjectId(event_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"ok": True}


@api.post("/events/{event_id}/photos")
async def upload_event_photo(
    event_id: str,
    file: UploadFile = File(...),
    admin: dict = Depends(require_admin),
):
    ev = await db.events.find_one({"_id": ObjectId(event_id)})

    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")

    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
        "image/heic",
        "image/heif",
    ):
        raise HTTPException(
            status_code=415,
            detail="Only JPEG/PNG/WEBP/HEIC accepted",
        )

    # Read maximum 20 MB
    raw = await file.read(20 * 1024 * 1024 + 1)

    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image too large (max 20MB)",
        )

    # -----------------------------
    # FACE PROCESSING
    # -----------------------------
    try:
        width, height, locations, encodings = await asyncio.to_thread(
            process_image_bytes,
            raw,
        )

    except Exception as exc:
        logging.exception("Face processing failed")

        raise HTTPException(
            status_code=400,
            detail=f"Image processing failed: {exc}",
        )

    # -----------------------------
    # CREATE UNIQUE R2 KEY
    # -----------------------------
    extension = "jpg"

    if file.filename and "." in file.filename:
        extension = file.filename.rsplit(".", 1)[-1].lower()

    unique_name = f"{uuid.uuid4().hex}.{extension}"

    r2_key = f"events/{event_id}/photos/{unique_name}"

    # -----------------------------
    # UPLOAD PHOTO TO CLOUDFLARE R2
    # -----------------------------
    try:
        await asyncio.to_thread(
            r2_upload_file,
            raw,
            r2_key,
            file.content_type,
        )

    except Exception as exc:
        logging.exception("R2 upload failed")

        raise HTTPException(
            status_code=500,
            detail=f"Photo storage failed: {exc}",
        )

    # -----------------------------
    # SAVE PHOTO METADATA TO MONGODB
    # -----------------------------
    now = now_iso()

    photo_doc = {
        "event_id": event_id,
        "filename": file.filename or "upload",
        "content_type": file.content_type,
        "r2_key": r2_key,
        "width": width,
        "height": height,
        "face_count": len(encodings),
        "created_at": now,
    }

    result = await db.event_photos.insert_one(photo_doc)

    pid = str(result.inserted_id)

    # -----------------------------
    # SAVE FACE EMBEDDINGS
    # -----------------------------
    enc_docs = []

    for enc, loc in zip(encodings, locations):
        t, r, b, l = loc

        enc_docs.append(
            {
                "event_id": event_id,
                "photo_id": pid,
                "model": "face_recognition-dlib-128-v1",
                "embedding": [float(x) for x in enc],
                "location": {
                    "top": int(t),
                    "right": int(r),
                    "bottom": int(b),
                    "left": int(l),
                },
                "created_at": now,
            }
        )

    if enc_docs:
        await db.face_encodings.insert_many(enc_docs)

    # -----------------------------
    # RESPONSE
    # -----------------------------
    photo_doc["_id"] = result.inserted_id

    out = serialize(photo_doc)

    # Temporary URL for displaying the photo
    out["image_url"] = create_download_url(
        r2_key,
        expires_in=3600,
    )

    return out
@api.get("/events/{event_id}/photos")
async def list_event_photos(
    event_id: str,
    admin: dict = Depends(require_admin),
):
    photos = (
        await db.event_photos.find({"event_id": event_id})
        .sort("created_at", -1)
        .to_list(1000)
    )

    result = []

    for photo in photos:
        photo = serialize(photo)

        # Generate temporary Cloudflare R2 URL
        r2_key = photo.get("r2_key")

        if r2_key:
            try:
                photo["image_url"] = create_download_url(
                    r2_key,
                    expires_in=3600,
                )
            except Exception:
                logging.exception("Failed to create R2 download URL")
                photo["image_url"] = None
        else:
            photo["image_url"] = None

        result.append(photo)

    return result


@api.delete("/events/{event_id}/photos/{photo_id}")
async def delete_event_photo(event_id: str, photo_id: str, admin: dict = Depends(require_admin)):
    await db.face_encodings.delete_many({"photo_id": photo_id})
    result = await db.event_photos.delete_one({"_id": ObjectId(photo_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Photo not found")
    return {"ok": True}


@api.post("/events/{event_id}/qr")
async def generate_client_qr(
    event_id: str,
    booking_id: str = Form(...),
    admin: dict = Depends(require_admin),
):
    ev = await db.events.find_one({"_id": ObjectId(event_id)})
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    b = await db.bookings.find_one({"_id": ObjectId(booking_id)})
    if not b:
        raise HTTPException(status_code=404, detail="Booking not found")

    token = _make_client_token(event_id=event_id, booking_id=booking_id)
    base_url = os.environ.get("PORTAL_BASE_URL", "").rstrip("/")
    url = f"{base_url}/client/scan?token={token}" if base_url else f"/client/scan?token={token}"

    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return {"qr": data_url, "url": url, "token": token, "client_name": b.get("client_name")}


@api.post("/client/scan")
async def client_scan(token: str = Form(...)):
    payload = _verify_client_token(token)
    ev = await db.events.find_one({"_id": ObjectId(payload["event_id"])})
    b = await db.bookings.find_one({"_id": ObjectId(payload["booking_id"])})
    if not ev or not b:
        raise HTTPException(status_code=404, detail="Event or booking missing")
    resp = JSONResponse(
        content={
            "event": {"id": str(ev["_id"]), "title": ev.get("title")},
            "booking": {"id": str(b["_id"]), "client_name": b.get("client_name")},
            "token": token,
        }
    )
    resp.set_cookie(
        key="client_token",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
        path="/",
    )
    return resp


@api.get("/client/me")
async def client_me(claims: dict = Depends(get_client)):
    ev = await db.events.find_one({"_id": ObjectId(claims["event_id"])})
    b = await db.bookings.find_one({"_id": ObjectId(claims["booking_id"])})
    return {
        "event": serialize(ev) if ev else None,
        "booking": serialize(b) if b else None,
    }


@api.post("/client/logout")
async def client_logout():
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie("client_token", path="/")
    return resp


@api.get("/client/me/photos")
async def client_list_photos(claims: dict = Depends(get_client)):
    """Return ALL event photos for the client's event (used by album selection)."""
    photos = (
        await db.event_photos.find({"event_id": claims["event_id"]})
        .sort("created_at", -1)
        .to_list(2000)
    )
    return [serialize(p) for p in photos]


@api.post("/client/me/photos/search")
async def client_search_by_selfie(
    file: UploadFile = File(...),
    claims: dict = Depends(get_client),
):
    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
        "image/heic",
        "image/heif",
    ):
        raise HTTPException(status_code=415, detail="Only JPEG/PNG/WEBP/HEIC selfies accepted")
    raw = await file.read(15 * 1024 * 1024 + 1)
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Selfie too large")

    try:
        encoding, count = await asyncio.to_thread(encode_selfie, raw)
    except Exception as exc:
        logging.exception("Selfie processing failed")
        raise HTTPException(status_code=400, detail=f"Selfie processing failed: {exc}")

    if encoding is None:
        raise HTTPException(
            status_code=400,
            detail=f"Selfie must contain exactly one face (found {count})",
        )

    # Get event's threshold
    ev = await db.events.find_one({"_id": ObjectId(claims["event_id"])})
    threshold = float(ev.get("match_threshold", 0.52)) if ev else 0.52

    stored = []
    async for row in db.face_encodings.find({"event_id": claims["event_id"]}):
        stored.append((row["photo_id"], row["embedding"]))

    photo_scores = await asyncio.to_thread(match_encodings, encoding, stored, threshold)

    if not photo_scores:
        return {"matches": [], "threshold": threshold, "total_faces_scanned": len(stored)}

    ids = [ObjectId(pid) for pid in photo_scores.keys()]
    photos = await db.event_photos.find({"_id": {"$in": ids}}).to_list(500)
    out = []
    for p in photos:
        sp = serialize(p)
        sp["distance"] = photo_scores.get(str(p["_id"]))
        out.append(sp)
    out.sort(key=lambda x: x.get("distance", 999))
    return {"matches": out, "threshold": threshold, "total_faces_scanned": len(stored)}


# ---------- Album Selection (Bride/Groom portal) ----------
@api.get("/client/me/album")
async def client_get_album(claims: dict = Depends(get_client)):
    sel = await db.album_selections.find_one(
        {"event_id": claims["event_id"], "booking_id": claims["booking_id"]}
    )
    return serialize(sel) if sel else {"photo_ids": [], "submitted": False}


@api.post("/client/me/album")
async def client_save_album(body: AlbumSelectionIn, claims: dict = Depends(get_client)):
    now = now_iso()
    await db.album_selections.update_one(
        {"event_id": claims["event_id"], "booking_id": claims["booking_id"]},
        {
            "$set": {
                "event_id": claims["event_id"],
                "booking_id": claims["booking_id"],
                "photo_ids": body.photo_ids,
                "submitted": True,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    sel = await db.album_selections.find_one(
        {"event_id": claims["event_id"], "booking_id": claims["booking_id"]}
    )
    return serialize(sel)


@api.get("/events/{event_id}/album-selections")
async def admin_list_album_selections(event_id: str, admin: dict = Depends(require_admin)):
    sels = await db.album_selections.find({"event_id": event_id}).to_list(200)
    out = []
    for s in sels:
        s = serialize(s)
        if s.get("booking_id"):
            try:
                b = await db.bookings.find_one({"_id": ObjectId(s["booking_id"])})
                if b:
                    s["client_name"] = b.get("client_name")
                    s["client_email"] = b.get("client_email")
            except Exception:
                pass
        out.append(s)
    return out


app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Startup: seed admin + default packages ----------
DEFAULT_PACKAGES = [
    {
        "name": "Signature Wedding",
        "category": "Wedding",
        "description": "Full-day wedding coverage with cinematic edits, second shooter, and premium album.",
        "price": 2500.0,
        "features": ["10 hours coverage", "2 photographers", "500+ edited photos", "Premium album", "Online gallery"],
        "duration": "10 hours",
    },
    {
        "name": "Pre-Wedding Story",
        "category": "Pre-wedding",
        "description": "Romantic pre-wedding shoot in your favourite locations with cinematic storytelling.",
        "price": 850.0,
        "features": ["4 hours session", "2 outfits", "100+ edited photos", "1 minute highlight reel"],
        "duration": "4 hours",
    },
    {
        "name": "Editorial Portrait",
        "category": "Portrait",
        "description": "Studio or on-location portrait session with editorial-grade retouching.",
        "price": 450.0,
        "features": ["2 hours session", "Studio or outdoor", "30+ edited photos", "High-res files"],
        "duration": "2 hours",
    },
    {
        "name": "Event Coverage",
        "category": "Event",
        "description": "Corporate events, birthdays, and parties captured with a documentary approach.",
        "price": 700.0,
        "features": ["Up to 5 hours", "150+ edited photos", "Next-day preview", "Online gallery"],
        "duration": "5 hours",
    },
    {
        "name": "Commercial Brand",
        "category": "Commercial",
        "description": "Product, brand, and lifestyle imagery tailored to your marketing needs.",
        "price": 1200.0,
        "features": ["Custom scope", "Concept & moodboard", "Full commercial license", "Fast turnaround"],
        "duration": "Custom",
    },
]


@app.on_event("startup")
async def startup_event():
    await db.users.create_index("email", unique=True)
    await db.bookings.create_index("created_at")
    await db.gallery.create_index("category")
    await db.event_photos.create_index("event_id")
    await db.face_encodings.create_index("event_id")
    await db.face_encodings.create_index("photo_id")
    await db.album_selections.create_index([("event_id", 1), ("booking_id", 1)])

    # Seed admin
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@lensstudio.com").lower()
    admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
    existing = await db.users.find_one({"email": admin_email})
    if not existing:
        await db.users.insert_one(
            {
                "name": "Studio Admin",
                "email": admin_email,
                "password_hash": hash_password(admin_password),
                "role": "admin",
                "specialization": "Owner",
                "created_at": now_iso(),
            }
        )
    else:
        # keep password in sync if changed
        if not verify_password(admin_password, existing["password_hash"]):
            await db.users.update_one(
                {"email": admin_email}, {"$set": {"password_hash": hash_password(admin_password)}}
            )

    # Seed packages if empty
    pkg_count = await db.packages.count_documents({})
    if pkg_count == 0:
        for p in DEFAULT_PACKAGES:
            p2 = dict(p)
            p2["created_at"] = now_iso()
            await db.packages.insert_one(p2)


@app.on_event("shutdown")
async def shutdown_event():
    client.close()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lensstudio")

