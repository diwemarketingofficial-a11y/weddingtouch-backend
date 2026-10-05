from dotenv import load_dotenv
from pathlib import Path
import os
import io
import uuid
import base64
import asyncio
import logging
import tempfile
import requests
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
from fastapi.responses import JSONResponse, Response
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, EmailStr, Field, BeforeValidator, ConfigDict

from face_service import process_image_bytes, encode_selfie, match_encodings
from google_drive_storage import (
    extract_folder_id as drive_extract_folder_id,
    list_images as drive_list_images,
    download_file as drive_download_file,
    get_file_metadata as drive_get_file_metadata,
)

from r2_storage import (
    upload_file as r2_upload_file,
    download_file as r2_download_file,
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
    if user.get("role") not in ("admin", "super_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


async def require_super_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "super_admin":
        raise HTTPException(status_code=403, detail="Super admin access required")
    return user


# ---------- Models ----------
class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TeamMemberCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = "team"
    specialization: Optional[str] = None
    mobile: Optional[str] = None
    team_part: Optional[str] = None
    work_role: Optional[str] = None
    joining_date: Optional[str] = None
    main_balance: float = 0.0
    monthly_payment: Optional[float] = None  # legacy compatibility
    payment_details: Optional[str] = None


class TeamMemberUpdate(BaseModel):
    name: Optional[str] = None
    mobile: Optional[str] = None
    role: Optional[str] = None
    specialization: Optional[str] = None
    team_part: Optional[str] = None
    work_role: Optional[str] = None
    joining_date: Optional[str] = None
    main_balance: Optional[float] = None
    monthly_payment: Optional[float] = None  # legacy compatibility
    payment_details: Optional[str] = None


class TeamPasswordResetIn(BaseModel):
    password: str


class TeamPaymentIn(BaseModel):
    amount: float
    pay_date: str
    month: Optional[str] = None
    payment_method: Optional[str] = None
    reference: Optional[str] = None
    notes: Optional[str] = None

class TeamBalanceUpdateIn(BaseModel):
    main_balance: float
    notes: Optional[str] = None


class PackageIn(BaseModel):
    name: str
    category: str
    description: str
    price: float
    features: List[str] = []
    duration: Optional[str] = None
    image_data: Optional[str] = None
    image_key: Optional[str] = None


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
    assigned_to: Optional[str] = None  # legacy single user id
    assigned_to_ids: Optional[List[str]] = None  # multiple team member ids
    total_amount: Optional[float] = None
    advance_paid: Optional[float] = None
    notes: Optional[str] = None
    task_name: Optional[str] = None
    internal_note: Optional[str] = None
    event_date: Optional[str] = None
    event_time: Optional[str] = None
    location: Optional[str] = None


class GalleryImageIn(BaseModel):
    title: Optional[str] = ""
    category: str
    image_data: Optional[str] = None  # legacy/base64 compatibility
    image_key: Optional[str] = None


class GalleryImageUpdate(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    is_active: Optional[bool] = None


class PortfolioCategoryIn(BaseModel):
    name: str
    sort_order: int = 0
    is_active: bool = True


class PortfolioCategoryUpdate(BaseModel):
    name: Optional[str] = None
    sort_order: Optional[int] = None
    is_active: Optional[bool] = None


class PackageCategoryIn(BaseModel):
    name: str
    sort_order: int = 0
    is_active: bool = True


class PackageCategoryUpdate(BaseModel):
    name: Optional[str] = None
    sort_order: Optional[int] = None
    is_active: Optional[bool] = None



class ReelIn(BaseModel):
    title: str
    tag: Optional[str] = "WEDDING FILM"
    image_data: Optional[str] = None
    image_key: Optional[str] = None
    video_url: str
    is_active: bool = True

class ReelUpdate(BaseModel):
    title: Optional[str] = None
    tag: Optional[str] = None
    image_data: Optional[str] = None
    image_key: Optional[str] = None
    video_url: Optional[str] = None
    is_active: Optional[bool] = None

class BlogPostIn(BaseModel):
    title: str
    slug: str
    excerpt: str
    content: str
    image_data: Optional[str] = None  # DB fallback when R2 is unavailable
    image_key: Optional[str] = None
    tag: Optional[str] = "Guide"
    is_active: bool = True

class ReviewIn(BaseModel):
    name: str
    image_data: Optional[str] = None
    rating: int = Field(ge=1, le=5)
    review: str
    image_key: Optional[str] = None
    event_type: Optional[str] = None
    is_active: bool = True

class SuperAdminCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    mobile: Optional[str] = None


# ---------- App ----------
app = FastAPI()
api = APIRouter(prefix="/api")
print("ENV_CHECK TEST_RENDER_ENV:", bool(os.getenv("TEST_RENDER_ENV")))
print("ENV_CHECK GDRIVE_JSON_B64:", bool(os.getenv("GDRIVE_JSON_B64")))


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
        secure=os.getenv("COOKIE_SECURE", "true").lower() == "true",
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
async def list_team(admin: dict = Depends(require_admin)):
    members = await db.users.find({"role": {"$ne": "super_admin"}}).to_list(200)
    return [serialize(m) for m in members]


@api.get("/team/me/profile")
async def my_team_profile(user: dict = Depends(get_current_user)):
    """Read-only profile for the logged-in team member."""
    return user


@api.get("/team/me/payments")
async def my_team_payments(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    member_id = user["id"]
    member = await db.users.find_one({"_id": ObjectId(member_id)})
    if not member:
        raise HTTPException(status_code=404, detail="Team member not found")
    query = {"member_id": member_id}
    if from_date or to_date:
        date_query = {}
        if from_date:
            date_query["$gte"] = from_date
        if to_date:
            date_query["$lte"] = to_date
        query["pay_date"] = date_query
    history = await db.team_payments.find(query).sort([("pay_date", -1), ("created_at", -1)]).to_list(1000)
    all_payments = await db.team_payments.find({"member_id": member_id, "transaction_type": {"$in": [None, "payment"]}}).to_list(5000)
    total_paid = sum(float(p.get("amount") or 0) for p in all_payments)
    main_balance = float(member.get("main_balance", member.get("monthly_payment", 0)) or 0)
    return {
        "payments": [serialize(p) for p in history],
        "history": [serialize(p) for p in history],
        "main_balance": main_balance,
        "total_paid": total_paid,
        "current_due": max(main_balance - total_paid, 0),
        "total_received": total_paid,
    }


@api.post("/team")
async def create_team_member(body: TeamMemberCreate, admin: dict = Depends(require_admin)):
    email = body.email.lower()
    existing = await db.users.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")
    opening_balance = float(body.main_balance if body.main_balance is not None else (body.monthly_payment or 0))
    doc = {
        "name": body.name,
        "email": email,
        "password_hash": hash_password(body.password),
        "role": body.role if body.role in ("admin", "team") else "team",
        "specialization": body.specialization,
        "mobile": body.mobile,
        "team_part": body.team_part,
        "work_role": body.work_role,
        "joining_date": body.joining_date,
        "main_balance": opening_balance,
        "payment_details": body.payment_details,
        "created_at": now_iso(),
    }
    result = await db.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    if opening_balance != 0:
        await db.team_payments.insert_one({
            "member_id": str(result.inserted_id),
            "transaction_type": "balance_adjustment",
            "amount": opening_balance,
            "balance_before": 0.0,
            "balance_after": opening_balance,
            "pay_date": now_iso()[:10],
            "notes": "Opening main balance",
            "created_by": admin["id"],
            "created_at": now_iso(),
        })
    return serialize(doc)


@api.patch("/team/{member_id}")
async def update_team_member(member_id: str, body: TeamMemberUpdate, admin: dict = Depends(require_admin)):
    member = await db.users.find_one({"_id": ObjectId(member_id), "role": {"$ne": "super_admin"}})
    if not member:
        raise HTTPException(status_code=404, detail="Team member not found")
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    requested_balance = updates.pop("main_balance", None)
    legacy_balance = updates.pop("monthly_payment", None)
    if requested_balance is None and legacy_balance is not None:
        requested_balance = legacy_balance
    if updates.get("role") not in (None, "admin", "team"):
        updates["role"] = "team"
    if requested_balance is not None:
        old_balance = float(member.get("main_balance", member.get("monthly_payment", 0)) or 0)
        new_balance = float(requested_balance)
        if new_balance < old_balance:
            raise HTTPException(status_code=400, detail="Main balance can only be increased, not decreased")
        updates["main_balance"] = new_balance
        if new_balance > old_balance:
            await db.team_payments.insert_one({
                "member_id": member_id,
                "transaction_type": "balance_adjustment",
                "amount": new_balance - old_balance,
                "balance_before": old_balance,
                "balance_after": new_balance,
                "pay_date": now_iso()[:10],
                "notes": "Main balance increased by admin",
                "created_by": admin["id"],
                "created_at": now_iso(),
            })
    updates["updated_at"] = now_iso()
    await db.users.update_one({"_id": ObjectId(member_id)}, {"$set": updates, "$unset": {"monthly_payment": ""}})
    member = await db.users.find_one({"_id": ObjectId(member_id)})
    return serialize(member)


@api.get("/team/{member_id}/payments")
async def list_team_payments(member_id: str, admin: dict = Depends(require_admin)):
    member = await db.users.find_one({"_id": ObjectId(member_id), "role": {"$ne": "super_admin"}})
    if not member:
        raise HTTPException(status_code=404, detail="Team member not found")
    history = await db.team_payments.find({"member_id": member_id}).sort([("pay_date", -1), ("created_at", -1)]).to_list(1000)
    paid_docs = [p for p in history if p.get("transaction_type") in (None, "payment")]
    total_paid = sum(float(p.get("amount") or 0) for p in paid_docs)
    main_balance = float(member.get("main_balance", member.get("monthly_payment", 0)) or 0)
    return {
        "history": [serialize(x) for x in history],
        "payments": [serialize(x) for x in paid_docs],
        "main_balance": main_balance,
        "total_paid": total_paid,
        "current_due": max(main_balance - total_paid, 0),
    }


@api.post("/team/{member_id}/payments")
async def add_team_payment(member_id: str, body: TeamPaymentIn, admin: dict = Depends(require_admin)):
    member = await db.users.find_one({"_id": ObjectId(member_id), "role": {"$ne": "super_admin"}})
    if not member:
        raise HTTPException(status_code=404, detail="Team member not found")
    if body.amount <= 0:
        raise HTTPException(status_code=400, detail="Payment amount must be greater than zero")
    history = await db.team_payments.find({"member_id": member_id}).to_list(5000)
    total_paid = sum(float(p.get("amount") or 0) for p in history if p.get("transaction_type") in (None, "payment"))
    main_balance = float(member.get("main_balance", member.get("monthly_payment", 0)) or 0)
    due_before = max(main_balance - total_paid, 0)
    if body.amount > due_before:
        raise HTTPException(status_code=400, detail=f"Payment cannot exceed current due ({due_before:.2f})")
    doc = body.model_dump()
    doc.update({
        "member_id": member_id,
        "transaction_type": "payment",
        "due_before": due_before,
        "due_after": max(due_before - float(body.amount), 0),
        "created_by": admin["id"],
        "created_at": now_iso(),
    })
    result = await db.team_payments.insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize(doc)


@api.post("/team/{member_id}/reset-password")
async def reset_team_password(member_id: str, body: TeamPasswordResetIn, admin: dict = Depends(require_admin)):
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    result = await db.users.update_one({"_id": ObjectId(member_id), "role": {"$ne": "super_admin"}}, {"$set": {"password_hash": hash_password(body.password), "updated_at": now_iso()}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Team member not found")
    return {"ok": True}


# IMPORTANT: keep the literal /team/me/tasks route BEFORE /team/{member_id}/tasks.
# FastAPI matches routes in declaration order; otherwise "me" is treated as a member_id.
@api.get("/team/me/tasks")
async def my_team_tasks(user: dict = Depends(get_current_user)):
    member_id = user["id"]
    query = {"$or": [{"assigned_to": member_id}, {"assigned_to_ids": member_id}]}
    bookings = await db.bookings.find(query).to_list(500)
    bookings.sort(key=lambda b: (str(b.get("event_date") or ""), str(b.get("event_time") or "")))
    return [serialize(b) for b in bookings]


@api.get("/team/{member_id}/tasks")
async def team_member_tasks(member_id: str, user: dict = Depends(get_current_user)):
    # Team members can see their own tasks; admins can inspect anyone.
    if user.get("role") not in ("admin", "super_admin") and user.get("id") != member_id:
        raise HTTPException(status_code=403, detail="You can only view your own tasks")
    query = {"$or": [{"assigned_to": member_id}, {"assigned_to_ids": member_id}]}
    bookings = await db.bookings.find(query).to_list(500)
    bookings.sort(key=lambda b: (str(b.get("event_date") or ""), str(b.get("event_time") or "")))
    return [serialize(b) for b in bookings]


@api.delete("/team/{member_id}")
async def delete_team_member(member_id: str, admin: dict = Depends(require_admin)):
    if member_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    result = await db.users.delete_one({"_id": ObjectId(member_id), "role": {"$ne": "super_admin"}})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}
@api.post("/events/{event_id}/photos/presign")
async def presign_event_photo(
    event_id: str,
    file_name: str = Form(...),
    content_type: str = Form(...),
    user: dict = Depends(get_current_user),
):
    # Check that the event exists
    try:
        event_oid = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid event ID")

    event = await db.events.find_one({"_id": event_oid})

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Validate image type
    allowed_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    }

    if content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG/PNG/WEBP/HEIC accepted",
        )

    # Keep original extension
    extension = "jpg"

    if file_name and "." in file_name:
        extension = file_name.rsplit(".", 1)[-1].lower()

    # Generate unique R2 object name
    unique_name = f"{uuid.uuid4().hex}.{extension}"

    r2_key = f"events/{event_id}/photos/{unique_name}"

    # Generate temporary direct-upload URL
    try:
        upload_url = create_upload_url(
            key=r2_key,
            content_type=content_type,
            expires_in=3600,
        )
    except Exception as exc:
        logging.exception("Failed to create R2 upload URL")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create upload URL: {exc}",
        )

    return {
        "upload_url": upload_url,
        "r2_key": r2_key,
        "key": r2_key,
    }

# ---------- Optimized website media ----------
def _media_doc(doc: dict) -> dict:
    row = serialize(doc)
    key = row.get("image_key")
    if key:
        try:
            row["image_data"] = create_download_url(key, expires_in=86400)
        except Exception:
            logging.exception("Could not sign website media URL")
    return row

@api.post("/media/website")
async def upload_website_media(
    file: UploadFile = File(...),
    kind: str = Form("gallery"),
    admin: dict = Depends(require_admin),
):
    """Optimize website artwork to WebP and store it in R2.
    Originals are intentionally not served to public pages.
    """
    allowed = {"gallery", "portfolio", "service", "reel", "website", "blog", "review"}
    if kind not in allowed:
        raise HTTPException(status_code=400, detail="Invalid media type")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10MB or smaller")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        # Website artwork never needs camera-original dimensions.
        max_side = 1920 if kind == "website" else 1400
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, "white"); bg.paste(img, mask=img.getchannel("A")); img = bg
        out = io.BytesIO()
        img.save(out, format="WEBP", quality=78, method=4, optimize=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
    optimized = out.getvalue()
    key = f"website/{kind}/{uuid.uuid4().hex}.webp"
    try:
        await asyncio.to_thread(r2_upload_file, optimized, key, "image/webp")
        return {"image_key": key, "image_data": None, "image_url": create_download_url(key, expires_in=86400), "storage": "r2", "bytes": len(optimized)}
    except Exception as exc:
        # Keep website administration usable if R2 credentials/bucket/network are
        # temporarily unavailable. Optimized WebP is small enough for MongoDB and
        # existing public serializers already support image_data.
        logging.exception("R2 website-media upload failed; using database fallback")
        data_url = "data:image/webp;base64," + base64.b64encode(optimized).decode("ascii")
        return {"image_key": None, "image_data": data_url, "image_url": data_url, "storage": "database", "storage_warning": str(exc), "bytes": len(optimized)}

# ---------- Package Routes ----------
@api.get("/packages")
async def list_packages():
    packages = await db.packages.find({}).to_list(200)
    return [_media_doc(p) for p in packages]


@api.post("/packages")
async def create_package(body: PackageIn, admin: dict = Depends(require_admin)):
    doc = body.model_dump()
    doc["created_at"] = now_iso()
    result = await db.packages.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _media_doc(doc)


@api.put("/packages/{package_id}")
async def update_package(package_id: str, body: PackageIn, admin: dict = Depends(require_admin)):
    await db.packages.update_one({"_id": ObjectId(package_id)}, {"$set": body.model_dump()})
    updated = await db.packages.find_one({"_id": ObjectId(package_id)})
    if not updated:
        raise HTTPException(status_code=404, detail="Not found")
    return _media_doc(updated)


@api.delete("/packages/{package_id}")
async def delete_package(package_id: str, admin: dict = Depends(require_admin)):
    result = await db.packages.delete_one({"_id": ObjectId(package_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ---------- Booking PDF + WhatsApp helpers ----------
def _money(value) -> str:
    try:
        return f"INR {float(value or 0):,.2f}"
    except Exception:
        return "INR 0.00"


def _booking_pdf_bytes(booking: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    total = float(booking.get("total_amount") or 0)
    paid = float(booking.get("advance_paid") or 0)
    due = max(total - paid, 0)
    rows = [
        ["Client", booking.get("client_name", "")],
        ["Phone", booking.get("client_phone", "")],
        ["Event", booking.get("event_type", "")],
        ["Event date", booking.get("event_date", "")],
        ["Location", booking.get("location", "") or ""],
        ["Status", booking.get("status", "")],
        ["Total booking amount", _money(total)],
        ["Payment amount", _money(paid)],
        ["Due amount", _money(due)],
    ]
    table = Table(rows, colWidths=[55*mm, 105*mm])
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("BACKGROUND", (0,0), (0,-1), colors.whitesmoke),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 7),
    ]))
    story = [Paragraph("Wedding Touch - Booking Update", styles["Title"]), Spacer(1, 8*mm), table]
    doc.build(story)
    return buf.getvalue()


def _send_whatsapp_booking_sync(booking: dict) -> dict:
    token = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
    api_version = os.getenv("WHATSAPP_API_VERSION", "v23.0").strip()
    phone = "".join(ch for ch in str(booking.get("client_phone") or "") if ch.isdigit())
    if phone.startswith("0") and os.getenv("WHATSAPP_DEFAULT_COUNTRY_CODE"):
        phone = os.getenv("WHATSAPP_DEFAULT_COUNTRY_CODE").strip() + phone.lstrip("0")
    if not token or not phone_number_id or not phone:
        return {"status": "not_configured"}

    total = float(booking.get("total_amount") or 0)
    paid = float(booking.get("advance_paid") or 0)
    due = max(total - paid, 0)
    message = (
        f"Wedding Touch booking update\n"
        f"Client: {booking.get('client_name', '')}\n"
        f"Event date: {booking.get('event_date', '')}\n"
        f"Total booking amount: {_money(total)}\n"
        f"Payment amount: {_money(paid)}\n"
        f"Due amount: {_money(due)}\n"
        f"Status: {booking.get('status', '')}"
    )
    base = f"https://graph.facebook.com/{api_version}/{phone_number_id}"
    headers = {"Authorization": f"Bearer {token}"}
    text_resp = requests.post(
        f"{base}/messages",
        headers={**headers, "Content-Type": "application/json"},
        json={"messaging_product": "whatsapp", "to": phone, "type": "text", "text": {"body": message}},
        timeout=30,
    )
    text_resp.raise_for_status()

    pdf = _booking_pdf_bytes(booking)
    media_resp = requests.post(
        f"{base}/media",
        headers=headers,
        data={"messaging_product": "whatsapp", "type": "application/pdf"},
        files={"file": (f"booking-{booking.get('_id', 'update')}.pdf", pdf, "application/pdf")},
        timeout=60,
    )
    media_resp.raise_for_status()
    media_id = media_resp.json()["id"]
    doc_resp = requests.post(
        f"{base}/messages",
        headers={**headers, "Content-Type": "application/json"},
        json={"messaging_product": "whatsapp", "to": phone, "type": "document", "document": {"id": media_id, "filename": "Wedding-Touch-Booking.pdf", "caption": "Your latest Wedding Touch booking details"}},
        timeout=30,
    )
    doc_resp.raise_for_status()
    return {"status": "sent"}


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
        # attach assigned team member names (supports legacy single assignment)
        assigned_ids = b.get("assigned_to_ids") or ([b.get("assigned_to")] if b.get("assigned_to") else [])
        assigned_ids = [x for x in assigned_ids if x]
        names = []
        for member_id in assigned_ids:
            try:
                mem = await db.users.find_one({"_id": ObjectId(member_id)})
                if mem:
                    names.append(mem.get("name"))
            except Exception:
                pass
        b["assigned_names"] = names
        b["assigned_name"] = ", ".join(names) if names else None
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

    # Every successful booking-sheet update can notify the client. If WhatsApp
    # credentials are not configured, the booking update still succeeds.
    whatsapp = {"status": "not_configured"}
    try:
        whatsapp = await asyncio.to_thread(_send_whatsapp_booking_sync, updated)
    except Exception as exc:
        logging.exception("WhatsApp booking notification failed")
        whatsapp = {"status": "failed", "detail": str(exc)}
    result = serialize(updated)
    result["whatsapp"] = whatsapp
    return result


@api.get("/bookings/{booking_id}/pdf")
async def booking_pdf(booking_id: str, user: dict = Depends(get_current_user)):
    booking = await db.bookings.find_one({"_id": ObjectId(booking_id)})
    if not booking:
        raise HTTPException(status_code=404, detail="Not found")
    pdf = await asyncio.to_thread(_booking_pdf_bytes, booking)
    return Response(content=pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="booking-{booking_id}.pdf"'})


@api.delete("/bookings/{booking_id}")
async def delete_booking(booking_id: str, admin: dict = Depends(require_admin)):
    result = await db.bookings.delete_one({"_id": ObjectId(booking_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ---------- Gallery + Website Master Data Routes ----------
DEFAULT_PORTFOLIO_CATEGORIES = ["Wedding", "Pre-wedding", "Portrait", "Event", "Commercial"]

@api.get("/master/portfolio-categories")
async def list_portfolio_categories(include_inactive: bool = False):
    query = {} if include_inactive else {"is_active": {"$ne": False}}
    rows = await db.portfolio_categories.find(query).sort([("sort_order", 1), ("name", 1)]).to_list(200)
    if not rows:
        return [{"id": f"default-{i}", "name": n, "sort_order": i + 1, "is_active": True} for i, n in enumerate(DEFAULT_PORTFOLIO_CATEGORIES)]
    return [_media_doc(x) for x in rows]

@api.post("/master/portfolio-categories")
async def create_portfolio_category(body: PortfolioCategoryIn, super_admin: dict = Depends(require_super_admin)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Category name is required")
    if await db.portfolio_categories.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}}):
        raise HTTPException(status_code=400, detail="Category already exists")
    doc = {"name": name, "sort_order": body.sort_order, "is_active": body.is_active, "created_at": now_iso()}
    result = await db.portfolio_categories.insert_one(doc); doc["_id"] = result.inserted_id
    return serialize(doc)

@api.patch("/master/portfolio-categories/{category_id}")
async def update_portfolio_category(category_id: str, body: PortfolioCategoryUpdate, super_admin: dict = Depends(require_super_admin)):
    if category_id.startswith("default-"):
        raise HTTPException(status_code=400, detail="Save a new category first; default categories cannot be edited until initialized")
    old = await db.portfolio_categories.find_one({"_id": ObjectId(category_id)})
    if not old: raise HTTPException(status_code=404, detail="Category not found")
    updates = {k:v for k,v in body.model_dump().items() if v is not None}
    if "name" in updates:
        updates["name"] = updates["name"].strip()
        if not updates["name"]: raise HTTPException(status_code=400, detail="Category name is required")
        await db.gallery.update_many({"category": old["name"]}, {"$set": {"category": updates["name"]}})
    updates["updated_at"] = now_iso()
    await db.portfolio_categories.update_one({"_id": old["_id"]}, {"$set": updates})
    return serialize(await db.portfolio_categories.find_one({"_id": old["_id"]}))

@api.delete("/master/portfolio-categories/{category_id}")
async def delete_portfolio_category(category_id: str, super_admin: dict = Depends(require_super_admin)):
    if category_id.startswith("default-"): raise HTTPException(status_code=400, detail="Default category cannot be deleted until master data is initialized")
    cat = await db.portfolio_categories.find_one({"_id": ObjectId(category_id)})
    if not cat: raise HTTPException(status_code=404, detail="Category not found")
    count = await db.gallery.count_documents({"category": cat["name"]})
    if count: raise HTTPException(status_code=400, detail=f"Category has {count} gallery image(s). Move or delete those images first, or disable the category.")
    await db.portfolio_categories.delete_one({"_id": cat["_id"]})
    return {"ok": True}

DEFAULT_PACKAGE_CATEGORIES = ["Wedding", "Pre-wedding", "Portrait", "Event", "Commercial"]

@api.get("/master/package-categories")
async def list_package_categories(include_inactive: bool = False, user: dict = Depends(get_current_user)):
    query = {} if (include_inactive and user.get("role") == "super_admin") else {"is_active": {"$ne": False}}
    rows = await db.package_categories.find(query).sort([("sort_order", 1), ("name", 1)]).to_list(200)
    if not rows:
        return [{"id": f"default-package-{i}", "name": n, "sort_order": i + 1, "is_active": True} for i, n in enumerate(DEFAULT_PACKAGE_CATEGORIES)]
    return [_media_doc(x) for x in rows]

@api.post("/master/package-categories")
async def create_package_category(body: PackageCategoryIn, super_admin: dict = Depends(require_super_admin)):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Category name is required")
    if await db.package_categories.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}}):
        raise HTTPException(status_code=400, detail="Category already exists")
    doc = {"name": name, "sort_order": body.sort_order, "is_active": body.is_active, "created_at": now_iso()}
    result = await db.package_categories.insert_one(doc); doc["_id"] = result.inserted_id
    return serialize(doc)

@api.patch("/master/package-categories/{category_id}")
async def update_package_category(category_id: str, body: PackageCategoryUpdate, super_admin: dict = Depends(require_super_admin)):
    if category_id.startswith("default-package-"):
        raise HTTPException(status_code=400, detail="Default categories are initialized when the backend starts. Restart once, then edit them.")
    old = await db.package_categories.find_one({"_id": ObjectId(category_id)})
    if not old:
        raise HTTPException(status_code=404, detail="Category not found")
    updates = {k:v for k,v in body.model_dump().items() if v is not None}
    if "name" in updates:
        updates["name"] = updates["name"].strip()
        if not updates["name"]:
            raise HTTPException(status_code=400, detail="Category name is required")
        duplicate = await db.package_categories.find_one({"_id": {"$ne": old["_id"]}, "name": {"$regex": f"^{updates['name']}$", "$options": "i"}})
        if duplicate:
            raise HTTPException(status_code=400, detail="Category already exists")
        await db.packages.update_many({"category": old["name"]}, {"$set": {"category": updates["name"]}})
    updates["updated_at"] = now_iso()
    await db.package_categories.update_one({"_id": old["_id"]}, {"$set": updates})
    return serialize(await db.package_categories.find_one({"_id": old["_id"]}))

@api.delete("/master/package-categories/{category_id}")
async def delete_package_category(category_id: str, super_admin: dict = Depends(require_super_admin)):
    if category_id.startswith("default-package-"):
        raise HTTPException(status_code=400, detail="Default category cannot be deleted until master data is initialized")
    cat = await db.package_categories.find_one({"_id": ObjectId(category_id)})
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")
    count = await db.packages.count_documents({"category": cat["name"]})
    if count:
        raise HTTPException(status_code=400, detail=f"Category is used by {count} package(s). Move those packages first, or disable the category.")
    await db.package_categories.delete_one({"_id": cat["_id"]})
    return {"ok": True}

@api.get("/super-admin/users")
async def list_super_admins(super_admin: dict = Depends(require_super_admin)):
    return [serialize(x) for x in await db.users.find({"role":"super_admin"}).to_list(50)]

@api.post("/super-admin/users")
async def create_super_admin(body: SuperAdminCreate, super_admin: dict = Depends(require_super_admin)):
    if len(body.password) < 8: raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    email=body.email.lower()
    if await db.users.find_one({"email":email}): raise HTTPException(status_code=400, detail="Email already exists")
    doc={"name":body.name,"email":email,"mobile":body.mobile,"password_hash":hash_password(body.password),"role":"super_admin","created_at":now_iso()}
    r=await db.users.insert_one(doc); doc["_id"]=r.inserted_id
    return serialize(doc)

# ---------- Blog ----------
@api.get("/blogs")
async def list_blogs():
    rows = await db.blog_posts.find({"is_active": {"$ne": False}}).sort("created_at", -1).to_list(100)
    return [_media_doc(x) for x in rows]

@api.get("/blogs/admin/all")
async def list_blogs_admin(admin: dict = Depends(require_admin)):
    rows = await db.blog_posts.find({}).sort("created_at", -1).to_list(500)
    return [_media_doc(x) for x in rows]

@api.get("/blogs/{slug}")
async def get_blog(slug: str):
    row = await db.blog_posts.find_one({"slug": slug, "is_active": {"$ne": False}})
    if not row: raise HTTPException(status_code=404, detail="Blog post not found")
    return _media_doc(row)

@api.post("/blogs")
async def create_blog(body: BlogPostIn, admin: dict = Depends(require_admin)):
    if await db.blog_posts.find_one({"slug": body.slug}): raise HTTPException(status_code=400, detail="Slug already exists")
    doc=body.model_dump(); doc["created_at"]=now_iso()
    r=await db.blog_posts.insert_one(doc); doc["_id"]=r.inserted_id
    return _media_doc(doc)

@api.delete("/blogs/{post_id}")
async def delete_blog(post_id: str, admin: dict = Depends(require_admin)):
    r=await db.blog_posts.delete_one({"_id":ObjectId(post_id)})
    if not r.deleted_count: raise HTTPException(status_code=404, detail="Blog post not found")
    return {"ok":True}

# ---------- Client Reviews ----------
@api.get("/reviews")
async def list_reviews():
    rows=await db.reviews.find({"is_active":{"$ne":False}}).sort("created_at",-1).to_list(100)
    return [_media_doc(x) for x in rows]

@api.post("/reviews")
async def create_review(body: ReviewIn):
    doc=body.model_dump(); doc["created_at"]=now_iso()
    r=await db.reviews.insert_one(doc); doc["_id"]=r.inserted_id
    return _media_doc(doc)

@api.get("/reviews/admin/all")
async def list_reviews_admin(admin: dict = Depends(require_admin)):
    rows = await db.reviews.find({}).sort("created_at", -1).to_list(500)
    return [_media_doc(x) for x in rows]

@api.patch("/reviews/{review_id}")
async def update_review(review_id: str, body: dict, admin: dict = Depends(require_admin)):
    updates = {}
    if "is_active" in body: updates["is_active"] = bool(body["is_active"])
    if not updates: raise HTTPException(status_code=400, detail="No supported fields supplied")
    updates["updated_at"] = now_iso()
    result = await db.reviews.update_one({"_id": ObjectId(review_id)}, {"$set": updates})
    if not result.matched_count: raise HTTPException(status_code=404, detail="Review not found")
    return _media_doc(await db.reviews.find_one({"_id": ObjectId(review_id)}))

@api.delete("/reviews/{review_id}")
async def delete_review(review_id: str, admin: dict = Depends(require_admin)):
    r=await db.reviews.delete_one({"_id":ObjectId(review_id)})
    if not r.deleted_count: raise HTTPException(status_code=404, detail="Review not found")
    return {"ok":True}

# ---------- Website Gallery (separate from Portfolio) ----------
@api.get("/website-gallery")
async def list_website_gallery(category: Optional[str] = None):
    query = {"is_active": {"$ne": False}}
    if category: query["category"] = category
    images = await db.website_gallery.find(query).sort("created_at", -1).to_list(500)
    return [_media_doc(i) for i in images]

@api.get("/website-gallery/admin/all")
async def list_website_gallery_admin(admin: dict = Depends(require_admin)):
    return [_media_doc(i) for i in await db.website_gallery.find({}).sort("created_at", -1).to_list(200)]

@api.post("/website-gallery")
async def upload_website_gallery(body: GalleryImageIn, admin: dict = Depends(require_admin)):
    doc = body.model_dump(); doc["is_active"] = True; doc["created_at"] = now_iso()
    result = await db.website_gallery.insert_one(doc); doc["_id"] = result.inserted_id
    return _media_doc(doc)

@api.patch("/website-gallery/{image_id}")
async def update_website_gallery(image_id: str, body: GalleryImageUpdate, admin: dict = Depends(require_admin)):
    updates={k:v for k,v in body.model_dump().items() if v is not None}; updates["updated_at"]=now_iso()
    result=await db.website_gallery.update_one({"_id":ObjectId(image_id)},{"$set":updates})
    if result.matched_count==0: raise HTTPException(status_code=404, detail="Not found")
    return _media_doc(await db.website_gallery.find_one({"_id":ObjectId(image_id)}))

@api.delete("/website-gallery/{image_id}")
async def delete_website_gallery(image_id: str, admin: dict = Depends(require_admin)):
    result = await db.website_gallery.delete_one({"_id": ObjectId(image_id)})
    if result.deleted_count == 0: raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}

@api.get("/gallery")
async def list_gallery(category: Optional[str] = None):
    query = {"is_active": {"$ne": False}}
    if category: query["category"] = category
    images = await db.gallery.find(query).sort("created_at", -1).to_list(500)
    return [_media_doc(i) for i in images]

@api.get("/gallery/admin/all")
async def list_gallery_admin(admin: dict = Depends(require_admin)):
    return [_media_doc(i) for i in await db.gallery.find({}).sort("created_at", -1).to_list(200)]

@api.post("/gallery")
async def upload_gallery(body: GalleryImageIn, admin: dict = Depends(require_admin)):
    doc = body.model_dump(); doc["is_active"] = True; doc["created_at"] = now_iso()
    result = await db.gallery.insert_one(doc); doc["_id"] = result.inserted_id
    return _media_doc(doc)

@api.patch("/gallery/{image_id}")
async def update_gallery(image_id: str, body: GalleryImageUpdate, admin: dict = Depends(require_admin)):
    updates={k:v for k,v in body.model_dump().items() if v is not None}; updates["updated_at"]=now_iso()
    result=await db.gallery.update_one({"_id":ObjectId(image_id)},{"$set":updates})
    if result.matched_count==0: raise HTTPException(status_code=404, detail="Not found")
    return _media_doc(await db.gallery.find_one({"_id":ObjectId(image_id)}))

@api.delete("/gallery/{image_id}")
async def delete_gallery(image_id: str, admin: dict = Depends(require_admin)):
    result = await db.gallery.delete_one({"_id": ObjectId(image_id)})
    if result.deleted_count == 0: raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


# ---------- Reels (public + Super Admin management) ----------
@api.get("/reels")
async def list_reels():
    rows = await db.reels.find({"is_active": {"$ne": False}}).sort("created_at", -1).to_list(100)
    return [_media_doc(x) for x in rows]

@api.get("/reels/admin/all")
async def list_reels_admin(admin: dict = Depends(require_super_admin)):
    return [_media_doc(x) for x in await db.reels.find({}).sort("created_at", -1).to_list(100)]

@api.post("/reels")
async def create_reel(body: ReelIn, admin: dict = Depends(require_super_admin)):
    doc=body.model_dump(); doc["created_at"]=now_iso(); r=await db.reels.insert_one(doc); doc["_id"]=r.inserted_id; return _media_doc(doc)

@api.patch("/reels/{reel_id}")
async def update_reel(reel_id: str, body: ReelUpdate, admin: dict = Depends(require_super_admin)):
    updates={k:v for k,v in body.model_dump().items() if v is not None}; updates["updated_at"]=now_iso()
    await db.reels.update_one({"_id":ObjectId(reel_id)},{"$set":updates})
    row=await db.reels.find_one({"_id":ObjectId(reel_id)});
    if not row: raise HTTPException(status_code=404, detail="Not found")
    return _media_doc(row)

@api.delete("/reels/{reel_id}")
async def delete_reel(reel_id: str, admin: dict = Depends(require_super_admin)):
    r=await db.reels.delete_one({"_id":ObjectId(reel_id)})
    if not r.deleted_count: raise HTTPException(status_code=404, detail="Not found")
    return {"ok":True}


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


class DriveFolderIn(BaseModel):
    folder_url: str


class AlbumSelectionIn(BaseModel):
    photo_ids: List[str]

class PhotoRegisterIn(BaseModel):
    r2_key: str
    file_name: str
    content_type: str

def _client_token_secret() -> str:
    # Reuse main JWT secret but with a distinct token "type" claim
    return os.environ["JWT_SECRET"]


def _make_client_token(event_id: str, booking_id: str, access_mode: str = "guest", minutes: int = 60 * 24 * 30) -> str:
    payload = {
        "type": "client",
        "event_id": event_id,
        "booking_id": booking_id,
        "access_mode": access_mode,
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
async def create_event(body: EventCreate, user: dict = Depends(get_current_user)):
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
async def list_events(user: dict = Depends(get_current_user)):
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
async def get_event(event_id: str, user: dict = Depends(get_current_user)):
    e = await db.events.find_one({"_id": ObjectId(event_id)})
    if not e:
        raise HTTPException(status_code=404, detail="Event not found")
    return serialize(e)
def _event_photo_url(photo: dict) -> Optional[str]:
    if photo.get("storage") == "google_drive" and photo.get("drive_file_id"):
        api_base = os.getenv("PUBLIC_API_BASE_URL", "https://api.weddingtouch.in").rstrip("/")
        return f"{api_base}/api/photos/{str(photo.get('_id') or photo.get('id'))}/content"
    if photo.get("r2_key"):
        try:
            return create_download_url(photo["r2_key"], expires_in=3600)
        except Exception:
            logging.exception("Failed to create R2 download URL")
    return None


async def process_drive_photo(photo_id: str, event_id: str, drive_file_id: str):
    async with FACE_PROCESSING_SEMAPHORE:
        try:
            await db.event_photos.update_one(
                {"_id": ObjectId(photo_id)},
                {"$set": {"processing_status": "processing", "processing_error": None}},
            )
            raw = await asyncio.to_thread(drive_download_file, drive_file_id)
            width, height, locations, encodings = await asyncio.to_thread(process_image_bytes, raw)
            now = now_iso()
            await db.face_encodings.delete_many({"photo_id": photo_id})
            docs = []
            for enc, loc in zip(encodings, locations):
                t, r, b, l = loc
                docs.append({
                    "event_id": event_id,
                    "photo_id": photo_id,
                    "model": "face_recognition-dlib-128-v1",
                    "embedding": [float(x) for x in enc],
                    "location": {"top": int(t), "right": int(r), "bottom": int(b), "left": int(l)},
                    "created_at": now,
                })
            if docs:
                await db.face_encodings.insert_many(docs)
            await db.event_photos.update_one(
                {"_id": ObjectId(photo_id)},
                {"$set": {
                    "width": width, "height": height, "face_count": len(encodings),
                    "processing_status": "ready", "processing_error": None, "processed_at": now,
                }},
            )
        except Exception as exc:
            logging.exception("Google Drive face processing failed for %s", photo_id)
            await db.event_photos.update_one(
                {"_id": ObjectId(photo_id)},
                {"$set": {"processing_status": "failed", "processing_error": str(exc)}},
            )


@api.get("/debug/google-drive-env")
async def debug_google_drive_env(admin: dict = Depends(require_admin)):
    raw = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
    secret_path = os.getenv(
        "GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE",
        "/etc/secrets/google-drive-service-account.json",
    )
    secret_exists = os.path.isfile(secret_path)
    secret_size = os.path.getsize(secret_path) if secret_exists else 0

    try:
        secret_files = sorted(os.listdir("/etc/secrets")) if os.path.isdir("/etc/secrets") else []
    except Exception:
        secret_files = []

    matching_keys = sorted(
        key for key in os.environ.keys()
        if "GOOGLE" in key.upper() or "DRIVE" in key.upper() or key == "TEST_RENDER_ENV"
    )

    return {
        "configured_env": bool(raw and raw.strip()),
        "test_render_env_present": os.getenv("TEST_RENDER_ENV") is not None,
        "env_length": len(raw) if raw else 0,
        "secret_path": secret_path,
        "secret_exists": secret_exists,
        "secret_size": secret_size,
        "secret_files": secret_files,
        "matching_env_keys": matching_keys,
        "render_service_name": os.getenv("RENDER_SERVICE_NAME"),
        "render_external_hostname": os.getenv("RENDER_EXTERNAL_HOSTNAME"),
        "render_git_commit": os.getenv("RENDER_GIT_COMMIT"),
    }

@api.post("/events/{event_id}/drive-folder")
async def connect_event_drive_folder(
    event_id: str,
    body: DriveFolderIn,
    admin: dict = Depends(require_admin),
):
    try:
        event_oid = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid event ID")
    if not await db.events.find_one({"_id": event_oid}):
        raise HTTPException(status_code=404, detail="Event not found")
    try:
        folder_id = drive_extract_folder_id(body.folder_url)
        files = await asyncio.to_thread(drive_list_images, folder_id)
    except Exception as exc:
        logging.exception("Could not read Google Drive folder")
        raise HTTPException(status_code=400, detail=f"Could not read Google Drive folder: {exc}")

    await db.events.update_one(
        {"_id": event_oid},
        {"$set": {"drive_folder_id": folder_id, "drive_folder_url": body.folder_url, "drive_connected_at": now_iso()}},
    )

    added = 0
    skipped = 0
    for item in files:
        file_id = item["id"]
        existing = await db.event_photos.find_one({"event_id": event_id, "drive_file_id": file_id})
        if existing:
            skipped += 1
            continue
        doc = {
            "event_id": event_id,
            "filename": item.get("name") or "drive-photo",
            "content_type": item.get("mimeType") or "application/octet-stream",
            "storage": "google_drive",
            "drive_file_id": file_id,
            "drive_folder_id": folder_id,
            "width": None,
            "height": None,
            "face_count": 0,
            "processing_status": "pending",
            "processing_error": None,
            "created_at": now_iso(),
        }
        result = await db.event_photos.insert_one(doc)
        asyncio.create_task(process_drive_photo(str(result.inserted_id), event_id, file_id))
        added += 1

    return {"ok": True, "folder_id": folder_id, "found": len(files), "added": added, "skipped": skipped}


@api.post("/events/{event_id}/drive-sync")
async def sync_event_drive_folder(event_id: str, admin: dict = Depends(require_admin)):
    event = await db.events.find_one({"_id": ObjectId(event_id)})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    folder_id = event.get("drive_folder_id")
    if not folder_id:
        raise HTTPException(status_code=400, detail="No Google Drive folder connected")
    return await connect_event_drive_folder(
        event_id,
        DriveFolderIn(folder_url=folder_id),
        admin,
    )


@api.get("/photos/{photo_id}/content")
async def get_photo_content(photo_id: str, request: Request):
    # A photo may be requested either by an authenticated admin/team user or
    # by a client holding the event QR cookie/token.
    event_id = None
    try:
        user = await get_current_user(request)
        if user:
            event_id = None
    except HTTPException:
        try:
            claims = await get_client(request)
            event_id = claims.get("event_id")
        except HTTPException:
            raise HTTPException(status_code=401, detail="Not authenticated")

    photo = await db.event_photos.find_one({"_id": ObjectId(photo_id)})
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    if event_id and photo.get("event_id") != event_id:
        raise HTTPException(status_code=403, detail="Photo not available for this event")
    if photo.get("storage") != "google_drive" or not photo.get("drive_file_id"):
        raise HTTPException(status_code=400, detail="Photo is not stored in Google Drive")
    try:
        raw = await asyncio.to_thread(drive_download_file, photo["drive_file_id"])
    except Exception as exc:
        logging.exception("Could not download Google Drive photo")
        raise HTTPException(status_code=502, detail=f"Could not download photo: {exc}")
    return Response(
        content=raw,
        media_type=photo.get("content_type") or "application/octet-stream",
        headers={"Cache-Control": "private, max-age=300"},
    )



# Limit face processing so many uploads do not overload the server
FACE_PROCESSING_SEMAPHORE = asyncio.Semaphore(2)


async def process_registered_photo(
    photo_id: str,
    event_id: str,
    r2_key: str,
):
    async with FACE_PROCESSING_SEMAPHORE:
        try:
            # Mark processing
            await db.event_photos.update_one(
                {"_id": ObjectId(photo_id)},
                {
                    "$set": {
                        "processing_status": "processing",
                        "processing_error": None,
                    }
                },
            )

            logging.info("Starting face processing: %s", photo_id)

            # Download original image from R2
            raw = await asyncio.to_thread(
                r2_download_file,
                r2_key,
            )

            # Detect faces + create encodings
            width, height, locations, encodings = await asyncio.to_thread(
                process_image_bytes,
                raw,
            )

            now = now_iso()

            # Remove old encodings if photo is reprocessed
            await db.face_encodings.delete_many({
                "photo_id": photo_id
            })

            enc_docs = []

            for enc, loc in zip(encodings, locations):
                t, r, b, l = loc

                enc_docs.append({
                    "event_id": event_id,
                    "photo_id": photo_id,
                    "model": "face_recognition-dlib-128-v1",
                    "embedding": [float(x) for x in enc],
                    "location": {
                        "top": int(t),
                        "right": int(r),
                        "bottom": int(b),
                        "left": int(l),
                    },
                    "created_at": now,
                })

            if enc_docs:
                await db.face_encodings.insert_many(enc_docs)

            # Update photo
            await db.event_photos.update_one(
                {"_id": ObjectId(photo_id)},
                {
                    "$set": {
                        "width": width,
                        "height": height,
                        "face_count": len(encodings),
                        "processing_status": "ready",
                        "processing_error": None,
                        "processed_at": now,
                    }
                },
            )

            logging.info(
                "Face processing complete: %s - %d faces",
                photo_id,
                len(encodings),
            )

        except Exception as exc:
            logging.exception(
                "Face processing failed for photo %s",
                photo_id,
            )

            await db.event_photos.update_one(
                {"_id": ObjectId(photo_id)},
                {
                    "$set": {
                        "processing_status": "failed",
                        "processing_error": str(exc),
                    }
                },
            )
@api.post("/events/{event_id}/photos/register")
async def register_event_photo(
    event_id: str,
    body: PhotoRegisterIn,
    user: dict = Depends(get_current_user),
):
    # Validate event ID
    try:
        event_oid = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid event ID")

    event = await db.events.find_one({"_id": event_oid})

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Security: object must belong to this event
    expected_prefix = f"events/{event_id}/photos/"

    if not body.r2_key.startswith(expected_prefix):
        raise HTTPException(status_code=400, detail="Invalid R2 key")

    # Avoid registering same R2 object twice
    existing = await db.event_photos.find_one({
        "r2_key": body.r2_key
    })

    if existing:
        out = serialize(existing)
        out["image_url"] = create_download_url(
            body.r2_key,
            expires_in=3600,
        )
        return out

    # Create MongoDB record immediately.
    # Face recognition will happen separately.
    photo_doc = {
        "event_id": event_id,
        "filename": body.file_name,
        "content_type": body.content_type,
        "r2_key": body.r2_key,
        "width": None,
        "height": None,
        "face_count": 0,
        "processing_status": "pending",
        "processing_error": None,
        "created_at": now_iso(),
    }

    result = await db.event_photos.insert_one(photo_doc)

    photo_doc["_id"] = result.inserted_id

    # Start face recognition in background
    asyncio.create_task(
        process_registered_photo(
            str(result.inserted_id),
            event_id,
            body.r2_key,
        )
    )

    out = serialize(photo_doc)

    out["image_url"] = create_download_url(
        body.r2_key,
        expires_in=3600,
    )

    return out
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
    user: dict = Depends(get_current_user),
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
    user: dict = Depends(get_current_user),
):
    photos = (
        await db.event_photos.find({"event_id": event_id})
        .sort("created_at", -1)
        .to_list(1000)
    )

    result = []

    for photo in photos:
        photo = serialize(photo)

        photo["image_url"] = _event_photo_url(photo)

        result.append(photo)

    return result


@api.get("/events/{event_id}/photo-summary")
async def get_event_photo_summary(event_id: str, user: dict = Depends(get_current_user)):
    try:
        event_oid = ObjectId(event_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid event id")
    if not await db.events.find_one({"_id": event_oid}):
        raise HTTPException(status_code=404, detail="Event not found")
    photo_count = await db.event_photos.count_documents({"event_id": event_id})
    pipeline = [{"$match":{"event_id":event_id}}, {"$group":{"_id":None,"total":{"$sum":{"$ifNull":["$face_count",0]}}}}]
    result = await db.event_photos.aggregate(pipeline).to_list(1)
    face_count = result[0]["total"] if result else 0
    pending_count = await db.event_photos.count_documents({"event_id":event_id,"processing_status":{"$in":["pending","processing"]}})
    failed_count = await db.event_photos.count_documents({"event_id":event_id,"processing_status":"failed"})
    return {"photo_count":photo_count,"face_count":face_count,"pending_count":pending_count,"failed_count":failed_count}


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
    access_mode: str = Form("guest"),
    user: dict = Depends(get_current_user),
):
    ev = await db.events.find_one({"_id": ObjectId(event_id)})
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    b = await db.bookings.find_one({"_id": ObjectId(booking_id)})
    if not b:
        raise HTTPException(status_code=404, detail="Booking not found")

    access_mode = "album" if access_mode == "album" else "guest"
    token = _make_client_token(event_id=event_id, booking_id=booking_id, access_mode=access_mode)
    base_url = os.environ.get("PORTAL_BASE_URL", "").rstrip("/")
    url = f"{base_url}/client/scan?token={token}" if base_url else f"/client/scan?token={token}"

    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return {"qr": data_url, "url": url, "token": token, "client_name": b.get("client_name"), "access_mode": access_mode}


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
            "access_mode": payload.get("access_mode", "guest"),
        }
    )
    resp.set_cookie(
        key="client_token",
        value=token,
        httponly=True,
        secure=os.getenv("COOKIE_SECURE", "true").lower() == "true",
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
        "access_mode": claims.get("access_mode", "guest"),
    }


@api.post("/client/logout")
async def client_logout():
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie("client_token", path="/")
    return resp


@api.get("/client/me/photos")
async def client_list_photos(claims: dict = Depends(get_client)):
    """Return all event photos only to the dedicated Album Selection QR."""
    if claims.get("access_mode") != "album":
        raise HTTPException(status_code=403, detail="Use the Album Selection QR")
    photos = (
        await db.event_photos.find({"event_id": claims["event_id"]})
        .sort("created_at", -1)
        .to_list(2000)
    )
    out = []
    for p in photos:
        item = serialize(p)
        if p.get("r2_key"):
            item["image_url"] = create_download_url(p["r2_key"], expires_in=3600)
        out.append(item)
    return out


@api.post("/client/me/photos/search")
async def client_search_by_selfie(
    file: UploadFile = File(...),
    claims: dict = Depends(get_client),
):
    if claims.get("access_mode", "guest") == "album":
        raise HTTPException(status_code=403, detail="Album QR does not allow selfie search")
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
        if p.get("r2_key"):
            sp["image_url"] = create_download_url(p["r2_key"], expires_in=3600)
        out.append(sp)
    out.sort(key=lambda x: x.get("distance", 999))
    return {"matches": out, "threshold": threshold, "total_faces_scanned": len(stored)}


# ---------- Album Selection (Bride/Groom portal) ----------
@api.get("/client/me/album")
async def client_get_album(claims: dict = Depends(get_client)):
    if claims.get("access_mode") != "album":
        raise HTTPException(status_code=403, detail="Use the Album Selection QR")
    sel = await db.album_selections.find_one(
        {"event_id": claims["event_id"], "booking_id": claims["booking_id"]}
    )
    return serialize(sel) if sel else {"photo_ids": [], "submitted": False}


@api.post("/client/me/album")
async def client_save_album(body: AlbumSelectionIn, claims: dict = Depends(get_client)):
    if claims.get("access_mode") != "album":
        raise HTTPException(status_code=403, detail="Use the Album Selection QR")
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
async def admin_list_album_selections(event_id: str, user: dict = Depends(get_current_user)):
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


@api.get("/events/{event_id}/album-selections/{selection_id}/photos")
async def admin_album_selection_photos(event_id: str, selection_id: str, user: dict = Depends(get_current_user)):
    sel = await db.album_selections.find_one({"_id": ObjectId(selection_id), "event_id": event_id})
    if not sel:
        raise HTTPException(status_code=404, detail="Album selection not found")
    ids = []
    for pid in sel.get("photo_ids", []):
        try:
            ids.append(ObjectId(pid))
        except Exception:
            pass
    photos = await db.event_photos.find({"_id": {"$in": ids}, "event_id": event_id}).to_list(2000) if ids else []
    by_id = {str(p["_id"]): p for p in photos}
    out = []
    for pid in sel.get("photo_ids", []):
        p = by_id.get(pid)
        if not p:
            continue
        item = serialize(p)
        if p.get("r2_key"):
            item["image_url"] = create_download_url(p["r2_key"], expires_in=3600)
        out.append(item)
    return out


app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://weddingtouch.in",
        "https://www.weddingtouch.in",
    ],
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
    await db.website_gallery.create_index("category")
    await db.blog_posts.create_index("slug", unique=True)
    await db.reviews.create_index("created_at")
    await db.event_photos.create_index("event_id")
    await db.face_encodings.create_index("event_id")
    await db.face_encodings.create_index("photo_id")
    await db.album_selections.create_index([("event_id", 1), ("booking_id", 1)])
    await db.team_payments.create_index([("member_id", 1), ("month", 1)])

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

    # Seed package category master data
    await db.package_categories.create_index("name", unique=True)
    if await db.package_categories.count_documents({}) == 0:
        existing_package_categories = await db.packages.distinct("category")
        names = [x for x in existing_package_categories if x] or DEFAULT_PACKAGE_CATEGORIES
        for idx, name in enumerate(names, 1):
            await db.package_categories.insert_one({"name": name, "sort_order": idx, "is_active": True, "created_at": now_iso()})

    # Seed portfolio category master data
    await db.portfolio_categories.create_index("name", unique=True)
    if await db.portfolio_categories.count_documents({}) == 0:
        for idx, name in enumerate(DEFAULT_PORTFOLIO_CATEGORIES, 1):
            await db.portfolio_categories.insert_one({"name": name, "sort_order": idx, "is_active": True, "created_at": now_iso()})

    # Optional first Super Admin bootstrap. Configure both variables in Render/local .env.
    super_email = os.environ.get("SUPER_ADMIN_EMAIL", "").strip().lower()
    super_password = os.environ.get("SUPER_ADMIN_PASSWORD", "").strip()
    if super_email and super_password:
        existing_super = await db.users.find_one({"email": super_email})
        if not existing_super:
            await db.users.insert_one({"name": os.environ.get("SUPER_ADMIN_NAME", "Wedding Touch Owner"), "email": super_email, "password_hash": hash_password(super_password), "role": "super_admin", "created_at": now_iso()})
        elif existing_super.get("role") != "super_admin":
            await db.users.update_one({"_id": existing_super["_id"]}, {"$set": {"role": "super_admin"}})

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

