import os
import uuid
import hashlib
import hmac
import time
import base64
import struct
import re
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse, parse_qs

import numpy as np
import cv2
import httpx
from libsql_client import create_client
from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    APP_TITLE,
    APP_VERSION,
    UPSTREAM_BASE_URL,
    UPSTREAM_TOKEN,
    UPSTREAM_JOB_ID,
    CREDIT_PER_CONFIRMED,
    DB_URL,
    DB_TOKEN,
    HOST,
    PORT,
    RELOAD,
    APP_SECRET,
    SESSION_COOKIE,
    RECOVERY_DOMAIN,
)

UPSTREAM_BASE_URL = UPSTREAM_BASE_URL.rstrip("/")

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# =========================
# DB / Connection
# =========================
db_client = create_client(DB_URL, auth_token=DB_TOKEN)


def _rows_to_dicts(result) -> List[Dict[str, Any]]:
    if not result or not result.rows:
        return []
    columns = []
    if result.columns:
        columns = [c.name if hasattr(c, "name") else c for c in result.columns]
    if not columns:
        return [dict(enumerate(row)) for row in result.rows]
    return [dict(zip(columns, row)) for row in result.rows]


def db_fetchone(sql: str, args: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
    result = db_client.execute(sql, args)
    rows = _rows_to_dicts(result)
    return rows[0] if rows else None


def db_fetchall(sql: str, args: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    result = db_client.execute(sql, args)
    return _rows_to_dicts(result)


def db_execute(sql: str, args: Tuple[Any, ...] = ()) -> None:
    db_client.execute(sql, args)


def db_fetchone_client(client, sql: str, args: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
    result = client.execute(sql, args)
    rows = _rows_to_dicts(result)
    return rows[0] if rows else None


def _table_has_column(table: str, column: str) -> bool:
    rows = db_fetchall(f"PRAGMA table_info({table});")
    cols = {r.get("name") for r in rows}
    return column in cols


def with_write_tx(fn):
    attempts = 5
    for i in range(attempts):
        try:
            db_execute("BEGIN;")
            result = fn(db_client)
            db_execute("COMMIT;")
            return result
        except Exception as e:
            try:
                db_execute("ROLLBACK;")
            except Exception:
                pass
            msg = str(e).lower()
            if "database is locked" in msg or "locked" in msg:
                if i < attempts - 1:
                    time.sleep(0.15 * (i + 1))
                    continue
            raise


def db_init() -> None:
    # Users
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # Admins
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS admins (
            admin_id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # Sessions
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,          -- 'user' | 'admin'
            principal_id TEXT NOT NULL,  -- user_id/admin_id
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    db_execute("CREATE INDEX IF NOT EXISTS ix_sessions_kind ON sessions(kind);")
    db_execute("CREATE INDEX IF NOT EXISTS ix_sessions_principal ON sessions(principal_id);")

    # user_tasks
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS user_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            job_task_id TEXT NOT NULL,
            status_raw TEXT,
            status_norm TEXT,
            is_final INTEGER DEFAULT 0,
            last_sync_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    if not _table_has_column("user_tasks", "is_final"):
        db_execute("ALTER TABLE user_tasks ADD COLUMN is_final INTEGER DEFAULT 0;")
    if not _table_has_column("user_tasks", "status_raw"):
        db_execute("ALTER TABLE user_tasks ADD COLUMN status_raw TEXT;")
    if not _table_has_column("user_tasks", "status_norm"):
        db_execute("ALTER TABLE user_tasks ADD COLUMN status_norm TEXT;")
    if not _table_has_column("user_tasks", "last_sync_at"):
        db_execute("ALTER TABLE user_tasks ADD COLUMN last_sync_at DATETIME;")

    db_execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_user_jobtask ON user_tasks(user_id, job_task_id);")
    db_execute("CREATE INDEX IF NOT EXISTS ix_user_tasks_userid ON user_tasks(user_id);")
    db_execute("CREATE INDEX IF NOT EXISTS ix_user_tasks_final ON user_tasks(is_final);")

    # Draft table for "gmail + password + recovery + secret"
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS user_drafts (
            user_id TEXT PRIMARY KEY,
            gmail TEXT,
            gen_password TEXT,
            recovery_email TEXT,
            secret TEXT,
            otp_digits INTEGER DEFAULT 6,
            otp_period INTEGER DEFAULT 30,
            qr_raw TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    if not _table_has_column("user_drafts", "otp_digits"):
        db_execute("ALTER TABLE user_drafts ADD COLUMN otp_digits INTEGER DEFAULT 6;")
    if not _table_has_column("user_drafts", "otp_period"):
        db_execute("ALTER TABLE user_drafts ADD COLUMN otp_period INTEGER DEFAULT 30;")
    if not _table_has_column("user_drafts", "qr_raw"):
        db_execute("ALTER TABLE user_drafts ADD COLUMN qr_raw TEXT;")

    # Ledger
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS user_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            kind TEXT NOT NULL,          -- 'earn' | 'admin_credit' | 'withdraw'
            amount INTEGER NOT NULL,     -- positive integer
            ref TEXT,
            meta TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    db_execute("CREATE INDEX IF NOT EXISTS ix_ledger_user ON user_ledger(user_id);")
    db_execute("CREATE INDEX IF NOT EXISTS ix_ledger_kind ON user_ledger(kind);")
    db_execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_ledger_ref_once
        ON user_ledger(user_id, kind, ref)
        WHERE ref IS NOT NULL;
        """
    )

    # Withdrawals
    db_execute(
        """
        CREATE TABLE IF NOT EXISTS withdrawals (
            withdrawal_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            amount INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',   -- pending | paid | rejected
            method TEXT NOT NULL,
            number TEXT NOT NULL,
            meta TEXT,
            admin_txid TEXT,
            admin_note TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME,
            paid_at DATETIME
        );
        """
    )
    db_execute("CREATE INDEX IF NOT EXISTS ix_withdrawals_user ON withdrawals(user_id);")
    db_execute("CREATE INDEX IF NOT EXISTS ix_withdrawals_status ON withdrawals(status);")

    # Default admin (dev/test): admin/admin
    row = db_fetchone("SELECT COUNT(*) AS c FROM admins;")
    if row and int(row["c"]) == 0:
        admin_id = str(uuid.uuid4())
        db_execute(
            "INSERT INTO admins(admin_id, username, password_hash) VALUES (?, ?, ?);",
            (admin_id, "admin", hash_password("admin")),
        )


@app.on_event("startup")
def on_startup():
    db_init()


# =========================
# Password / Session
# =========================
def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")
    salt = APP_SECRET.encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", password_bytes, salt, 120_000)
    return dk.hex()


def verify_password(password: str, password_hash: str) -> bool:
    return hmac.compare_digest(hash_password(password), password_hash)


def create_session(kind: str, principal_id: str) -> str:
    session_id = str(uuid.uuid4())

    def _tx(client):
        client.execute(
            "INSERT INTO sessions(session_id, kind, principal_id) VALUES (?, ?, ?);",
            (session_id, kind, principal_id),
        )

    with_write_tx(_tx)
    return session_id


def delete_session(session_id: str) -> None:
    def _tx(client):
        client.execute("DELETE FROM sessions WHERE session_id=?;", (session_id,))

    with_write_tx(_tx)


def get_session(session_id: Optional[str]) -> Optional[Dict[str, str]]:
    if not session_id:
        return None
    return db_fetchone(
        "SELECT session_id, kind, principal_id FROM sessions WHERE session_id=?;",
        (session_id,),
    )


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    return db_fetchone(
        "SELECT user_id, username, password_hash FROM users WHERE username=?;",
        (username,),
    )


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    return db_fetchone(
        "SELECT user_id, username FROM users WHERE user_id=?;",
        (user_id,),
    )


def create_user(username: str, password: str) -> Dict[str, Any]:
    user_id = str(uuid.uuid4())

    def _tx(client):
        client.execute(
            "INSERT INTO users(user_id, username, password_hash) VALUES (?, ?, ?);",
            (user_id, username, hash_password(password)),
        )

    with_write_tx(_tx)
    return {"user_id": user_id, "username": username}


def get_admin_by_username(username: str) -> Optional[Dict[str, Any]]:
    return db_fetchone(
        "SELECT admin_id, username, password_hash FROM admins WHERE username=?;",
        (username,),
    )


def get_admin_by_id(admin_id: str) -> Optional[Dict[str, Any]]:
    return db_fetchone(
        "SELECT admin_id, username FROM admins WHERE admin_id=?;",
        (admin_id,),
    )


def require_user(request: Request) -> Dict[str, Any]:
    sid = request.cookies.get(SESSION_COOKIE)
    s = get_session(sid)
    if not s or s["kind"] != "user":
        raise HTTPException(status_code=401, detail="User not authenticated")
    u = get_user_by_id(s["principal_id"])
    if not u:
        raise HTTPException(status_code=401, detail="Invalid user session")
    return u


def require_admin(request: Request) -> Dict[str, Any]:
    sid = request.cookies.get(SESSION_COOKIE)
    s = get_session(sid)
    if not s or s["kind"] != "admin":
        raise HTTPException(status_code=401, detail="Admin not authenticated")
    a = get_admin_by_id(s["principal_id"])
    if not a:
        raise HTTPException(status_code=401, detail="Invalid admin session")
    return a


# =========================
# Upstream / Task
# =========================
def require_upstream_config():
    missing = []
    if not UPSTREAM_TOKEN:
        missing.append("UPSTREAM_TOKEN")
    if not UPSTREAM_JOB_ID:
        missing.append("UPSTREAM_JOB_ID")
    if missing:
        raise HTTPException(status_code=500, detail=f"Server misconfigured. Missing env vars: {', '.join(missing)}")


def upstream_headers() -> Dict[str, str]:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {UPSTREAM_TOKEN}"}


def normalize_status(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    s = str(raw).strip().lower()
    if s in ("confirmed", "confirm", "approved", "complete", "completed", "success", "succeeded"):
        return "confirmed"
    if s in ("pending", "wait", "waiting", "queued", "queue"):
        return "pending"
    if s in ("processing", "in_progress", "in-progress", "running", "review", "reviewing"):
        return "processing"
    if s in ("declined", "deny", "denied", "rejected", "reject", "failed", "fail", "canceled", "cancelled", "invalid"):
        return "declined"
    return "unknown"


def is_final_status(norm: str) -> bool:
    return norm in ("confirmed", "declined")


async def call_upstream_submit(job_proof: str) -> Dict[str, Any]:
    url = f"{UPSTREAM_BASE_URL}/api/v2/tasks/submit"
    payload = {"job_id": UPSTREAM_JOB_ID, "job_proof": job_proof}
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, headers=upstream_headers(), json=payload)

    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={"message": "Upstream error", "status_code": resp.status_code, "response": data},
        )
    return data


async def call_upstream_details(task_id: str) -> Dict[str, Any]:
    url = f"{UPSTREAM_BASE_URL}/api/v2/tasks/details"
    params = {"task_id": task_id}
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, headers=upstream_headers(), params=params, json={})

    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={"message": "Upstream error", "status_code": resp.status_code, "response": data},
        )
    return data


def upsert_user_task(user_id: str, job_id: str, job_task_id: str) -> None:
    def _tx(client):
        client.execute(
            "INSERT OR IGNORE INTO user_tasks(user_id, job_id, job_task_id) VALUES (?, ?, ?);",
            (user_id, job_id, job_task_id),
        )

    with_write_tx(_tx)


def update_task_status(
    client,
    user_id: str,
    job_task_id: str,
    status_raw: Optional[str],
    status_norm: str,
    final: bool,
) -> None:
    client.execute(
        """
        UPDATE user_tasks
        SET status_raw=?, status_norm=?, is_final=?, last_sync_at=CURRENT_TIMESTAMP
        WHERE user_id=? AND job_task_id=?;
        """,
        (status_raw, status_norm, 1 if final else 0, user_id, job_task_id),
    )


def get_tasks_for_user(user_id: str) -> List[Dict[str, Any]]:
    return db_fetchall(
        """
        SELECT job_task_id, status_norm, is_final, last_sync_at, created_at
        FROM user_tasks
        WHERE user_id=?
        ORDER BY created_at DESC, id DESC;
        """,
        (user_id,),
    )


def get_non_final_task_ids(user_id: str) -> List[str]:
    rows = db_fetchall(
        """
        SELECT job_task_id
        FROM user_tasks
        WHERE user_id=? AND (is_final IS NULL OR is_final=0);
        """,
        (user_id,),
    )
    return [str(r["job_task_id"]) for r in rows]


def task_stats(user_id: str) -> Dict[str, int]:
    total = db_fetchone(
        "SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks WHERE user_id=?;",
        (user_id,),
    )["c"]

    confirmed = db_fetchone(
        "SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks WHERE user_id=? AND status_norm='confirmed';",
        (user_id,),
    )["c"]

    declined = db_fetchone(
        "SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks WHERE user_id=? AND status_norm='declined';",
        (user_id,),
    )["c"]

    processing = db_fetchone(
        "SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks WHERE user_id=? AND status_norm='processing';",
        (user_id,),
    )["c"]

    pending = db_fetchone(
        "SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks WHERE user_id=? AND status_norm='pending';",
        (user_id,),
    )["c"]

    return {
        "total": int(total),
        "confirmed": int(confirmed),
        "declined": int(declined),
        "processing": int(processing),
        "pending": int(pending),
    }


def admin_overview_stats() -> Dict[str, int]:
    total_users = db_fetchone("SELECT COALESCE(COUNT(*),0) AS c FROM users;")["c"]
    total_tasks = db_fetchone("SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks;")["c"]
    total_withdrawals = db_fetchone("SELECT COALESCE(COUNT(*),0) AS c FROM withdrawals;")["c"]
    pending_withdrawals = db_fetchone(
        "SELECT COALESCE(COUNT(*),0) AS c FROM withdrawals WHERE status='pending';"
    )["c"]
    return {
        "total_users": int(total_users),
        "total_tasks": int(total_tasks),
        "total_withdrawals": int(total_withdrawals),
        "pending_withdrawals": int(pending_withdrawals),
    }


# =========================
# Draft helpers (gmail/password/recovery/secret)
# =========================
def get_draft(user_id: str) -> Optional[Dict[str, Any]]:
    return db_fetchone(
        """
        SELECT user_id, gmail, gen_password, recovery_email, secret, otp_digits, otp_period, qr_raw, created_at, updated_at
        FROM user_drafts
        WHERE user_id=?;
        """,
        (user_id,),
    )


def clear_draft(user_id: str) -> None:
    def _tx(client):
        client.execute("DELETE FROM user_drafts WHERE user_id=?;", (user_id,))

    with_write_tx(_tx)


def _rand_letters(n: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    b = os.urandom(n)
    return "".join(alphabet[x % len(alphabet)] for x in b)


def _rand_digits(n: int) -> str:
    digits = "0123456789"
    b = os.urandom(n)
    return "".join(digits[x % len(digits)] for x in b)


def generate_pretty_password() -> str:
    # Example style: behivgusez@1074
    return f"{_rand_letters(10)}@{_rand_digits(4)}"


def generate_recovery_email() -> str:
    # Example style: gopemlixiw155@xneko.xyz
    return f"{_rand_letters(9)}{_rand_digits(3)}@{RECOVERY_DOMAIN}"


def start_or_reset_draft(user_id: str, gmail: str) -> Dict[str, Any]:
    gmail = (gmail or "").strip().lower()
    if not gmail or "@" not in gmail:
        raise HTTPException(status_code=400, detail="Valid Gmail is required")

    gen_password = generate_pretty_password()
    recovery_email = generate_recovery_email()

    def _tx(client):
        client.execute(
            """
            INSERT INTO user_drafts(user_id, gmail, gen_password, recovery_email, secret, otp_digits, otp_period, qr_raw, updated_at)
            VALUES (?, ?, ?, ?, NULL, 6, 30, NULL, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                gmail=excluded.gmail,
                gen_password=excluded.gen_password,
                recovery_email=excluded.recovery_email,
                secret=NULL,
                otp_digits=6,
                otp_period=30,
                qr_raw=NULL,
                updated_at=CURRENT_TIMESTAMP;
            """,
            (user_id, gmail, gen_password, recovery_email),
        )

    with_write_tx(_tx)
    return get_draft(user_id) or {}


def save_secret_to_draft(user_id: str, secret: str, otp_digits: int, otp_period: int, qr_raw: str) -> None:
    def _tx(client):
        client.execute(
            """
            UPDATE user_drafts
            SET secret=?, otp_digits=?, otp_period=?, qr_raw=?, updated_at=CURRENT_TIMESTAMP
            WHERE user_id=?;
            """,
            (secret, int(otp_digits), int(otp_period), qr_raw, user_id),
        )

    with_write_tx(_tx)


# =========================
# QR Decode + TOTP
# =========================
BASE32_RE = re.compile(r"^[A-Z2-7]+=*$")


def _clean_base32(s: str) -> str:
    s = (s or "").strip().replace(" ", "").replace("-", "")
    s = s.upper()
    return s


def decode_qr_payload_from_image(image_bytes: bytes) -> str:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    detector = cv2.QRCodeDetector()

    # Try single decode
    data, points, _ = detector.detectAndDecode(img)
    if data and data.strip():
        return data.strip()

    # Try scale-up and grayscale
    img2 = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    data, points, _ = detector.detectAndDecode(gray)
    if data and data.strip():
        return data.strip()

    # Try multi decode (if supported)
    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(img2)
        if ok and decoded_info:
            for d in decoded_info:
                if d and d.strip():
                    return d.strip()
    except Exception:
        pass

    raise ValueError("QR not detected. Please upload a clear QR-only screenshot (zoom QR).")


def extract_secret_from_qr_payload(payload: str) -> Tuple[str, int, int]:
    """
    Returns: (secret_base32, digits, period)
    """
    payload = (payload or "").strip()

    digits = 6
    period = 30

    # Typical QR contains otpauth://totp/...?...secret=XXXX&digits=6&period=30
    if payload.lower().startswith("otpauth://"):
        u = urlparse(payload)
        q = parse_qs(u.query)
        sec = q.get("secret", [None])[0]
        if q.get("digits"):
            try:
                digits = int(q["digits"][0])
            except Exception:
                pass
        if q.get("period"):
            try:
                period = int(q["period"][0])
            except Exception:
                pass
        if not sec:
            raise ValueError("QR decoded but secret param not found")
        sec = _clean_base32(sec)
        return sec, digits, period

    # Sometimes QR is just the secret
    sec = _clean_base32(payload)
    if not sec:
        raise ValueError("Empty QR payload")

    # Basic base32 validation (allow missing padding)
    test = sec + ("=" * ((8 - (len(sec) % 8)) % 8))
    if not BASE32_RE.match(test):
        # still allow some authenticators which embed extra text; try to find secret=...
        m = re.search(r"secret=([A-Za-z2-7=]+)", payload)
        if m:
            sec2 = _clean_base32(m.group(1))
            test2 = sec2 + ("=" * ((8 - (len(sec2) % 8)) % 8))
            if BASE32_RE.match(test2):
                return sec2, digits, period
        raise ValueError("QR payload is not a valid OTP secret")

    return sec, digits, period


def totp_now(secret_b32: str, digits: int = 6, period: int = 30) -> Tuple[str, int]:
    secret_b32 = _clean_base32(secret_b32)
    if not secret_b32:
        raise ValueError("Missing secret")

    # pad for base32 decoding
    padded = secret_b32 + ("=" * ((8 - (len(secret_b32) % 8)) % 8))
    key = base64.b32decode(padded, casefold=True)

    t = int(time.time())
    counter = t // int(period)
    remaining = int(period) - (t % int(period))

    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code_int = (
        ((digest[offset] & 0x7F) << 24)
        | ((digest[offset + 1] & 0xFF) << 16)
        | ((digest[offset + 2] & 0xFF) << 8)
        | (digest[offset + 3] & 0xFF)
    )
    code = code_int % (10 ** int(digits))
    return str(code).zfill(int(digits)), remaining


def format_job_proof(draft: Dict[str, Any]) -> str:
    gmail = (draft.get("gmail") or "").strip()
    pw = (draft.get("gen_password") or "").strip()
    rec = (draft.get("recovery_email") or "").strip()
    sec = (draft.get("secret") or "").strip()
    if not (gmail and pw and rec and sec):
        raise HTTPException(status_code=400, detail="Draft incomplete (gmail/password/recovery/secret required)")
    return f"{gmail}:{pw}:{rec}:{sec}"


# =========================
# Wallet / Ledger / Withdrawals
# =========================
def ledger_sum(client, user_id: str, kind: str) -> int:
    row = db_fetchone_client(
        client,
        "SELECT COALESCE(SUM(amount), 0) AS total FROM user_ledger WHERE user_id=? AND kind=?;",
        (user_id, kind),
    )
    return int(row["total"] if row else 0)


def reserved_withdraw_sum(client, user_id: str) -> int:
    row = db_fetchone_client(
        client,
        "SELECT COALESCE(SUM(amount), 0) AS total FROM withdrawals WHERE user_id=? AND status='pending';",
        (user_id,),
    )
    return int(row["total"] if row else 0)


def hold_balance(user_id: str) -> int:
    row = db_fetchone(
        """
        SELECT COALESCE(COUNT(*), 0) AS c
        FROM user_tasks
        WHERE user_id=?
          AND (is_final IS NULL OR is_final=0)
          AND status_norm IN ('pending','processing');
        """,
        (user_id,),
    )
    return int(row["c"]) * CREDIT_PER_CONFIRMED


def balances(user_id: str) -> Dict[str, int]:
    earned = ledger_sum(db_client, user_id, "earn") + ledger_sum(db_client, user_id, "admin_credit")
    withdrawn = ledger_sum(db_client, user_id, "withdraw")
    reserved = reserved_withdraw_sum(db_client, user_id)
    available = earned - withdrawn - reserved
    if available < 0:
        available = 0
    return {
        "available_balance": int(available),
        "hold_balance": hold_balance(user_id),
        "total_earned": int(earned),
        "total_withdrawn": int(withdrawn),
        "reserved_withdraw_balance": int(reserved),
    }


def ledger_add_once(
    client,
    user_id: str,
    kind: str,
    amount: int,
    ref: Optional[str],
    meta: Optional[str],
) -> bool:
    result = client.execute(
        "INSERT OR IGNORE INTO user_ledger(user_id, kind, amount, ref, meta) VALUES (?, ?, ?, ?, ?);",
        (user_id, kind, int(amount), ref, meta),
    )
    return result.rows_affected == 1


def create_withdraw_request(user_id: str, amount: int, method: str, number: str) -> str:
    if amount <= 0:
        raise HTTPException(status_code=400, detail="amount must be > 0")
    method = (method or "").strip()
    number = (number or "").strip()
    if not method or not number:
        raise HTTPException(status_code=400, detail="method and number are required")

    withdrawal_id = str(uuid.uuid4())

    def _tx(client):
        earned = ledger_sum(client, user_id, "earn") + ledger_sum(client, user_id, "admin_credit")
        withdrawn = ledger_sum(client, user_id, "withdraw")
        reserved = reserved_withdraw_sum(client, user_id)
        available = earned - withdrawn - reserved
        if available < amount:
            raise HTTPException(
                status_code=400,
                detail={"message": "Insufficient balance", "available": max(int(available), 0)},
            )

        client.execute(
            """
            INSERT INTO withdrawals(withdrawal_id, user_id, amount, status, method, number, updated_at)
            VALUES (?, ?, ?, 'pending', ?, ?, CURRENT_TIMESTAMP);
            """,
            (withdrawal_id, user_id, int(amount), method, number),
        )

    with_write_tx(_tx)
    return withdrawal_id


def list_withdrawals(user_id: str) -> List[Dict[str, Any]]:
    return db_fetchall(
        """
        SELECT withdrawal_id, amount, status, method, number, meta, admin_txid, admin_note, created_at, updated_at, paid_at
        FROM withdrawals
        WHERE user_id=?
        ORDER BY created_at DESC;
        """,
        (user_id,),
    )


def list_all_withdrawals(status: Optional[str] = None) -> List[Dict[str, Any]]:
    if status:
        return db_fetchall(
            """
            SELECT withdrawal_id, user_id, amount, status, method, number, meta, admin_txid, admin_note, created_at, updated_at, paid_at
            FROM withdrawals
            WHERE status=?
            ORDER BY created_at DESC;
            """,
            (status,),
        )
    return db_fetchall(
        """
        SELECT withdrawal_id, user_id, amount, status, method, number, meta, admin_txid, admin_note, created_at, updated_at, paid_at
        FROM withdrawals
        ORDER BY created_at DESC;
        """
    )


def admin_confirm_withdraw(withdrawal_id: str, txid: str, note: str) -> None:
    txid = (txid or "").strip()
    note = (note or "").strip()

    def _tx(client):
        w = db_fetchone_client(
            client,
            "SELECT withdrawal_id, user_id, amount, status FROM withdrawals WHERE withdrawal_id=?;",
            (withdrawal_id,),
        )
        if not w:
            raise HTTPException(status_code=404, detail="Withdrawal not found")
        if w["status"] != "pending":
            raise HTTPException(status_code=400, detail={"message": "Not pending", "status": w["status"]})

        ok = ledger_add_once(
            client=client,
            user_id=str(w["user_id"]),
            kind="withdraw",
            amount=int(w["amount"]),
            ref=str(withdrawal_id),
            meta=f"txid={txid};note={note}",
        )
        if not ok:
            raise HTTPException(status_code=409, detail="Already finalized in ledger")

        client.execute(
            """
            UPDATE withdrawals
            SET status='paid', admin_txid=?, admin_note=?, paid_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
            WHERE withdrawal_id=?;
            """,
            (txid or None, note or None, withdrawal_id),
        )

    with_write_tx(_tx)


def admin_reject_withdraw(withdrawal_id: str, note: str) -> None:
    note = (note or "").strip()

    def _tx(client):
        w = db_fetchone_client(
            client,
            "SELECT withdrawal_id, status FROM withdrawals WHERE withdrawal_id=?;",
            (withdrawal_id,),
        )
        if not w:
            raise HTTPException(status_code=404, detail="Withdrawal not found")
        if w["status"] != "pending":
            raise HTTPException(status_code=400, detail={"message": "Not pending", "status": w["status"]})

        client.execute(
            """
            UPDATE withdrawals
            SET status='rejected', admin_note=?, updated_at=CURRENT_TIMESTAMP
            WHERE withdrawal_id=?;
            """,
            (note or None, withdrawal_id),
        )

    with_write_tx(_tx)


# =========================
# Sync logic (final tasks never requested again)
# =========================
async def sync_user_tasks(user_id: str) -> None:
    require_upstream_config()
    task_ids = get_non_final_task_ids(user_id)
    if not task_ids:
        return

    for tid in task_ids:
        detail = await call_upstream_details(task_id=tid)
        raw_status = detail.get("status")
        norm = normalize_status(raw_status)
        final = is_final_status(norm)

        def _tx(client):
            update_task_status(
                client,
                user_id=user_id,
                job_task_id=tid,
                status_raw=raw_status,
                status_norm=norm,
                final=final,
            )
            if final and norm == "confirmed":
                ledger_add_once(client, user_id=user_id, kind="earn", amount=CREDIT_PER_CONFIRMED, ref=tid, meta=None)

        with_write_tx(_tx)


# =========================
# Web Pages
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# --- USER AUTH ---
@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request, "error": None})


@app.post("/signup")
def signup_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    username = username.strip().lower()
    if len(username) < 3 or len(password) < 4:
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Invalid username/password"})

    if get_user_by_username(username):
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Username already exists"})

    u = create_user(username, password)
    sid = create_session("user", u["user_id"])
    resp = RedirectResponse(url="/user/dashboard", status_code=302)
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    username = username.strip().lower()
    u = get_user_by_username(username)
    if not u or not verify_password(password, u["password_hash"]):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

    sid = create_session("user", u["user_id"])
    resp = RedirectResponse(url="/user/dashboard", status_code=302)
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


@app.get("/logout")
def logout(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        delete_session(sid)
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# --- ADMIN AUTH ---
@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": None})


@app.post("/admin/login")
def admin_login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    username = username.strip().lower()
    a = get_admin_by_username(username)
    if not a or not verify_password(password, a["password_hash"]):
        return templates.TemplateResponse("admin_login.html", {"request": request, "error": "Invalid credentials"})

    sid = create_session("admin", a["admin_id"])
    resp = RedirectResponse(url="/admin", status_code=302)
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


@app.get("/admin/logout")
def admin_logout(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        delete_session(sid)
    resp = RedirectResponse(url="/admin/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# =========================
# USER PAGES
# =========================
@app.get("/user/dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request):
    user = require_user(request)
    try:
        await sync_user_tasks(user["user_id"])
    except Exception:
        pass

    b = balances(user["user_id"])
    stats = task_stats(user["user_id"])

    return templates.TemplateResponse(
        "user_dashboard.html",
        {"request": request, "user": user, "balances": b, "stats": stats},
    )


@app.get("/user/gmail", response_class=HTMLResponse)
def user_gmail_page(request: Request):
    user = require_user(request)
    draft = get_draft(user["user_id"])
    return templates.TemplateResponse(
        "user_gmail.html",
        {"request": request, "user": user, "draft": draft, "error": None, "success": None},
    )


@app.post("/user/gmail/start")
def user_gmail_start(request: Request, gmail: str = Form(...)):
    user = require_user(request)
    try:
        start_or_reset_draft(user["user_id"], gmail)
    except HTTPException as e:
        draft = get_draft(user["user_id"])
        return templates.TemplateResponse(
            "user_gmail.html",
            {"request": request, "user": user, "draft": draft, "error": str(e.detail), "success": None},
        )

    return RedirectResponse(url="/user/gmail", status_code=302)


@app.post("/user/gmail/upload-qr")
async def user_gmail_upload_qr(request: Request, qr_image: UploadFile = File(...)):
    user = require_user(request)
    draft = get_draft(user["user_id"])
    if not draft or not draft.get("gmail"):
        return templates.TemplateResponse(
            "user_gmail.html",
            {"request": request, "user": user, "draft": draft, "error": "Please enter Gmail first", "success": None},
        )

    try:
        img_bytes = await qr_image.read()
        payload = decode_qr_payload_from_image(img_bytes)
        secret, digits, period = extract_secret_from_qr_payload(payload)
        save_secret_to_draft(user["user_id"], secret=secret, otp_digits=digits, otp_period=period, qr_raw=payload)
    except Exception as e:
        draft = get_draft(user["user_id"])
        return templates.TemplateResponse(
            "user_gmail.html",
            {"request": request, "user": user, "draft": draft, "error": f"QR decode failed: {e}", "success": None},
        )

    return RedirectResponse(url="/user/gmail", status_code=302)


@app.get("/user/gmail/totp")
def user_gmail_totp(request: Request):
    user = require_user(request)
    draft = get_draft(user["user_id"])
    if not draft or not draft.get("secret"):
        return JSONResponse({"ok": False, "error": "No secret yet"}, status_code=400)

    try:
        code, remaining = totp_now(
            secret_b32=draft["secret"],
            digits=int(draft.get("otp_digits") or 6),
            period=int(draft.get("otp_period") or 30),
        )
        return {"ok": True, "code": code, "remaining": remaining, "period": int(draft.get("otp_period") or 30)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/user/gmail/submit")
async def user_gmail_submit(request: Request):
    user = require_user(request)
    require_upstream_config()

    draft = get_draft(user["user_id"])
    if not draft:
        return RedirectResponse(url="/user/gmail", status_code=302)

    try:
        job_proof = format_job_proof(draft)
        data = await call_upstream_submit(job_proof=job_proof)

        job_task_id = data.get("job_task_id") or data.get("job_taskId") or data.get("jobTaskId")
        job_id = data.get("job_id") or UPSTREAM_JOB_ID
        if not job_task_id:
            raise HTTPException(status_code=502, detail="Upstream missing job_task_id")

        upsert_user_task(user_id=user["user_id"], job_id=str(job_id), job_task_id=str(job_task_id))
    except HTTPException as e:
        draft = get_draft(user["user_id"])
        return templates.TemplateResponse(
            "user_gmail.html",
            {"request": request, "user": user, "draft": draft, "error": str(e.detail), "success": None},
        )

    return RedirectResponse(url="/user/tasks", status_code=302)


@app.post("/user/gmail/reset")
def user_gmail_reset(request: Request):
    user = require_user(request)
    clear_draft(user["user_id"])
    return RedirectResponse(url="/user/gmail", status_code=302)


@app.get("/user/withdraw", response_class=HTMLResponse)
def user_withdraw_page(request: Request):
    user = require_user(request)
    b = balances(user["user_id"])
    w = list_withdrawals(user["user_id"])
    return templates.TemplateResponse(
        "user_withdraw.html",
        {"request": request, "user": user, "balances": b, "withdrawals": w, "error": None},
    )


@app.post("/user/withdraw")
def user_withdraw_action(request: Request, amount: int = Form(...), method: str = Form(...), number: str = Form(...)):
    user = require_user(request)
    try:
        create_withdraw_request(user_id=user["user_id"], amount=int(amount), method=method, number=number)
    except HTTPException as e:
        b = balances(user["user_id"])
        w = list_withdrawals(user["user_id"])
        return templates.TemplateResponse(
            "user_withdraw.html",
            {"request": request, "user": user, "balances": b, "withdrawals": w, "error": str(e.detail)},
        )

    return RedirectResponse(url="/user/withdraw", status_code=302)


@app.get("/user/tasks", response_class=HTMLResponse)
async def user_tasks_page(request: Request):
    user = require_user(request)
    try:
        await sync_user_tasks(user["user_id"])
    except Exception:
        pass
    tasks = get_tasks_for_user(user["user_id"])
    return templates.TemplateResponse("user_tasks.html", {"request": request, "user": user, "tasks": tasks})


# =========================
# ADMIN DASH
# =========================
@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    admin = require_admin(request)
    return templates.TemplateResponse(
        "admin_dashboard.html",
        {"request": request, "admin": admin, "stats": admin_overview_stats()},
    )


@app.get("/admin/credit", response_class=HTMLResponse)
def admin_credit_page(request: Request):
    admin = require_admin(request)
    return templates.TemplateResponse(
        "admin_credit.html",
        {"request": request, "admin": admin, "error": None, "success": None},
    )


@app.post("/admin/credit")
def admin_add_credit(request: Request, target_username: str = Form(...), amount: int = Form(...), note: str = Form("")):
    admin = require_admin(request)
    target_username = target_username.strip().lower()
    u = get_user_by_username(target_username)
    if not u:
        return templates.TemplateResponse(
            "admin_credit.html",
            {
                "request": request,
                "admin": admin,
                "error": "User not found",
                "success": None,
            },
        )

    amount = int(amount)
    if amount <= 0:
        return templates.TemplateResponse(
            "admin_credit.html",
            {
                "request": request,
                "admin": admin,
                "error": "Amount must be > 0",
                "success": None,
            },
        )

    ref = str(uuid.uuid4())

    def _tx(client):
        ledger_add_once(client, u["user_id"], "admin_credit", amount, ref=ref, meta=f"note={note}")

    with_write_tx(_tx)

    return templates.TemplateResponse(
        "admin_credit.html",
        {
            "request": request,
            "admin": admin,
            "error": None,
            "success": f"Credited {amount} to {target_username}",
        },
    )


@app.get("/admin/withdrawals", response_class=HTMLResponse)
def admin_withdrawals_page(request: Request):
    admin = require_admin(request)
    w_all = list_all_withdrawals(status=None)
    return templates.TemplateResponse(
        "admin_withdrawals.html",
        {"request": request, "admin": admin, "withdrawals": w_all, "error": None, "success": None},
    )


@app.post("/admin/withdrawals/confirm")
def admin_confirm_withdrawal_web(request: Request, withdrawal_id: str = Form(...), txid: str = Form(""), note: str = Form("")):
    require_admin(request)
    admin_confirm_withdraw(withdrawal_id=withdrawal_id, txid=txid, note=note)
    return RedirectResponse(url="/admin/withdrawals", status_code=302)


@app.post("/admin/withdrawals/reject")
def admin_reject_withdrawal_web(request: Request, withdrawal_id: str = Form(...), note: str = Form("")):
    require_admin(request)
    admin_reject_withdraw(withdrawal_id=withdrawal_id, note=note)
    return RedirectResponse(url="/admin/withdrawals", status_code=302)


# =========================
# Direct run
# =========================
def run():
    import uvicorn
    if RELOAD:
        uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
    else:
        uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    run()
