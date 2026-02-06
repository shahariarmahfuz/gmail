import os
import uuid
import hashlib
import hmac
import time
import base64
import struct
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List, Tuple, Callable
from urllib.parse import urlparse, parse_qs

import numpy as np
import httpx
import libsql
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

APP_LOGGER_NAME = "task_proxy_wallet"

UPSTREAM_BASE_URL = (UPSTREAM_BASE_URL or "").rstrip("/")

# Local embedded replica file path (required for libsql embedded replicas).
# You can override it via env without touching config.py.
DB_FILE = os.getenv("DB_FILE", os.getenv("DB_PATH", "./local.db")).strip() or "./local.db"
DB_SYNC_INTERVAL = int(os.getenv("DB_SYNC_INTERVAL", "30"))  # seconds, 0 disables auto-sync

# =========================
# App
# =========================
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# =========================
# DB (libsql embedded replica, single-threaded executor)
# =========================
_db_executor: Optional[ThreadPoolExecutor] = None
_db_conn: Optional[libsql.Connection] = None


def _db_columns(cursor) -> List[str]:
    try:
        if cursor.description:
            return [d[0] for d in cursor.description]
    except Exception:
        pass
    return []


def _rows_to_dicts(cursor, rows) -> List[Dict[str, Any]]:
    cols = _db_columns(cursor)
    if not rows:
        return []
    if not cols:
        return [dict(enumerate(r)) for r in rows]
    return [dict(zip(cols, r)) for r in rows]


def _db_connect_sync() -> libsql.Connection:
    if not DB_URL or not DB_TOKEN:
        raise RuntimeError("DB_URL/DB_TOKEN missing (Turso/libSQL credentials required)")

    # Embedded replica: local file + sync_url/auth_token
    # Writes are delegated to the remote primary, reads are local.
    conn = libsql.connect(
        DB_FILE,
        sync_url=DB_URL,
        auth_token=DB_TOKEN,
        sync_interval=DB_SYNC_INTERVAL if DB_SYNC_INTERVAL > 0 else None,
    )

    try:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=8000;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        # Some pragmas may be ignored depending on environment; do not fail startup.
        pass

    return conn


async def _run_db(fn: Callable[[], Any]) -> Any:
    loop = asyncio.get_running_loop()
    if _db_executor is None:
        raise RuntimeError("DB executor not initialized")
    return await loop.run_in_executor(_db_executor, fn)


def _db_execute_sync(sql: str, args: Tuple[Any, ...] = ()):
    if _db_conn is None:
        raise RuntimeError("DB connection not initialized")
    return _db_conn.execute(sql, args)


def _db_fetchone_sync(sql: str, args: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
    cur = _db_execute_sync(sql, args)
    row = cur.fetchone()
    if row is None:
        return None
    cols = _db_columns(cur)
    if cols:
        return dict(zip(cols, row))
    return dict(enumerate(row))


def _db_fetchall_sync(sql: str, args: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    cur = _db_execute_sync(sql, args)
    rows = cur.fetchall()
    return _rows_to_dicts(cur, rows)


def _db_changes_sync() -> int:
    r = _db_fetchone_sync("SELECT changes() AS c;")
    return int((r or {}).get("c") or 0)


async def db_fetchone(sql: str, args: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
    return await _run_db(lambda: _db_fetchone_sync(sql, args))


async def db_fetchall(sql: str, args: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    return await _run_db(lambda: _db_fetchall_sync(sql, args))


async def db_execute(sql: str, args: Tuple[Any, ...] = ()) -> None:
    await _run_db(lambda: _db_execute_sync(sql, args))


async def _table_has_column(table: str, column: str) -> bool:
    rows = await db_fetchall(f"PRAGMA table_info({table});")
    cols = {str(r.get("name")) for r in rows if r.get("name") is not None}
    return column in cols


async def with_write_tx(fn: Callable[[], Any]) -> Any:
    attempts = 5
    for i in range(attempts):
        try:
            return await _run_db(fn)
        except Exception as e:
            msg = str(e).lower()
            if "database is locked" in msg or "locked" in msg:
                if i < attempts - 1:
                    await asyncio.sleep(0.15 * (i + 1))
                    continue
            raise


def _db_init_sync() -> None:
    if _db_conn is None:
        raise RuntimeError("DB connection not initialized")

    # Pull remote -> local at startup (best-effort)
    try:
        _db_conn.sync()
    except Exception:
        pass

    # Users
    _db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            is_banned INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    # Admins
    _db_conn.execute(
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
    _db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            principal_id TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    _db_conn.execute("CREATE INDEX IF NOT EXISTS ix_sessions_kind ON sessions(kind);")
    _db_conn.execute("CREATE INDEX IF NOT EXISTS ix_sessions_principal ON sessions(principal_id);")

    # user_tasks
    _db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            job_task_id TEXT NOT NULL,
            gmail TEXT,
            gen_password TEXT,
            recovery_email TEXT,
            status_raw TEXT,
            status_norm TEXT,
            is_final INTEGER DEFAULT 0,
            last_sync_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    _db_conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_user_jobtask ON user_tasks(user_id, job_task_id);")
    _db_conn.execute("CREATE INDEX IF NOT EXISTS ix_user_tasks_userid ON user_tasks(user_id);")
    _db_conn.execute("CREATE INDEX IF NOT EXISTS ix_user_tasks_final ON user_tasks(is_final);")

    # user_drafts
    _db_conn.execute(
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

    # Ledger
    _db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            amount INTEGER NOT NULL,
            ref TEXT,
            meta TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    _db_conn.execute("CREATE INDEX IF NOT EXISTS ix_ledger_user ON user_ledger(user_id);")
    _db_conn.execute("CREATE INDEX IF NOT EXISTS ix_ledger_kind ON user_ledger(kind);")
    _db_conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_ledger_ref_once
        ON user_ledger(user_id, kind, ref)
        WHERE ref IS NOT NULL;
        """
    )

    # Withdrawals
    _db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS withdrawals (
            withdrawal_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            amount INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
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
    _db_conn.execute("CREATE INDEX IF NOT EXISTS ix_withdrawals_user ON withdrawals(user_id);")
    _db_conn.execute("CREATE INDEX IF NOT EXISTS ix_withdrawals_status ON withdrawals(status);")

    # Backfill columns (safe)
    # users.is_banned
    try:
        cur = _db_conn.execute("PRAGMA table_info(users);")
        cols = {r[1] for r in cur.fetchall()}  # (cid, name, type, notnull, dflt, pk)
        if "is_banned" not in cols:
            _db_conn.execute("ALTER TABLE users ADD COLUMN is_banned INTEGER DEFAULT 0;")
    except Exception:
        pass

    # user_drafts otp fields
    try:
        cur = _db_conn.execute("PRAGMA table_info(user_drafts);")
        cols = {r[1] for r in cur.fetchall()}
        if "otp_digits" not in cols:
            _db_conn.execute("ALTER TABLE user_drafts ADD COLUMN otp_digits INTEGER DEFAULT 6;")
        if "otp_period" not in cols:
            _db_conn.execute("ALTER TABLE user_drafts ADD COLUMN otp_period INTEGER DEFAULT 30;")
        if "qr_raw" not in cols:
            _db_conn.execute("ALTER TABLE user_drafts ADD COLUMN qr_raw TEXT;")
    except Exception:
        pass

    # Default admin
    row = _db_fetchone_sync("SELECT COALESCE(COUNT(*),0) AS c FROM admins;")
    if row and int(row["c"]) == 0:
        admin_id = str(uuid.uuid4())
        _db_conn.execute(
            "INSERT INTO admins(admin_id, username, password_hash) VALUES (?, ?, ?);",
            (admin_id, "admin", hash_password("admin")),
        )

    _db_conn.commit()
    try:
        _db_conn.sync()
    except Exception:
        pass


@app.on_event("startup")
async def on_startup():
    global _db_executor, _db_conn
    _db_executor = ThreadPoolExecutor(max_workers=1)
    _db_conn = await _run_db(_db_connect_sync)
    await _run_db(_db_init_sync)


@app.on_event("shutdown")
async def on_shutdown():
    global _db_executor, _db_conn
    if _db_conn is not None:
        def _close():
            try:
                _db_conn.commit()
            except Exception:
                pass
            try:
                _db_conn.close()
            except Exception:
                pass
        try:
            await _run_db(_close)
        except Exception:
            pass
        _db_conn = None
    if _db_executor is not None:
        _db_executor.shutdown(wait=False, cancel_futures=True)
        _db_executor = None


# =========================
# Password / Session
# =========================
def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")
    salt = (APP_SECRET or "").encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", password_bytes, salt, 120_000)
    return dk.hex()


def verify_password(password: str, password_hash: str) -> bool:
    return hmac.compare_digest(hash_password(password), password_hash)


async def create_session(kind: str, principal_id: str) -> str:
    session_id = str(uuid.uuid4())

    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            _db_conn.execute(
                "INSERT INTO sessions(session_id, kind, principal_id) VALUES (?, ?, ?);",
                (session_id, kind, principal_id),
            )
            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
        except Exception:
            _db_conn.rollback()
            raise

    await with_write_tx(_tx)
    return session_id


async def delete_session(session_id: str) -> None:
    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            _db_conn.execute("DELETE FROM sessions WHERE session_id=?;", (session_id,))
            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
        except Exception:
            _db_conn.rollback()
            raise

    await with_write_tx(_tx)


async def get_session(session_id: Optional[str]) -> Optional[Dict[str, str]]:
    if not session_id:
        return None
    row = await db_fetchone(
        "SELECT session_id, kind, principal_id FROM sessions WHERE session_id=?;",
        (session_id,),
    )
    return row  # already dict


async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    return await db_fetchone(
        "SELECT user_id, username, password_hash, is_banned FROM users WHERE username=?;",
        (username,),
    )


async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    return await db_fetchone(
        "SELECT user_id, username, is_banned FROM users WHERE user_id=?;",
        (user_id,),
    )


async def create_user(username: str, password: str) -> Dict[str, Any]:
    user_id = str(uuid.uuid4())

    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            _db_conn.execute(
                "INSERT INTO users(user_id, username, password_hash, is_banned) VALUES (?, ?, ?, 0);",
                (user_id, username, hash_password(password)),
            )
            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
        except Exception:
            _db_conn.rollback()
            raise

    await with_write_tx(_tx)
    return {"user_id": user_id, "username": username}


async def get_admin_by_username(username: str) -> Optional[Dict[str, Any]]:
    return await db_fetchone(
        "SELECT admin_id, username, password_hash FROM admins WHERE username=?;",
        (username,),
    )


async def get_admin_by_id(admin_id: str) -> Optional[Dict[str, Any]]:
    return await db_fetchone(
        "SELECT admin_id, username FROM admins WHERE admin_id=?;",
        (admin_id,),
    )


async def require_user(request: Request) -> Dict[str, Any]:
    sid = request.cookies.get(SESSION_COOKIE)
    s = await get_session(sid)
    if not s or s.get("kind") != "user":
        raise HTTPException(status_code=401, detail="User not authenticated")
    u = await get_user_by_id(str(s.get("principal_id") or ""))
    if not u:
        raise HTTPException(status_code=401, detail="Invalid user session")
    if int(u.get("is_banned") or 0) == 1:
        raise HTTPException(status_code=403, detail="User is banned")
    return u


async def require_admin(request: Request) -> Dict[str, Any]:
    sid = request.cookies.get(SESSION_COOKIE)
    s = await get_session(sid)
    if not s or s.get("kind") != "admin":
        raise HTTPException(status_code=401, detail="Admin not authenticated")
    a = await get_admin_by_id(str(s.get("principal_id") or ""))
    if not a:
        raise HTTPException(status_code=401, detail="Invalid admin session")
    return a


# =========================
# Upstream / Task
# =========================
def require_upstream_config():
    missing = []
    if not (UPSTREAM_TOKEN or "").strip():
        missing.append("UPSTREAM_TOKEN")
    if not (UPSTREAM_JOB_ID or "").strip():
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


async def upsert_user_task(
    user_id: str,
    job_id: str,
    job_task_id: str,
    gmail: str,
    gen_password: str,
    recovery_email: str,
) -> None:
    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            _db_conn.execute(
                """
                INSERT OR IGNORE INTO user_tasks(user_id, job_id, job_task_id, gmail, gen_password, recovery_email)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (user_id, job_id, job_task_id, gmail, gen_password, recovery_email),
            )
            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
        except Exception:
            _db_conn.rollback()
            raise

    await with_write_tx(_tx)


async def update_task_status(
    user_id: str,
    job_task_id: str,
    status_raw: Optional[str],
    status_norm: str,
    final: bool,
) -> None:
    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            _db_conn.execute(
                """
                UPDATE user_tasks
                SET status_raw=?, status_norm=?, is_final=?, last_sync_at=CURRENT_TIMESTAMP
                WHERE user_id=? AND job_task_id=?;
                """,
                (status_raw, status_norm, 1 if final else 0, user_id, job_task_id),
            )
            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
        except Exception:
            _db_conn.rollback()
            raise

    await with_write_tx(_tx)


async def get_tasks_for_user(user_id: str) -> List[Dict[str, Any]]:
    return await db_fetchall(
        """
        SELECT job_task_id, status_norm, is_final, last_sync_at, created_at
        FROM user_tasks
        WHERE user_id=?
        ORDER BY created_at DESC, id DESC;
        """,
        (user_id,),
    )


async def get_non_final_task_ids(user_id: str) -> List[str]:
    rows = await db_fetchall(
        """
        SELECT job_task_id
        FROM user_tasks
        WHERE user_id=? AND (is_final IS NULL OR is_final=0);
        """,
        (user_id,),
    )
    return [str(r.get("job_task_id")) for r in rows if r.get("job_task_id") is not None]


async def get_latest_task_for_gmail(user_id: str, gmail: str) -> Optional[Dict[str, Any]]:
    return await db_fetchone(
        """
        SELECT job_task_id, status_norm, status_raw, is_final, gen_password, recovery_email, created_at
        FROM user_tasks
        WHERE user_id=? AND gmail=?
        ORDER BY created_at DESC, id DESC
        LIMIT 1;
        """,
        (user_id, gmail),
    )


async def task_stats(user_id: str) -> Dict[str, int]:
    total = (await db_fetchone(
        "SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks WHERE user_id=?;",
        (user_id,),
    ))["c"]

    confirmed = (await db_fetchone(
        "SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks WHERE user_id=? AND status_norm='confirmed';",
        (user_id,),
    ))["c"]

    declined = (await db_fetchone(
        "SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks WHERE user_id=? AND status_norm='declined';",
        (user_id,),
    ))["c"]

    processing = (await db_fetchone(
        "SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks WHERE user_id=? AND status_norm='processing';",
        (user_id,),
    ))["c"]

    pending = (await db_fetchone(
        "SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks WHERE user_id=? AND status_norm='pending';",
        (user_id,),
    ))["c"]

    return {
        "total": int(total),
        "confirmed": int(confirmed),
        "declined": int(declined),
        "processing": int(processing),
        "pending": int(pending),
    }


async def admin_overview_stats() -> Dict[str, int]:
    total_users = (await db_fetchone("SELECT COALESCE(COUNT(*),0) AS c FROM users;"))["c"]
    total_tasks = (await db_fetchone("SELECT COALESCE(COUNT(*),0) AS c FROM user_tasks;"))["c"]
    total_withdrawals = (await db_fetchone("SELECT COALESCE(COUNT(*),0) AS c FROM withdrawals;"))["c"]
    pending_withdrawals = (await db_fetchone("SELECT COALESCE(COUNT(*),0) AS c FROM withdrawals WHERE status='pending';"))["c"]
    return {
        "total_users": int(total_users),
        "total_tasks": int(total_tasks),
        "total_withdrawals": int(total_withdrawals),
        "pending_withdrawals": int(pending_withdrawals),
    }


async def admin_user_list() -> List[Dict[str, Any]]:
    return await db_fetchall(
        """
        SELECT user_id, username, is_banned, created_at
        FROM users
        ORDER BY created_at DESC;
        """
    )


async def admin_user_tasks(user_id: str) -> List[Dict[str, Any]]:
    return await db_fetchall(
        """
        SELECT job_id, job_task_id, gmail, gen_password, recovery_email, status_raw, status_norm, is_final, last_sync_at, created_at
        FROM user_tasks
        WHERE user_id=?
        ORDER BY created_at DESC, id DESC;
        """,
        (user_id,),
    )


async def admin_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    return await db_fetchone(
        """
        SELECT u.user_id,
               u.username,
               u.is_banned,
               u.created_at,
               d.gmail,
               d.recovery_email,
               d.secret,
               d.created_at AS draft_created_at,
               d.updated_at AS draft_updated_at
        FROM users u
        LEFT JOIN user_drafts d ON d.user_id = u.user_id
        WHERE u.user_id=?;
        """,
        (user_id,),
    )


async def admin_user_withdrawals(user_id: str) -> List[Dict[str, Any]]:
    return await db_fetchall(
        """
        SELECT withdrawal_id, amount, status, method, number, admin_txid, admin_note, created_at, updated_at, paid_at
        FROM withdrawals
        WHERE user_id=?
        ORDER BY created_at DESC;
        """,
        (user_id,),
    )


async def admin_user_gmail_list() -> List[Dict[str, Any]]:
    return await db_fetchall(
        """
        SELECT u.user_id,
               u.username,
               d.gmail,
               d.recovery_email,
               d.secret,
               d.created_at,
               d.updated_at,
               COALESCE(SUM(CASE WHEN t.status_norm='confirmed' THEN 1 ELSE 0 END), 0) AS confirmed_count,
               COALESCE(SUM(CASE WHEN t.status_norm='pending' THEN 1 ELSE 0 END), 0) AS pending_count,
               COALESCE(SUM(CASE WHEN t.status_norm='processing' THEN 1 ELSE 0 END), 0) AS processing_count,
               COALESCE(SUM(CASE WHEN t.status_norm='declined' THEN 1 ELSE 0 END), 0) AS declined_count
        FROM users u
        LEFT JOIN user_drafts d ON d.user_id = u.user_id
        LEFT JOIN user_tasks t ON t.user_id = u.user_id
        GROUP BY u.user_id, u.username, d.gmail, d.recovery_email, d.secret, d.created_at, d.updated_at
        ORDER BY d.updated_at DESC NULLS LAST, u.created_at DESC;
        """
    )


async def set_user_ban(user_id: str, banned: bool) -> None:
    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            _db_conn.execute("UPDATE users SET is_banned=? WHERE user_id=?;", (1 if banned else 0, user_id))
            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
        except Exception:
            _db_conn.rollback()
            raise

    await with_write_tx(_tx)


# =========================
# Draft helpers (gmail/password/recovery/secret)
# =========================
async def get_draft(user_id: str) -> Optional[Dict[str, Any]]:
    return await db_fetchone(
        """
        SELECT user_id, gmail, gen_password, recovery_email, secret, otp_digits, otp_period, qr_raw, created_at, updated_at
        FROM user_drafts
        WHERE user_id=?;
        """,
        (user_id,),
    )


async def clear_draft(user_id: str) -> None:
    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            _db_conn.execute("DELETE FROM user_drafts WHERE user_id=?;", (user_id,))
            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
        except Exception:
            _db_conn.rollback()
            raise

    await with_write_tx(_tx)


def _rand_letters(n: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    b = os.urandom(n)
    return "".join(alphabet[x % len(alphabet)] for x in b)


def _rand_digits(n: int) -> str:
    digits = "0123456789"
    b = os.urandom(n)
    return "".join(digits[x % len(digits)] for x in b)


async def generate_pretty_password() -> str:
    while True:
        candidate = f"{_rand_letters(10)}@{_rand_digits(4)}"
        exists = await db_fetchone(
            """
            SELECT 1 FROM user_tasks WHERE gen_password=?
            UNION
            SELECT 1 FROM user_drafts WHERE gen_password=?
            LIMIT 1;
            """,
            (candidate, candidate),
        )
        if not exists:
            return candidate


async def generate_recovery_email() -> str:
    while True:
        candidate = f"{_rand_letters(9)}{_rand_digits(3)}@{RECOVERY_DOMAIN}"
        exists = await db_fetchone(
            """
            SELECT 1 FROM user_tasks WHERE recovery_email=?
            UNION
            SELECT 1 FROM user_drafts WHERE recovery_email=?
            LIMIT 1;
            """,
            (candidate, candidate),
        )
        if not exists:
            return candidate


async def start_or_reset_draft(user_id: str, gmail: str) -> Dict[str, Any]:
    gmail = (gmail or "").strip().lower()
    if not gmail or "@" not in gmail:
        raise HTTPException(status_code=400, detail="Valid Gmail is required")

    last_task = await get_latest_task_for_gmail(user_id, gmail)
    if last_task and last_task.get("status_norm") in ("confirmed", "pending", "processing"):
        raise HTTPException(status_code=400, detail="This Gmail is already submitted or in progress")

    if last_task and last_task.get("status_norm") == "declined":
        gen_password = (last_task.get("gen_password") or "").strip()
        recovery_email = (last_task.get("recovery_email") or "").strip()
        if not gen_password or not recovery_email:
            gen_password = await generate_pretty_password()
            recovery_email = await generate_recovery_email()
    else:
        gen_password = await generate_pretty_password()
        recovery_email = await generate_recovery_email()

    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            _db_conn.execute(
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
            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
        except Exception:
            _db_conn.rollback()
            raise

    await with_write_tx(_tx)
    return await get_draft(user_id) or {}


async def save_secret_to_draft(user_id: str, secret: str, otp_digits: int, otp_period: int, qr_raw: str) -> None:
    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            _db_conn.execute(
                """
                UPDATE user_drafts
                SET secret=?, otp_digits=?, otp_period=?, qr_raw=?, updated_at=CURRENT_TIMESTAMP
                WHERE user_id=?;
                """,
                (secret, int(otp_digits), int(otp_period), qr_raw, user_id),
            )
            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
        except Exception:
            _db_conn.rollback()
            raise

    await with_write_tx(_tx)


# =========================
# QR Decode + TOTP
# =========================
BASE32_RE = re.compile(r"^[A-Z2-7]+=*$")


def _clean_base32(s: str) -> str:
    s = (s or "").strip().replace(" ", "").replace("-", "")
    return s.upper()


def decode_qr_payload_from_image(image_bytes: bytes) -> str:
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    detector = cv2.QRCodeDetector()

    data, _, _ = detector.detectAndDecode(img)
    if data and data.strip():
        return data.strip()

    img2 = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    data, _, _ = detector.detectAndDecode(gray)
    if data and data.strip():
        return data.strip()

    try:
        ok, decoded_info, _, _ = detector.detectAndDecodeMulti(img2)
        if ok and decoded_info:
            for d in decoded_info:
                if d and d.strip():
                    return d.strip()
    except Exception:
        pass

    raise ValueError("QR not detected. Please upload a clear QR-only screenshot (zoom QR).")


def extract_secret_from_qr_payload(payload: str) -> Tuple[str, int, int]:
    payload = (payload or "").strip()
    digits = 6
    period = 30

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
        return _clean_base32(sec), digits, period

    sec = _clean_base32(payload)
    if not sec:
        raise ValueError("Empty QR payload")

    test = sec + ("=" * ((8 - (len(sec) % 8)) % 8))
    if not BASE32_RE.match(test):
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
async def ledger_sum(user_id: str, kind: str) -> int:
    row = await db_fetchone(
        "SELECT COALESCE(SUM(amount), 0) AS total FROM user_ledger WHERE user_id=? AND kind=?;",
        (user_id, kind),
    )
    return int((row or {}).get("total") or 0)


async def reserved_withdraw_sum(user_id: str) -> int:
    row = await db_fetchone(
        "SELECT COALESCE(SUM(amount), 0) AS total FROM withdrawals WHERE user_id=? AND status='pending';",
        (user_id,),
    )
    return int((row or {}).get("total") or 0)


async def hold_balance(user_id: str) -> int:
    row = await db_fetchone(
        """
        SELECT COALESCE(COUNT(*), 0) AS c
        FROM user_tasks
        WHERE user_id=?
          AND (is_final IS NULL OR is_final=0)
          AND status_norm IN ('pending','processing');
        """,
        (user_id,),
    )
    return int((row or {}).get("c") or 0) * int(CREDIT_PER_CONFIRMED)


async def balances(user_id: str) -> Dict[str, int]:
    earned = await ledger_sum(user_id, "earn")
    earned += await ledger_sum(user_id, "admin_credit")
    withdrawn = await ledger_sum(user_id, "withdraw")
    reserved = await reserved_withdraw_sum(user_id)
    available = earned - withdrawn - reserved
    if available < 0:
        available = 0
    return {
        "available_balance": int(available),
        "hold_balance": int(await hold_balance(user_id)),
        "total_earned": int(earned),
        "total_withdrawn": int(withdrawn),
        "reserved_withdraw_balance": int(reserved),
    }


async def ledger_add_once(user_id: str, kind: str, amount: int, ref: Optional[str], meta: Optional[str]) -> bool:
    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            _db_conn.execute(
                "INSERT OR IGNORE INTO user_ledger(user_id, kind, amount, ref, meta) VALUES (?, ?, ?, ?, ?);",
                (user_id, kind, int(amount), ref, meta),
            )
            inserted = _db_changes_sync() == 1
            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
            return inserted
        except Exception:
            _db_conn.rollback()
            raise

    return bool(await with_write_tx(_tx))


async def create_withdraw_request(user_id: str, amount: int, method: str, number: str) -> str:
    if amount <= 0:
        raise HTTPException(status_code=400, detail="amount must be > 0")
    method = (method or "").strip()
    number = (number or "").strip()
    if not method or not number:
        raise HTTPException(status_code=400, detail="method and number are required")

    withdrawal_id = str(uuid.uuid4())

    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            # compute available inside same tx
            r1 = _db_fetchone_sync(
                "SELECT COALESCE(SUM(amount),0) AS t FROM user_ledger WHERE user_id=? AND kind IN ('earn','admin_credit');",
                (user_id,),
            )
            earned = int((r1 or {}).get("t") or 0)

            r2 = _db_fetchone_sync(
                "SELECT COALESCE(SUM(amount),0) AS t FROM user_ledger WHERE user_id=? AND kind='withdraw';",
                (user_id,),
            )
            withdrawn = int((r2 or {}).get("t") or 0)

            r3 = _db_fetchone_sync(
                "SELECT COALESCE(SUM(amount),0) AS t FROM withdrawals WHERE user_id=? AND status='pending';",
                (user_id,),
            )
            reserved = int((r3 or {}).get("t") or 0)

            available = earned - withdrawn - reserved
            if available < amount:
                raise HTTPException(
                    status_code=400,
                    detail={"message": "Insufficient balance", "available": max(int(available), 0)},
                )

            _db_conn.execute(
                """
                INSERT INTO withdrawals(withdrawal_id, user_id, amount, status, method, number, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?, CURRENT_TIMESTAMP);
                """,
                (withdrawal_id, user_id, int(amount), method, number),
            )
            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
            return withdrawal_id
        except Exception:
            _db_conn.rollback()
            raise

    return str(await with_write_tx(_tx))


async def list_withdrawals(user_id: str) -> List[Dict[str, Any]]:
    return await db_fetchall(
        """
        SELECT withdrawal_id, amount, status, method, number, meta, admin_txid, admin_note, created_at, updated_at, paid_at
        FROM withdrawals
        WHERE user_id=?
        ORDER BY created_at DESC;
        """,
        (user_id,),
    )


async def list_all_withdrawals(status: Optional[str] = None) -> List[Dict[str, Any]]:
    if status:
        return await db_fetchall(
            """
            SELECT withdrawal_id, user_id, amount, status, method, number, meta, admin_txid, admin_note, created_at, updated_at, paid_at
            FROM withdrawals
            WHERE status=?
            ORDER BY created_at DESC;
            """,
            (status,),
        )
    return await db_fetchall(
        """
        SELECT withdrawal_id, user_id, amount, status, method, number, meta, admin_txid, admin_note, created_at, updated_at, paid_at
        FROM withdrawals
        ORDER BY created_at DESC;
        """
    )


async def admin_withdrawals_overview(status: Optional[str] = None) -> List[Dict[str, Any]]:
    if status:
        return await db_fetchall(
            """
            SELECT w.withdrawal_id,
                   w.user_id,
                   u.username,
                   u.is_banned,
                   d.gmail,
                   d.recovery_email,
                   w.amount,
                   w.status,
                   w.method,
                   w.number,
                   w.meta,
                   w.admin_txid,
                   w.admin_note,
                   w.created_at,
                   w.updated_at,
                   w.paid_at
            FROM withdrawals w
            JOIN users u ON u.user_id = w.user_id
            LEFT JOIN user_drafts d ON d.user_id = w.user_id
            WHERE w.status=?
            ORDER BY w.created_at DESC;
            """,
            (status,),
        )
    return await db_fetchall(
        """
        SELECT w.withdrawal_id,
               w.user_id,
               u.username,
               u.is_banned,
               d.gmail,
               d.recovery_email,
               w.amount,
               w.status,
               w.method,
               w.number,
               w.meta,
               w.admin_txid,
               w.admin_note,
               w.created_at,
               w.updated_at,
               w.paid_at
        FROM withdrawals w
        JOIN users u ON u.user_id = w.user_id
        LEFT JOIN user_drafts d ON d.user_id = w.user_id
        ORDER BY w.created_at DESC;
        """
    )


async def admin_confirm_withdraw(withdrawal_id: str, txid: str, note: str) -> None:
    txid = (txid or "").strip()
    note = (note or "").strip()

    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            w = _db_fetchone_sync(
                "SELECT withdrawal_id, user_id, amount, status FROM withdrawals WHERE withdrawal_id=?;",
                (withdrawal_id,),
            )
            if not w:
                raise HTTPException(status_code=404, detail="Withdrawal not found")
            if str(w.get("status")) != "pending":
                raise HTTPException(status_code=400, detail={"message": "Not pending", "status": w.get("status")})

            _db_conn.execute(
                "INSERT OR IGNORE INTO user_ledger(user_id, kind, amount, ref, meta) VALUES (?, 'withdraw', ?, ?, ?);",
                (str(w["user_id"]), int(w["amount"]), str(withdrawal_id), f"txid={txid};note={note}"),
            )
            if _db_changes_sync() != 1:
                raise HTTPException(status_code=409, detail="Already finalized in ledger")

            _db_conn.execute(
                """
                UPDATE withdrawals
                SET status='paid', admin_txid=?, admin_note=?, paid_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
                WHERE withdrawal_id=?;
                """,
                (txid or None, note or None, withdrawal_id),
            )

            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
        except Exception:
            _db_conn.rollback()
            raise

    await with_write_tx(_tx)


async def admin_reject_withdraw(withdrawal_id: str, note: str) -> None:
    note = (note or "").strip()

    def _tx():
        if _db_conn is None:
            raise RuntimeError("DB not ready")
        _db_conn.execute("BEGIN IMMEDIATE;")
        try:
            w = _db_fetchone_sync(
                "SELECT withdrawal_id, status FROM withdrawals WHERE withdrawal_id=?;",
                (withdrawal_id,),
            )
            if not w:
                raise HTTPException(status_code=404, detail="Withdrawal not found")
            if str(w.get("status")) != "pending":
                raise HTTPException(status_code=400, detail={"message": "Not pending", "status": w.get("status")})

            _db_conn.execute(
                """
                UPDATE withdrawals
                SET status='rejected', admin_note=?, updated_at=CURRENT_TIMESTAMP
                WHERE withdrawal_id=?;
                """,
                (note or None, withdrawal_id),
            )

            _db_conn.commit()
            try:
                _db_conn.sync()
            except Exception:
                pass
        except Exception:
            _db_conn.rollback()
            raise

    await with_write_tx(_tx)


# =========================
# Sync logic
# =========================
async def sync_user_tasks(user_id: str) -> None:
    require_upstream_config()
    task_ids = await get_non_final_task_ids(user_id)
    if not task_ids:
        return

    for tid in task_ids:
        detail = await call_upstream_details(task_id=tid)
        raw_status = detail.get("status")
        norm = normalize_status(raw_status)
        final = is_final_status(norm)

        # Update task and ledger in a single DB tx
        def _tx():
            if _db_conn is None:
                raise RuntimeError("DB not ready")
            _db_conn.execute("BEGIN IMMEDIATE;")
            try:
                _db_conn.execute(
                    """
                    UPDATE user_tasks
                    SET status_raw=?, status_norm=?, is_final=?, last_sync_at=CURRENT_TIMESTAMP
                    WHERE user_id=? AND job_task_id=?;
                    """,
                    (raw_status, norm, 1 if final else 0, user_id, tid),
                )
                if final and norm == "confirmed":
                    _db_conn.execute(
                        "INSERT OR IGNORE INTO user_ledger(user_id, kind, amount, ref, meta) VALUES (?, 'earn', ?, ?, NULL);",
                        (user_id, int(CREDIT_PER_CONFIRMED), tid),
                    )
                _db_conn.commit()
                try:
                    _db_conn.sync()
                except Exception:
                    pass
            except Exception:
                _db_conn.rollback()
                raise

        await with_write_tx(_tx)


# =========================
# Web Pages
# =========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        session = await get_session(sid)
        if session and session.get("kind") == "user":
            user = await get_user_by_id(str(session.get("principal_id") or ""))
            if user:
                return RedirectResponse(url="/user/dashboard", status_code=302)
        if session and session.get("kind") == "admin":
            admin = await get_admin_by_id(str(session.get("principal_id") or ""))
            if admin:
                return RedirectResponse(url="/admin", status_code=302)
    return templates.TemplateResponse("home.html", {"request": request})


# --- USER AUTH ---
@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request, "error": None})


@app.post("/signup")
async def signup_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    username = username.strip().lower()
    if len(username) < 3 or len(password) < 4:
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Invalid username/password"})

    if await get_user_by_username(username):
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Username already exists"})

    u = await create_user(username, password)
    sid = await create_session("user", u["user_id"])
    resp = RedirectResponse(url="/user/dashboard", status_code=302)
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    remember_me: Optional[str] = Form(None),
):
    username = username.strip().lower()
    u = await get_user_by_username(username)
    if not u or not verify_password(password, str(u.get("password_hash") or "")):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    if int(u.get("is_banned") or 0) == 1:
        return templates.TemplateResponse("login.html", {"request": request, "error": "User is banned"})

    sid = await create_session("user", str(u["user_id"]))
    resp = RedirectResponse(url="/user/dashboard", status_code=302)
    max_age = 30 * 24 * 60 * 60 if remember_me else None
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax", max_age=max_age)
    return resp


@app.get("/logout")
async def logout(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        await delete_session(sid)
    resp = RedirectResponse(url="/", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# --- ADMIN AUTH ---
@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": None})


@app.post("/admin/login")
async def admin_login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    username = username.strip().lower()
    a = await get_admin_by_username(username)
    if not a or not verify_password(password, str(a.get("password_hash") or "")):
        return templates.TemplateResponse("admin_login.html", {"request": request, "error": "Invalid credentials"})

    sid = await create_session("admin", str(a["admin_id"]))
    resp = RedirectResponse(url="/admin", status_code=302)
    resp.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return resp


@app.get("/admin/logout")
async def admin_logout(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        await delete_session(sid)
    resp = RedirectResponse(url="/admin/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# =========================
# USER PAGES
# =========================
@app.get("/user/dashboard", response_class=HTMLResponse)
async def user_dashboard(request: Request):
    user = await require_user(request)
    try:
        await sync_user_tasks(str(user["user_id"]))
    except Exception:
        pass

    b = await balances(str(user["user_id"]))
    stats = await task_stats(str(user["user_id"]))

    return templates.TemplateResponse(
        "user_dashboard.html",
        {"request": request, "user": user, "balances": b, "stats": stats},
    )


@app.get("/user/gmail", response_class=HTMLResponse)
async def user_gmail_page(request: Request):
    user = await require_user(request)
    draft = await get_draft(str(user["user_id"]))
    return templates.TemplateResponse(
        "user_gmail.html",
        {"request": request, "user": user, "draft": draft, "error": None, "success": None},
    )


@app.post("/user/gmail/start")
async def user_gmail_start(request: Request, gmail: str = Form(...)):
    user = await require_user(request)
    try:
        await start_or_reset_draft(str(user["user_id"]), gmail)
    except HTTPException as e:
        draft = await get_draft(str(user["user_id"]))
        return templates.TemplateResponse(
            "user_gmail.html",
            {"request": request, "user": user, "draft": draft, "error": str(e.detail), "success": None},
        )

    return RedirectResponse(url="/user/gmail", status_code=302)


@app.post("/user/gmail/upload-qr")
async def user_gmail_upload_qr(request: Request, qr_image: UploadFile = File(...)):
    user = await require_user(request)
    draft = await get_draft(str(user["user_id"]))
    if not draft or not draft.get("gmail"):
        return templates.TemplateResponse(
            "user_gmail.html",
            {"request": request, "user": user, "draft": draft, "error": "Please enter Gmail first", "success": None},
        )

    try:
        img_bytes = await qr_image.read()
        payload = decode_qr_payload_from_image(img_bytes)
        secret, digits, period = extract_secret_from_qr_payload(payload)
        await save_secret_to_draft(
            str(user["user_id"]),
            secret=secret,
            otp_digits=digits,
            otp_period=period,
            qr_raw=payload,
        )
    except Exception as e:
        draft = await get_draft(str(user["user_id"]))
        return templates.TemplateResponse(
            "user_gmail.html",
            {"request": request, "user": user, "draft": draft, "error": f"QR decode failed: {e}", "success": None},
        )

    return RedirectResponse(url="/user/gmail", status_code=302)


@app.get("/user/gmail/totp")
async def user_gmail_totp(request: Request):
    user = await require_user(request)
    draft = await get_draft(str(user["user_id"]))
    if not draft or not draft.get("secret"):
        return JSONResponse({"ok": False, "error": "No secret yet"}, status_code=400)

    try:
        code, remaining = totp_now(
            secret_b32=str(draft["secret"]),
            digits=int(draft.get("otp_digits") or 6),
            period=int(draft.get("otp_period") or 30),
        )
        return {"ok": True, "code": code, "remaining": remaining, "period": int(draft.get("otp_period") or 30)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/user/gmail/submit")
async def user_gmail_submit(request: Request):
    user = await require_user(request)
    require_upstream_config()

    draft = await get_draft(str(user["user_id"]))
    if not draft:
        return RedirectResponse(url="/user/gmail", status_code=302)

    existing_task = await get_latest_task_for_gmail(str(user["user_id"]), str(draft.get("gmail") or ""))
    if existing_task and existing_task.get("status_norm") in ("confirmed", "pending", "processing"):
        return templates.TemplateResponse(
            "user_gmail.html",
            {
                "request": request,
                "user": user,
                "draft": draft,
                "error": "This Gmail is already submitted or in progress",
                "success": None,
            },
        )

    try:
        job_proof = format_job_proof(draft)
        data = await call_upstream_submit(job_proof=job_proof)

        job_task_id = data.get("job_task_id") or data.get("job_taskId") or data.get("jobTaskId")
        job_id = data.get("job_id") or UPSTREAM_JOB_ID
        if not job_task_id:
            raise HTTPException(status_code=502, detail="Upstream missing job_task_id")

        await upsert_user_task(
            user_id=str(user["user_id"]),
            job_id=str(job_id),
            job_task_id=str(job_task_id),
            gmail=str(draft.get("gmail") or ""),
            gen_password=str(draft.get("gen_password") or ""),
            recovery_email=str(draft.get("recovery_email") or ""),
        )
    except HTTPException as e:
        draft = await get_draft(str(user["user_id"]))
        return templates.TemplateResponse(
            "user_gmail.html",
            {"request": request, "user": user, "draft": draft, "error": str(e.detail), "success": None},
        )

    return RedirectResponse(url="/user/tasks", status_code=302)


@app.post("/user/gmail/reset")
async def user_gmail_reset(request: Request):
    user = await require_user(request)
    await clear_draft(str(user["user_id"]))
    return RedirectResponse(url="/user/gmail", status_code=302)


@app.get("/user/withdraw", response_class=HTMLResponse)
async def user_withdraw_page(request: Request):
    user = await require_user(request)
    b = await balances(str(user["user_id"]))
    w = await list_withdrawals(str(user["user_id"]))
    return templates.TemplateResponse(
        "user_withdraw.html",
        {"request": request, "user": user, "balances": b, "withdrawals": w, "error": None},
    )


@app.post("/user/withdraw")
async def user_withdraw_action(
    request: Request,
    amount: int = Form(...),
    method: str = Form(...),
    number: str = Form(...),
):
    user = await require_user(request)
    try:
        await create_withdraw_request(user_id=str(user["user_id"]), amount=int(amount), method=method, number=number)
    except HTTPException as e:
        b = await balances(str(user["user_id"]))
        w = await list_withdrawals(str(user["user_id"]))
        return templates.TemplateResponse(
            "user_withdraw.html",
            {"request": request, "user": user, "balances": b, "withdrawals": w, "error": str(e.detail)},
        )

    return RedirectResponse(url="/user/withdraw", status_code=302)


@app.get("/user/tasks", response_class=HTMLResponse)
async def user_tasks_page(request: Request):
    user = await require_user(request)
    try:
        await sync_user_tasks(str(user["user_id"]))
    except Exception:
        pass
    tasks = await get_tasks_for_user(str(user["user_id"]))
    return templates.TemplateResponse("user_tasks.html", {"request": request, "user": user, "tasks": tasks})


# =========================
# ADMIN DASH
# =========================
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    admin = await require_admin(request)
    return templates.TemplateResponse(
        "admin_dashboard.html",
        {"request": request, "admin": admin, "stats": await admin_overview_stats()},
    )


@app.get("/admin/credit", response_class=HTMLResponse)
async def admin_credit_page(request: Request):
    admin = await require_admin(request)
    return templates.TemplateResponse(
        "admin_credit.html",
        {"request": request, "admin": admin, "error": None, "success": None},
    )


@app.post("/admin/credit")
async def admin_add_credit(
    request: Request,
    target_username: str = Form(...),
    amount: int = Form(...),
    note: str = Form(""),
):
    admin = await require_admin(request)
    target_username = target_username.strip().lower()
    u = await get_user_by_username(target_username)
    if not u:
        return templates.TemplateResponse(
            "admin_credit.html",
            {"request": request, "admin": admin, "error": "User not found", "success": None},
        )

    amount = int(amount)
    if amount <= 0:
        return templates.TemplateResponse(
            "admin_credit.html",
            {"request": request, "admin": admin, "error": "Amount must be > 0", "success": None},
        )

    ref = str(uuid.uuid4())
    ok = await ledger_add_once(str(u["user_id"]), "admin_credit", amount, ref=ref, meta=f"note={note}")
    if not ok:
        # extremely unlikely
        return templates.TemplateResponse(
            "admin_credit.html",
            {"request": request, "admin": admin, "error": "Credit already applied", "success": None},
        )

    return templates.TemplateResponse(
        "admin_credit.html",
        {"request": request, "admin": admin, "error": None, "success": f"Credited {amount} to {target_username}"},
    )


@app.get("/admin/withdrawals", response_class=HTMLResponse)
async def admin_withdrawals_page(request: Request):
    admin = await require_admin(request)
    w_all = await admin_withdrawals_overview(status=None)
    return templates.TemplateResponse(
        "admin_withdrawals.html",
        {"request": request, "admin": admin, "withdrawals": w_all, "error": None, "success": None},
    )


@app.post("/admin/withdrawals/confirm")
async def admin_confirm_withdrawal_web(
    request: Request,
    withdrawal_id: str = Form(...),
    txid: str = Form(""),
    note: str = Form(""),
):
    await require_admin(request)
    await admin_confirm_withdraw(withdrawal_id=withdrawal_id, txid=txid, note=note)
    return RedirectResponse(url="/admin/withdrawals", status_code=302)


@app.post("/admin/withdrawals/reject")
async def admin_reject_withdrawal_web(
    request: Request,
    withdrawal_id: str = Form(...),
    note: str = Form(""),
):
    await require_admin(request)
    await admin_reject_withdraw(withdrawal_id=withdrawal_id, note=note)
    return RedirectResponse(url="/admin/withdrawals", status_code=302)


@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(request: Request):
    admin = await require_admin(request)
    users = await admin_user_list()
    return templates.TemplateResponse(
        "admin_users.html",
        {"request": request, "admin": admin, "users": users, "error": None, "success": None},
    )


@app.post("/admin/users/ban")
async def admin_users_ban(request: Request, user_id: str = Form(...), action: str = Form(...)):
    await require_admin(request)
    action = (action or "").strip().lower()
    banned = action == "ban"
    await set_user_ban(user_id=user_id, banned=banned)
    return RedirectResponse(url="/admin/users", status_code=302)


@app.get("/admin/users/{user_id}", response_class=HTMLResponse)
async def admin_user_detail_page(request: Request, user_id: str):
    admin = await require_admin(request)
    profile = await admin_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    tasks = await admin_user_tasks(user_id)
    withdrawals = await admin_user_withdrawals(user_id)
    return templates.TemplateResponse(
        "admin_user_detail.html",
        {"request": request, "admin": admin, "profile": profile, "tasks": tasks, "withdrawals": withdrawals, "error": None, "success": None},
    )


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
