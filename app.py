import collections
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, make_response, render_template, request, redirect, url_for
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

from rag_engine import RAGEngine

load_dotenv(override=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

log = logging.getLogger(__name__)

SITE_PASSWORD = os.getenv("SITE_PASSWORD", "student")
TEACHER_PASSWORD = os.getenv("TEACHER_PASSWORD", "")

_IS_PRODUCTION = bool(os.getenv("RENDER") or os.getenv("VERCEL"))

if SITE_PASSWORD == "student":
    if _IS_PRODUCTION:
        raise RuntimeError(
            "SITE_PASSWORD is still set to the default 'student'. "
            "Set a strong password in your Vercel/Render environment variables."
        )
    else:
        log.warning("WARNING: SITE_PASSWORD is the default 'student'. Set a strong password before deploying.")

_SECRET = os.getenv("SECRET_KEY")
if not _SECRET:
    raise RuntimeError(
        "SECRET_KEY is not set. Add it to your .env file or Render environment variables."
    )

# ── Rate limiting (login brute-force protection) ────────────────────────────
# Stores (ip -> [timestamp, ...]) for failed attempts
_failed_attempts: dict = collections.defaultdict(list)
_MAX_ATTEMPTS = 5       # max failed tries
_LOCKOUT_SECONDS = 300  # 5 minutes

# ── Rate limiting (chat questions) ──────────────────────────────────────────
_chat_attempts: dict = collections.defaultdict(list)
_CHAT_MAX = 5           # max questions per hour per IP
_CHAT_WINDOW = 3600     # 1 hour in seconds


def _is_chat_limited(ip: str) -> tuple[bool, int]:
    now = time.time()
    hits = [t for t in _chat_attempts[ip] if now - t < _CHAT_WINDOW]
    _chat_attempts[ip] = hits
    stale = [k for k, v in _chat_attempts.items() if not v]
    for k in stale:
        _chat_attempts.pop(k, None)
    if len(hits) >= _CHAT_MAX:
        wait = int(_CHAT_WINDOW - (now - hits[0])) + 1
        return True, wait
    return False, 0


def _record_chat(ip: str):
    _chat_attempts[ip].append(time.time())


def _client_ip() -> str:
    return request.remote_addr or ""


def _is_rate_limited(ip: str) -> bool:
    now = time.time()
    attempts = [t for t in _failed_attempts[ip] if now - t < _LOCKOUT_SECONDS]
    _failed_attempts[ip] = attempts
    # Prune stale IPs to prevent unbounded memory growth
    stale = [k for k, v in _failed_attempts.items() if not v]
    for k in stale:
        _failed_attempts.pop(k, None)
    return len(attempts) >= _MAX_ATTEMPTS


def _record_failure(ip: str):
    _failed_attempts[ip].append(time.time())


def _clear_failures(ip: str):
    _failed_attempts.pop(ip, None)


# ── Auth helpers ────────────────────────────────────────────────────────────

def _make_token():
    return hmac.new(_SECRET.encode(), SITE_PASSWORD.encode(), hashlib.sha256).hexdigest()


def _make_teacher_token():
    return hmac.new(_SECRET.encode(), f"teacher:{TEACHER_PASSWORD}".encode(), hashlib.sha256).hexdigest()


def _is_authenticated():
    return hmac.compare_digest(request.cookies.get("auth", ""), _make_token())


def _is_teacher():
    if not TEACHER_PASSWORD:
        return False
    return hmac.compare_digest(request.cookies.get("teacher_auth", ""), _make_teacher_token())


def require_teacher():
    if not _is_teacher():
        return jsonify({"error": "Teacher access required"}), 403


# ── CSRF helpers ────────────────────────────────────────────────────────────

def _new_csrf_token() -> str:
    return secrets.token_hex(32)


def _csrf_valid() -> bool:
    cookie = request.cookies.get("csrf_token", "")
    header = request.headers.get("X-CSRF-Token", "")
    return bool(cookie) and hmac.compare_digest(cookie, header)


# ── Path param validation ────────────────────────────────────────────────────

def _safe_param(value: str) -> bool:
    return bool(value) and "\x00" not in value and ".." not in value \
        and "/" not in value and "\\" not in value


@app.after_request
def set_security_headers(response):
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if _IS_PRODUCTION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 50 MB."}), 413


@app.before_request
def require_login():
    if request.endpoint in ("login", "logout", "static", "ping"):
        return
    if not _is_authenticated():
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not authenticated"}), 401
        return redirect(url_for("login"))
    if request.method in ("POST", "DELETE", "PUT", "PATCH") and request.path.startswith("/api/"):
        if not _csrf_valid():
            return jsonify({"error": "CSRF validation failed"}), 403


# ── Auth routes ─────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        ip = _client_ip()
        mode = request.form.get("mode", "student")
        password = request.form.get("password", "")
        if _is_rate_limited(ip):
            error = "Too many failed attempts. Try again in 5 minutes."
        elif mode == "teacher" and TEACHER_PASSWORD and hmac.compare_digest(password, TEACHER_PASSWORD):
            _clear_failures(ip)
            csrf = _new_csrf_token()
            resp = make_response(redirect(url_for("index") + "?teacher=1"))
            resp.set_cookie("auth", _make_token(), httponly=True,
                            samesite="Lax", secure=_IS_PRODUCTION,
                            max_age=60 * 60 * 8)
            resp.set_cookie("teacher_auth", _make_teacher_token(), httponly=True,
                            samesite="Lax", secure=_IS_PRODUCTION,
                            max_age=60 * 60 * 8)
            resp.set_cookie("csrf_token", csrf, httponly=False,
                            samesite="Lax", secure=_IS_PRODUCTION,
                            max_age=60 * 60 * 8)
            return resp
        elif mode != "teacher" and hmac.compare_digest(password, SITE_PASSWORD):
            _clear_failures(ip)
            csrf = _new_csrf_token()
            resp = make_response(redirect(url_for("index")))
            resp.set_cookie("auth", _make_token(), httponly=True,
                            samesite="Lax", secure=_IS_PRODUCTION,
                            max_age=60 * 60 * 8)
            resp.set_cookie("csrf_token", csrf, httponly=False,
                            samesite="Lax", secure=_IS_PRODUCTION,
                            max_age=60 * 60 * 8)
            return resp
        else:
            _record_failure(ip)
            error = "Onjuist wachtwoord." if mode == "teacher" else "Incorrect password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    resp = make_response(redirect(url_for("login")))
    resp.delete_cookie("auth")
    resp.delete_cookie("teacher_auth")
    return resp


@app.route("/api/teacher-login", methods=["POST"])
def teacher_login():
    if not TEACHER_PASSWORD:
        return jsonify({"error": "Teacher access not configured"}), 403
    ip = _client_ip()
    if _is_rate_limited(ip):
        return jsonify({"error": "Too many failed attempts. Try again in 5 minutes."}), 429
    data = request.get_json(silent=True) or {}
    pw = str(data.get("password", ""))[:256]
    if not hmac.compare_digest(pw, TEACHER_PASSWORD):
        _record_failure(ip)
        return jsonify({"error": "Incorrect password"}), 403
    _clear_failures(ip)
    resp = make_response(jsonify({"success": True}))
    resp.set_cookie("teacher_auth", _make_teacher_token(), httponly=True,
                    samesite="Lax", secure=_IS_PRODUCTION,
                    max_age=60 * 60 * 8)
    return resp


# ── Data helpers ─────────────────────────────────────────────────────────────

_ON_VERCEL = bool(os.getenv("VERCEL"))
_WRITABLE_ROOT = Path("/tmp") if _ON_VERCEL else Path(".")

DATA_FOLDER = _WRITABLE_ROOT / "vectordb"
DATA_FOLDER.mkdir(exist_ok=True, parents=True)

UPLOAD_FOLDER = _WRITABLE_ROOT / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)

COURSES_FILE = DATA_FOLDER / "courses.json"
FILE_META_FILE = DATA_FOLDER / "file_meta.json"
STATS_FILE     = DATA_FOLDER / "stats.json"
FEEDBACK_FILE  = DATA_FOLDER / "feedback.json"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

MAX_QUESTION_LENGTH = 1000
MAX_COURSE_NAME_LENGTH = 100

# ── Supabase client ───────────────────────────────────────────────────────────

_SUPABASE_URL = os.getenv("SUPABASE_URL")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY")
_sb = None
if _SUPABASE_URL and _SUPABASE_KEY:
    from supabase import create_client as _create_sb
    _sb = _create_sb(_SUPABASE_URL, _SUPABASE_KEY)

_rag = None


def get_rag() -> RAGEngine:
    global _rag
    if _rag is None:
        _rag = RAGEngine()
    return _rag


def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ── Courses ───────────────────────────────────────────────────────────────────

def load_course_names() -> list:
    if _sb:
        result = _sb.table("courses").select("name").order("position").execute()
        return [r["name"] for r in (result.data or [])]
    if not COURSES_FILE.exists():
        return []
    try:
        return json.loads(COURSES_FILE.read_text())
    except Exception:
        return []


def save_course_name(name: str):
    if _sb:
        existing = [r["name"] for r in (_sb.table("courses").select("name").execute().data or [])]
        if name not in existing:
            pos = len(existing)
            _sb.table("courses").insert({"name": name, "position": pos}).execute()
        return
    names = load_course_names()
    if name not in names:
        names.append(name)
        COURSES_FILE.write_text(json.dumps(names))


def remove_course_name(name: str):
    if _sb:
        _sb.table("courses").delete().eq("name", name).execute()
        return
    names = [n for n in load_course_names() if n != name]
    COURSES_FILE.write_text(json.dumps(names))


# ── File metadata ─────────────────────────────────────────────────────────────

def load_file_meta() -> dict:
    if _sb:
        meta_rows = _sb.table("file_metadata").select("*").order("position").execute().data or []
        welcome_rows = _sb.table("course_welcome").select("*").execute().data or []
        result: dict = {}
        for row in meta_rows:
            course, filename = row["course"], row["filename"]
            result.setdefault(course, {"__order__": []})
            result[course][filename] = {
                "display_name": row.get("display_name") or filename,
                "upload_date": row.get("upload_date") or "",
                "file_size": row.get("file_size") or 0,
            }
            result[course]["__order__"].append(filename)
        for row in welcome_rows:
            course = row["course"]
            result.setdefault(course, {"__order__": []})
            result[course]["__welcome__"] = row.get("welcome_text", "")
        return result
    if not FILE_META_FILE.exists():
        return {}
    try:
        return json.loads(FILE_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_file_meta(meta: dict):
    if _sb:
        for course, course_data in meta.items():
            if not isinstance(course_data, dict):
                continue
            order = course_data.get("__order__", [])
            welcome = course_data.get("__welcome__")
            _sb.table("file_metadata").delete().eq("course", course).execute()
            rows = []
            for key, val in course_data.items():
                if key.startswith("__") or not isinstance(val, dict):
                    continue
                rows.append({
                    "course": course,
                    "filename": key,
                    "display_name": val.get("display_name", key),
                    "upload_date": val.get("upload_date", ""),
                    "file_size": val.get("file_size", 0),
                    "position": order.index(key) if key in order else 9999,
                })
            if rows:
                _sb.table("file_metadata").insert(rows).execute()
            if welcome is not None:
                _sb.table("course_welcome").upsert({"course": course, "welcome_text": welcome}).execute()
        return
    FILE_META_FILE.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


# ── Stats & feedback ──────────────────────────────────────────────────────────

def load_stats() -> list:
    if _sb:
        rows = _sb.table("stats").select("question,books,timestamp").order("id", desc=True).limit(500).execute().data or []
        rows.reverse()
        return [{"question": r["question"], "books": r["books"] or [], "timestamp": r["timestamp"]} for r in rows]
    if not STATS_FILE.exists():
        return []
    try:
        return json.loads(STATS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def load_feedback() -> list:
    if _sb:
        rows = _sb.table("feedback").select("question,answer,books,rating,timestamp").order("id", desc=True).limit(1000).execute().data or []
        rows.reverse()
        return [{"question": r["question"], "answer": r["answer"], "books": r["books"] or [], "rating": r["rating"], "timestamp": r["timestamp"]} for r in rows]
    if not FEEDBACK_FILE.exists():
        return []
    try:
        return json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def log_feedback(question: str, answer: str, books: list, rating: str):
    if _sb:
        _sb.table("feedback").insert({
            "question": question,
            "answer": answer,
            "books": books,
            "rating": rating,
            "timestamp": datetime.utcnow().isoformat(),
        }).execute()
        return
    feedback = load_feedback()
    feedback.append({"question": question, "answer": answer, "books": books, "rating": rating, "timestamp": datetime.utcnow().isoformat()})
    if len(feedback) > 1000:
        feedback = feedback[-1000:]
    FEEDBACK_FILE.write_text(json.dumps(feedback, ensure_ascii=False), encoding="utf-8")


def log_question(question: str, books: list):
    if _sb:
        _sb.table("stats").insert({
            "question": question,
            "books": books,
            "timestamp": datetime.utcnow().isoformat(),
        }).execute()
        return
    stats = load_stats()
    stats.append({"question": question, "books": books, "timestamp": datetime.utcnow().isoformat()})
    if len(stats) > 500:
        stats = stats[-500:]
    STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ping")
def ping():
    return jsonify({"ok": True})


# Courses

@app.route("/api/courses", methods=["GET"])
def get_courses():
    try:
        rag_courses = {c["name"]: c["files"] for c in get_rag().get_courses()}
        saved = load_course_names()
        extra = sorted(n for n in rag_courses if n not in saved)
        all_names = saved + extra
        meta = load_file_meta()
        result = []
        for n in all_names:
            raw_files = rag_courses.get(n, [])
            saved_order = meta.get(n, {}).get("__order__", [])
            ordered = [f for f in saved_order if f in raw_files] + [f for f in raw_files if f not in saved_order]
            course_meta = meta.get(n, {})
            files = [
                {"filename": f, **course_meta.get(f, {"display_name": f, "upload_date": "", "file_size": 0})}
                for f in ordered
            ]
            result.append({"name": n, "files": files, "welcome": meta.get(n, {}).get("__welcome__", "")})
        return jsonify(result)
    except Exception:
        log.exception("get_courses failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


@app.route("/api/courses/reorder", methods=["POST"])
def reorder_courses():
    err = require_teacher()
    if err: return err
    data = request.get_json(silent=True) or {}
    raw_order = data.get("order") or []
    if not isinstance(raw_order, list):
        return jsonify({"error": "Order must be a list"}), 400
    order = [str(item)[:MAX_COURSE_NAME_LENGTH] for item in raw_order[:200] if isinstance(item, str)]
    if _sb:
        for i, name in enumerate(order):
            _sb.table("courses").upsert({"name": name, "position": i}).execute()
    else:
        COURSES_FILE.write_text(json.dumps(order))
    return jsonify({"success": True})


@app.route("/api/courses", methods=["POST"])
def create_course():
    err = require_teacher()
    if err: return err
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()[:MAX_COURSE_NAME_LENGTH]
    if not name:
        return jsonify({"error": "Course name is required"}), 400
    try:
        save_course_name(name)
        return jsonify({"success": True})
    except Exception as e:
        log.exception("create_course failed")
        return jsonify({"error": f"Cursus aanmaken mislukt: {e}"}), 500


@app.route("/api/courses/<course_name>", methods=["DELETE"])
def delete_course(course_name):
    err = require_teacher()
    if err: return err
    if not _safe_param(course_name):
        return jsonify({"error": "Invalid course name"}), 400
    try:
        get_rag().delete_course(course_name)
        remove_course_name(course_name)
        return jsonify({"success": True})
    except Exception:
        log.exception("delete_course failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


@app.route("/api/courses/<course_name>/files/<filename>", methods=["DELETE"])
def delete_file(course_name, filename):
    err = require_teacher()
    if err: return err
    if not _safe_param(course_name) or not _safe_param(filename):
        return jsonify({"error": "Invalid parameters"}), 400
    try:
        get_rag().delete_file(course_name, filename)
        file_meta = load_file_meta()
        if course_name in file_meta:
            file_meta[course_name].pop(filename, None)
            # Also remove from __order__
            order = file_meta[course_name].get("__order__", [])
            if filename in order:
                file_meta[course_name]["__order__"] = [f for f in order if f != filename]
            save_file_meta(file_meta)
        return jsonify({"success": True})
    except Exception:
        log.exception("delete_file failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


# Upload

@app.route("/api/upload", methods=["POST"])
def upload():
    err = require_teacher()
    if err: return err
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    course_name = request.form.get("course", "").strip()[:MAX_COURSE_NAME_LENGTH]

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if not course_name:
        return jsonify({"error": "Course name is required"}), 400
    if not allowed(file.filename):
        return jsonify({"error": "Only PDF, Word (.docx), and .txt files are supported"}), 400

    filename = secure_filename(file.filename)
    file_path = UPLOAD_FOLDER / filename
    file.save(str(file_path))

    try:
        engine = get_rag()
        chunks = engine.add_document(str(file_path), course_name, filename)
        save_course_name(course_name)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        file_meta = load_file_meta()
        file_meta.setdefault(course_name, {})[filename] = {
            "display_name": filename,
            "upload_date": datetime.utcnow().isoformat(),
            "file_size": file_size,
        }
        save_file_meta(file_meta)
        return jsonify({
            "success": True,
            "chunks": chunks,
            "message": f'Added {chunks} sections from "{filename}" to {course_name}',
        })
    except Exception:
        log.exception("upload failed")
        return jsonify({"error": "Something went wrong processing the file."}), 500
    finally:
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass


# Chat (student)

@app.route("/api/chat", methods=["POST"])
def chat():
    ip = _client_ip()
    limited, wait = _is_chat_limited(ip)
    if limited:
        mins = (wait + 59) // 60
        return jsonify({"error": f"RATE_LIMIT:{mins}"}), 429

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()[:MAX_QUESTION_LENGTH]
    raw_books = data.get("books")
    if isinstance(raw_books, list):
        books = [str(b)[:200] for b in raw_books[:50] if isinstance(b, str)] or None
    else:
        books = None

    # Validate and sanitize history: max 10 entries, only role+content, content capped
    raw_history = data.get("history") or []
    history = []
    if isinstance(raw_history, list):
        for entry in raw_history[-10:]:
            if isinstance(entry, dict) and entry.get("role") in ("user", "assistant"):
                history.append({
                    "role": entry["role"],
                    "content": str(entry.get("content", ""))[:MAX_QUESTION_LENGTH],
                })

    lang = data.get("lang", "EN")
    if lang not in ("NL", "EN"):
        lang = "EN"

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        result = get_rag().query(question, books, history, lang=lang)
        _record_chat(ip)
        log_question(question, books or [])
        # Enrich sources with display_name from file_meta
        file_meta = load_file_meta()
        for src in result.get("sources", []):
            fname = src.get("filename", "")
            for course_meta in file_meta.values():
                if isinstance(course_meta, dict) and fname in course_meta:
                    src["display_name"] = course_meta[fname].get("display_name", fname)
                    break
        return jsonify(result)
    except Exception as e:
        log.exception("chat failed")
        return jsonify({"error": "Something went wrong. Please try again.", "detail": str(e)}), 500


@app.route("/api/extra", methods=["POST"])
def extra_info():
    if not _is_authenticated():
        return jsonify({"error": "Not authenticated"}), 401
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()[:MAX_QUESTION_LENGTH]
    lang = data.get("lang", "EN")
    if lang not in ("NL", "EN"):
        lang = "EN"
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    try:
        extra = get_rag().search_online(question, lang)
        return jsonify({"extra": extra})
    except Exception as e:
        log.exception("extra failed")
        return jsonify({"error": "Could not fetch online info", "detail": str(e)}), 500


# Search (teacher)

@app.route("/api/search", methods=["POST"])
def search():
    err = require_teacher()
    if err: return err
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()[:MAX_QUESTION_LENGTH]
    course = (data.get("course") or "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400
    if not course:
        return jsonify({"error": "Course is required"}), 400

    try:
        results = get_rag().search_content(query, course)
        return jsonify(results)
    except Exception:
        log.exception("search failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


# File metadata: rename

@app.route("/api/file-meta/rename", methods=["POST"])
def rename_file():
    err = require_teacher()
    if err: return err
    data = request.get_json(silent=True) or {}
    course = (data.get("course") or "").strip()[:MAX_COURSE_NAME_LENGTH]
    filename = (data.get("filename") or "").strip()[:200]
    display_name = (data.get("display_name") or "").strip()[:200]
    if not course or not filename or not display_name:
        return jsonify({"error": "course, filename and display_name are required"}), 400
    file_meta = load_file_meta()
    if course not in file_meta or filename not in file_meta[course]:
        return jsonify({"error": "File not found in metadata"}), 404
    file_meta[course][filename]["display_name"] = display_name
    save_file_meta(file_meta)
    return jsonify({"success": True})


# File order within a course

@app.route("/api/courses/<course_name>/files/reorder", methods=["POST"])
def reorder_files(course_name):
    err = require_teacher()
    if err: return err
    data = request.get_json(silent=True) or {}
    raw_order = data.get("order") or []
    if not isinstance(raw_order, list):
        return jsonify({"error": "Order must be a list"}), 400
    order = [str(item)[:200] for item in raw_order[:200] if isinstance(item, str)]
    file_meta = load_file_meta()
    file_meta.setdefault(course_name, {})["__order__"] = order
    save_file_meta(file_meta)
    return jsonify({"success": True})


# Statistics

@app.route("/api/stats", methods=["GET"])
def get_stats():
    err = require_teacher()
    if err: return err
    try:
        stats = load_stats()
        total = len(stats)
        per_course: dict = {}
        question_counts: dict = {}
        for entry in stats:
            for book in entry.get("books", []):
                per_course[book] = per_course.get(book, 0) + 1
            q = entry.get("question", "")
            question_counts[q] = question_counts.get(q, 0) + 1
        top_questions = sorted(question_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        recent = [
            {"question": e.get("question", ""), "books": e.get("books", []), "timestamp": e.get("timestamp", "")}
            for e in reversed(stats[-100:])
        ]
        feedback = load_feedback()
        thumbs_up = sum(1 for f in feedback if f.get("rating") == "up")
        thumbs_down = sum(1 for f in feedback if f.get("rating") == "down")
        total_feedback = thumbs_up + thumbs_down
        satisfaction_pct = round(thumbs_up / total_feedback * 100) if total_feedback else None

        today = datetime.utcnow().date()
        per_day = {(today - timedelta(days=i)).isoformat(): 0 for i in range(89, -1, -1)}
        for entry in stats:
            ts = entry.get("timestamp", "")
            day = ts[:10] if ts else ""
            if day in per_day:
                per_day[day] += 1

        if stats:
            first_day = stats[0].get("timestamp", "")[:10]
            try:
                days_active = max((today - datetime.fromisoformat(first_day).date()).days + 1, 1)
            except Exception:
                days_active = 1
        else:
            days_active = 1
        avg_per_day = round(total / days_active, 1)

        return jsonify({
            "total": total,
            "per_course": per_course,
            "top_questions": [{"question": q, "count": c} for q, c in top_questions],
            "recent": recent,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "satisfaction_pct": satisfaction_pct,
            "per_day": per_day,
            "avg_per_day": avg_per_day,
        })
    except Exception:
        log.exception("get_stats failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


# Feedback

@app.route("/api/feedback", methods=["POST"])
def post_feedback():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question") or "")[:MAX_QUESTION_LENGTH]
    answer = str(data.get("answer") or "")[:4000]
    raw_books = data.get("books")
    if isinstance(raw_books, list):
        books = [str(b)[:200] for b in raw_books[:50] if isinstance(b, str)]
    else:
        books = []
    rating = str(data.get("rating") or "")
    if rating not in ("up", "down"):
        return jsonify({"error": "rating must be 'up' or 'down'"}), 400
    log_feedback(question, answer, books, rating)
    return jsonify({"success": True})


# Welcome text

@app.route("/api/courses/<course>/welcome", methods=["POST"])
def set_welcome(course):
    err = require_teacher()
    if err: return err
    if not _safe_param(course):
        return jsonify({"error": "Invalid course name"}), 400
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or "")[:2000]
    file_meta = load_file_meta()
    file_meta.setdefault(course, {})["__welcome__"] = text
    save_file_meta(file_meta)
    return jsonify({"success": True})


# File preview

@app.route("/api/courses/<course>/files/<filename>/preview", methods=["GET"])
def preview_file(course, filename):
    err = require_teacher()
    if err: return err
    if not _safe_param(course) or not _safe_param(filename):
        return jsonify({"error": "Invalid parameters"}), 400
    try:
        collection = get_rag().collection
        results = collection.get(
            where={"$and": [{"course": {"$eq": course}}, {"filename": {"$eq": filename}}]},
            include=["documents", "metadatas"],
        )
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        chunks = [
            {"location": m.get("location", ""), "text": d[:600]}
            for d, m in sorted(zip(docs, metas), key=lambda x: x[1].get("location", ""))
        ]
        return jsonify(chunks)
    except Exception:
        log.exception("preview_file failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


# ── Startup ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "paste_your_key_here":
        print()
        print("  No API key found!")
        print("  Open the .env file and paste your OpenAI key.")
        print("  Example:  OPENAI_API_KEY=sk-proj-...")
        print()
    else:
        print()
        print("  SuperStudent is running.")
        print("  Open your browser at:  http://localhost:5000")
        print("  Press Ctrl+C to stop.")
        print()
        port = int(os.getenv("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False)
