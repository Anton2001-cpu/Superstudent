import collections
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path

from flask import Flask, jsonify, make_response, render_template, request, redirect, url_for
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from rag_engine import RAGEngine

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

log = logging.getLogger(__name__)

SITE_PASSWORD = os.getenv("SITE_PASSWORD", "student")
TEACHER_PASSWORD = os.getenv("TEACHER_PASSWORD", "")

if SITE_PASSWORD == "student":
    log.warning("WARNING: SITE_PASSWORD is still set to the default 'student'. Set a strong password in your environment variables.")

_SECRET = os.getenv("SECRET_KEY")
if not _SECRET:
    raise RuntimeError(
        "SECRET_KEY is not set. Add it to your .env file or Render environment variables."
    )

_IS_PRODUCTION = bool(os.getenv("RENDER"))

# ── Rate limiting (login brute-force protection) ────────────────────────────
# Stores (ip -> [timestamp, ...]) for failed attempts
_failed_attempts: dict = collections.defaultdict(list)
_MAX_ATTEMPTS = 5       # max failed tries
_LOCKOUT_SECONDS = 300  # 5 minutes


def _client_ip() -> str:
    return request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()


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
    return response


@app.before_request
def require_login():
    if request.endpoint in ("login", "logout", "static", "ping"):
        return
    if not _is_authenticated():
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not authenticated"}), 401
        return redirect(url_for("login"))


# ── Auth routes ─────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        ip = _client_ip()
        if _is_rate_limited(ip):
            error = "Too many failed attempts. Try again in 5 minutes."
        elif hmac.compare_digest(request.form.get("password", ""), SITE_PASSWORD):
            _clear_failures(ip)
            resp = make_response(redirect(url_for("index")))
            resp.set_cookie("auth", _make_token(), httponly=True,
                            samesite="Lax", secure=_IS_PRODUCTION,
                            max_age=60 * 60 * 8)
            return resp
        else:
            _record_failure(ip)
            error = "Incorrect password."
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
    data = request.get_json(force=True)
    pw = data.get("password", "")
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

DATA_FOLDER = Path("vectordb")
DATA_FOLDER.mkdir(exist_ok=True)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

COURSES_FILE = DATA_FOLDER / "courses.json"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

MAX_QUESTION_LENGTH = 1000
MAX_COURSE_NAME_LENGTH = 100

_rag = None


def get_rag() -> RAGEngine:
    global _rag
    if _rag is None:
        _rag = RAGEngine()
    return _rag


def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def load_course_names() -> list:
    if not COURSES_FILE.exists():
        return []
    try:
        return json.loads(COURSES_FILE.read_text())
    except Exception:
        return []


def save_course_name(name: str):
    names = load_course_names()
    if name not in names:
        names.append(name)
        COURSES_FILE.write_text(json.dumps(names))


def remove_course_name(name: str):
    names = [n for n in load_course_names() if n != name]
    COURSES_FILE.write_text(json.dumps(names))


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
        result = [{"name": n, "files": rag_courses.get(n, [])} for n in all_names]
        return jsonify(result)
    except Exception:
        log.exception("get_courses failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


@app.route("/api/courses/reorder", methods=["POST"])
def reorder_courses():
    err = require_teacher()
    if err: return err
    data = request.get_json(force=True)
    order = data.get("order") or []
    if not isinstance(order, list):
        return jsonify({"error": "Order must be a list"}), 400
    COURSES_FILE.write_text(json.dumps(order))
    return jsonify({"success": True})


@app.route("/api/courses", methods=["POST"])
def create_course():
    err = require_teacher()
    if err: return err
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()[:MAX_COURSE_NAME_LENGTH]
    if not name:
        return jsonify({"error": "Course name is required"}), 400
    save_course_name(name)
    return jsonify({"success": True})


@app.route("/api/courses/<course_name>", methods=["DELETE"])
def delete_course(course_name):
    err = require_teacher()
    if err: return err
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
    try:
        get_rag().delete_file(course_name, filename)
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
    data = request.get_json(force=True)
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

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        result = get_rag().query(question, books, history)
        return jsonify(result)
    except Exception:
        log.exception("chat failed")
        return jsonify({"error": "Something went wrong. Please try again."}), 500


# Search (teacher)

@app.route("/api/search", methods=["POST"])
def search():
    err = require_teacher()
    if err: return err
    data = request.get_json(force=True)
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
