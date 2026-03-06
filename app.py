import hashlib
import hmac
import json
import os
from pathlib import Path

from flask import Flask, jsonify, make_response, render_template, request, redirect, url_for, Response, stream_with_context
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from rag_engine import RAGEngine

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

SITE_PASSWORD = os.getenv("SITE_PASSWORD", "student")
_SECRET = os.getenv("SECRET_KEY", "supersecret-change-me")


def _make_token():
    return hmac.new(_SECRET.encode(), SITE_PASSWORD.encode(), hashlib.sha256).hexdigest()


def _is_authenticated():
    return hmac.compare_digest(request.cookies.get("auth", ""), _make_token())


@app.before_request
def require_login():
    if request.endpoint in ("login", "logout", "static", "ping"):
        return
    if not _is_authenticated():
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not authenticated"}), 401
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == SITE_PASSWORD:
            resp = make_response(redirect(url_for("index")))
            resp.set_cookie("auth", _make_token(), httponly=True,
                            samesite="Lax", max_age=60 * 60 * 24 * 30)
            return resp
        error = "Incorrect password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    resp = make_response(redirect(url_for("login")))
    resp.delete_cookie("auth")
    return resp

DATA_FOLDER = Path("vectordb")
DATA_FOLDER.mkdir(exist_ok=True)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

COURSES_FILE = DATA_FOLDER / "courses.json"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

_rag = None


def get_rag() -> RAGEngine:
    global _rag
    if _rag is None:
        _rag = RAGEngine()
    return _rag


def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ── courses.json helpers ───────────────────────────────────────────────────

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
        names.append(name)          # append to end, preserve custom order
        COURSES_FILE.write_text(json.dumps(names))


def remove_course_name(name: str):
    names = [n for n in load_course_names() if n != name]
    COURSES_FILE.write_text(json.dumps(names))


# ── Routes ─────────────────────────────────────────────────────────────────

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
        saved = load_course_names()                        # ordered list
        extra = sorted(n for n in rag_courses if n not in saved)
        all_names = saved + extra                          # saved order first
        result = [{"name": n, "files": rag_courses.get(n, [])} for n in all_names]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/courses/reorder", methods=["POST"])
def reorder_courses():
    data = request.get_json(force=True)
    order = data.get("order") or []
    if not isinstance(order, list):
        return jsonify({"error": "Order must be a list"}), 400
    COURSES_FILE.write_text(json.dumps(order))
    return jsonify({"success": True})


@app.route("/api/courses", methods=["POST"])
def create_course():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Course name is required"}), 400
    save_course_name(name)
    return jsonify({"success": True})


@app.route("/api/courses/<course_name>", methods=["DELETE"])
def delete_course(course_name):
    try:
        get_rag().delete_course(course_name)
        remove_course_name(course_name)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/courses/<course_name>/files/<filename>", methods=["DELETE"])
def delete_file(course_name, filename):
    try:
        get_rag().delete_file(course_name, filename)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Upload

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    course_name = request.form.get("course", "").strip()

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
        # Make sure course name is persisted even if it was created implicitly
        save_course_name(course_name)
        return jsonify({
            "success": True,
            "chunks": chunks,
            "message": f'Added {chunks} sections from "{filename}" to {course_name}',
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Chat (student)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    books = data.get("books") or None

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        result = get_rag().query(question, books)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    books = data.get("books") or None

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    def generate():
        try:
            for event in get_rag().stream_query(question, books):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Search (teacher)

@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    course = (data.get("course") or "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400
    if not course:
        return jsonify({"error": "Course is required"}), 400

    try:
        results = get_rag().search_content(query, course)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Startup ─────────────────────────────────────────────────────────────────

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
