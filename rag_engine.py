import os
import hashlib
import re
from pathlib import Path

import chromadb
from openai import OpenAI
import pypdf
from docx import Document as DocxDocument


class RAGEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        db_path = Path("vectordb")
        db_path.mkdir(exist_ok=True)

        self.chroma = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.chroma.get_or_create_collection(
            name="courses",
            metadata={"hnsw:space": "cosine"},
        )

        self.chunk_size = 1200
        self.chunk_overlap = 200

    # ── Text extraction ───────────────────────────────────────────────────

    def extract_with_meta(self, file_path: str) -> list:
        """Returns list of {text, location} dicts."""
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext in (".docx", ".doc"):
            return self._extract_docx(file_path)
        elif ext == ".txt":
            return self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _is_title_page(self, text: str, page_index: int) -> bool:
        """First page with little real content is likely a title page."""
        if page_index != 0:
            return False
        # Short pages with no real sentences
        if len(text.strip()) < 600:
            return True
        # Academic paper cover page: has email address + "abstract" heading
        t = text.lower()
        has_email = bool(re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text))
        has_abstract = "abstract" in t[:800]
        has_affiliation = any(w in t[:600] for w in [
            "university", "universiteit", "institute", "department", "faculty",
            "corresponding author", "delft", "technische"
        ])
        if has_email and (has_abstract or has_affiliation):
            return True
        return False

    def _is_reference_page(self, text: str) -> bool:
        """Detect bibliography / reference list pages."""
        first_lines = text.strip()[:300].lower()
        ref_headings = ["references", "bibliography", "referenties", "literatuur",
                        "bronnen", "literatuurlijst", "works cited", "bibliographie"]
        if any(first_lines.startswith(h) or f"\n{h}" in first_lines for h in ref_headings):
            return True
        # Page dominated by citation entries like [1], [2] … or numbered refs
        import re
        citation_hits = len(re.findall(r"^\s*\[\d+\]", text, re.MULTILINE))
        if citation_hits >= 4:
            return True
        return False

    def _extract_pdf(self, path: str) -> list:
        reader = pypdf.PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            if self._is_title_page(text, i):
                continue
            if self._is_reference_page(text):
                continue
            pages.append({"text": text.strip(), "location": f"Page {i + 1}"})
        return pages

    def _extract_docx(self, path: str) -> list:
        doc = DocxDocument(path)
        sections = []
        current_lines = []
        section_num = 1
        current_heading = f"Section {section_num}"

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if para.style.name.startswith("Heading"):
                if current_lines:
                    sections.append({
                        "text": "\n".join(current_lines),
                        "location": current_heading,
                    })
                    current_lines = []
                    section_num += 1
                current_heading = text or f"Section {section_num}"
                current_lines.append(text)
            else:
                current_lines.append(text)

        if current_lines:
            sections.append({
                "text": "\n".join(current_lines),
                "location": current_heading,
            })

        if not sections:
            full = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [{"text": full, "location": "Section 1"}] if full else []

        return sections

    def _extract_txt(self, path: str) -> list:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        lines = text.split("\n")
        blocks = []
        size = 60
        for i in range(0, len(lines), size):
            chunk = "\n".join(lines[i : i + size]).strip()
            if chunk:
                end = min(i + size, len(lines))
                blocks.append({"text": chunk, "location": f"Lines {i + 1}-{end}"})
        return blocks or [{"text": text.strip(), "location": "Full text"}]

    # ── Chunking ──────────────────────────────────────────────────────────

    def chunk_section(self, text: str, location: str) -> list:
        chunks = []
        start = 0
        text = text.strip()
        part = 1
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                loc = location if len(text) <= self.chunk_size else f"{location}, part {part}"
                chunks.append({"text": chunk, "location": loc})
                part += 1
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ── Document ingestion ────────────────────────────────────────────────

    def add_document(self, file_path: str, course_name: str, filename: str) -> int:
        sections = self.extract_with_meta(file_path)
        if not sections:
            raise ValueError("Could not extract any text from this file.")

        all_chunks = []
        for section in sections:
            all_chunks.extend(self.chunk_section(section["text"], section["location"]))

        if not all_chunks:
            raise ValueError("File appears to be empty after processing.")

        texts = [c["text"] for c in all_chunks]

        # Embed in batches of 100
        all_embeddings = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            response = self.client.embeddings.create(
                input=batch,
                model="text-embedding-3-small",
            )
            all_embeddings.extend([item.embedding for item in response.data])

        ids = [
            hashlib.md5(f"{course_name}|{filename}|{i}".encode()).hexdigest()
            for i in range(len(all_chunks))
        ]

        metadatas = [
            {
                "course": course_name,
                "filename": filename,
                "location": c["location"],
                "preview": c["text"][:900],
            }
            for c in all_chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(all_chunks)

    # ── Query (student) ───────────────────────────────────────────────────

    def _expand_query(self, text: str) -> str:
        """Add hyphen/space variants so 'e waste' matches 'e-waste' and vice versa."""
        variants = {text}
        variants.add(re.sub(r'(\w)-(\w)', lambda m: f"{m.group(1)} {m.group(2)}", text))
        variants.add(re.sub(r'(\b\w+)\s+(\w+\b)', lambda m: f"{m.group(1)}-{m.group(2)}", text))
        return " | ".join(v for v in variants if v != text) or text

    def query(self, question: str, books: list = None, history: list = None) -> dict:
        """books: list of filenames to restrict search to. None = all books."""
        total = self.collection.count()
        if total == 0:
            return {
                "answer": "I couldn't find anything — no course materials have been uploaded yet. Please ask your teacher.",
                "sources": [],
            }

        search_query = self._get_search_query(question, history or [])
        response = self.client.embeddings.create(
            input=[self._expand_query(search_query)],
            model="text-embedding-3-small",
        )
        query_embedding = response.data[0].embedding

        where = None
        if books and len(books) > 0:
            if len(books) == 1:
                where = {"filename": {"$eq": books[0]}}
            else:
                where = {"$or": [{"filename": {"$eq": b}} for b in books]}

        n_results = min(3, total)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )
        except Exception:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            return {
                "answer": "I couldn't find this in your course material, click below to check relevant information online.",
                "sources": [],
            }

        context_parts = []
        sources = []
        for doc, meta in zip(docs, metas):
            fname = meta.get("filename", "Unknown file")
            loc = meta.get("location", "")
            context_parts.append(
                f"[{fname} — {loc}]\n{doc}"
            )
            sources.append({
                "filename": fname,
                "location": loc,
                "preview": meta.get("preview", doc[:900]),
            })

        context = "\n\n---\n\n".join(context_parts)

        answer_prompt = (
            "You are a study assistant for students. "
            "IMPORTANT: Always respond in the same language as the student's question. "
            "If the question is in Dutch, answer in Dutch. If in English, answer in English. "
            "Do not use emojis. "
            "STRICT RULE: Give a maximum of 4-5 sentences. Use bullet points only when listing multiple items. "
            "Be direct — give the core answer only, no unnecessary intro or conclusion. "
            "Answer using ONLY the course material provided. Do not invent information. "
            "Either give a short answer, OR say exactly (in the student's language): "
            "\"I couldn't find this in your course material, click below to check relevant information online.\" "
            "Never mix an answer with the fallback phrase. Bold key terms with **term**. "
            "EXCEPTION: If the student explicitly asks for more detail (e.g. 'leg beter uit', 'meer uitleg', 'elaborate'), you may give a longer answer."
        )

        messages = [{"role": "system", "content": answer_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": f"Course materials:\n\n{context}\n\n---\n\nQuestion: {question}"})

        chat_response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=600,
        )
        answer = chat_response.choices[0].message.content.strip()

        fallback_old = "I couldn't find this in your course materials. Please ask your teacher."
        if fallback_old in answer and len(answer) > len(fallback_old) + 10:
            answer = answer.replace(fallback_old, "").strip().rstrip("\n").strip()
        no_answer = "couldn't find" in answer.lower()

        # Separate call for online extra info — uses web search
        try:
            extra_response = self.client.responses.create(
                model="gpt-4o-mini",
                tools=[{"type": "web_search_preview"}],
                input=(
                    f"The student asked: \"{question}\". "
                    "Search online and give a short, helpful answer (2-4 sentences) about this topic. "
                    "Respond in the same language as the question. "
                    "Be informative — give real content, not just a list of links."
                ),
            )
            extra = extra_response.output_text.strip()
        except Exception:
            extra = ""

        return {
            "answer": answer,
            "extra": extra,
            "sources": [] if no_answer else sources,
        }

    _FOLLOWUP_PHRASES = [
        "explain better", "explain more", "elaborate", "more detail", "more info",
        "tell me more", "can you explain", "what do you mean", "give an example",
        "i don't understand", "i dont understand", "don't get it", "dont get it",
        "leg beter uit", "meer uitleg", "vertel meer", "wat bedoel je", "meer detail",
        "geef een voorbeeld", "kun je uitleggen", "kan je uitleggen", "meer informatie",
        "verduidelijk", "hoe bedoel je", "ik snap het niet", "snap het niet",
        "ik begrijp het niet", "begrijp het niet", "wat betekent dit", "wat betekent dat",
        "niet begrepen", "leg uit", "uitleggen",
    ]

    def _is_followup(self, question: str) -> bool:
        q = question.lower().strip()
        return any(phrase in q for phrase in self._FOLLOWUP_PHRASES)

    def _get_search_query(self, question: str, history: list) -> str:
        """If followup question, use last user question from history as search query."""
        if self._is_followup(question) and history:
            for msg in reversed(history):
                if msg.get("role") == "user":
                    return msg["content"]
        return question

    def stream_query(self, question: str, books: list = None, history: list = None):
        """Stream the answer token by token. Yields dicts for SSE."""
        total = self.collection.count()
        if total == 0:
            yield {"type": "token", "text": "I couldn't find anything — no course materials have been uploaded yet. Please ask your teacher."}
            yield {"type": "done", "sources": []}
            return

        search_query = self._get_search_query(question, history or [])
        response = self.client.embeddings.create(
            input=[self._expand_query(search_query)],
            model="text-embedding-3-small",
        )
        query_embedding = response.data[0].embedding

        where = None
        if books and len(books) > 0:
            if len(books) == 1:
                where = {"filename": {"$eq": books[0]}}
            else:
                where = {"$or": [{"filename": {"$eq": b}} for b in books]}

        n_results = min(5, total)
        try:
            results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where)
        except Exception:
            results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            yield {"type": "token", "text": "I couldn't find this in your course material, click below to check relevant information online."}
            yield {"type": "done", "sources": []}
            return

        context_parts = []
        sources = []
        for doc, meta in zip(docs, metas):
            fname = meta.get("filename", "Unknown file")
            loc = meta.get("location", "")
            context_parts.append(f"[{fname} — {loc}]\n{doc}")
            sources.append({"filename": fname, "location": loc, "preview": meta.get("preview", doc[:900])})

        system_prompt = (
            "You are a study assistant for students. "
            "IMPORTANT: Always respond in the same language as the student's question. "
            "If the question is in Dutch, answer in Dutch. If in English, answer in English. "
            "Answer the question using ONLY the course material provided below. "
            "Do not use any outside knowledge or invent information. "
            "Do not use emojis. "
            "Either give a complete answer based on the material, OR say exactly (in the student's language): "
            "\"I couldn't find this in your course materials. Please ask your teacher.\" "
            "Never do both in the same response. "
            "Be clear and concise. Use bullet points for lists. Bold key terms with **term**. "
            "If the student explicitly asks for more info, extra explanation, or more detail (e.g. 'vertel meer', 'meer uitleg', 'tell me more', 'elaborate'), "
            "give a more detailed answer still based strictly on the course material. "
            "Only after a full course-material answer, if there is genuinely useful extra information available online, "
            "you may add 1-2 relevant links under the heading '**Extra info:**' (in Dutch) or '**Extra info:**' (in English), "
            "but ONLY when the student explicitly asked for more info. Never add links otherwise."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": f"Course materials:\n\n{chr(10).join(context_parts)}\n\n---\n\nQuestion: {question}"})

        stream = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=1200,
            stream=True,
        )

        full_answer = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_answer += token
                yield {"type": "token", "text": token}

        no_answer = "couldn't find" in full_answer.lower()
        yield {"type": "done", "sources": [] if no_answer else sources}

    # ── Search (teacher) ──────────────────────────────────────────────────

    def search_content(self, query: str, course_name: str) -> list:
        """Semantic search through a course's content. Returns matching passages."""
        total = self.collection.count()
        if total == 0:
            return []

        response = self.client.embeddings.create(
            input=[self._expand_query(query)],
            model="text-embedding-3-small",
        )
        query_embedding = response.data[0].embedding

        where = {"course": {"$eq": course_name}}
        n_results = min(8, total)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
            )
        except Exception:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        return [
            {
                "filename": meta.get("filename", "Unknown file"),
                "location": meta.get("location", ""),
                "preview": doc[:500],
            }
            for doc, meta in zip(docs, metas)
        ]

    # ── Course management ─────────────────────────────────────────────────

    def get_courses(self) -> list:
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["metadatas"])
        courses: dict = {}

        for meta in results["metadatas"]:
            course = meta["course"]
            filename = meta["filename"]
            if course not in courses:
                courses[course] = set()
            courses[course].add(filename)

        return [
            {"name": name, "files": sorted(files)}
            for name, files in sorted(courses.items())
        ]

    def delete_course(self, course_name: str):
        results = self.collection.get(
            where={"course": {"$eq": course_name}},
            include=["metadatas"],
        )
        if results["ids"]:
            self.collection.delete(ids=results["ids"])

    def delete_file(self, course_name: str, filename: str):
        results = self.collection.get(
            where={
                "$and": [
                    {"course": {"$eq": course_name}},
                    {"filename": {"$eq": filename}},
                ]
            },
            include=["metadatas"],
        )
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
