# app.py
import os
import re
import json
import math
import urllib.request
from collections import Counter
import streamlit as st
from google import genai

# -----------------------------
# Config
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/krayF1sh/SMS-Spam-Collection/master/SMSSpamCollection.txt"
DEFAULT_DATA_PATH = os.path.join("data", "SMSSpamCollection.txt")
DEFAULT_MODEL = "gemini-3-flash-preview"

TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)

# -----------------------------
# API key (PRIVATE)
# -----------------------------
def get_api_key():
    # 1) Streamlit secrets (local/deploy)
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    # 2) Env vars (any platform)
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# -----------------------------
# Data loading + parsing (file + line)
# -----------------------------
def download_default_if_needed():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DEFAULT_DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DEFAULT_DATA_PATH)

def parse_text_with_lines(text: str, filename: str):
    """
    Returns rows:
    {file, line, label, text}
    label is ham/spam if dataset format matches, else "unknown"
    line is 1-indexed
    """
    rows = []
    base = os.path.basename(filename)

    for ln, raw in enumerate(text.splitlines(), start=1):
        s = raw.strip()
        if not s:
            continue

        label = "unknown"
        msg = s

        # If looks like SMS dataset: "ham\tmessage" or "spam\tmessage"
        if "\t" in s:
            first, rest = s.split("\t", 1)
            if first.lower() in ("ham", "spam") and rest.strip():
                label = first.lower()
                msg = rest.strip()
        elif " " in s:
            first, rest = s.split(" ", 1)
            if first.lower() in ("ham", "spam") and rest.strip():
                label = first.lower()
                msg = rest.strip()

        rows.append({"file": base, "line": ln, "label": label, "text": msg})

    return rows

def load_rows_from_path(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return parse_text_with_lines(f.read(), path)

# -----------------------------
# BM25 Retrieval (local, no embedding quota)
# -----------------------------
def tokenize(s: str):
    return TOKEN_RE.findall(s.lower())

def build_bm25_index(rows, k1=1.5, b=0.75):
    docs_tokens = [tokenize(r["text"]) for r in rows]
    doc_tfs = [Counter(toks) for toks in docs_tokens]
    doc_lens = [len(toks) for toks in docs_tokens]
    N = len(rows)
    avgdl = (sum(doc_lens) / N) if N else 0.0

    df = Counter()
    for toks in docs_tokens:
        df.update(set(toks))

    idf = {term: math.log(1 + (N - dfi + 0.5) / (dfi + 0.5)) for term, dfi in df.items()}

    return {
        "rows": rows,
        "doc_tfs": doc_tfs,
        "doc_lens": doc_lens,
        "idf": idf,
        "N": N,
        "avgdl": avgdl,
        "k1": k1,
        "b": b,
    }

def bm25_score(index, query_tokens, doc_i):
    tf = index["doc_tfs"][doc_i]
    dl = index["doc_lens"][doc_i]
    avgdl = index["avgdl"]
    k1 = index["k1"]
    b = index["b"]
    idf = index["idf"]

    score = 0.0
    for t in query_tokens:
        if t not in tf:
            continue
        f = tf[t]
        denom = f + k1 * (1 - b + b * (dl / avgdl if avgdl else 0.0))
        score += idf.get(t, 0.0) * (f * (k1 + 1) / (denom if denom else 1.0))
    return score

def retrieve_topk(index, query, k=8):
    qtok = tokenize(query)
    if not qtok:
        return []

    scores = []
    for i in range(index["N"]):
        s = bm25_score(index, qtok, i)
        if s > 0:
            scores.append((i, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:k]

    out = []
    for i, s in top:
        r = index["rows"][i]
        out.append({
            "idx": i,
            "score": round(float(s), 4),
            "file": r["file"],
            "line": r["line"],
            "label": r["label"],
            "text": r["text"],
        })
    return out

# -----------------------------
# Gemini helpers
# -----------------------------
def extract_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start:end+1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None

def make_prompt(user_msg: str, history: list, sources: list, lang: str):
    hist_txt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-6:]])

    return f"""
You are an SMS spam classifier using RAG (retrieval-augmented generation).

You MUST decide if NEW_MESSAGE is "spam" or "ham".
You MUST use ONLY the provided SOURCES (lines from a .txt file) as evidence.
If the sources are not enough, still make your best guess but lower confidence.

Return STRICT JSON only with:
{{
  "label": "spam" | "ham",
  "confidence": number (0..1),
  "reply": "string in {lang} (short explanation)",
  "sources": [{{"file":"...", "line": 123}}, ...],
  "how_i_used_sources": "short explanation in {lang}"
}}

Rules:
- Cite at least 1 source in "sources".
- "sources" must reference ONLY lines that appear in SOURCES below.

CHAT_HISTORY:
{hist_txt}

SOURCES (each item includes file + line + label):
{json.dumps(sources, ensure_ascii=False, indent=2)}

NEW_MESSAGE:
{user_msg}
""".strip()
# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG + Gemini (file + line sources)", layout="wide")
st.title("RAG Chat (Gemini Flash) + Sources (file + line)")

with st.sidebar:
    st.subheader("Settings")
    model = st.text_input("Gemini model", value=DEFAULT_MODEL)
    topk = st.slider("Top-K retrieved lines", 3, 15, 8)
    lang = st.selectbox("Reply language", ["Darija", "English", "Français"], index=0)

    st.divider()
    st.subheader("Data source")
    uploaded = st.file_uploader("Upload .txt (optional)", type=["txt"])
    st.caption("If no upload: app uses SMS dataset in data/ (downloads if missing).")

# --- API key init (private)
api_key = get_api_key()
if not api_key:
    st.error("API key missing. Put it in Streamlit Secrets (GEMINI_API_KEY) or env var.")
    st.stop()

client = genai.Client(api_key=api_key)

# --- Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Load / rebuild corpus when changed
def get_corpus_and_id():
    if uploaded is not None:
        raw = uploaded.getvalue()
        text = raw.decode("utf-8", errors="ignore")
        corpus_id = f"upload:{uploaded.name}:{len(raw)}"
        rows = parse_text_with_lines(text, uploaded.name)
        return rows, corpus_id, os.path.basename(uploaded.name)

    download_default_if_needed()
    mtime = os.path.getmtime(DEFAULT_DATA_PATH)
    corpus_id = f"path:{DEFAULT_DATA_PATH}:{mtime}"
    rows = load_rows_from_path(DEFAULT_DATA_PATH)
    return rows, corpus_id, os.path.basename(DEFAULT_DATA_PATH)

rows, corpus_id, corpus_name = get_corpus_and_id()

if st.session_state.get("corpus_id") != corpus_id:
    st.session_state.corpus_id = corpus_id
    st.session_state.rows = rows
    st.session_state.bm25 = build_bm25_index(rows)
    st.session_state.data_file = corpus_name
    st.session_state.n_rows = len(rows)
    st.session_state.messages = []  # reset chat when corpus changes

st.caption(f"Loaded **{st.session_state.n_rows}** lines from **{st.session_state.data_file}**.")

# --- Render chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if "sources_view" in m:
            with st.expander("Sources used (file + line)"):
                for s in m["sources_view"]:
                    st.write(f"- {s['file']} : line {s['line']}  [{s['label']}] {s['text']}")

user_msg = st.chat_input("Kteb message hna...")

if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    sources = retrieve_topk(st.session_state.bm25, user_msg, k=int(topk))

    try:
        prompt = make_prompt(user_msg, st.session_state.messages, sources, lang)
        resp = client.models.generate_content(model=model, contents=prompt)
        data = extract_json(resp.text)

        if not data or "reply" not in data:
            assistant_text = "Model رجّع output ماشي JSON مضبوط. جرّب عاود."
            used_sources_view = sources
        else:
            assistant_text = data["reply"]
            cited = {(s.get("file"), s.get("line")) for s in data.get("sources", [])}
            used_sources_view = [s for s in sources if (s["file"], s["line"]) in cited] or sources

        with st.chat_message("assistant"):
            st.write(assistant_text)
            with st.expander("Sources used (file + line)"):
                for s in used_sources_view:
                    st.write(f"- {s['file']} : line {s['line']}  [{s['label']}] {s['text']}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_text,
            "sources_view": used_sources_view
        })

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})