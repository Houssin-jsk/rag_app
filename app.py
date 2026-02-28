import os
import re
import json
import urllib.request
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# -----------------------------
# Config
# -----------------------------
APP_LINK = "https://ragapp-e72nlxxoepjymfmbnybxpt.streamlit.app"
DEFAULT_MODEL = "gemini-3-flash-preview"

SMS_URL = "https://raw.githubusercontent.com/krayF1sh/SMS-Spam-Collection/master/SMSSpamCollection.txt"
SMS_PATH = os.path.join("data", "SMSSpamCollection.txt")
EMAIL_PATH = os.path.join("data", "spam_ham_dataset.txt")

# -----------------------------
# API key (PRIVATE)
# -----------------------------
def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# -----------------------------
# Data loaders
# -----------------------------
def ensure_sms_downloaded():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(SMS_PATH):
        urllib.request.urlretrieve(SMS_URL, SMS_PATH)

def load_sms_rows(path: str):
    """
    Format: ham<TAB>message or spam<TAB>message
    Returns docs with file + line (1-indexed)
    """
    docs = []
    base = os.path.basename(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                label, text = line.split("\t", 1)
            elif " " in line:
                label, text = line.split(" ", 1)
            else:
                continue
            label = label.strip().lower()
            text = text.strip()
            if label in ("ham", "spam") and text:
                docs.append({
                    "doc_id": f"sms_{ln}",
                    "file": base,
                    "line_start": ln,
                    "line_end": ln,
                    "label": label,
                    "text": text
                })
    return docs

EMAIL_END_RE = re.compile(r"^\s*(\d+)(ham|spam)\s+.*\s+([01])\s*$", re.IGNORECASE)

def load_email_rows(path: str):
    """
    The uploaded email dataset looks like a multi-line printed table:
    - Emails span multiple lines (Subject:, body...)
    - Each email block ends with a line like: "4685spam   ...   1"
    We parse blocks and keep line_start/line_end for citations.
    """
    docs = []
    base = os.path.basename(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = list(f.readlines())

    # Skip the first header line if it contains 'label' and 'text'
    start_idx = 1 if lines and "label" in lines[0].lower() and "text" in lines[0].lower() else 0

    cur = []
    block_start_line = start_idx + 1  # 1-indexed
    for i in range(start_idx, len(lines)):
        ln = i + 1
        raw = lines[i].rstrip("\n")
        if not raw.strip():
            # keep blank lines inside block (optional)
            cur.append(raw)
            continue

        cur.append(raw)

        m = EMAIL_END_RE.match(raw)
        if m:
            num_id = m.group(1)
            label = m.group(2).lower()
            # Build email text from the whole block
            text = "\n".join([x for x in cur if x.strip()])

            docs.append({
                "doc_id": f"email_{num_id}",
                "file": base,
                "line_start": block_start_line,
                "line_end": ln,
                "label": label,
                "text": text
            })

            # reset for next block
            cur = []
            block_start_line = ln + 1

    return docs

# -----------------------------
# Vector RAG index (TF-IDF)
# -----------------------------
@st.cache_resource
def build_tfidf_index(texts, min_df=2):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=min_df, max_df=0.95, stop_words="english")
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def retrieve_topk_vector(vectorizer, X, docs, query: str, k: int = 8):
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).ravel()
    idx = np.argsort(-sims)[:k]
    out = []
    for i in idx:
        d = docs[int(i)]
        out.append({
            "doc_id": d["doc_id"],
            "file": d["file"],
            "line_start": d["line_start"],
            "line_end": d["line_end"],
            "label": d["label"],
            "score": float(round(sims[int(i)], 4)),
            "text": d["text"]
        })
    return out

# -----------------------------
# Gemini prompt + JSON parse
# -----------------------------
def extract_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            return None
    return None

def make_prompt_classify(user_msg: str, sources: list, lang: str):
    """
    Forces: label + confidence + explanation + citations (file + line range)
    """
    return f"""
You are a spam/ham classifier using Vector RAG.

Task:
- Predict label for NEW_MESSAGE: "spam" or "ham".
- Use ONLY SOURCES as evidence.
- If evidence is weak, still guess but lower confidence.

Return STRICT JSON only:
{{
  "label": "spam" | "ham",
  "confidence": number (0..1),
  "reply": "short explanation in {lang}",
  "sources": [{{"file":"...", "line_start": 1, "line_end": 2, "doc_id":"..."}}, ...]
}}

Rules:
- Cite at least 1 source.
- sources must reference ONLY items present in SOURCES list below.

SOURCES:
{json.dumps(sources, ensure_ascii=False, indent=2)}

NEW_MESSAGE:
{user_msg}
""".strip()

# -----------------------------
# Metrics helpers
# -----------------------------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return acc, prec, rec, f1

def plot_confusion(cm, labels=("ham","spam"), title="Confusion Matrix"):
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)

    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    return fig

def plot_roc(y_true, y_score, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1])
    ax.set_title(f"{title} (AUC={auc:.3f})")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    plt.tight_layout()
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Vector RAG Spam/Ham (Gemini)", layout="wide")
st.title("Vector RAG Spam/Ham (Gemini Flash)")
st.caption(f"App link: {APP_LINK}")

api_key = get_api_key()
if not api_key:
    st.error("API key missing. Add GEMINI_API_KEY in Streamlit Secrets أو env var.")
    st.stop()

client = genai.Client(api_key=api_key)

with st.sidebar:
    page = st.radio("Page", ["Chat", "Tests & Metrics"], index=0)

    st.divider()
    st.subheader("Corpus")
    corpus_choice = st.selectbox("Dataset", ["SMS (short)", "Emails (long)"], index=0)

    st.subheader("RAG settings")
    model = st.text_input("Gemini model", value=DEFAULT_MODEL)
    topk = st.slider("Top-K sources", 3, 20, 8)
    lang = st.selectbox("Reply language", ["Darija", "Français", "English"], index=0)

# Load corpus
@st.cache_data
def load_corpus(corpus_choice: str):
    if corpus_choice.startswith("SMS"):
        ensure_sms_downloaded()
        docs = load_sms_rows(SMS_PATH)
        return docs
    else:
        if not os.path.exists(EMAIL_PATH):
            return None
        docs = load_email_rows(EMAIL_PATH)
        return docs

docs = load_corpus(corpus_choice)
if docs is None:
    st.error(f"Email dataset not found. Put it here: `{EMAIL_PATH}`")
    st.stop()

texts = [d["text"] for d in docs]
labels = [1 if d["label"] == "spam" else 0 for d in docs]  # spam=1, ham=0

vectorizer, X = build_tfidf_index(texts, min_df=2)

st.caption(f"Loaded {len(docs)} documents from `{docs[0]['file']}`")

# -----------------------------
# Page: Chat
# -----------------------------
if page == "Chat":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if "sources_view" in m:
                with st.expander("Sources (file + line range)"):
                    for s in m["sources_view"]:
                        st.write(f"- {s['file']} : lines {s['line_start']}-{s['line_end']} | {s['label']} | score={s['score']}")
                        st.code(s["text"][:900])

    user_msg = st.chat_input("كتب رسالة/إيميل هنا باش نصنّفها…")
    if user_msg:
        st.session_state.messages.append({"role":"user","content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        sources = retrieve_topk_vector(vectorizer, X, docs, user_msg, k=int(topk))
        prompt = make_prompt_classify(user_msg, sources, lang)

        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            data = extract_json(resp.text)

            if not data:
                assistant_text = "Output ماشي JSON مضبوط."
                assistant_label = "unknown"
                assistant_conf = None
                used_sources = sources
            else:
                assistant_label = data.get("label", "unknown")
                assistant_conf = data.get("confidence", None)
                assistant_text = data.get("reply", "")

                cited = set()
                for s in data.get("sources", []):
                    cited.add((s.get("file"), s.get("line_start"), s.get("line_end"), s.get("doc_id")))

                used_sources = [
                    s for s in sources
                    if (s["file"], s["line_start"], s["line_end"], s["doc_id"]) in cited
                ] or sources

            with st.chat_message("assistant"):
                st.markdown(f"**Label:** `{assistant_label}`" + (f"  |  **Confidence:** `{assistant_conf}`" if assistant_conf is not None else ""))
                st.write(assistant_text)
                with st.expander("Sources (file + line range)"):
                    for s in used_sources:
                        st.write(f"- {s['file']} : lines {s['line_start']}-{s['line_end']} | {s['label']} | score={s['score']}")
                        st.code(s["text"][:900])

            st.session_state.messages.append({
                "role":"assistant",
                "content": f"Label: {assistant_label} | Confidence: {assistant_conf}\n\n{assistant_text}",
                "sources_view": used_sources
            })

        except Exception as e:
            st.error(f"Gemini error: {e}")

# -----------------------------
# Page: Tests & Metrics
# -----------------------------
else:
    st.subheader("Tests & Metrics")

    st.markdown("هاد الصفحة كتبيّن performance ديال classifier.")
    st.caption("ملاحظة: تقييم Gemini+RAG كيحتاج API calls؛ baseline كيحسب بسرعة بلا API.")

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test split (%)", 10, 40, 20)
        seed = st.number_input("Random seed", value=42, step=1)
    with col2:
        baseline_c = st.slider("Baseline LogisticRegression C", 0.1, 10.0, 2.0)
        eval_n = st.slider("Gemini eval sample size (from test set)", 5, 50, 20)

    # Split
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(docs)),
        np.array(labels),
        test_size=test_size/100.0,
        random_state=int(seed),
        stratify=np.array(labels)
    )

    train_docs = [docs[i] for i in X_train_idx]
    test_docs = [docs[i] for i in X_test_idx]

    train_texts = [d["text"] for d in train_docs]
    test_texts = [d["text"] for d in test_docs]

    # Baseline model: TF-IDF + Logistic Regression
    st.markdown("### 1) Baseline (TF-IDF + Logistic Regression)")
    vec_b, Xtr = build_tfidf_index(train_texts, min_df=2)
    Xte = vec_b.transform(test_texts)

    clf = LogisticRegression(max_iter=2000, C=float(baseline_c))
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    y_score = clf.predict_proba(Xte)[:,1]

    acc, prec, rec, f1 = compute_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("Precision", f"{prec:.3f}")
    m3.metric("Recall", f"{rec:.3f}")
    m4.metric("F1", f"{f1:.3f}")

    st.pyplot(plot_confusion(cm, labels=("ham","spam"), title="Baseline Confusion Matrix"))
    st.pyplot(plot_roc(y_test, y_score, title="Baseline ROC"))

    st.divider()

    # Gemini RAG evaluation (optional)
    st.markdown("### 2) Gemini + Vector RAG evaluation (sample)")
    st.caption("كنبني retrieval index على TRAIN فقط، ومن بعد كنقيّمو على TEST باش ما يكونش leakage.")

    run_eval = st.button("Run Gemini evaluation on sample")

    if run_eval:
        # Build retrieval index on TRAIN docs
        train_vec, train_X = build_tfidf_index(train_texts, min_df=2)

        # Sample from test set
        rng = np.random.default_rng(int(seed))
        idxs = rng.choice(len(test_docs), size=min(int(eval_n), len(test_docs)), replace=False)

        y_true_llm = []
        y_pred_llm = []
        rows_out = []

        for j in idxs:
            msg = test_docs[j]["text"]
            true_label = 1 if test_docs[j]["label"] == "spam" else 0

            sources = retrieve_topk_vector(train_vec, train_X, train_docs, msg, k=int(topk))
            prompt = make_prompt_classify(msg, sources, lang)

            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                data = extract_json(resp.text)
                pred_label = (data.get("label","ham") if data else "ham").lower()
            except Exception:
                pred_label = "ham"

            pred_num = 1 if pred_label == "spam" else 0

            y_true_llm.append(true_label)
            y_pred_llm.append(pred_num)

            rows_out.append({
                "true": "spam" if true_label==1 else "ham",
                "pred": pred_label,
                "doc_id": test_docs[j]["doc_id"],
                "file": test_docs[j]["file"],
                "lines": f"{test_docs[j]['line_start']}-{test_docs[j]['line_end']}",
            })

        acc2, prec2, rec2, f12 = compute_metrics(np.array(y_true_llm), np.array(y_pred_llm))
        cm2 = confusion_matrix(y_true_llm, y_pred_llm)

        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Accuracy (Gemini)", f"{acc2:.3f}")
        mm2.metric("Precision (Gemini)", f"{prec2:.3f}")
        mm3.metric("Recall (Gemini)", f"{rec2:.3f}")
        mm4.metric("F1 (Gemini)", f"{f12:.3f}")

        st.pyplot(plot_confusion(cm2, labels=("ham","spam"), title="Gemini+VectorRAG Confusion Matrix"))
        st.dataframe(rows_out, use_container_width=True)
