import os
import re
import json
import streamlit as st
import nltk
from PyPDF2 import PdfReader
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ensure_nltk_tokenizer():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # Newer NLTK versions also require punkt_tab
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except:
            pass  # some NLTK versions don't have punkt_tab

ensure_nltk_tokenizer()
# -----------------------
# USER CONFIG - EDIT THESE
# -----------------------
GEMINI_API_KEY = "AIzaSyBnNcI3bPJWgdbbTGBqqRr3tM9hSWYvWaQ"   # <<-- REPLACE with your Gemini key
PDF_PATH = "NFHS-5_Phase-II_0.pdf"          # <<-- REPLACE with your PDF filename/path
GEMINI_MODEL = "gemini-2.5-flash-lite"

# -----------------------
# Environment & UI setup
# -----------------------
nltk.download("punkt", quiet=True)
st.set_page_config(page_title="Multi-Pipeline PDF Q/A", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background: white; color: #0f172a; }
      .block-container { padding: 1.25rem 1.5rem; }
      h1 { color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("📚 Multi-chunking PDF Q/A ")

# -----------------------
# Gemini helpers (safe)
# -----------------------
def configure_gemini():
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("PUT_"):
        return False, "Set GEMINI_API_KEY variable at the top of app.py"
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return True, ""
    except Exception as e:
        return False, f"genai.configure() failed: {e}"

def gemini_generate(prompt: str, model_name: str = GEMINI_MODEL, max_output_tokens: int = 512):
    ok, msg = configure_gemini()
    if not ok:
        return f"[Gemini not configured] {msg}"
    try:
        model = genai.GenerativeModel(model_name, generation_config={"max_output_tokens": max_output_tokens})
    except Exception as e:
        return f"[Gemini init error] {e}"
    try:
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None) or getattr(resp, "output_text", None) or str(resp)
        return text
    except Exception as e:
        return f"[Gemini call failed] {e}"

# -----------------------
# PDF loader (no uploader)
# -----------------------
def read_pdf_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        reader = PdfReader(path)
        txt = ""
        for p in reader.pages:
            page_text = p.extract_text() or ""
            txt += page_text + "\n\n"
        return txt.strip()
    except Exception:
        return ""

doc_text = read_pdf_text(PDF_PATH)
if not doc_text:
    st.error(f"Cannot load PDF at `{PDF_PATH}` — place PDF next to app.py or set correct path.")
    st.stop()

# -----------------------
# Chunkers (FAST mode - smaller chunks)
# -----------------------
def fixed_chunking(text: str, chars: int = 800):
    text = text.replace("\n", " ")
    return [text[i:i+chars].strip() for i in range(0, len(text), chars) if text[i:i+chars].strip()]

def recursive_chunking(text: str, chunk_size: int = 800, overlap: int = 120):
    step = max(chunk_size - overlap, 1)
    chunks = []
    for i in range(0, len(text), step):
        chunks.append(text[i:i+chunk_size].strip())
    return [c for c in chunks if c]

def sentence_chunking(text: str, sentences_per_chunk: int = 5):
    sents = nltk.sent_tokenize(text)
    return [" ".join(sents[i:i+sentences_per_chunk]).strip() for i in range(0, len(sents), sentences_per_chunk)]

def paragraph_chunking(text: str, min_len: int = 150):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    merged = []
    buf = ""
    for p in paras:
        if len(p) < min_len:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                merged.append((buf + " " + p).strip())
                buf = ""
            else:
                merged.append(p)
    if buf:
        merged.append(buf)
    return merged

def semantic_chunking_embedding_window(text: str, window_sentences: int = 4, sim_threshold: float = 0.64):
    sents = nltk.sent_tokenize(text)
    if not sents:
        return []
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sent_embs = model.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    sent_embs = sent_embs / (np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-10)
    chunks = []
    i = 0
    n = len(sents)
    while i < n:
        j = min(i + window_sentences, n)
        while j < n:
            window_mean = sent_embs[i:j].mean(axis=0)
            next_sent = sent_embs[j]
            sim = float(np.dot(window_mean, next_sent))
            if sim >= sim_threshold and (j - i) < (window_sentences * 3):
                j += 1
            else:
                break
        chunks.append(" ".join(sents[i:j]).strip())
        i = j
    # merge very small chunks
    merged = []
    for c in chunks:
        if len(c) < 40 and merged:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)
    return merged

# -----------------------
# Retrieval strategies (separate for each pipeline)
# -----------------------
def retrieval_tfidf(chunks, question, top_k=3):
    try:
        vect = TfidfVectorizer().fit_transform(chunks + [question])
        sims = cosine_similarity(vect[-1], vect[:-1]).flatten()
        idxs = np.argsort(sims)[-top_k:][::-1]
        return [chunks[i] for i in idxs]
    except Exception:
        # fallback lexical
        scores = [sum(w.lower() in c.lower() for w in question.split()) for c in chunks]
        idxs = np.argsort(scores)[-top_k:][::-1]
        return [chunks[i] for i in idxs]

def retrieval_embedding(chunks, question, top_k=3):
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        chunk_embs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        q_emb = model.encode([question], convert_to_numpy=True)[0]
        sims = (chunk_embs @ q_emb) / (np.linalg.norm(chunk_embs, axis=1) * (np.linalg.norm(q_emb) + 1e-10) + 1e-10)
        idxs = np.argsort(sims)[-top_k:][::-1]
        return [chunks[i] for i in idxs]
    except Exception:
        # fallback to tfidf
        return retrieval_tfidf(chunks, question, top_k=top_k)

def retrieval_lexical(chunks, question, top_k=3):
    scores = [sum(w.lower() in c.lower() for w in question.split()) for c in chunks]
    idxs = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in idxs]

def retrieval_hybrid(chunks, question, top_k=3):
    # try embeddings, then combine with TF-IDF if necessary
    try:
        emb_top = retrieval_embedding(chunks, question, top_k=top_k)
        if len(emb_top) >= top_k:
            return emb_top
    except Exception:
        pass
    return retrieval_tfidf(chunks, question, top_k=top_k)

# -----------------------
# Per-pipeline prompts (distinct)
# -----------------------
def prompt_tfidf_style(context_chunks, question):
    ctx = "\n\n---\n\n".join(context_chunks)
    return f"""You are an academic assistant. Use ONLY the context below.
Context:
{ctx}

Question:
{question}

Produce a clear, structured answer (3-6 bullet points) citing where the information came from in the context. If not present, say "Not clearly answered in the document."""

def prompt_embedding_style(context_chunks, question):
    ctx = "\n\n".join(context_chunks)
    return f"""You are a precise summarizer. Use ONLY the provided context to answer the question directly. Do not hallucinate.
Context:
{ctx}

Question:
{question}

Answer in 4–6 sentences, focusing on facts from the context. If not present, say "Not clearly answered in the document."""

def prompt_semantic_style(context_chunks, question):
    ctx = "\n\n".join(context_chunks)
    return f"""You are a semantic researcher. Given the context, synthesize an evidence-based answer for the question.
Context:
{ctx}

Question:
{question}

Give: (1) short answer (1 line), (2) 3 supporting facts from the context, (3) short suggestion for follow-up query. If info missing, say "Not clearly answered in the document."""

def prompt_sentence_style(context_chunks, question):
    ctx = "\n\n".join(context_chunks)
    return f"""You are a sentence-level responder. Use ONLY the context sentences to answer.
Context:
{ctx}

Question:
{question}

Give a concise answer (1-2 lines). If uncertain, say "Not clearly answered in the document."""

def prompt_paragraph_style(context_chunks, question):
    ctx = "\n\n".join(context_chunks)
    return f"""You are a paragraph-level analyst. Use ONLY the context paragraphs to answer.
Context:
{ctx}

Question:
{question}

Provide a 4-bullet concise answer with explicit facts pulled from the context. If info missing, say "Not clearly answered in the document."""

def prompt_agentic_selection(selected_chunks, question):
    ctx = "\n\n".join(selected_chunks)
    return f"""You are an assistant that must answer using ONLY the selected chunks.
Selected context:
{ctx}

Question:
{question}

Answer thoroughly (4-6 bullet points) using only the selected context. If not present, say "Not clearly answered in the document."""

# -----------------------
# Agentic selection helper (LLM chooses indices)
# -----------------------
def agentic_select_chunks(candidates, question, max_candidates=8, selection_limit=4):
    # prepare numbered list of candidates (truncated)
    numbered = "\n".join([f"{i+1}. {c[:700].replace('\\n',' ')}" for i, c in enumerate(candidates[:max_candidates])])
    sel_prompt = f"""You are a selection assistant. The user question is below.
Return ONLY a JSON array of the 1-based indices of the candidate chunks (best first).
Question:
{question}

Candidates:
{numbered}

Return e.g. [2,5]"""
    sel_resp = gemini_generate(sel_prompt, max_output_tokens=180)
    # parse JSON array
    try:
        m = re.search(r"(\[.*?\])", sel_resp, flags=re.S)
        if not m:
            return [candidates[0]]
        arr = json.loads(m.group(1))
        chosen = []
        for idx in arr[:selection_limit]:
            if isinstance(idx, int) and 1 <= idx <= len(candidates[:max_candidates]):
                chosen.append(candidates[idx-1])
        return chosen if chosen else [candidates[0]]
    except Exception:
        return [candidates[0]]

# -----------------------
# Pipeline implementations (completely separate)
# -----------------------
def pipeline_fixed(text, question):
    # chunking: fixed windows
    chunks = fixed_chunking(text, chars=800)
    # retrieval: TF-IDF
    top = retrieval_tfidf(chunks, question, top_k=3)
    # prompt: TF-IDF style
    prompt = prompt_tfidf_style(top, question)
    return gemini_generate(prompt, max_output_tokens=450)

def pipeline_recursive(text, question):
    # chunking: overlapping recursive-style
    chunks = recursive_chunking(text, chunk_size=800, overlap=160)
    # retrieval: embedding similarity
    top = retrieval_embedding(chunks, question, top_k=3)
    # prompt: embedding style
    prompt = prompt_embedding_style(top, question)
    return gemini_generate(prompt, max_output_tokens=450)

def pipeline_semantic(text, question):
    # chunking: semantic embedding-window
    chunks = semantic_chunking_embedding_window(text, window_sentences=4, sim_threshold=0.62)
    if not chunks:
        chunks = fixed_chunking(text, chars=800)
    # retrieval: hybrid (embedding preferred)
    top = retrieval_hybrid(chunks, question, top_k=3)
    # prompt: semantic style
    prompt = prompt_semantic_style(top, question)
    return gemini_generate(prompt, max_output_tokens=500)

def pipeline_sentence(text, question):
    # chunking: sentence-based
    chunks = sentence_chunking(text, sentences_per_chunk=5)
    # retrieval: lexical overlap (fast)
    top = retrieval_lexical(chunks, question, top_k=2)
    prompt = prompt_sentence_style(top, question)
    # use smaller max tokens for concise reply
    return gemini_generate(prompt, max_output_tokens=220)

def pipeline_paragraph(text, question):
    # chunking: paragraph merging
    chunks = paragraph_chunking(text, min_len=120)
    # retrieval: TF-IDF with larger context
    top = retrieval_tfidf(chunks, question, top_k=3)
    prompt = prompt_paragraph_style(top, question)
    return gemini_generate(prompt, max_output_tokens=480)

def pipeline_agentic(text, question):
    # chunking: paragraph candidates fallback to sentence/fixed
    candidates = paragraph_chunking(text, min_len=120)
    if len(candidates) < 6:
        candidates = sentence_chunking(text, sentences_per_chunk=5)
    if len(candidates) < 1:
        candidates = fixed_chunking(text, chars=800)
    # LLM selects indices
    selected = agentic_select_chunks(candidates, question, max_candidates=12, selection_limit=5)
    # final LLM answer using selected chunks
    prompt = prompt_agentic_selection(selected, question)
    return gemini_generate(prompt, max_output_tokens=600)

# -----------------------
# UI (single question, choose pipeline)
# -----------------------
st.sidebar.header("Choose pipeline (research mode)")
choice = st.sidebar.radio("Pipeline", [
    "Fixed-size chunking",
    "Recursive chunking",
    "Semantic chunking",
    "Sentence-based chunking",
    "Paragraph-based chunking",
    "Agentic chunking"
])

st.sidebar.markdown("---")
st.sidebar.write(f"PDF: `{PDF_PATH}`")
st.sidebar.write("Display: Only final answer (research mode)")

st.markdown("### Enter your question (applies to selected pipeline)")
question = st.text_input("Question")

if st.button("Run pipeline"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running selected pipeline — this may take a few seconds..."):
            if choice == "Fixed-size chunking":
                result = pipeline_fixed(doc_text, question)
                st.success("Answer — Fixed-size chunking")
                st.write(result)
            elif choice == "Recursive chunking":
                result = pipeline_recursive(doc_text, question)
                st.success("Answer — Recursive chunking")
                st.write(result)
            elif choice == "Semantic chunking":
                result = pipeline_semantic(doc_text, question)
                st.success("Answer — Semantic chunking")
                st.write(result)
            elif choice == "Sentence-based chunking":
                result = pipeline_sentence(doc_text, question)
                st.success("Answer — Sentence-based chunking")
                st.write(result)
            elif choice == "Paragraph-based chunking":
                result = pipeline_paragraph(doc_text, question)
                st.success("Answer — Paragraph-based chunking")
                st.write(result)
            else:  # Agentic
                result = pipeline_agentic(doc_text, question)
                st.success("Answer — Agentic chunking")
                st.write(result)

st.markdown("---")
st.caption("Notes: Research mode = independent pipelines & prompts. Set GEMINI_API_KEY and PDF_PATH at top of this file.")

