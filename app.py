"""PDF Chat Dashboard — Streamlit UI for the simple RAG pipeline in rag.py."""

import html
import os
from datetime import datetime
from typing import List

import streamlit as st

try:  # optional: read .env before rag/providers read their settings
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from providers import (  # noqa: E402
    ALL_PROVIDERS,
    NOT_FOUND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PROVIDER_ENV_VARS,
    PROVIDER_OLLAMA,
    PROVIDER_SIGNUP_URLS,
    build_context,
    cite_label,
    ollama_models,
    resolve_api_key,
    simple_local_answer,
    sources_footer,
    stream_answer,
)
from rag import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    TESSDATA,
    RAGStore,
    build_indexes,
    chunk_blocks,
    delete_store,
    ensure_compatible,
    extract_pdf,
    file_hash,
    list_saved_indexes,
    load_embedding_model,
    load_reranker,
    load_store,
    save_store,
    search,
)

# The store and settings live independently of widgets, so page navigation preserves them.
st.set_page_config(page_title="ManualMate", page_icon=":material/build_circle:", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: #f5f7fb; }
.block-container { max-width: 1300px; padding: 2.5rem 3rem 4rem; }
h1, h2, h3 { letter-spacing: -.035em; }
h1 { font-weight: 700; }
[data-testid="stSidebar"] { background: #101a2e; }
[data-testid="stSidebar"] p { color: #d8e1f0; }
[data-testid="stSidebar"] .stButton button { width: 100%; min-height: 2.9rem; border-radius: 9px; justify-content: flex-start; }
[data-testid="stSidebar"] .stButton button[kind="secondary"] { background: transparent; color: #c4d0e3; border-color: transparent; }
[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover { background: #1d2b43; color: white; }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background: #2457D6; color: white; border-color: #2457D6; }
.brand { display: flex; align-items: center; gap: 11px; margin: 4px 0 28px; }
.brand-symbol { display: grid; place-items: center; width: 39px; height: 39px; border-radius: 12px; background: #2457d6; color: white; font-size: 18px; font-weight: 800; }
.brand-name { color: white; font-size: 19px; font-weight: 700; letter-spacing: -.04em; }
.brand-caption { color: #8da0bb; font-size: 10px; font-weight: 600; letter-spacing: .13em; }
.workspace-kicker { color: #2457d6; font-size: 11px; font-weight: 700; letter-spacing: .13em; margin-bottom: 6px; }
.landing-nav { display:flex; align-items:center; flex-wrap:wrap; gap:24px; min-height:42px; color:#51617b; font-size:.88rem; }
.landing-nav a { color:inherit; text-decoration:none; font-weight:600; }
.landing-nav a:hover { color:#2457d6; }
.landing-hero { padding:4rem 0 2rem; }
.landing-pill { display:inline-block; padding:7px 12px; border:1px solid #d7e4ff; border-radius:999px; background:#edf3ff; color:#2457d6; font-size:.72rem; font-weight:700; letter-spacing:.08em; }
.landing-hero h1 { max-width:740px; font-size:clamp(2.7rem, 6vw, 5rem); line-height:1.04; margin:1.3rem 0; letter-spacing:-.055em; }
.landing-hero h1 span { color:#2457d6; }
.landing-hero p { color:#61718a; max-width:590px; font-size:1.12rem; line-height:1.7; }
.landing-section { padding:3rem 0 1rem; scroll-margin-top:5rem; }
.landing-section h2 { font-size:2rem; margin:.4rem 0; }
.landing-section p { color:#66758d; }
.landing-card { height:100%; min-height:165px; background:white; border:1px solid #e2e8f1; border-radius:16px; padding:1.4rem; }
.landing-card .number { display:grid; place-items:center; width:38px; height:38px; background:#edf3ff; border-radius:11px; color:#2457d6; font-weight:700; margin-bottom:1rem; }
.landing-card h3 { font-size:1rem; margin:0 0 .45rem; }
.landing-card p { font-size:.9rem; color:#66758d; line-height:1.6; margin:0; }
.landing-banner { background:#142238; color:white; border-radius:18px; padding:2rem; margin-top:3rem; }
.landing-banner h2 { color:white; margin:0 0 .4rem; }
.landing-banner p { color:#c0cce0; margin:0; }
.library-stat { display:flex; align-items:center; gap:16px; min-height:100px; padding:18px 20px; background:#fff; border:1px solid #e2e8f1; border-radius:15px; box-shadow:0 5px 22px rgba(22,40,75,.035); }
.library-stat-icon { display:grid; place-items:center; flex-shrink:0; width:48px; height:48px; border-radius:12px; background:#eaf1ff; color:#2457d6; }
.library-stat-icon svg { width:24px; height:24px; stroke:currentColor; fill:none; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.library-stat-label { color:#52627a; font-size:.82rem; font-weight:600; }
.library-stat-value { color:#142238; font-size:1.6rem; font-weight:700; line-height:1.2; }
.library-empty { text-align:center; padding:1.4rem .5rem 1rem; }
.library-empty-icon { display:grid; place-items:center; width:60px; height:60px; margin:0 auto 1rem; border-radius:18px; color:#2457d6; background:#edf3ff; }
.library-empty-icon svg { width:30px; height:30px; fill:none; stroke:currentColor; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }
.library-empty h3 { margin:0 0 .35rem; font-size:1.1rem; }
.library-empty p { color:#66758d; margin:0; }
[data-testid="stFileUploaderDropzone"] { background:#f4f8ff; border:1px dashed #a6bff4; border-radius:12px; }
div[data-testid="stVerticalBlockBorderWrapper"] > div { background: #fff; border: 1px solid #e2e8f1; border-radius: 15px; box-shadow: 0 5px 22px rgba(22, 40, 75, .035); }
[data-testid="stChatMessage"] { background: #f7f9fd; border-radius: 12px; }
.evidence { background: #f9fbff; border: 1px solid #e2e8f1; border-radius: 12px; padding: 1rem; margin-bottom: .7rem; overflow-wrap: anywhere; }
.evidence-label { color: #2457D6; font-weight: 700; }
.evidence-meta { color: #66758d; font-size: .86rem; margin: .3rem 0 .6rem; }
.evidence-text { white-space: pre-wrap; }
.stButton button, .stDownloadButton button { border-radius: 9px; }
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="combobox"]:focus-visible { outline: 3px solid #2457D6 !important; outline-offset: 2px; }
@media (max-width: 768px) { .block-container { padding: 1.25rem 1rem 3rem; } }
</style>
""", unsafe_allow_html=True)

defaults = {
    "store": RAGStore(), "chat_history": [], "search_history": [], "seen_uploads": set(),
    "page": "Home", "active_library": "Current session", "manual_states": {},
    "settings": {"provider": ALL_PROVIDERS[0], "keys": {}, "ollama_model": OLLAMA_MODEL,
                 "hybrid": True, "rerank": False, "threshold": 0.35, "count": 6},
    "selected_manuals": None, "delete_target": None, "library_notice": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return load_embedding_model()

@st.cache_resource(show_spinner=False)
def get_reranker():
    return load_reranker()

@st.cache_data(ttl=30, show_spinner=False)
def get_ollama_models():
    return ollama_models()

def ingest_pdf(file_name: str, file_bytes: bytes, store: RAGStore, notes: list) -> int:
    """Extract (in memory, OCR fallback) → structure-aware chunks → store. Returns # chunks added."""
    blocks, info = extract_pdf(file_bytes)
    chunks, chunk_meta = chunk_blocks(blocks)
    if not chunks:
        hint = " (scanned PDF — install Tesseract for OCR)" if info["ocr_unavailable"] else ""
        notes.append(("warning", f"⚠️ No text extracted from {file_name}{hint}"))
        return 0
    text = "\n".join(chunks)
    store.add_document(
        file_name,
        chunks,
        {
            "total_text_length": len(text),
            "num_chunks": len(chunks),
            "sample_text": text[:500],
            "pages": info["pages"],
            "ocr_pages": info["ocr_pages"],
            "tables": info["tables"],
            "sha1": file_hash(file_bytes),
        },
        chunk_meta,
    )
    if info["ocr_unavailable"]:
        notes.append(("warning", f"⚠️ {file_name}: {len(info['ocr_unavailable'])} scanned page(s) skipped — Tesseract not found."))
    return len(chunks)


def chat_to_markdown(history: list) -> str:
    lines = [f"# Chat Transcript\n\n_Generated {datetime.now().isoformat(timespec='seconds')}_\n"]
    for msg in history:
        lines.append(f"## {msg['role'].capitalize()}\n\n{msg['content']}\n")
    return "\n".join(lines)


def chat_to_pdf_bytes(history: list):
    """Return PDF bytes if fpdf2 is installed, else None."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Chat Transcript", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated {datetime.now().isoformat(timespec='seconds')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for msg in history:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, msg["role"].capitalize(), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        # Default core fonts only support latin-1; replace anything else gracefully.
        safe = msg["content"].encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe)
        pdf.ln(2)
    return bytes(pdf.output(dest="S"))



def go_to(page):
    st.session_state.page = page


def landing():
    nav, action = st.columns([5, 1])
    with nav:
        st.markdown('<nav class="landing-nav" aria-label="Page sections"><a href="#features" target="_self">Features</a><a href="#how-it-works" target="_self">How it works</a><a href="#use-cases" target="_self">Use cases</a></nav>', unsafe_allow_html=True)
    with action:
        st.button("Open workspace", type="primary", on_click=go_to, args=("Troubleshoot",), use_container_width=True)

    st.markdown('''<div class="landing-hero"><div class="landing-pill">YOUR MANUALS, MADE USEFUL</div>
        <h1>Find the fix<br><span>in your manual.</span></h1>
        <p>Turn equipment manuals into practical guidance. Describe the problem, find relevant instructions, and check the pages they came from.</p></div>''', unsafe_allow_html=True)
    start, library_link, _ = st.columns([1.4, 1.4, 4])
    with start:
        st.button("Start troubleshooting", type="primary", on_click=go_to, args=("Troubleshoot",), use_container_width=True)
    with library_link:
        st.button("Add a manual", on_click=go_to, args=("Manual Library",), use_container_width=True)
    st.caption("Works with text PDFs and scanned pages when Tesseract OCR is installed. Check the original manual before performing a procedure.")

    st.markdown('<div class="landing-section" id="features"><div class="workspace-kicker">THE TOOLKIT</div><h2>Less searching. More solving.</h2><p>Everything you need to navigate technical documentation with confidence.</p></div>', unsafe_allow_html=True)
    features = [
        ("01", "Search with context", "Find relevant passages by describing symptoms, a model, or an error code."),
        ("02", "See the source", "Review the manual name, page, section, and passage behind an answer."),
        ("03", "Keep a library", "Save and reopen collections of manuals for the equipment you support."),
        ("04", "Choose your method", "Use direct excerpts, a local Ollama model, or a supported cloud provider."),
    ]
    for col, (number, title, description) in zip(st.columns(4), features):
        with col:
            st.markdown(f'<div class="landing-card"><div class="number">{number}</div><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="landing-section" id="how-it-works"><div class="workspace-kicker">HOW IT WORKS</div><h2>Three steps to a useful answer.</h2><p>Start with a PDF and follow the evidence back to the source.</p></div>', unsafe_allow_html=True)
    steps = [("01", "Add your manual", "Upload a PDF or open a saved library."),
             ("02", "Describe the issue", "Include the device model, symptom, or error code."),
             ("03", "Review the guidance", "Read the relevant passages and check their page references.")]
    for col, (number, title, description) in zip(st.columns(3), steps):
        with col:
            st.markdown(f'<div class="landing-card"><div class="number">{number}</div><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="landing-section" id="use-cases"><div class="workspace-kicker">MADE FOR REAL EQUIPMENT</div><h2>One workspace, many manuals.</h2><p>Keep the instructions for the devices you work with close at hand.</p></div>', unsafe_allow_html=True)
    uses = [("01", "Printers", "Look up setup instructions, error codes, and paper-feed problems."),
            ("02", "Appliances", "Find operating instructions and maintenance information."),
            ("03", "Equipment", "Search technical manuals for documented procedures and precautions.")]
    for col, (number, title, description) in zip(st.columns(3), uses):
        with col:
            st.markdown(f'<div class="landing-card"><div class="number">{number}</div><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="landing-banner"><h2>Ready to get started?</h2><p>Bring your manual and find the guidance you need.</p></div>', unsafe_allow_html=True)
    st.button("Go to workspace", on_click=go_to, args=("Troubleshoot",), type="primary")
    st.markdown("---")
    st.caption("ManualMate | Manual-powered troubleshooting | Project documentation is available in README.md")


def ready_store(store):
    return store.index is not None and store.index.ntotal == len(store.chunks) and bool(store.chunks)


def render_evidence(sources):
    if not sources:
        st.caption("Sources will appear here after a search.")
        return
    for number, source in enumerate(sources, 1):
        name = html.escape(source["doc"])
        page = f"Page {source['page']}" if source.get("page") else "Page unavailable"
        section = f" | {html.escape(source['section'])}" if source.get("section") else ""
        text = html.escape(source["text"])
        st.markdown(f'<div class="evidence"><span class="evidence-label">[{number}]</span> <strong>{name}</strong><div class="evidence-meta">{page}{section}</div><div class="evidence-text" dir="auto">{text}</div></div>', unsafe_allow_html=True)


def run_query(prompt, store, selected, settings):
    st.session_state.search_history.append(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    try:
        if not selected:
            raise ValueError("Select at least one manual to search.")
        with st.spinner("Searching the selected manuals..."):
            results = search(store, get_embedding_model(), prompt, k=settings["count"],
                hybrid=settings["hybrid"], doc_filter=selected, min_score=settings["threshold"],
                reranker=get_reranker() if settings["rerank"] else None)
        if not results:
            answer = "No sufficiently relevant passages were found. Try adding the device model, error code, or a relevant manual."
            sources = []
        else:
            context, count = build_context(results)
            sources = results[:count]
            provider = settings["provider"]
            key = resolve_api_key(provider, settings["keys"].get(provider, ""))
            if provider == ALL_PROVIDERS[0] or (provider in PROVIDER_ENV_VARS and not key):
                answer = "**Matching manual excerpts**\n\n" + "\n\n".join(
                    f"[{i}] {item['text']}" for i, item in enumerate(sources, 1))
                if provider != ALL_PROVIDERS[0]:
                    answer = "No API key was found; displaying manual excerpts.\n\n" + answer
            else:
                try:
                    with st.spinner("Preparing guidance..."):
                        answer = "".join(stream_answer(provider, prompt, context, key, settings["ollama_model"]))
                except Exception as exc:
                    answer = f"Answer provider error: {exc}. Showing matching excerpts instead.\n\n" + "\n\n".join(
                        f"[{i}] {item['text']}" for i, item in enumerate(sources, 1))
            answer += sources_footer(answer, sources)
    except Exception as exc:
        answer, sources = f"Could not search the manuals: {exc}", []
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})


def troubleshoot():
    store = st.session_state.store
    settings = st.session_state.settings
    st.markdown('<div class="workspace-kicker">YOUR WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Troubleshoot")
    st.caption("Find clear guidance in the manuals you trust.")
    if not ready_store(store):
        with st.container(border=True):
            st.subheader("Add a manual to begin")
            st.write("Upload a device manual or open a saved library to find troubleshooting guidance.")
            left, right = st.columns(2)
            left.button("Upload manuals", type="primary", on_click=go_to, args=("Manual Library",), use_container_width=True)
            right.button("Open saved library", on_click=go_to, args=("Manual Library",), use_container_width=True)
        return
    st.caption(f"Active library: {st.session_state.active_library} | {len(store.documents)} manuals | Answer method: {settings['provider']}" + (f" | {settings['ollama_model']}" if settings['provider'] == PROVIDER_OLLAMA else ""))
    selected = st.session_state.selected_manuals
    if selected is None:
        selected = store.documents
    selected = [name for name in selected if name in store.documents]
    selected = st.multiselect("Active manuals", store.documents, default=selected)
    st.session_state.selected_manuals = selected
    with st.container(border=True):
        with st.form("issue_form", clear_on_submit=True):
            prompt = st.text_input("Describe the issue", placeholder="Enter a symptom, device model, or error code.")
            submitted = st.form_submit_button("Find guidance", type="primary")
    if submitted:
        if prompt.strip():
            run_query(prompt.strip(), store, selected, settings)
        else:
            st.warning("Describe the issue before searching.")
    answers = [m for m in st.session_state.chat_history if m["role"] == "assistant"]
    left, right = st.columns([3, 2], gap="medium")
    with left:
        with st.container(border=True):
            st.subheader("Troubleshooting guidance")
            if not st.session_state.chat_history:
                st.caption("Your guidance will appear here after you describe an issue.")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            st.markdown("**Conversation tools**")
            a, b, c = st.columns(3)
            with a:
                if st.button("Clear conversation", disabled=not st.session_state.chat_history):
                    st.session_state.chat_history = []
                    st.rerun()
            with b:
                st.download_button("Markdown", chat_to_markdown(st.session_state.chat_history), file_name="manualmate-notes.md", mime="text/markdown", disabled=not st.session_state.chat_history)
            with c:
                pdf = chat_to_pdf_bytes(st.session_state.chat_history) if st.session_state.chat_history else None
                st.download_button("PDF", pdf or b"", file_name="manualmate-notes.pdf", mime="application/pdf", disabled=pdf is None)
    with right:
        with st.container(border=True):
            st.subheader("Manual sources")
            st.caption("Evidence for the latest answer. Source numbers match the guidance.")
            render_evidence(answers[-1].get("sources", []) if answers else [])


def process_uploads(files):
    import faiss
    original = st.session_state.store
    pending = [f for f in files if f.file_id not in st.session_state.seen_uploads]
    if not pending:
        return
    # Work on an isolated copy; failed indexing must not corrupt an existing library.
    candidate = RAGStore(chunks=list(original.chunks), doc_names=list(original.doc_names),
        chunk_meta=list(original.chunk_meta), pdf_metadata=dict(original.pdf_metadata),
        embedding_model=original.embedding_model,
        index=faiss.clone_index(original.index) if original.index is not None else None, bm25=original.bm25)
    added = []
    status = st.empty()
    notes = []
    for file in pending:
        st.session_state.seen_uploads.add(file.file_id)
        data = file.getvalue()
        if candidate.has_document(file.name) or candidate.find_by_hash(file_hash(data)):
            st.session_state.manual_states[file.name] = "Already in library"
            continue
        st.session_state.manual_states[file.name] = "Reading PDF"
        try:
            with status.container():
                with st.spinner("Reading PDF: " + file.name):
                    count = ingest_pdf(file.name, data, candidate, notes)
            status.empty()
            if count:
                added.append(file.name)
                st.session_state.manual_states[file.name] = "Waiting for indexing"
            else:
                st.session_state.manual_states[file.name] = "No readable passages"
        except Exception as exc:
            status.empty()
            st.session_state.manual_states[file.name] = f"Reading failed: {exc}"
    if added:
        try:
            with status.container():
                with st.spinner("Loading embedding model (first use may download model files)..."):
                    model = get_embedding_model()
            status.empty()
            for name in added:
                st.session_state.manual_states[name] = "Indexing passages"
            with status.container():
                with st.spinner("Indexing passages..."):
                    build_indexes(candidate, model)
            status.empty()
            st.session_state.store = candidate
            st.session_state.selected_manuals = None
            for name in added:
                st.session_state.manual_states[name] = "Ready"
        except Exception as exc:
            status.empty()
            for name in added:
                st.session_state.manual_states[name] = f"Indexing failed: {exc}"
            # Allow retry by removing and re-adding a failed upload.
            for file in pending:
                if file.name in added:
                    st.session_state.seen_uploads.discard(file.file_id)
            notes.append(("error", f"Indexing failed; the existing library was preserved: {exc}"))
    st.session_state.library_notice = notes


def library():
    st.title("Manual library")
    st.caption("Manage the PDFs used for troubleshooting.")
    overview = st.empty()
    with st.container(border=True):
        st.subheader("Add equipment manuals")
        files = st.file_uploader("Upload PDF manuals", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
        st.caption("PDF manuals are processed locally. The first model load may download files.")
        if files:
            process_uploads(files)
        for kind, message in st.session_state.library_notice or []:
            getattr(st, kind)(message)
    store = st.session_state.store
    with overview.container():
        metrics = [
            ("Manuals", len(store.documents), '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H11v16H6.5A2.5 2.5 0 0 0 4 21z"/><path d="M20 5.5A2.5 2.5 0 0 0 17.5 3H11v16h6.5A2.5 2.5 0 0 1 20 21z"/>'),
            ("Searchable passages", store.index.ntotal if store.index is not None else 0, '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/>'),
            ("Questions asked", len(st.session_state.search_history), '<path d="M5 4h14a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-7l-5 4v-4H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/>'),
        ]
        for column, (label, value, icon) in zip(st.columns(3), metrics):
            with column:
                st.markdown(f'<div class="library-stat"><div class="library-stat-icon"><svg viewBox="0 0 24 24" aria-hidden="true">{icon}</svg></div><div><div class="library-stat-label">{label}</div><div class="library-stat-value">{value}</div></div></div>', unsafe_allow_html=True)
    with st.container(border=True):
        names = list(dict.fromkeys(store.documents + list(st.session_state.manual_states)))
        if not names:
            st.markdown('<div class="library-empty"><div class="library-empty-icon"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 5.5A2.5 2.5 0 0 1 5.5 3H11v16H5.5A2.5 2.5 0 0 0 3 21z"/><path d="M21 5.5A2.5 2.5 0 0 0 18.5 3H11v16h7.5A2.5 2.5 0 0 1 21 21z"/></svg></div><h3>Start troubleshooting</h3><p>Upload a manual or open a saved library to ask your first question.</p></div>', unsafe_allow_html=True)
        else:
            st.subheader("Current manuals")
            st.caption(f"{st.session_state.active_library} | {len(store.documents)} manuals")
        for name in names:
            state = st.session_state.manual_states.get(name, "Ready" if name in store.documents and ready_store(store) else "Not indexed")
            st.write(f"**{name}**  -  {state}")
    with st.container(border=True):
        st.subheader("Saved libraries")
        saved = list_saved_indexes()
        selected = st.selectbox("Choose a saved library", [""] + saved, format_func=lambda x: x or "Select a library")
        a, b = st.columns(2)
        with a:
            if st.button("Open library", disabled=not selected, use_container_width=True):
                try:
                    with st.spinner("Loading embedding model and checking library compatibility..."):
                        loaded = load_store(selected)
                        ensure_compatible(loaded, get_embedding_model())
                    st.session_state.store = loaded
                    st.session_state.active_library = selected
                    st.session_state.selected_manuals = None
                    st.session_state.manual_states = {name: "Ready" for name in loaded.documents}
                    st.session_state.seen_uploads = set()
                    st.session_state.delete_target = None
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not open library: {exc}")
        with b:
            if st.button("Delete library", disabled=not selected, use_container_width=True):
                st.session_state.delete_target = selected
        if st.session_state.delete_target:
            target = st.session_state.delete_target
            st.warning(f"Delete saved library '{target}'? This cannot be undone.")
            c, d = st.columns(2)
            if c.button("Confirm delete", type="primary"):
                try:
                    delete_store(target)
                    st.session_state.delete_target = None
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete library: {exc}")
            if d.button("Cancel"):
                st.session_state.delete_target = None
                st.rerun()
        name = st.text_input("Save current library as", placeholder="e.g. office-printers")
        if st.button("Save library", disabled=not name.strip() or not ready_store(store)):
            try:
                path = save_store(store, name.strip())
                st.session_state.active_library = path.name
                st.success(f"Saved library: {path.name}")
            except Exception as exc:
                st.error(f"Could not save library: {exc}")


def settings_page():
    settings = st.session_state.settings
    st.title("Settings")
    st.caption("Choose how answers are generated and how manuals are searched.")
    with st.container(border=True):
        st.subheader("Answer method")
        provider = st.selectbox("Provider", ALL_PROVIDERS, index=ALL_PROVIDERS.index(settings["provider"]))
        settings["provider"] = provider
        if provider in PROVIDER_ENV_VARS:
            settings["keys"][provider] = st.text_input("API key", value=settings["keys"].get(provider, ""), type="password", help=f"Leave blank to use {PROVIDER_ENV_VARS[provider]} from the environment.")
        elif provider == PROVIDER_OLLAMA:
            models = get_ollama_models()
            if models is None:
                st.error(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Start the Ollama server and try again.")
            elif not models:
                st.warning("Ollama is connected but has no installed models. Pull a model first.")
            else:
                current = settings["ollama_model"]
                settings["ollama_model"] = st.selectbox("Ollama model", models, index=models.index(current) if current in models else 0)
                st.success("Ollama connected.")
        else:
            st.caption("Displays retrieved excerpts directly, without a language model.")
    with st.expander("Advanced search"):
        settings["hybrid"] = st.checkbox("Combine keyword and semantic search", value=settings["hybrid"])
        settings["rerank"] = st.checkbox("Rerank results", value=settings["rerank"])
        settings["threshold"] = st.slider("Minimum relevance", 0.0, 1.0, float(settings["threshold"]), 0.05)
        settings["count"] = st.slider("Passages to search", 1, 20, settings["count"])

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-symbol">M</div><div><div class="brand-name">ManualMate</div><div class="brand-caption">TECHNICIAN WORKSPACE</div></div></div>', unsafe_allow_html=True)
    for page in ("Home", "Troubleshoot", "Manual Library", "Settings"):
        st.button(page, key="nav_" + page, type="primary" if st.session_state.page == page else "secondary", on_click=go_to, args=(page,), use_container_width=True)
    st.divider()
    st.caption(f"Active library: {st.session_state.active_library}")
    st.caption(f"{len(st.session_state.store.documents)} manuals")

if st.session_state.page == "Home":
    landing()
elif st.session_state.page == "Manual Library":
    library()
elif st.session_state.page == "Settings":
    settings_page()
else:
    troubleshoot()
