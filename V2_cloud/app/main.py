import os
import sys
import time
import streamlit as ui

# Keep tokenizer threads controlled for local execution stability
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

# Allow imports from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.summarizer import generate_summary

load_dotenv()


# -------------------------------------------------
# Page Setup
# -------------------------------------------------
ui.set_page_config(
    page_title="UWindsor RateMyProf Agent",
    page_icon="🎓",
    layout="wide"
)


# -------------------------------------------------
# Session State
# -------------------------------------------------
if "analysis_data" not in ui.session_state:
    ui.session_state.analysis_data = None

if "prof_metadata" not in ui.session_state:
    ui.session_state.prof_metadata = None


# -------------------------------------------------
# Premium Light UI Styling
# -------------------------------------------------
ui.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Nunito:wght@700;800&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(99, 102, 241, 0.08), transparent 26%),
        radial-gradient(circle at top right, rgba(16, 185, 129, 0.06), transparent 22%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 50%, #ffffff 100%) !important;
    color: #1e293b !important;
}

/* Main width */
.block-container {
    max-width: 1120px !important;
    padding-top: 4.2rem !important;
    padding-bottom: 3rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

/* ---------- Hero ---------- */
.hero-shell {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 30px 30px 24px 30px;
    margin-bottom: 1.3rem;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(241, 245, 249, 0.92));
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow:
        0 10px 40px rgba(15, 23, 42, 0.08),
        inset 0 1px 0 rgba(255,255,255,0.6);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}

.hero-shell::before {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    top: -90px;
    right: -70px;
    background: radial-gradient(circle, rgba(99,102,241,0.12), transparent 70%);
    pointer-events: none;
}

.hero-shell::after {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    bottom: -100px;
    left: -40px;
    background: radial-gradient(circle, rgba(16,185,129,0.10), transparent 70%);
    pointer-events: none;
}

.hero-badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #4338ca;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.14);
    margin-bottom: 14px;
}

.hero-title {
    font-family: 'Nunito', sans-serif !important;
    font-size: 2.65rem;
    line-height: 1.08;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1rem;
    color: #475569;
    line-height: 1.75;
    max-width: 760px;
    margin-top: 0.2rem;
}

/* ---------- Labels / text ---------- */
p, .stMarkdown p, label, .stTextInput label, .stTextArea label {
    color: #334155 !important;
    line-height: 1.7 !important;
    font-size: 0.96rem !important;
}

h1, h2, h3, h4 {
    color: #0f172a !important;
    letter-spacing: -0.01em;
}

/* ---------- Inputs ---------- */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 16px !important;
    color: #0f172a !important;
    padding: 0.88rem 1rem !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.8),
        0 4px 12px rgba(15, 23, 42, 0.04) !important;
    transition: all 0.22s ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border: 1px solid rgba(99, 102, 241, 0.45) !important;
    box-shadow:
        0 0 0 3px rgba(99, 102, 241, 0.08),
        0 8px 20px rgba(15, 23, 42, 0.06) !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder,
.stTextArea textarea::placeholder {
    color: #94a3b8 !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border: 0 !important;
    border-radius: 14px !important;
    padding: 0.78rem 1.4rem !important;
    font-size: 0.90rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #ffffff !important;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 55%, #4338ca 100%) !important;
    box-shadow:
        0 10px 26px rgba(79, 70, 229, 0.18),
        inset 0 1px 0 rgba(255,255,255,0.14) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
}

/* Force inner button text to stay white */
.stButton > button *,
.stButton > button span,
.stButton > button p,
.stButton > button div {
    color: #ffffff !important;
}

/* Hover */
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    filter: brightness(1.04) !important;
    color: #ffffff !important;
    box-shadow:
        0 16px 34px rgba(79, 70, 229, 0.22),
        0 0 18px rgba(99, 102, 241, 0.22),
        inset 0 1px 0 rgba(255,255,255,0.16) !important;
}

/* Keep text white on hover too */
.stButton > button:hover *,
.stButton > button:hover span,
.stButton > button:hover p,
.stButton > button:hover div {
    color: #ffffff !important;
}

.stButton > button:active {
    transform: translateY(0px) scale(0.995) !important;
}

/* ---------- Status / alerts ---------- */
.stStatus {
    background: rgba(255, 255, 255, 0.78) !important;
    border: 1px solid rgba(99, 102, 241, 0.14) !important;
    border-radius: 18px !important;
    color: #1e293b !important;
}

.stAlert {
    border-radius: 14px !important;
    border: 1px solid rgba(239, 68, 68, 0.20) !important;
    background: rgba(254, 242, 242, 0.95) !important;
    color: #991b1b !important;
}

/* ---------- Metrics ---------- */
.metric-panel {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 26px 18px;
    border-radius: 22px;
    background: linear-gradient(180deg, #ffffff, #f8fafc);
    border: 1px solid rgba(148,163,184,0.16);
    box-shadow:
        0 12px 26px rgba(15,23,42,0.06),
        inset 0 1px 0 rgba(255,255,255,0.75);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    transition: all 0.24s ease;
    min-height: 142px;
}

.metric-panel::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent, rgba(99,102,241,0.05), transparent);
    transform: translateX(-100%);
    transition: transform 0.7s ease;
}

.metric-panel:hover::before {
    transform: translateX(100%);
}

.metric-panel:hover {
    transform: translateY(-4px);
    border-color: rgba(99,102,241,0.22);
    box-shadow:
        0 16px 40px rgba(79, 70, 229, 0.08),
        0 12px 26px rgba(15,23,42,0.08);
}

.metric-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.metric-number {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800;
    font-size: 2.45rem;
    line-height: 1;
    color: #0f172a;
}

.metric-note {
    font-size: 0.86rem;
    color: #64748b;
    margin-top: 10px;
}

/* ---------- Section title ---------- */
.section-chip {
    display: inline-block;
    margin-top: 1.1rem;
    margin-bottom: 0.6rem;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.12);
    color: #4338ca;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ---------- Report container ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, #ffffff, #f8fafc) !important;
    border: 1px solid rgba(148,163,184,0.16) !important;
    border-radius: 22px !important;
    padding: 1.5rem 1.6rem !important;
    box-shadow:
        0 12px 30px rgba(15,23,42,0.06),
        inset 0 1px 0 rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

/* Markdown inside report */
[data-testid="stVerticalBlockBorderWrapper"] p,
[data-testid="stVerticalBlockBorderWrapper"] li {
    color: #334155 !important;
    font-size: 0.98rem !important;
    line-height: 1.9 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] strong {
    color: #0f172a !important;
    font-weight: 700 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] h1,
[data-testid="stVerticalBlockBorderWrapper"] h2,
[data-testid="stVerticalBlockBorderWrapper"] h3,
[data-testid="stVerticalBlockBorderWrapper"] h4 {
    font-family: 'Nunito', sans-serif !important;
    color: #0f172a !important;
    margin-top: 1.35rem !important;
    margin-bottom: 0.7rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid rgba(99,102,241,0.10) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] ul {
    padding-left: 1.25rem !important;
}

[data-testid="stVerticalBlockBorderWrapper"] li {
    margin-bottom: 0.38rem !important;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #e2e8f0;
}
::-webkit-scrollbar-thumb {
    background: #94a3b8;
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
    background: #6366f1;
}

/* ---------- Divider ---------- */
hr {
    border: none !important;
    border-top: 1px solid rgba(99, 102, 241, 0.10) !important;
    margin: 1.2rem 0 !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Hero
# -------------------------------------------------
ui.markdown("""
<div class="hero-shell">
    <div class="hero-badge">Local RAG • Ollama • Pinecone • Cross-Encoder</div>
    <div class="hero-title">🎓 UWindsor RateMyProf Agent</div>
    <div class="hero-subtitle">
        A polished local retrieval-augmented generation assistant that finds professor review data,
        reranks the most relevant evidence, and produces a grounded analysis with a clean dashboard-style UI.
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Load Models
# -------------------------------------------------
@ui.cache_resource
def initialize_models():
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embedding_model,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )

    rerank_model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    language_model = ChatOllama(
        model="llama3.2:1b",
        temperature=0
    )

    return vector_store, rerank_model, language_model


vector_store, rerank_model, language_model = initialize_models()


# -------------------------------------------------
# Input Area
# -------------------------------------------------
ui.markdown("<div class='section-chip'>Input</div>", unsafe_allow_html=True)

professor_input = ui.text_input(
    "Professor Name",
    placeholder="e.g. Ziad Kobti"
)

question_input = ui.text_area(
    "Specific Question (Optional)",
    placeholder="Ask something specific, or leave blank to generate a general review summary."
)


# -------------------------------------------------
# Run Analysis
# -------------------------------------------------
if ui.button("Run Agentic Analysis"):
    if not professor_input.strip():
        ui.error("Please enter a professor name.")
    else:
        with ui.status("🧠 Agent Reasoning: Fetching, matching, and reranking...", expanded=True) as reasoning_box:
            formatted_name = professor_input.strip().title()

            matched_results = vector_store.similarity_search(
                "placeholder",
                k=1,
                filter={"prof_name": formatted_name}
            )

            if not matched_results:
                ui.write(f"⚠️ '{formatted_name}' was not found exactly. Searching similar professor names...")

                candidate_results = vector_store.similarity_search(formatted_name, k=3)

                if candidate_results:
                    resolved_name = candidate_results[0].metadata.get("prof_name")
                    ui.write(f"✅ Closest professor identified: **{resolved_name}**")

                    selected_professor = resolved_name
                    matched_results = vector_store.similarity_search(
                        "placeholder",
                        k=1,
                        filter={"prof_name": resolved_name}
                    )
                else:
                    reasoning_box.update(label="Professor not found.", state="error")
                    ui.stop()
            else:
                selected_professor = formatted_name

            final_query = (
                question_input.strip()
                if question_input.strip()
                else f"Summarize reviews for {selected_professor}"
            )

            ui.session_state.analysis_data = generate_summary(
                final_query,
                selected_professor,
                vector_store,
                rerank_model,
                language_model
            )

            ui.session_state.prof_metadata = matched_results[0].metadata

            reasoning_box.update(
                label=f"Analysis complete for {selected_professor}!",
                state="complete"
            )


# -------------------------------------------------
# Results Display
# -------------------------------------------------
if ui.session_state.analysis_data:
    result_payload = ui.session_state.analysis_data
    professor_meta = ui.session_state.prof_metadata
    summary_text = result_payload.get("summary", "")

    ui.markdown("<div class='section-chip'>Overview</div>", unsafe_allow_html=True)
    ui.subheader(f"📊 {professor_meta.get('prof_name')} | {professor_meta.get('dept')}")

    metric_col_1, metric_col_2, metric_col_3 = ui.columns(3)

    with metric_col_1:
        ui.markdown(f"""
        <div class="metric-panel">
            <div class="metric-label">⭐ Quality</div>
            <div class="metric-number">{professor_meta.get('avg_rating')}/5</div>
            <div class="metric-note">Average student rating</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col_2:
        ui.markdown(f"""
        <div class="metric-panel">
            <div class="metric-label">🔄 Retake</div>
            <div class="metric-number">{professor_meta.get('would_take_again')}</div>
            <div class="metric-note">Would students take again?</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col_3:
        ui.markdown(f"""
        <div class="metric-panel">
            <div class="metric-label">📈 Difficulty</div>
            <div class="metric-number">{professor_meta.get('avg_difficulty')}/5</div>
            <div class="metric-note">Reported course challenge</div>
        </div>
        """, unsafe_allow_html=True)

    ui.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    ui.markdown("<div class='section-chip'>Analysis Report</div>", unsafe_allow_html=True)

    if "|||" in summary_text:
        split_output = summary_text.split("|||", 1)
        direct_answer = split_output[0].strip()
        remaining_report = split_output[1].strip() if len(split_output) > 1 else ""
        display_output = f"{direct_answer}\n\n\n\n{remaining_report}"
    else:
        display_output = summary_text

    if "REVIEWS:" in display_output:
        display_output = display_output.split("REVIEWS:")[0].strip()

    if display_output:
        with ui.container(border=True):
            ui.markdown(display_output)