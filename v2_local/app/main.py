import os
import sys
import time
import streamlit as st
# Force local-only settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.summarizer import generate_summary

load_dotenv()

st.set_page_config(page_title="UWindsor RateMyProf Agent", page_icon="🎓", layout="wide")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "meta" not in st.session_state:
    st.session_state.meta = None

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* ── App Background ── */
    .stApp {
        background: #080c14 !important;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99, 102, 241, 0.12) 0%, transparent 60%),
            radial-gradient(ellipse 40% 30% at 80% 80%, rgba(16, 185, 129, 0.05) 0%, transparent 50%) !important;
    }

    /* ── Main content block ── */
    .block-container {
        padding: 3rem 4rem !important;
        max-width: 1100px !important;
    }

    /* ── Title ── */
    h1 {
        font-family: 'Nunito', sans-serif !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #f1f5f9 !important;
        letter-spacing: -0.01em !important;
        line-height: 1.15 !important;
        margin-bottom: 0.25rem !important;
    }

    /* ── Subheaders ── */
    h2, h3 {
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        letter-spacing: -0.01em !important;
    }

    /* ── Body text ── */
    p, .stMarkdown p, label, .stTextInput label, .stTextArea label {
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
    }

    /* ── Input Fields ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.08) !important;
        outline: none !important;
    }

    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #475569 !important;
    }

    /* ── Run Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 2rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.45) !important;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Status Box ── */
    .stStatus {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 12px !important;
        color: #94a3b8 !important;
    }

    /* ── Error / Alert ── */
    .stAlert {
        background: rgba(239, 68, 68, 0.08) !important;
        border: 1px solid rgba(239, 68, 68, 0.25) !important;
        border-radius: 10px !important;
        color: #fca5a5 !important;
    }

    /* ── Metric Cards ── */
    .metric-box {
        text-align: center;
        padding: 28px 16px;
        background: rgba(15, 23, 42, 0.75);
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.04);
        position: relative;
        overflow: hidden;
    }

    .metric-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.6), transparent);
        opacity: 0;
        transition: opacity 0.25s ease;
    }

    .metric-box:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15), 0 4px 24px rgba(0, 0, 0, 0.4);
    }

    .metric-box:hover::before {
        opacity: 1;
    }

    .metric-label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }

    .metric-value {
        font-family: 'Nunito', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1;
        display: block;
        letter-spacing: -0.01em;
    }

    /* ── Summary Container (st.container with border) ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2) !important;
    }

    /* ── Markdown content inside summary ── */
    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] li {
        color: #cbd5e1 !important;
        font-size: 0.95rem !important;
        line-height: 1.8 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] strong {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] h2,
    [data-testid="stVerticalBlockBorderWrapper"] h3,
    [data-testid="stVerticalBlockBorderWrapper"] h4 {
        font-family: 'Nunito', sans-serif !important;
        color: #f1f5f9 !important;
        margin-top: 1.5rem !important;
        border-bottom: 1px solid rgba(99, 102, 241, 0.1) !important;
        padding-bottom: 0.5rem !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] ul {
        padding-left: 1.2rem !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] li {
        margin-bottom: 0.4rem !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #6366f1;
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid rgba(99, 102, 241, 0.12) !important;
        margin: 1.5rem 0 !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_db = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = ChatOllama(model="llama3.1:8b-instruct-q8_0", temperature=0)
    return vector_db, reranker, llm

vector_db, reranker, llm = load_models()

st.title("🎓 UWindsor RateMyProf Agent")
st.write("Specialized Local RAG Agent using Llama-3.1 and Agentic Reasoning.")

prof_name_input = st.text_input("Professor Name", placeholder="e.g. Ziad Kobti")
query_input = st.text_area("Specific Question (Optional)")

if st.button("Run Agentic Analysis"):
    if not prof_name_input:
        st.error("Please enter a professor name.")
    else:
        with st.status("🧠 Agent Reasoning: Fetching & Reranking...", expanded=True) as status:
            user_input = prof_name_input.strip().title()
            
            # Step 1: Try exact match first
            results = vector_db.similarity_search("placeholder", k=1, filter={"prof_name": user_input})
            
            # Step 2: Agentic Fallback
            if not results:
                st.write(f"⚠️ '{user_input}' not found exactly. Searching for similar names...")
                broad_results = vector_db.similarity_search(user_input, k=3)
                
                if broad_results:
                    best_match = broad_results[0].metadata.get('prof_name')
                    st.write(f"✅ Agent identified: **{best_match}**")
                    user_name_to_use = best_match
                    results = vector_db.similarity_search("placeholder", k=1, filter={"prof_name": best_match})
                else:
                    status.update(label="Professor not found.", state="error")
                    st.stop()
            else:
                user_name_to_use = user_input

            # Step 3: Run RAG
            query = query_input if query_input.strip() else f"Summarize reviews for {user_name_to_use}"
            st.session_state.analysis_result = generate_summary(query, user_name_to_use, vector_db, reranker, llm)
            st.session_state.meta = results[0].metadata
            status.update(label=f"Analysis Complete for {user_name_to_use}!", state="complete")

if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    meta = st.session_state.meta
    summary_text = res.get('summary', "")

    # --- 1. Metrics Section (TOP) ---
    st.subheader(f"📊 {meta.get('prof_name')} | {meta.get('dept')}")
    col1, col2, col3 = st.columns(3)
    #_, col1, col2, col3, _ = st.columns([0.5, 2, 2, 2, 0.5])

    with col1:
        st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>⭐ Quality</div>
                <span class='metric-value'>{meta.get('avg_rating')}/5</span>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>🔄 Retake</div>
                <span class='metric-value'>{meta.get('would_take_again')}</span>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>📈 Difficulty</div>
                <span class='metric-value'>{meta.get('avg_difficulty')}/5</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # --- 2. Logic to Combine the Report ---
    if "|||" in summary_text:
        parts = summary_text.split("|||", 1)
        # We keep the "Agent's Direct Response" header from the prompt
        direct_answer = parts[0].strip()
        remaining_report = parts[1].strip() if len(parts) > 1 else ""
        # Combine them with a clean horizontal rule
        final_display_text = f"{direct_answer}\n\n\n\n{remaining_report}"
    else:
        final_display_text = summary_text

    # --- 3. Single Unified Container ---
    if final_display_text:
        # Extra safety split to remove context/reviews leakage
        if "REVIEWS:" in final_display_text:
            final_display_text = final_display_text.split("REVIEWS:")[0].strip()
            
        with st.container(border=True):
            st.markdown(final_display_text)