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
    .metric-box {
        text-align: center;
        padding: 15px;
        background: #0f172a;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
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
    summary_text = res.get('summary')

    # --- Metrics Section ---
    st.subheader(f"📊 {meta.get('prof_name')} | {meta.get('dept')}")
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f"<div class='metric-box'><small>⭐ Quality</small><br><b>{meta.get('avg_rating')}/5</b></div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='metric-box'><small>🔄 Retake</small><br><b>{meta.get('would_take_again')}</b></div>", unsafe_allow_html=True)
    with col3: st.markdown(f"<div class='metric-box'><small>📈 Difficulty</small><br><b>{meta.get('avg_difficulty')}/5</b></div>", unsafe_allow_html=True)

    if "### 🎯 DIRECT ANSWER" in summary_text:
        parts = summary_text.split("---", 1)
        direct_answer = parts[0].replace("### 🎯 DIRECT ANSWER", "").strip()
        remaining_report = parts[1] if len(parts) > 1 else ""
        
        st.markdown("### 🧠 Agent's Direct Response")
        with st.chat_message("assistant", avatar="🎓"):
            st.markdown(direct_answer)
        st.divider()
    else:
        remaining_report = summary_text