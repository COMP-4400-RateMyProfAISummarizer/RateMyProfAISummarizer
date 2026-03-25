import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.summarizer import generate_summary

load_dotenv()

st.set_page_config(page_title="UWindsor RateMyProf Agent", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=os.getenv("CLOUD_API_KEY"))
    return vector_db, reranker, llm

vector_db, reranker, llm = init_components()

st.title("🎓 UWindsor RateMyProf Agent")
prof_name_input = st.text_input("Professor name", placeholder="e.g. Ziad Kobti")
query_input = st.text_area("Your question (optional)")

if st.button("Generate Summary"):
    if not prof_name_input.strip():
        st.warning("Please enter a professor name.")
    else:
        user_input = prof_name_input.strip().title()
        results = vector_db.similarity_search("placeholder", k=1, filter={"prof_name": user_input})

        if not results:
            st.error("No data found.")
        else:
            meta = results[0].metadata
            st.subheader(f"📊 {meta.get('prof_name')} ({meta.get('dept')})")
            
            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f"<div class='metric-card'>⭐ Quality: {meta.get('avg_rating')}/5</div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div class='metric-card'>🔄 Retake: {meta.get('would_take_again')}</div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div class='metric-card'>📈 Difficulty: {meta.get('avg_difficulty')}/5</div>", unsafe_allow_html=True)

            with st.spinner("🤖 Generating AI Summary..."):
                query = query_input.strip() if query_input.strip() else f"Provide a review summary for {user_input}"
                result = generate_summary(query, user_input, vector_db, reranker, llm)
                
                st.markdown("### 📝 Summary")
                st.write(result["summary"])
                st.success("Generated using Google Gemini")