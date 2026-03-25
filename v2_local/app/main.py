import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

# Allow access to core folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.summarizer import generate_summary
from core.retriever import retrieve_reviews

load_dotenv()

st.set_page_config(
    page_title="UWindsor RateMyProf Agent",
    page_icon="🎓",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Main background adjustment */
    .stApp {
        background-color: var(--background-color);
    }

    /* Force Title and Text Visibility */
    h1, h2, h3, p, span, label {
        color: var(--text-color) !important;
    }

    /* Style the Input Boxes for better contrast */
    .stTextInput div[data-baseweb="input"], 
    .stTextArea div[data-baseweb="base-input"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid #4a5568 !important;
        border-radius: 8px;
    }

    /* Professional Metric Cards */
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Source Cards with subtle border */
    .source-card {
        background-color: var(--secondary-background-color);
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 14px;
        border-left: 5px solid #3182ce; /* UWindsor Blue accent */
        color: var(--text-color);
    }

    .small-note {
        color: #a0aec0;
        font-size: 0.92rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎓 UWindsor RateMyProf Agent")
st.write("Ask a question about a UWindsor professor based on student reviews.")


@st.cache_resource
def init_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    llm = ChatOllama(
        model="llama3",
        temperature=0.2
    )

    return vector_db, reranker, llm


def render_metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.95rem; color:#6b7280; margin-bottom:6px;">{title}</div>
            <div style="font-size:2rem; font-weight:700; color:#1f2937;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


vector_db, reranker, llm = init_components()

prof_name_input = st.text_input("Professor name", placeholder="e.g. Ziad Kobti")
query_input = st.text_area(
    "Your question (optional)",
    placeholder="e.g. How is the grading style and workload?"
)

if st.button("Generate Summary"):
    raw_input = prof_name_input.strip()

    if not raw_input:
        st.warning("Please enter a professor name.")
    else:
        user_input = raw_input.title()

        results = vector_db.similarity_search(
            "placeholder",
            k=1,
            filter={"prof_name": user_input}
        )

        if not results:
            st.error(f"No data found for '{user_input}'. Check spelling and try again.")
        else:
            meta = results[0].metadata
            prof_name = meta.get("prof_name", user_input)
            has_ratings = meta.get("avg_rating") != "N/A"

            st.subheader(f"📊 {prof_name} ({meta.get('dept', 'UWindsor')})")

            if has_ratings:
                c1, c2, c3 = st.columns(3)
                with c1:
                    render_metric_card("⭐ Quality", f"{meta.get('avg_rating')}/5")
                with c2:
                    render_metric_card("🔄 Would Take Again", f"{meta.get('would_take_again')}")
                with c3:
                    render_metric_card("📈 Difficulty", f"{meta.get('avg_difficulty')}/5")
            else:
                st.info(f"Profile found: {prof_name}")

            if not has_ratings or "No detailed student reviews" in results[0].page_content:
                st.warning(f"No ratings or reviews available for {prof_name}.")
            else:
                query = query_input.strip()
                if not query:
                    query = f"What is the grading style, workload, and overall vibe for {prof_name}?"

                st.write(f"🧠 Analyzing reviews for {prof_name}...")
                start_time = time.time()

                try:
                    result = generate_summary(query, prof_name, vector_db, reranker, llm)
                    retrieved_reviews = retrieve_reviews(query, prof_name, vector_db, reranker)
                    end_time = time.time()

                    seen_texts = set()
                    unique_reviews = []

                    for review in retrieved_reviews:
                        text = review.get("text", "").strip()
                        if text and text not in seen_texts:
                            seen_texts.add(text)
                            unique_reviews.append(text)

                    tab1, tab2 = st.tabs(["Summary", "Sources"])

                    with tab1:
                        summary = result["summary"]
                        if isinstance(summary, list):
                            st.write(summary[0].get("text"))
                        else:
                            st.write(summary)

                    with tab2:
                        st.markdown(
                            '<div class="small-note">Sources below are the top retrieved review snippets used to generate the summary.</div>',
                            unsafe_allow_html=True
                        )
                        st.caption(f"{len(unique_reviews)} unique snippets shown from {len(retrieved_reviews)} retrieved.")

                        for i, text in enumerate(unique_reviews, start=1):
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <div style="font-weight:700; margin-bottom:10px;">Source {i}</div>
                                    <div style="line-height:1.6;">{text}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.success(f"Answer generated in {end_time - start_time:.2f} seconds")

                except Exception as e:
                    st.error(f"Error during analysis: {e}")