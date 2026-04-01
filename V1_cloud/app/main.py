import os
import sys
import time
import streamlit as ui
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

# Allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.summarizer import generate_summary

# Load environment variables
load_dotenv()


# -------------------------------
# Streamlit Page Setup
# -------------------------------
ui.set_page_config(
    page_title="UWindsor RateMyProf Agent",
    page_icon="🎓",
    layout="wide"
)


# -------------------------------
# Styling
# -------------------------------
ui.markdown("""
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


# -------------------------------
# Initialize models + database
# -------------------------------
@ui.cache_resource
def setup_pipeline():

    # Embedding model
    embed_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Vector database connection
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embed_model,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )

    # Reranker model
    ranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Language model (Gemini)
    language_model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=os.getenv("CLOUD_API_KEY")
    )

    return vector_store, ranking_model, language_model


# Load components
vector_store, ranking_model, language_model = setup_pipeline()


# -------------------------------
# UI Inputs
# -------------------------------
ui.title("🎓 UWindsor RateMyProf Agent")

professor_input = ui.text_input(
    "Professor name",
    placeholder="e.g. Ziad Kobti"
)

question_input = ui.text_area(
    "Your question (optional)"
)


# -------------------------------
# Generate Button Logic
# -------------------------------
if ui.button("Generate Summary"):

    if not professor_input.strip():
        ui.warning("Please enter a professor name.")
    
    else:
        formatted_name = professor_input.strip().title()

        # Fetch professor metadata
        search_results = vector_store.similarity_search(
            "placeholder",
            k=1,
            filter={"prof_name": formatted_name}
        )

        if not search_results:
            ui.error("No data found.")
        
        else:
            professor_data = search_results[0].metadata

            ui.subheader(f"📊 {professor_data.get('prof_name')} ({professor_data.get('dept')})")
            
            col_a, col_b, col_c = ui.columns(3)

            with col_a:
                ui.markdown(
                    f"<div class='metric-card'>⭐ Quality: {professor_data.get('avg_rating')}/5</div>",
                    unsafe_allow_html=True
                )

            with col_b:
                ui.markdown(
                    f"<div class='metric-card'>🔄 Retake: {professor_data.get('would_take_again')}</div>",
                    unsafe_allow_html=True
                )

            with col_c:
                ui.markdown(
                    f"<div class='metric-card'>📈 Difficulty: {professor_data.get('avg_difficulty')}/5</div>",
                    unsafe_allow_html=True
                )

            # Generate summary
            with ui.spinner("🤖 Generating AI Summary..."):

                final_query = (
                    question_input.strip()
                    if question_input.strip()
                    else f"Provide a review summary for {formatted_name}"
                )

                summary_output = generate_summary(
                    final_query,
                    formatted_name,
                    vector_store,
                    ranking_model,
                    language_model
                )

                ui.markdown("### 📝 Summary")
                ui.write(summary_output["summary"])

                ui.success("Generated using Google Gemini")