import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

# Ensure 'core' is visible if running from the app folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.summarizer import generate_summary

load_dotenv()

# 1. Setup the "Memory" (Pinecone)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# 2. Setup the "Reranker"
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 3. Setup the "AI" (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("CLOUD_API_KEY")
)

def main():
    print("--- UWindsor RateMyProf RAG System ---")

    while True:
        user_input = input("\n🔍 Enter Professor Name: ").strip()
        
        # Validation 1: Empty or Null input -> RE-PROMPT
        if not user_input:
            print("⚠️ Error: Please enter a name.")
            continue

        # Exit command
        if user_input.lower() == 'exit':
            break

        # Validation 2: Check if professor exists in Vector DB
        results = vector_db.similarity_search("placeholder", k=1, filter={"prof_name": user_input})
        
        # Prof does not exist -> RE-PROMPT
        if not results:
            print(f"❌ No data found for '{user_input}'. Check spelling and try again.")
            continue

        # --- PROFESSOR FOUND: EXITING AFTER THIS BLOCK ---
        meta = results[0].metadata
        prof_name = meta.get('prof_name', user_input)
        has_ratings = meta.get('avg_rating') != "N/A"

        # Display Header
        if has_ratings:
            print(f"\n📊 --- {prof_name} ({meta.get('dept', 'UWindsor')}) ---")
            print(f"⭐ Quality: {meta.get('avg_rating')}/5")
            print(f"🔄 Would Take Again: {meta.get('would_take_again')}")
            print(f"📈 Difficulty: {meta.get('avg_difficulty')}/5")
            print("-" * 40)
        else:
            print(f"\nℹ️  Profile Found: {prof_name} ({meta.get('dept', 'UWindsor')})")

        # Case A: Prof exists but no ratings/reviews -> EXIT
        if not has_ratings or "No detailed student reviews" in results[0].page_content:
            print(f"\n❗️ NOTE: There are currently no ratings or reviews for {prof_name} on RateMyProfessors.")
            print("Check back later or consult the department syllabus for more info.")
            return # Exit program

        # Case B: Prof exists and has reviews -> ANALYZE THEN EXIT
        print(f"🧠 Analyzing reviews for {prof_name}...")
        query = f"What is the grading style, workload, and overall vibe for {prof_name}?"
        
        try:
            result = generate_summary(query, prof_name, vector_db, reranker, llm)
            print(f"\n🎓 AI SUMMARY FOR {prof_name}")
            print("=" * 40)
            
            summary = result["summary"]
            print(summary[0].get('text') if isinstance(summary, list) else summary)
            print("\n📚 SOURCES USED:", len(result["sources"]))
            
        except Exception as e:
            print(f"❌ An error occurred during analysis: {e}")
        
        # Exit program after showing results
        return 

if __name__ == "__main__":
    main()