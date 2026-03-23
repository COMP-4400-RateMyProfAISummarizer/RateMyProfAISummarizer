import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

from core.summarizer import generate_summary

load_dotenv()

# 1. Setup the "Memory" (Pinecone)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# 2. Setup the "Reranker" (The quality filter)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 3. Setup the "AI" (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    google_api_key=os.getenv("CLOUD_API_KEY")
)

def main():
    prof_name = input("\n🔍 Enter Professor Name: ")
    query = f"What is the grading style and workload for {prof_name}?"

    print(f"🧠 Analyzing reviews for {prof_name}...")
    
    result = generate_summary(query, prof_name, vector_db, reranker, llm)

    if result["sources"]:
        # Grab metadata from the first retrieved source
        meta = result["sources"][0] 
        print(f"\n📊 --- {prof_name} ({meta.get('dept', 'UWindsor')}) ---")
        print(f"⭐ Quality: {meta.get('avg_rating', 'N/A')}/5")
        print(f"🔄 Would Take Again: {meta.get('would_take_again', 'N/A')}")
        print(f"📈 Difficulty: {meta.get('avg_difficulty', 'N/A')}/5")
        print("-" * 40)
    # -----------------------------

    print(f"\n🎓 SUMMARY FOR {prof_name}")
    print("=" * 40)
    
    # Clean output handling
    summary = result["summary"]
    if isinstance(summary, list):
        print(summary[0].get('text', 'No text found.'))
    else:
        print(summary)
        
    print("\n📚 SOURCES USED:", len(result["sources"]))

if __name__ == "__main__":
    main()