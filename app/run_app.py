import os
import sys
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
import time

# Import your Task 1 logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.embeddings_manager import get_embeddings
from core.summarizer import generate_summary

load_dotenv()

# 1. Setup Local Embeddings & Vector DB
# Using the function you created in core/embeddings_manager.py
embeddings = get_embeddings()

vector_db = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# 2. Setup the "Reranker"
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 3. Initialize Local LLM (Llama 3.1)
print("🦙 Connecting to local Llama instance...")
llm = ChatOllama(
    model="llama3.1",
    temperature=0, 
)

def main():
    print("\n--- UWindsor RateMyProf RAG System (LOCAL) ---")

    while True:
        raw_input = input("\n🔍 Enter Professor Name: ").strip()
        
        if not raw_input:
            print("⚠️ Error: Please enter a name.")
            continue

        if raw_input.lower() == 'exit':
            break

        # Case-insensitivity fix
        user_input = raw_input.title()

        # Check for existence in Pinecone
        results = vector_db.similarity_search("placeholder", k=1, filter={"prof_name": user_input})
        
        if not results:
            print(f"🔍 '{user_input}' not found. Agent is attempting Fuzzy Name Match...")
            # Agentic Loop: Try a broader search to find the correct name
            broad_search = vector_db.similarity_search(user_input, k=3)
            
            if broad_search:
                suggested = broad_search[0].metadata.get('prof_name')
                print(f"🔄 Loop found a match: '{suggested}'")
                
                user_input = suggested
                results = vector_db.similarity_search("placeholder", k=1, filter={"prof_name": user_input})
            else:
                print("❌ No matching professors found.")
                continue

        meta = results[0].metadata
        prof_name = meta.get('prof_name', user_input)
        has_ratings = meta.get('avg_rating') != "N/A"

        # Display Stats
        if has_ratings:
            print(f"\n📊 --- {prof_name} ({meta.get('dept', 'UWindsor')}) ---")
            print(f"⭐ Quality: {meta.get('avg_rating')}/5")
            print(f"🔄 Would Take Again: {meta.get('would_take_again')}")
            print(f"📈 Difficulty: {meta.get('avg_difficulty')}/5")
            print("-" * 40)
        else:
            print(f"\nℹ️  Profile Found: {prof_name} ({meta.get('dept', 'UWindsor')})")

        # Short-circuit for empty profiles
        if not has_ratings or "No detailed student reviews" in results[0].page_content:
            print(f"\n❗️ NOTE: There are currently no ratings or reviews for {prof_name} on RateMyProfessors.")
            return 

        # Proceed to Local AI Analysis
        print(f"🧠 Local Llama is analyzing reviews for {prof_name}...")
        query = f"What is the grading style, workload, and overall vibe for {prof_name}?"
        
        try:
            start_time = time.time() # Start Benchmark
            
            result = generate_summary(query, prof_name, vector_db, reranker, llm)
            
            end_time = time.time() # End Benchmark
            inference_duration = end_time - start_time

            print(f"\n🎓 AI SUMMARY (LOCAL MODEL)")
            print("=" * 40)
            print(result["summary"])
            print("-" * 40)
            print(f"⏱️  Inference Time: {inference_duration:.2f} seconds") # <--- Engineering Metric
            print(f"📚 Sources Used: {len(result['sources'])}")
            
        except Exception as e:
            print(f"❌ An error occurred during analysis: {e}")
        
        return 

if __name__ == "__main__":
    main()