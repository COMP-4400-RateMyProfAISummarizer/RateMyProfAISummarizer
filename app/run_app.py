import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

# Import your teammate's logic
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
# This model is small and runs for free on your Mac
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

    print("\n" + "="*50)
    print(f"🎓 SUMMARY FOR {prof_name}")
    print("="*50)
    
    # Check if summary is a list (multimodal format) or a string
    summary = result["summary"]
    if isinstance(summary, list):
        print(summary[0].get('text', 'No text found.'))
    else:
        print(summary)
        
    print("\n📚 SOURCES USED:", len(result["sources"]))

if __name__ == "__main__":
    main()