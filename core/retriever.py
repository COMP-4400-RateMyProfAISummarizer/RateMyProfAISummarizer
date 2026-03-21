import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

def analyze_professor(prof_name):
    # 1. Initialize Embeddings (HuggingFace is free/local)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Connect to Pinecone
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"), 
        embedding=embeddings
    )

    # 3. Retrieval with Metadata Filtering
    # This ensures we ONLY pull reviews for the specific professor
    results = vector_store.similarity_search(
        f"Detailed teaching style of {prof_name}", 
        k=5, 
        filter={"prof_name": prof_name}
    )

    if not results:
        print(f"❌ No data found for '{prof_name}'. Check your spelling or Pinecone index.")
        return

    # 4. Extract Stats from Metadata (Step C requirement)
    meta = results[0].metadata
    print(f"\n📊 --- {prof_name} ({meta.get('dept', 'UWindsor')}) ---")
    print(f"⭐ Quality: {meta.get('avg_rating')}/5")
    print(f"🔄 Would Take Again: {meta.get('would_take_again')}")
    print(f"📈 Difficulty: {meta.get('avg_difficulty')}/5")
    print("-" * 40)

    # 5. Prepare context for the Free LLM
    context_text = "\n".join([res.page_content for res in results])

    # 6. Setup the Analysis Prompt
    template = """
    You are an AI Student Advisor at the University of Windsor. 
    Using the provided professor reviews, create a balanced Pros/Cons analysis.
    
    PROFESSOR: {professor}
    STUDENT REVIEWS:
    {context}
    
    RESPONSE FORMAT:
    PROS:
    * (Bullet points)
    
    CONS:
    * (Bullet points)
    
    FINAL VERDICT:
    (A one-sentence recommendation for a student considering this prof)
    """
    
    prompt = PromptTemplate(template=template, input_variables=["professor", "context"])
    
    # 7. Use Gemini (Free Tier)
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0, google_api_key=os.getenv("CLOUD_API_KEY"))
    
    # 8. Run the Chain
    formatted_prompt = prompt.format(professor=prof_name, context=context_text)
    response = llm.invoke(formatted_prompt)
    
    print(response.content)

if __name__ == "__main__":
    name = input("\n🔍 Enter Professor Name (e.g., Ziad Kobti): ")
    analyze_professor(name)