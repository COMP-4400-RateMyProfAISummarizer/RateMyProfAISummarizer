import os
import sys
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

# Allow access to core folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.summarizer import generate_summary
from core.retriever import retrieve_reviews

load_dotenv()

# -----------------------------
# 1. GOLDEN DATASET
# -----------------------------
# You can add more professors/questions later.
GOLDEN_DATASET = [
    {
        "prof_name": "Ziad Kobti",
        "query": "How is the grading style, workload, and overall vibe?",
        "expected_keywords": ["grading", "workload", "midterm", "feedback"]
    },
    {
        "prof_name": "Pooya Zadeh",
        "query": "How is the grading style, workload, and overall vibe?",
        "expected_keywords": ["lecture", "clear", "assignment", "workload"]
    },
    {
        "prof_name": "Jianguo Lu",
        "query": "How is the grading style, workload, and overall vibe?",
        "expected_keywords": ["grading", "assignment", "exam", "teaching"]
    }
]


# -----------------------------
# 2. BACKEND INITIALIZATION
# -----------------------------
def init_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=os.getenv("CLOUD_API_KEY")
    )

    return vector_db, reranker, llm


# -----------------------------
# 3. SIMPLE SCORING HELPERS
# -----------------------------
def normalize_summary_text(summary):
    if isinstance(summary, list):
        if summary and isinstance(summary[0], dict):
            return summary[0].get("text", "")
        return str(summary)
    return str(summary)


def compute_keyword_score(summary_text, expected_keywords):
    matched_keywords = [
        kw for kw in expected_keywords
        if kw.lower() in summary_text.lower()
    ]

    score = 0.0
    if expected_keywords:
        score = len(matched_keywords) / len(expected_keywords)

    return round(score, 2), matched_keywords


def deduplicate_reviews(retrieved_reviews):
    seen = set()
    unique_reviews = []

    for review in retrieved_reviews:
        text = review.get("text", "").strip()
        if text and text not in seen:
            seen.add(text)
            unique_reviews.append(text)

    return unique_reviews


# -----------------------------
# 4. MAIN EVALUATION
# -----------------------------
def evaluate_rag(vector_db, reranker, llm):
    results = []

    print("\n📊 Running RAG Evaluation...\n")

    for i, test in enumerate(GOLDEN_DATASET, start=1):
        prof_name = test["prof_name"]
        query = test["query"]
        expected_keywords = test["expected_keywords"]

        print("=" * 60)
        print(f"Test Case {i}")
        print(f"Professor: {prof_name}")
        print(f"Query: {query}")

        start_time = time.time()

        try:
            output = generate_summary(query, prof_name, vector_db, reranker, llm)
            retrieved_reviews = retrieve_reviews(query, prof_name, vector_db, reranker)

            end_time = time.time()
            latency = round(end_time - start_time, 2)

            summary_text = normalize_summary_text(output["summary"])
            unique_reviews = deduplicate_reviews(retrieved_reviews)

            score, matched_keywords = compute_keyword_score(summary_text, expected_keywords)

            result = {
                "prof_name": prof_name,
                "query": query,
                "latency": latency,
                "score": score,
                "matched_keywords": matched_keywords,
                "expected_keywords": expected_keywords,
                "retrieved_count": len(retrieved_reviews),
                "unique_retrieved_count": len(unique_reviews),
                "summary_text": summary_text
            }

            results.append(result)

            print(f"✅ Score: {score}")
            print(f"⏱ Inference Time: {latency}s")
            print(f"🔑 Matched Keywords: {matched_keywords}")
            print(f"📚 Retrieved Snippets: {len(retrieved_reviews)}")
            print(f"📚 Unique Snippets: {len(unique_reviews)}")
            print("\n📝 Summary:")
            print(summary_text[:800] + ("..." if len(summary_text) > 800 else ""))

        except Exception as e:
            print(f"❌ Error while evaluating {prof_name}: {e}")

    return results


# -----------------------------
# 5. FINAL REPORT
# -----------------------------
def print_report(results):
    if not results:
        print("\nNo evaluation results to report.")
        return

    print("\n" + "=" * 60)
    print("📈 FINAL EVALUATION REPORT")
    print("=" * 60)

    for r in results:
        print(f"\nProfessor: {r['prof_name']}")
        print(f"Query: {r['query']}")
        print(f"Score: {r['score']}")
        print(f"Inference Time: {r['latency']}s")
        print(f"Matched Keywords: {r['matched_keywords']}")
        print(f"Retrieved Snippets: {r['retrieved_count']}")
        print(f"Unique Snippets: {r['unique_retrieved_count']}")
        print("-" * 40)

    avg_score = round(sum(r["score"] for r in results) / len(results), 2)
    avg_latency = round(sum(r["latency"] for r in results) / len(results), 2)

    print("\n🎯 OVERALL PERFORMANCE")
    print(f"Average Score: {avg_score}")
    print(f"Average Inference Time: {avg_latency}s")
    print("=" * 60)


# -----------------------------
# 6. RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    vector_db, reranker, llm = init_components()
    results = evaluate_rag(vector_db, reranker, llm)
    print_report(results)