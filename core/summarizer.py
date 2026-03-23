from core.retriever import retrieve_reviews
from core.prompt_templates import build_summary_prompt

def generate_summary(query, prof_name, vector_db, reranker, llm):
    reviews = retrieve_reviews(query, prof_name, vector_db, reranker)

    if not reviews:
        return {
            "summary": "No review data available for this professor.",
            "sources": []
        }
    prompt = build_summary_prompt(prof_name, reviews)

    response = llm.invoke(prompt)

    summary_text = response.content if hasattr(response, "content") else response

    return {
        "summary": summary_text,
        "sources": [r["metadata"] for r in reviews]
    }
