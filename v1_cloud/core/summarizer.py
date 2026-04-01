from core.retriever import retrieve_reviews
from core.prompt_templates import build_summary_prompt
from langchain_core.runnables import RunnableLambda

def generate_summary(query, prof_name, vector_db, reranker, llm):
    reviews = retrieve_reviews(query, prof_name, vector_db, reranker)

    if not reviews:
        return {
            "summary": "No review data available for this professor.",
            "sources": []
        }
    prompt_chain = RunnableLambda(
        lambda x: build_summary_prompt(x["prof_name"], x["reviews"])
    )

    chain = prompt_chain | llm

    response = chain.invoke({
        "prof_name": prof_name,
        "reviews": reviews
    })

    summary_text = response.content if hasattr(response, "content") else response

    return {
        "summary": summary_text,
        "sources": [r["metadata"] for r in reviews]
    }
