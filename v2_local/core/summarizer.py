from core.retriever import retrieve_reviews
from core.prompt_templates import build_summary_prompt
from langchain_core.runnables import RunnableLambda


def generate_summary(query, prof_name, vector_db, reranker, llm):
    # Step 1: retrieve reviews
    reviews = retrieve_reviews(query, prof_name, vector_db, reranker)

    if not reviews:
        return {
            "summary": "No review data available for this professor.",
            "sources": []
        }

    # Step 2: build prompt
    prompt_chain = RunnableLambda(
        lambda x: build_summary_prompt(
            x["prof_name"],
            x["reviews"],
            x["query"]
        )
    )

    # Step 3: connect to LLM
    chain = prompt_chain | llm

    # Step 4: invoke model
    response = chain.invoke({
        "prof_name": prof_name,
        "reviews": reviews,
        "query": query
    })

    # Step 5: extract text safely
    summary_text = response.content if hasattr(response, "content") else str(response)

    return {
        "summary": summary_text,
        "sources": [r["metadata"] for r in reviews]
    }