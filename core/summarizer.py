from core.retriever import retrieve_reviews
from core.prompt_templates import build_summary_prompt
from langchain_core.runnables import RunnableLambda

def generate_summary(query, prof_name, vector_db, reranker, llm):
    # 1. Get the reviews (Returns a list of DICTS with 'content' and 'metadata')
    reviews = retrieve_reviews(query, prof_name, vector_db, reranker)

    if not reviews:
        return {
            "summary": "No review data available for this professor.",
            "sources": []
        }

    # 2. Extract just the text content for the LLM to read
    # This prevents the 'string indices' error by ensuring we pass a list of strings
    review_texts = [r["content"] for r in reviews]

    # 3. Build the chain
    # We pass the strings directly to the prompt builder
    prompt_chain = RunnableLambda(
        lambda x: build_summary_prompt(x["prof_name"], x["review_texts"])
    )

    chain = prompt_chain | llm

    # 4. Invoke the local LLM
    response = chain.invoke({
        "prof_name": prof_name,
        "review_texts": review_texts
    })

    summary_text = response.content if hasattr(response, "content") else str(response)

    return {
        "summary": summary_text,
        "sources": [r["metadata"] for r in reviews]
    }