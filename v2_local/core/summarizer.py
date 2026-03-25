from core.retriever import retrieve_reviews
from core.prompt_templates import build_summary_prompt  # Ensure this filename matches
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def generate_summary(query, prof_name, vector_db, reranker, llm):
    # 1. Fetch the reviews
    reviews = retrieve_reviews(query, prof_name, vector_db, reranker)
    
    # 2. Define the Chain
    # RunnablePassthrough allows us to pass the dictionary into the lambda
    chain = (
        RunnablePassthrough()
        | (lambda x: build_summary_prompt(x["prof_name"], x["reviews"], x["query"]))
        | llm
        | StrOutputParser()
    )

    # 3. Run the Chain
    response = chain.invoke({
        "prof_name": prof_name,
        "reviews": reviews,
        "query": query
    })

    return {
        "summary": response,
        "sources": reviews
    }