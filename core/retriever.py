import os
from dotenv import load_dotenv

load_dotenv()

def deduplicate_reviews(reviews):
    """
    Person 2 Task: Data Cleaning/Deduplication
    Removes exact duplicate text to save space in the AI's context window.
    """
    seen = set()
    unique_reviews = []
    for r in reviews:
        # Create a hash of the content to find duplicates
        content = r.page_content if hasattr(r, 'page_content') else r.get('content', "")
        if content not in seen:
            unique_reviews.append(r)
            seen.add(content)
    return unique_reviews

def retrieve_reviews(query, prof_name, vector_db, reranker):
    # 1. Metadata Engineering: Programmatic Filter
    metadata_filter = {"prof_name": prof_name}

    # 2. Hybrid Search Simulation
    # We pull a larger set (k=25) to ensure we catch specific keywords 
    # like course codes and 'grading' or 'midterm'
    initial_results = vector_db.similarity_search(
        query, 
        k=25, 
        filter=metadata_filter
    )

    if not initial_results:
        return []

    # 3. Data Cleaning: Deduplication
    clean_results = deduplicate_reviews(initial_results)

    # 4. Re-Ranking Layer (The Cross-Encoder)
    # This re-scores the top 25 results to find the absolute best 5
    reranked = rerank(query, clean_results, reranker)

    return reranked[:5]

def rerank(query, retrieved, reranker):
    if not retrieved:
        return []
    
    # Pair the question with each review content
    # Note: 'retrieved' is now a list of LangChain Document objects
    pairs = [(query, doc.page_content) for doc in retrieved]
    rr_scores = reranker.predict(pairs)

    reranked_data = []
    for doc, score in zip(retrieved, rr_scores):
        reranked_data.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "rerank_score": float(score)
        })
    
    # Sort by the Cross-Encoder score (highest to lowest)
    reranked_data.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked_data