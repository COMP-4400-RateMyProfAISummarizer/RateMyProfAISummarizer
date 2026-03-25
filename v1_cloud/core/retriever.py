from dotenv import load_dotenv

# Look for the .env file and load the variables
load_dotenv()

def retrieve_reviews(query, prof_name, vector_db, reranker):
    # Metadata filtering
    metadata_filter = {"prof_name": prof_name}

    # Retrieve top candidate reviews
    results = vector_db.similarity_search(query, k=20, filter=metadata_filter)

    # Rerank results
    reranked = rerank(query, results, reranker)

    # Keep top 5
    top5_results = reranked[:5]

    return top5_results

def rerank(query, retrieved, reranker):
    if not retrieved:
        return []
    
    # Extract review text from each retrieved result
    pairs = [(query, review.page_content) for review in retrieved]

    # Get reranker scores
    rr_scores = reranker.predict(pairs)

    reranked = []

    for review, score in zip(retrieved, rr_scores):
        reranked.append({
            "text": review.page_content,
            "metadata": review.metadata,
            "rerank_score": float(score)
        })
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    return reranked
