from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# Look for the .env file and load the variables
load_dotenv()

def retrieve_reviews(query, prof_name, vector_db, reranker):
    # Metadata filtering
    metadata_filter = {"prof_name": prof_name}

    # Retrieve top candidate reviews
    results = vector_db.similarity_search(query, k=20, filter=metadata_filter)

    # Deduplication pass
    unique_results = []
    seen_texts = set()

    for result in results:
        text = result.page_content.strip()

        if text not in seen_texts:
            seen_texts.add(text)
            unique_results.append(result)
    
    results = unique_results

    # Hybrid BM25 keyword reordering
    results = hybrid_rerank(query, results)

    # Cross-encoder reranking
    reranked = rerank(query, results, reranker)

    # Keep top 5
    top5_results = reranked[:5]

    return top5_results

def hybrid_rerank(query, results):
    if not results:
        return []
    
    tokenized_docs = [doc.page_content.lower().split() for doc in results]
    bm25 = BM25Okapi(tokenized_docs)

    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    combined = []

    for doc, bm25_score in zip(results, bm25_scores):
        combined.append({
            "doc": doc,
            "bm25_score": float(bm25_score)
        })
    combined.sort(key=lambda x: x["bm25_score"], reverse=True)

    return [item["doc"] for item in combined]

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
