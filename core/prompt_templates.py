def build_summary_prompt(prof_name, reviews):
    if not reviews:
        return "No review data available for this professor."
    
    context = "\n\n---\n\n".join([r["text"] for r in reviews])

    prompt = f"""
You are a UWindsor Academic Assistant.

Use ONLY the provided student reviews below to answer. Do not use outside knowledge.

Student reviews for {prof_name}:
{context}

Summarize the reviews into the following categories:

1. Grading Style:
2. Workload:
3. Overall Vibe:

Each summary should be based only on the provided reviews.

If any category is not mentioned in the reviews, write "Information not available".
"""
    return prompt
