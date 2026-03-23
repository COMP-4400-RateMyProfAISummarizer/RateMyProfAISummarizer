def build_summary_prompt(prof_name, reviews):
    if not reviews:
        return "No review data available for this professor."
    
    # Joining the top 5 reranked reviews
    context = "\n\n---\n\n".join([r["text"] for r in reviews])

    prompt = f"""
You are a UWindsor Academic Assistant.

Use ONLY the provided student reviews below to answer. Do not use outside knowledge.

Student reviews for {prof_name}:
{context}

---
### ⚖️ QUICK COMPARISON
**PROS:**
* (Bullet points of the most positive aspects mentioned)

**CONS:**
* (Bullet points of the most common complaints or challenges)

---
### 📝 DETAILED ANALYSIS
1. **Grading Style:** (Summarize how they grade and provide feedback)
2. **Workload:** (Summarize the volume of assignments, readings, and exams)
3. **Overall Vibe:** (Summarize the classroom atmosphere and communication style)

### 🏁 FINAL VERDICT
(A one-sentence recommendation for a student considering this professor)

Each summary should be based only on the provided reviews.

If any category is not mentioned in the reviews, write "Information not available".
"""
    return prompt
