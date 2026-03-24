def build_summary_prompt(prof_name, review_texts):
    if not review_texts:
        return "No review data available for this professor."
    
    # review_texts is now a list of strings [ "review 1...", "review 2..." ]
    context = "\n\n---\n\n".join(review_texts)

    prompt = f"""
You are a UWindsor Academic Assistant.
Use ONLY the provided student reviews below to answer. Do not use outside knowledge.

Student reviews for {prof_name}:
{context}

---
### ⚖️ QUICK COMPARISON
**PROS:**
* (Bullet points)
**CONS:**
* (Bullet points)

---
### 📝 DETAILED ANALYSIS
1. **Grading Style:** (Summarize)
2. **Workload:** (Summarize)
3. **Overall Vibe:** (Summarize)

### 🏁 FINAL VERDICT
(One-sentence recommendation)
"""
    return prompt