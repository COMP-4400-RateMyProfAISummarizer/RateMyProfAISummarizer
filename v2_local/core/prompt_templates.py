def build_summary_prompt(prof_name, reviews, query):
    if not reviews:
        return "No review data available for this professor."
    
    context = "\n\n---\n\n".join([r["text"] for r in reviews])
    is_generic = query.startswith("Summarize reviews for")

    target_section = ""
    if not is_generic:
        target_section = f"""
### 🎯 DIRECT ANSWER
**Question:** {query}
(Answer the question in 2-3 sentences based on reviews.)

|||
"""

    prompt = f"""
You are a UWindsor Academic Assistant. 

{target_section}

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

---
REVIEWS FOR {prof_name}:
{context}

INSTRUCTIONS:
- If a question was asked, you MUST place '|||' on its own line after the Direct Answer.
- DO NOT include the "STUDENT REVIEWS" or "REVIEWS FOR..." text in your final output.
- Use ONLY the provided reviews.
- If a category is missing data, write "Information not available."
"""
    return prompt