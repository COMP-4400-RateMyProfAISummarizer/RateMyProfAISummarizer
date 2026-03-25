def build_summary_prompt(prof_name, reviews, query):
    if not reviews:
        return "No review data available for this professor."
    
    context = "\n\n---\n\n".join([r["text"] for r in reviews])

    # Check if the query is just the generic fallback
    is_generic = query.startswith("Summarize reviews for")

    # The Direct Answer is now the first section of the prompt
    target_section = ""
    if not is_generic:
        target_section = f"""
### 🎯 DIRECT ANSWER
**Question:** {query}

(Provide a clear, 2-3 sentence answer here based on the reviews. If the reviews don't mention this topic, state that information is limited.)

---
"""

    prompt = f"""
You are a UWindsor Academic Assistant. 

{target_section}

### ⚖️ QUICK COMPARISON
**PROS:**
* (Bullet points of the most positive aspects mentioned)

**CONS:**
* (Bullet points of the most common complaints or challenges)

---
### 📝 GENERAL ANALYSIS
1. **Grading Style:** (Summarize how they grade and provide feedback)
2. **Workload:** (Summarize the volume of assignments, readings, and exams)
3. **Overall Vibe:** (Summarize the classroom atmosphere and communication style)

### 🏁 FINAL VERDICT
(A one-sentence recommendation for a student considering this professor)

---
STUDENT REVIEWS FOR {prof_name}:
{context}

INSTRUCTIONS:
- Put the DIRECT ANSWER at the very top if a specific question was asked.
- Use ONLY the provided reviews.
- If a category is missing data, write "Information not available."
"""
    return prompt