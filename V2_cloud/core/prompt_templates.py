def create_summary_prompt(professor_name, review_list, user_query):

    # If no reviews exist, return fallback message
    if not review_list:
        return "No review data available for this professor."
    
    # Combine all review texts into one formatted context block
    combined_reviews = "\n\n---\n\n".join([item["text"] for item in review_list])

    # Check if query is generic (default system query)
    is_default_query = user_query.startswith("Summarize reviews for")

    # Optional section: direct answer (only if user asked a custom question)
    direct_answer_block = ""
    if not is_default_query:
        direct_answer_block = f"""
### 🎯 DIRECT ANSWER
**Question:** {user_query}
(Answer the question in 2-3 sentences based on reviews.)

|||
"""

    # Final prompt template
    final_prompt = f"""
You are a UWindsor Academic Assistant. 

{direct_answer_block}

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
REVIEWS FOR {professor_name}:
{combined_reviews}

INSTRUCTIONS:
- If a question was asked, you MUST place '|||' on its own line after the Direct Answer.
- DO NOT include the "STUDENT REVIEWS" or "REVIEWS FOR..." text in your final output.
- Use ONLY the provided reviews.
- If a category is missing data, write "Information not available."
"""

    return final_prompt