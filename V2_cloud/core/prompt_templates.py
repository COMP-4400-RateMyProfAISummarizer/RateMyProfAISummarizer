def build_summary_prompt(professor_name, review_list, user_query):
    if not review_list:
        return "No review data available for this professor."
    

    combined_reviews = "\n\n---\n\n".join([item["text"] for item in review_list])
    is_default_query = user_query.startswith("Summarize reviews for")

    if is_default_query:
        final_prompt = f"""
You are a UWindsor Academic Assistant.

Use ONLY the reviews below.
Do NOT invent details.
Be specific and detailed.

Return your answer in exactly this format:

### ⚖️ QUICK COMPARISON
**PROS:**
- bullet
- bullet
- bullet

**CONS:**
- bullet
- bullet

### 📝 DETAILED ANALYSIS
1. **Grading Style:** ...
2. **Workload:** ...
3. **Overall Vibe:** ...

### 🏁 FINAL VERDICT
...

Rules:
- Write 3 to 5 pros if enough evidence exists.
- Write 2 to 5 cons if enough evidence exists.
- If no strong negatives are supported, write: "No major complaints were consistently mentioned in the provided reviews."
- If a section lacks enough information, write: "Information not available."
- Do NOT include the text "REVIEWS FOR" in the final answer.

REVIEWS FOR {professor_name}:
{combined_reviews}
"""
    else:
        final_prompt = f"""
You are a UWindsor Academic Assistant.

Use ONLY the reviews below.
Do NOT invent details.
Be specific and detailed.

Return your answer in exactly this format:

### 🎯 DIRECT ANSWER
Answer this question in 2-3 specific sentences: {user_query}

|||

### ⚖️ QUICK COMPARISON
**PROS:**
- bullet
- bullet
- bullet

**CONS:**
- bullet
- bullet

### 📝 DETAILED ANALYSIS
1. **Grading Style:** ...
2. **Workload:** ...
3. **Overall Vibe:** ...

### 🏁 FINAL VERDICT
...

Rules:
- The line ||| must appear exactly once, on its own line, after the direct answer.
- Answer the question directly first.
- Write 3 to 5 pros if enough evidence exists.
- Write 2 to 5 cons if enough evidence exists.
- If no strong negatives are supported, write: "No major complaints were consistently mentioned in the provided reviews."
- If the reviews do not fully answer the question, say so clearly.
- If a section lacks enough information, write: "Information not available."
- Do NOT include the text "REVIEWS FOR" in the final answer.

REVIEWS FOR {professor_name}:
{combined_reviews}
"""

    return final_prompt