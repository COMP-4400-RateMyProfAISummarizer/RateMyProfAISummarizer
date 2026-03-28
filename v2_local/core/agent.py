import json
from difflib import get_close_matches

from langchain.tools import tool
from langchain.agents import create_agent

from v2_local.core.retriever import retrieve_reviews

def expand_query(user_question: str):
    q = user_question.strip()

    if not q:
        return [
            "overall teaching quality",
            "grading style",
            "workload and exams"
        ]
    
    return [
        q,
        f"{q} grading style",
        f"{q} exams and assignments",
        f"{q} workload and difficulty"
    ]

def find_best_prof_name(user_input: str, vector_db):
    """
    Resolve partial professor names like 'Ziad' implying 'Ziad Kobti'
    """
    if not user_input:
        return None
    
    cleaned_input = user_input.strip().title()
    
    # Try exact metadata match first
    exact = vector_db.similarity_search(
        "placeholder",
        k=1,
        filter={"prof_name": cleaned_input}
    )
    if exact:
        return cleaned_input
    
    # Pull a wider sample and collect professor names
    broad_results = vector_db.similarity_search(cleaned_input, k=200)
    
    prof_names = []
    for doc in broad_results:
        name = doc.metadata.get("prof_name")
        if name and name not in prof_names:
            prof_names.append(name)
    
    # Partial substring match
    lowered = cleaned_input.lower()
    for name in prof_names:
        if lowered in name.lower():
            return name
        
    # Fuzzy closest match
    matches = get_close_matches(cleaned_input, prof_names, n=1, cutoff=0.50)
    if matches:
        return matches[0]
            
    return None

def build_search_tool(vector_db, reranker):
    @tool
    def search_uwindsor_reviews(prof_name: str, question: str) -> str:
        """Search UWindsor professor reviews for a professor and question."""

        if not prof_name:
            return "No professor name was provided."
        
        resolved_name = find_best_prof_name(prof_name, vector_db)
        if not resolved_name:
            return f"No professor match found for '{prof_name}'."
        
        queries = expand_query(question)

        all_results = []
        seen = set()

        for q in queries:
            results = retrieve_reviews(q, resolved_name, vector_db, reranker)
            for item in results:
                text = item.get("text", "").strip()
                if text and text not in seen:
                    seen.add(text)
                    all_results.append(item)

        if not all_results:
            return f"No reviews found for {resolved_name}."
        
        snippets = [item["text"] for item in all_results[:8]]

        return json.dumps({
            "resolved_prof_name": resolved_name,
            "expanded_queries": queries,
            "review_snippets": snippets,
        }, indent=2)
    
    return search_uwindsor_reviews

def get_system_prompt():
    return """
You are a UWindsor RateMyProf assistant.

Rules:
- Do not answer from memory.
- If the user asks about a professor, call search_uwindsor_reviews first.
- If the professor name is partial, unclear, or likely incomplete, use the tool anyway and rely on the resolved name.
- Base your answer only on returned review snippets.
- Separate clearly supported positives and negatives.
- If no reviews are found, say so and suggest checking the course syllabus or official university resources.
- Do not invent details not supported by the tool output.
"""

def build_agent(llm, vector_db, reranker):
    tool_obj = build_search_tool(vector_db, reranker)

    agent = create_agent(
        model=llm,
        tools=[tool_obj],
        system_prompt=get_system_prompt()
    )
    return agent
