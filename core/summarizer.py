from core.retriever import retrieve_reviews
from core.prompt_templates import build_summary_prompt
from langchain_core.messages import SystemMessage, HumanMessage

def generate_summary(query, prof_name, vector_db, reranker, llm):
    """
    Agentic Summarizer: 
    1. Expands the user query into multiple search perspectives.
    2. Iteratively gathers context for each perspective.
    3. Applies System Logic for a controlled academic response.
    """

    # --- 1. Query Expansion (Person 3 Task) ---
    # We ask the LLM to identify 3 specific areas to search for based on the user's query.
    expansion_instruction = (
        f"The user wants to know: '{query}' about {prof_name}. "
        "List 3 short search queries to find info on: 1. Grading/Exams, 2. Workload, 3. Personality."
    )
    
    print(f"🧠 Agent is expanding query...")
    expansion_response = llm.invoke(expansion_instruction)
    # Split by lines and clean up to get 3 search strings
    expanded_queries = [q.strip() for q in expansion_response.content.split('\n') if q.strip()][:3]

    all_reviews = []
    
    # --- 2. Iterative Reasoning Loop (Person 3 Task) ---
    # The agent "loops" through the expanded queries to gather a multi-faceted context.
    for sub_query in expanded_queries:
        print(f"🔄 Agent searching for: {sub_query}")
        sub_results = retrieve_reviews(sub_query, prof_name, vector_db, reranker)
        all_reviews.extend(sub_results)

    if not all_reviews:
        return {
            "summary": "The Agent found no specific review data. Suggestion: Check the official University of Windsor Syllabus or UWinsite.",
            "sources": []
        }

    # Deduplicate based on content to keep the context window efficient
    unique_texts = list(set([r["content"] for r in all_reviews]))
    unique_metadata = [r["metadata"] for r in all_reviews[:5]] # Capture top 5 metadata samples

    # --- 3. Prompt Engineering & Control (Person 3 Task) ---
    # We use a SystemMessage to enforce "If/Then" logic and boundaries.
    system_message = SystemMessage(content=f"""
        You are a UWindsor Academic Assistant.
        IF the reviews suggest the professor is unresponsive, advise the student to visit Office Hours.
        IF the reviews are contradictory, present both perspectives clearly.
        Answer using ONLY the provided reviews. Do not use external knowledge.
    """)

    # --- 4. Final Reasoning & Synthesis ---
    # We use the template you already built to keep the formatting consistent
    final_prompt_text = build_summary_prompt(prof_name, unique_texts)
    
    print(f"📝 Synthesizing final report...")
    response = llm.invoke([system_message, HumanMessage(content=final_prompt_text)])
    
    summary_text = response.content if hasattr(response, "content") else str(response)

    return {
        "summary": summary_text,
        "sources": unique_metadata
    }