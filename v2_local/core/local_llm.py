import time
from langchain_ollama import ChatOllama

def get_local_llm():
    return ChatOllama(
        model="llama3.2:1b",
        temperature=0.2
    )

def test_local_llm():
    llm = get_local_llm()
    start = time.perf_counter()
    response = llm.invoke("Explain retrieval-augmented generation (RAG) in 2 short sentences.")
    end = time.perf_counter()

    print(response.content if hasattr(response, "content") else response)
    print("Inference time:", round(end - start, 2), "seconds")

if __name__ == "__main__":
    test_local_llm()
