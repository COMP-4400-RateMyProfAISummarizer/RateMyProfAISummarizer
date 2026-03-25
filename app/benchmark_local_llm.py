import time
from core.local_llm import get_local_llm

def test_three_prompts():
    llm = get_local_llm()
    start1 = time.perf_counter()
    response1 = llm.invoke("Define in two sentences retrieval-augmented generation.")
    end1 = time.perf_counter()
    print(response1.content if hasattr(response1, "content") else response1)
    print("Inference time:", round(end1 - start1, 2), "seconds")
    start2 = time.perf_counter()
    response2 = llm.invoke("Define in two sentences large language model embedding.")
    end2 = time.perf_counter()
    print(response2.content if hasattr(response2, "content") else response2)
    print("Inference time:", round(end2 - start2, 2), "seconds")
    start3 = time.perf_counter()
    response3 = llm.invoke("Define in two sentences vector database.")
    end3 = time.perf_counter()
    print(response3.content if hasattr(response3, "content") else response3)
    print("Inference time:", round(end3 - start3, 2), "seconds")

if __name__ == "__main__":
    test_three_prompts()
