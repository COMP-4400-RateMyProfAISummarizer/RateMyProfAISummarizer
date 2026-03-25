import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv

from core.summarizer import generate_summary

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaLLM

load_dotenv()

def init_components():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_API_KEY in .env")
    if not pinecone_index_name:
        raise ValueError("Missing PINECONE_INDEX_NAME in .env")
    if not openai_api_key:
        raise ValueError("Missing OPENAI_API_KEY in .env")

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key
    )

    vector_db = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = OllamaLLM(model="llama3.1:8b-instruct-q8_0")

    return vector_db, reranker, llm

golden_dataset = [
    {
        "prof_name": "Ziad Kobti",
        "query": "How is the grading style?",
        "ground_truth": "Check whether the answer matches the retrieved reviews."
    },
    {
        "prof_name": "Ziad Kobti",
        "query": "What is the workload like?",
        "ground_truth": "Check whether the answer matches the retrieved reviews."
    }
]

def run_evaluation():
    vector_db, reranker, llm = init_components()

    for i, item in enumerate(golden_dataset, start=1):
        start = time.time()

        result = generate_summary(
            query=item["query"],
            prof_name=item["prof_name"],
            vector_db=vector_db,
            reranker=reranker,
            llm=llm
        )

        end = time.time()

        print("\n==============================")
        print(f"Test Case {i}")
        print("Professor:", item["prof_name"])
        print("Question:", item["query"])
        print("Summary:", result["summary"])
        print("Sources:", result["sources"])
        print("Ground Truth:", item["ground_truth"])
        print(f"Inference Time: {end - start:.2f} seconds")

if __name__ == "__main__":
    run_evaluation()