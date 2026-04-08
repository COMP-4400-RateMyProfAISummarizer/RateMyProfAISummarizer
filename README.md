## 🎓 UWindsor RateMyProf Agentic RAG System
### *University of Windsor – Advanced Faculty Review Analysis*

This project implements an **Agentic Retrieval-Augmented Generation (RAG)** pipeline designed to provide grounded, factual summaries of UWindsor professor reviews. By utilizing **Local Inference**, **Iterative Reasoning**, and **Cross-Encoder Re-ranking**, the system eliminates hallucinations and provides high-precision academic insights for student planning.

---

## 🏗️ System Architecture
The system follows a modular agentic workflow to ensure data integrity and reasoning depth:
1.  **Fuzzy Input Handling:** Automatically corrects partial or misspelled names (e.g., "Ziad" → "Ziad Kobti") via a retrieval loop.
2.  **Multi-Query Expansion:** Decomposes a single user question into three targeted search vectors (e.g., Grading, Workload, and Classroom Vibe).
3.  **Hybrid-Style Retrieval:** Uses metadata-filtered Pinecone searches followed by a **Cross-Encoder Re-ranker** to prioritize the most informative snippets.
4.  **Local Synthesis:** Generates a structured report using an **8-bit Quantized Llama-3.1** model running locally on Ollama.

---

## 📺 Demo
Check out the system in action: https://www.youtube.com/watch?v=1cAJ8rb8KzY

---

## 📂 Project Structure & Team Roles

### 1. Data & Vector Engineering (`/ingestion`)
* **Assigned to:** Aakanksha Mandal
* **Goal:** Create the "Memory" of the bot.
* **Tasks:**
    * Implement **Semantic Chunking** to keep review context intact.
    * Manage the **Vector Space** using Pinecone.
    * Apply **Metadata Tagging** (`prof_name`, `dept`, `difficulty`) for filtered retrieval.

### 2. Retrieval Architecture (`/core`)
* **Assigned to:** Noor Haddad
* **Goal:** Build the "Brain" and the Summarizer.
* **Tasks:**
    * Develop **Metadata-filtered Retrieval** logic.
    * Design **System Prompts** for specialized academic summarization.
    * Implement **Source Attribution** to cite specific student reviews.

### 3. Ops & Agentic Deployment (`/app`)
* **Assigned to:** Hanan Senah
* **Goal:** Deployment and Quality Assurance.
* **Tasks:**
    * Manage **LLM Inference** and application state.
    * Serve the model and manage user session flow.
    * Conduct **Faithfulness & Relevance** tests to ensure zero hallucinations.

---

## 🛠️ Shared Technical Milestones (Completed by All Members)

Every team member contributed to the core engineering of the following three pillars:

#### **Task 1: The Infrastructure & Local Model Lead**
*Focus: Local LLM Deployment, Quantization, and Embedding Optimization.*
- [x] **Local Model Environment:** Set up Ollama to run **Llama-3.1-8B** locally.
- [x] **Model Quantization:** Implemented **8-bit quantization** (`q8_0`) to ensure high-precision inference on standard laptops.
- [x] **Embedding Pipeline:** Replaced generic embeddings with the specialized local model **BAAI/bge-small-en-v1.5**.
- [x] **Performance Benchmarking:** Created scripts to log **Inference Time** as a core "Engineering Metric" for the final report.

#### **Task 2: The Retrieval & Semantic Search Lead**
*Focus: Advanced Database Logic, Hybrid Search, and Metadata Filtering.*
- [x] **Hybrid Search Implementation:** Moved beyond simple vector search to handle specific terms like course codes (e.g., "COMP-1410").
- [x] **Metadata Engineering:** Programmatic filtering by `dept`, `difficulty`, and `prof_name`.
- [x] **Re-Ranking Layer:** Implemented a **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) to re-score results, ensuring only the most relevant reviews reach the LLM.
- [x] **Data Cleaning:** Scripted a **Deduplication pass** to optimize the AI’s context window.

#### **Task 3: The Agentic Logic & Tool Lead**
*Focus: Function Calling, Iterative Loops, and Query Expansion.*
- [x] **Function Calling (Tools):** Wrapped the database search into a LangChain Tool.
- [x] **Iterative Reasoning (The Loop):** Created a reasoning loop to handle partial name matches (e.g., "Ziad" → "Ziad Kobti").
- [x] **Query Expansion:** Programmed the agent to turn one user question into three internal targeted searches.
- [x] **Prompt Engineering & Control:** Developed "If/Then" system logic to manage edge cases and fallback instructions (e.g., "Suggest checking the University Syllabus").

---

## 🚀 Getting Started

### 1. Prerequisites
* [Ollama](https://ollama.com/) installed and running.
* Download the model: `ollama pull llama3.1:8b-instruct-q8_0`

### 2. Setup
```bash
git clone https://github.com/COMP-4400-RateMyProfAISummarizer/RateMyProfAISummarizer.git
cd RateMyProfAISummarizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=your_index_name
```

### 4. Running the Code
```bash
# To run the streamlit app, navigate to either v1_cloud or v2_local folder and run the following command
streamlit run app/main.py
```

---

## 🛠️ Tech Stack
* **LLM:** Llama-3.1-8B (Local via Ollama)
* **Embeddings:** BAAI/bge-small-en-v1.5 (Local/HuggingFace)
* **Reranker:** Cross-Encoder MS-Marco MiniLM
* **Vector DB:** Pinecone (Serverless)
* **Orchestration:** LangChain (LCEL)
