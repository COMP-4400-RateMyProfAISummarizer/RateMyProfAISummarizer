# 🎓 UWindsor Rate My Prof Specialized RAG Agent
**University of Windsor – Faculty Reviews Analysis System**

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to provide grounded, factual summaries of University of Windsor professor reviews on Rate My Prof. By using specialized AI logic instead of general LLM responses, we eliminate hallucinations and provide direct source attribution for academic planning.

---

## 🏗️ System Architecture
The system follows a standard RAG pattern: **Ingestion** (Vectorizing reviews), **Retrieval** (Finding relevant context), and **Generation** (Summarizing with an LLM).

---

## 📂 Project Structure & Team Roles

### **1. Data & Vector Engineering** (`/ingestion`)
**Assigned to: Aakanksha Mandal**
* **Goal:** Create the "Memory" of the bot.
* **Tasks:** * Implement **Semantic Chunking** to keep review context intact.
    * Manage the **Vector Space** using Pinecone/ChromaDB.
    * Apply **Metadata Tagging** (`prof_name`, `dept`) for filtered retrieval.

### **2. Retrieval Architecture** (`/core`)
**Assigned to: Noor Haddad**
* **Goal:** Build the "Brain" and the Summarizer.
* **Tasks:** * Develop **Metadata-filtered Retrieval** logic.
    * Design **System Prompts** for specialized summarization.
    * Implement **Source Attribution** to cite specific reviews.

### **3. Ops & Evaluation** (`/app`)
**Assigned to: Hanan Senah**
* **Goal:** Deployment and Quality Assurance.
* **Tasks:** * Serve the model using **Streamlit**.
    * Manage **LLM Inference** (via Ollama or OpenAI).
    * Conduct **Faithfulness & Relevance** tests to ensure zero hallucinations.

---

## 🚀 Getting Started

### **1. Clone & Setup**
```bash
git clone [repository-url]
cd RateMyProfAISummarizer
pip install -r requirements.txt
```

### **2. Environment Variables (Security Protocol)**
We use a `.env` file to manage private API keys. **Never commit your `.env` file to GitHub.**
1. Copy the template: `cp .env.example .env`
2. Open `.env` and paste your specific keys:
   * `PINECONE_API_KEY`
   * `OLLAMA_API_KEY`

### **3. Running the Pipeline**
1. **Ingest Data:** `python ingestion/upload_to_pinecone.py`
2. **Launch App:** `streamlit run app/main.py`

---

## 🛠️ Tech Stack
* **LLM:** Llama 3 / GPT-4
* **Orchestration:** LangChain (LCEL)
* **Vector DB:** Pinecone
* **Interface:** Streamlit
* **Embedding Model:** `text-embedding-3-small`

---

## 📝 Metadata Contract (Internal Use)
To ensure the **Data** and **Retrieval** parts connect, all vectors must use the following keys:
* `prof_name`: Full name of the instructor.
* `course_code`: e.g., COMP-3710.
* `review_text`: The raw student feedback.
