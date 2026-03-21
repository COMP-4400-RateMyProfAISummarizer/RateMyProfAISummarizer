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
git clone https://github.com/COMP-4400-RateMyProfAISummarizer/RateMyProfAISummarizer.git
cd RateMyProfAISummarizer
python3 -m venv comp4400.venv
source comp4400.venv/bin/activate
pip install -r requirements.txt
```

### **2. Environment Variables (Security Protocol)**
We use a `.env` file to manage private API keys. **Never commit your `.env` file to GitHub.**
1. Copy the template: `cp .env.example .env`
2. Open `.env` and paste your specific keys:
   * `PINECONE_API_KEY`
   * `CLOUD_API_KEY`

### **3. Running the Pipeline**
1. **Ingest Data:** `python ingestion/upload_to_pinecone.py`
2. **Launch App:** `streamlit run app/main.py`

---

## 🛠️ Tech Stack
* **LLM:** Google Gemini 3 Flash (Preview)
* **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (Local/Free)
* **Orchestration:** LangChain (LCEL)
* **Vector DB:** Pinecone
* **Interface:** Streamlit
* **CI/CD:** GitHub Actions

---

## 📝 Metadata Contract (Internal Use)
To ensure the **Data** and **Retrieval** parts connect, all vectors must use the following keys:
* `prof_name`: Full name of the instructor.
* `dept`: Academic department at UWindsor.
* `avg_rating`: Numerical score (1-5).
* `would_take_again`: Percentage feedback.
* `avg_difficulty`: Numerical score (1-5).
