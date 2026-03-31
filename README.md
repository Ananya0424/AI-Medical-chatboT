# 🏥 AI Medical Chatbot

An intelligent medical chatbot powered by **LangChain**, **FAISS**, **Ollama (Phi-3)**, and **Flask** — capable of answering medical questions using a custom knowledge base built from medical PDFs.

---

## 📌 Features

- 💬 Conversational AI with memory (multi-turn chat)
- 📚 Retrieval-Augmented Generation (RAG) using FAISS vector store
- 🧠 Local LLM via Ollama (Phi-3) — no API cost, runs offline
- 🔍 Semantic search using HuggingFace sentence-transformers
- 🌐 Clean Flask web interface
- 🗂️ PDF ingestion pipeline for custom medical knowledge base

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML, CSS, JavaScript (Flask Templates) |
| **Backend** | Python, Flask |
| **LLM** | Ollama — Phi-3 (runs locally) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (local) |
| **RAG Framework** | LangChain |

---

## 📁 Project Structure

```
AI-Medical-chatbot/
│
├── src/
│   ├── helper.py          # PDF loading, chunking, embedding utils
│   └── prompt.py          # System prompt for the chatbot
│
├── templates/
│   └── index.html         # Chat UI
│
├── data/                  # Place your medical PDFs here
│
├── faiss_index/           # Auto-generated vector store (after store_index.py)
│
├── app.py                 # Main Flask application
├── store_index.py         # Script to build FAISS index from PDFs
├── requirements.txt
├── setup.py
├── .env                   # Environment variables (not committed)
└── README.md
```

---

## ⚙️ Local Setup & Run

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Ananya0424/AI-Medical-chatboT.git
cd AI-Medical-chatboT
```

---

### Step 2 — Create Virtual Environment

```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Setup Environment Variables

Create a `.env` file in the root folder:

```env
# No API keys needed for local setup
# Ollama runs fully offline
```

---

### Step 5 — Add Medical PDFs

Place your PDF files inside the `data/` folder.

---

### Step 6 — Build the FAISS Vector Index

```bash
python store_index.py
```

> This reads all PDFs from `data/`, creates embeddings, and saves the FAISS index locally. Run this once.

---

### Step 7 — Pull and Start Ollama (Phi-3)

Open **Terminal 1** and run:

```bash
ollama pull phi3
ollama run phi3
```

> Keep this terminal open. Ollama must be running at `http://localhost:11434`

---

### Step 8 — Start the Flask App

Open **Terminal 2** and run:

```bash
python app.py
```

---

### Step 9 — Open in Browser

```
http://localhost:5000
```

Your AI Medical Chatbot is live! 🎉

---

## 🔁 How It Works

```
User Question
     │
     ▼
Flask /get endpoint
     │
     ▼
FAISS Retriever → Top 3 relevant chunks from medical PDFs
     │
     ▼
LangChain ConversationalRetrievalChain
     │
     ▼
Ollama Phi-3 LLM (local) → Generates answer using context
     │
     ▼
Response sent back to UI
```

---

## 📦 Requirements

```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-huggingface
faiss-cpu
flask
flask-cors
sentence-transformers
pypdf
python-dotenv
gunicorn
-e .
```

---

## 🙋‍♀️ Author

**Ananya Sharma**
- GitHub: [@Ananya0424](https://github.com/Ananya0424)
- Email: ananyasharma242004@gmail.com

---

## 📄 License

This project is licensed under the MIT License.