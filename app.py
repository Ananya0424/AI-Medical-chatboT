import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from langchain_core.prompts import PromptTemplate
from src.prompt import system_prompt

load_dotenv()
app = Flask(__name__)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local(
        "faiss_index", embedding, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    print("FAISS index not found! Run store_index.py first.")
    vectorstore = None
    retriever = None

llm = OllamaLLM(model="phi3", base_url="http://localhost:11434")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

CUSTOM_PROMPT = PromptTemplate(
    template=system_prompt + "\nQuestion: {question}\nAnswer:",
    input_variables=["context", "chat_history", "question"]
)

if retriever:
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        return_source_documents=True
    )
else:
    qa = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/status")
def status():
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            return jsonify({"status": "Online", "model": "phi3"})
        return jsonify({"status": "Ollama Offline"}), 503
    except:
        return jsonify({"status": "Disconnected"}), 503

@app.route("/clear", methods=["POST"])
def clear_chat():
    memory.clear()
    return jsonify({"status": "History cleared"})

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg") or request.form.get("input") or request.form.get("message")
    if not msg and request.is_json:
        msg = request.json.get("msg") or request.json.get("input") or request.json.get("message")
    if not msg:
        return jsonify({"error": "No question received"}), 400
    if not qa:
        return jsonify({"error": "Database not initialized."}), 503
    try:
        result = qa.invoke({"question": msg})
        return jsonify({
            "answer": result["answer"],
            "source": [doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)