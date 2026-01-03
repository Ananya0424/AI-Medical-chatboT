import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
# ----------------------
# Load ENV
# ----------------------
load_dotenv()



# ----------------------
# Flask
# ----------------------
app = Flask(__name__)

# ----------------------
# Pinecone init (NEW API)
# ----------------------



# ----------------------
# Embeddings
# ----------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------
# VectorStore
# ----------------------
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ----------------------
# Ollama
# ----------------------
from langchain_community.llms import Ollama

llm = Ollama(model="phi3", base_url="http://localhost:11434")



# ----------------------
# Prompt
# ----------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# ----------------------
# Routes
# ----------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    # Accept form data or JSON
    msg = request.form.get("msg") or request.form.get("input") or request.form.get("message")
    if not msg and request.is_json:
        msg = request.json.get("msg") or request.json.get("input") or request.json.get("message")

    if not msg:
        return jsonify({"error": "No question received"}), 400

    # Get answer from QA
    result = qa.invoke({"query": msg})
    return jsonify({"answer": result["result"]})


# Run Flask
if __name__ == "__main__":
    app.run(debug=True)