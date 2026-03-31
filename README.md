# AI-Medical-chatboT
Medical Chatbot using Generative AI

# Build-a-Complete-Medical-Chatbot-with-LangChain-FAISS-Flask-a locally hosted LLM via Ollama (phi3).

#
FEATURES

.Medical Question Answering
.Local LLM using Ollama
.Vector Search using FAISS
.Backend using Flask
.No paid APIs required
.Beginner-friendly Generative AI project


#
TECH STACK 

.Python
.LangChain
.FAISS
.Flask
.Ollama (Local LLM – phi3)
.HTML, CSS, JavaScript (Frontend)

#
PROJECT STRUCTURE

AI-Medical-chatbot/
│
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── faiss_index/
├── requirements.txt
└── README.md

#
PREREQUISITES

.Python 3.10
.Conda (recommended)
.Ollama installed locally
👉 Install Ollama from:
https://ollama.com

# How to run?
### STEPS:

Clone the repository

```bash
git clonehttps://github.com/Ananya0424/AI-Medical-chatboT.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```
### STEP 03-Ollama Setup
 Pull the required model:
  ```bash
    ollama pull phi3
  ```

```bash
# run Ollama
ollama run phi3
```
Ollama will run at:http://localhost:11434

### STEP 04-Run the Flask App
```bash
# Open a new terminal (keep Ollama running):
python app.py
```
### STEP 05-Open in Browser
text- http://127.0.0.1:5000

🎉 Your Medical Chatbot is now running!

# Deployment
 This project is deployed on **GitHub** for source code hosting and version control.

The application is designed for **local deployment**, where:
- The backend runs using **Flask**
- The language model is served locally using **Ollama (phi3)**

No cloud services or paid APIs are required.