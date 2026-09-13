# LangChain RAG Debugging

A beginner-friendly **end-to-end RAG implementation using LangChain**, focused on understanding and debugging each stage of the pipeline.

## 🔄 RAG Flow

```text
Document
   ↓
Text Splitting
   ↓
Embeddings
   ↓
Chroma Vector DB
   ↓
Retriever
   ↓
Relevant Context
   ↓
Prompt
   ↓
LLM
   ↓
Answer
```

## 🛠️ Tech Stack

* Python
* LangChain
* Hugging Face Embeddings
* Sentence Transformers
* ChromaDB
* OpenRouter
* LangSmith

## 🚀 Setup

```bash
git clone https://github.com/Charanm24/langChain-rag-debugging.git
cd langChain-rag-debugging

python -m venv .venv
```

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env`:

```env
OPENROUTER_API_KEY=your_api_key

LANGSMITH_API_KEY=your_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langchain-rag-debugging
```

Run:

```bash
python main.py
```

## 🔍 LangSmith Debugging

LangSmith lets you see what happens inside the RAG pipeline.

```text
User Question
      ↓
   Retriever
      ↓
Retrieved Chunks
      ↓
Context + Prompt
      ↓
     LLM
      ↓
   Answer
```

In LangSmith, inspect each step to find problems such as:

* ❌ Wrong documents retrieved
* ❌ Poor chunking
* ❌ Missing context
* ❌ Incorrect prompt
* ❌ Unexpected LLM response
* ⏱️ Slow components

The goal is to answer:

> **Why did my RAG application give this answer?**

## 🎯 What You Will Learn

* How documents are loaded and split
* How text becomes embeddings
* How vectors are stored in Chroma
* How retrieval works
* Why `format_docs()` is needed
* How LangChain connects retrieval → prompt → LLM
* How to debug RAG using LangSmith
* How to identify whether a problem is caused by **chunking, retrieval, prompt, or the LLM**

##  Future Improvements

* Persistent ChromaDB
* Better/semantic chunking
* LangSmith tracing & debugging
* RAG evaluation
* Hybrid search
* Reranking

> **Goal:** Understand every stage of a RAG pipeline instead of treating RAG as a black box.
