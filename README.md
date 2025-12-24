# 📊 Financial Assistant with Google GenAI and 10-K Filings

This project demonstrates how to build an **AI-powered financial assistant** that analyzes SEC 10-K filings and answers questions about a company’s financial data using **Google's Generative AI (Gemini)**.

> 🔗 **Medium Article**: [From PDF to AI: Building a Financial Assistant with Google GenAI and 10-K Filings](https://medium.com/@rushyanth55/from-pdf-to-ai-building-a-financial-assistant-with-google-genai-and-10-k-filings-f1410d3cc620)  
> 📺 **YouTube Demo**: [Watch the Demo](https://www.youtube.com/watch?v=goC5lFe9fcQ&t=3s)


## 📌 Overview

This application extracts, processes, and analyzes financial data from **SEC 10-K filings** to provide intelligent insights and responses through a structured pipeline:

- 📄 Extract text from PDF documents  
- 🧠 Convert unstructured text to structured JSON using **Google Gemini** with few-shot prompting  
- 🔍 Create semantic embeddings with **Sentence Transformers**  
- 🗂 Store embeddings in **ChromaDB** (a vector database)  
- 🔁 Implement a **Retrieval-Augmented Generation (RAG)** pipeline  
- ✍️ Generate structured and controlled responses via **LLM prompting**  

---

## 🚀 Features

- **Text Extraction**: Pulls raw text directly from 10-K PDF filings  
- **Structured Data Conversion**: Converts unstructured text into a structured JSON format using few-shot prompts  
- **Semantic Search**: Uses Sentence Transformers to enable intelligent querying and context-aware retrieval  
- **RAG Pipeline**: Implements a Retrieval-Augmented Generation framework for contextual, relevant responses  
- **Controlled Generation**: Ensures responses are structured, consistent, and relevant through prompt engineering

---

## 🛠️ Technical Architecture

### 📥 Data Extraction & Processing
- Direct PDF text extraction using `PyPDF2`
- Text preprocessing & segmentation
- Structured conversion using **Google Gemini** + few-shot prompting

### 📐 Embedding & Storage
- Embedding generation via **Sentence Transformers**
- Vector storage and similarity search with **ChromaDB**

### 🔁 RAG Implementation
- Retrieve relevant content using semantic search
- Integrate context with queries
- Generate final answers using few-shot prompting and Gemini LLM

---

## 📋 Prerequisites

- **Python** ≥ 3.9  
- **Google API key** (for Gemini access)  
- **Required Libraries**:
  - `sentence-transformers`
  - `chromadb`
  - `PyPDF2`
  - `langchain`
  - `google-generativeai`


---

## 🔮 Future Work

The broader vision includes:

- ✅ **LLM Evaluation**: Compare responses against ground truth to assess accuracy and factuality  
- 📊 **Data Expansion**: Extract a wider range of financial, strategic, and operational data  
- 🏢 **Multi-Company Support**: Extend parsing to multiple companies and include 10-Q reports  
- ⚙️ **Agent Pipelines**: Automate data extraction, embedding, and updating using AI agents  
- 🌐 **Web App**: Develop an interactive dashboard for exploring and querying filings  
- 🧠 **LLM Benchmarking**: Test models like GPT-4, Gemini, and Claude on financial document understanding tasks  

---

