# 📊 Financial Assistant with Google GenAI and 10-K Filings

This repo contains **3 notebook-based RAG pipelines** for querying SEC 10-K filings (Apple + Tesla samples included) using **ChromaDB** for retrieval and **Gemini** for answer generation (plus local-embedding options).

- **`RAG_with_llama.ipynb`**: LlamaParse (better structure/tables) → Gemini embeddings (`text-embedding-004`) → ChromaDB → Gemini 2.5 Flash answers  
- **`RAG.ipynb`**: PyMuPDF extraction → *local GPU* embeddings (`all-MiniLM-L6-v2`) → ChromaDB → Gemini answers  
- **`scratch_rag.ipynb`**: experimental pipeline (text + `pdfplumber` tables + multi-embedding configs)


## 📌 Overview

This application extracts, processes, and analyzes financial data from **SEC 10-K filings** to provide intelligent insights and responses through a structured pipeline:

- 📄 Extract text and tables from PDF documents (multiple extraction methods)
- 🧠 Process and chunk documents for optimal retrieval
- 🔍 Create semantic embeddings with **multiple embedding options** (Gemini, OpenAI, Local)
- 🗂 Store embeddings in **ChromaDB** (a vector database)
- 🔁 Implement a **Retrieval-Augmented Generation (RAG)** pipeline
- ✍️ Generate structured and controlled responses via **Gemini LLM**

---

## 🚀 Features

- **Multiple PDF Extraction Methods**:
  - **LlamaParse**: Advanced extraction with better table/structure handling (`RAG_with_llama.ipynb`)
  - **PyMuPDF (fitz)**: Fast local text extraction (`RAG.ipynb`, `scratch_rag.ipynb`)
  - **pdfplumber**: Dedicated table extraction from financial documents

- **Flexible Embedding Options**:
  - **Google Gemini** (`text-embedding-004`) - 768 dimensions
  - **OpenAI** (`text-embedding-3-small`) - 1536 dimensions
  - **Local GPU-accelerated** (`all-MiniLM-L6-v2`) - 384 dimensions, fast inference
  - **Local** (`intfloat/e5-large-v2`) - 1024 dimensions, high quality

- **Table Extraction**: Extracts and converts financial tables to text for better context

- **GPU Acceleration**: Local embedding models run on CUDA for fast processing

- **RAG Pipeline**: Full retrieval-augmented generation with source citations

- **Multi-Company Support**: Process multiple 10-K filings (Apple, Tesla, etc.)

---

## 🛠️ Technical Architecture

### 📥 Data Extraction & Processing
- PDF text extraction using **PyMuPDF** (`fitz`) or **LlamaParse**
- Table extraction using **pdfplumber** with DataFrame conversion
- Text cleaning, preprocessing & chunking (1200-1500 chars with overlap)

### 📐 Embedding & Storage
- **Gemini Embeddings** for cloud-based embedding generation
- **Sentence Transformers** for local GPU-accelerated embeddings
- Vector storage and similarity search with **ChromaDB**

### 🔁 RAG Implementation
- Semantic search with configurable top-k retrieval
- Filtering by year, company, or filing ID
- Context-aware prompt construction with system instructions
- Response generation using **Gemini 2.5 Flash/Pro** with retry logic

---

## 📋 Prerequisites

- **Python** ≥ 3.10
- **CUDA** (optional, for GPU-accelerated local embeddings)
- **API Keys**:
  - Google Gemini API key
  - LlamaCloud API key (for LlamaParse)
  - OpenAI API key (optional)

### Required Libraries

```bash
# Core dependencies
pip install python-dotenv chromadb google-genai

# PDF extraction
pip install PyMuPDF pdfplumber

# LlamaParse (advanced extraction)
pip install llama-cloud-services nest_asyncio

# Local embeddings (GPU-accelerated)
pip install sentence-transformers torch

# Optional
pip install pandas numpy
```

---

## 📁 Project Structure

```
RAG_for_Finance/
├── RAG_with_llama.ipynb    # LlamaParse + Gemini embeddings + Chroma + Gemini generation
├── RAG.ipynb               # PyMuPDF + local GPU embeddings + Chroma + Gemini generation
├── scratch_rag.ipynb       # Experimental: table extraction & multi-model comparison
├── GenAI_kaggle_Latest.ipynb
├── .env                    # API keys (GEMINI_API_KEY, LlamaParse, etc.)
├── NOV_2023.pdf            # Apple 10-K 2023
├── NOV_2024.pdf            # Apple 10-K 2024
├── OCT_2025.pdf            # Apple 10-K 2025
├── Tesla_2023.pdf          # Tesla 10-K 2023
├── Tesla_2023.docx
└── README.md
```

---

## 🚀 Quick Start

1. **Set up environment variables** in `.env`:
   ```
   GEMINI_API_KEY=your_gemini_key
   # LlamaParse key name differs across notebooks; set BOTH to avoid editing code
   LlamaParse=your_llamaparse_key
   LLAMA_CLOUD_API_KEY=your_llamaparse_key

   # Optional (only used in scratch experiments)
   OPENAI_API_KEY=your_openai_key
   OPEN_AI_API_KEY=your_openai_key
   ```

2. **Run a notebook**:
   - `RAG_with_llama.ipynb` - Best for structured financial documents
   - `RAG.ipynb` - Fast local GPU embeddings + Gemini generation
   - `scratch_rag.ipynb` - Table extraction experiments + embedding model configs

3. **Ask questions**:
   ```python
   answer("What was Apple's net income in 2023?")
   answer("Compare iPhone revenue between 2022 and 2023")
   answer("What are the main risk factors?")
   ```

---

## 🔮 Future Work

The broader vision includes:

- ✅ **LLM Evaluation**: Compare responses against ground truth to assess accuracy and factuality
- 📊 **Data Expansion**: Extract a wider range of financial, strategic, and operational data
- 🏢 **Multi-Company Support**: Extend parsing to multiple companies and include 10-Q reports
- ⚙️ **Agent Pipelines**: Automate data extraction, embedding, and updating using AI agents
- 🌐 **Web App**: Develop an interactive dashboard for exploring and querying filings
- 🧠 **LLM Benchmarking**: Test models like GPT-4, Gemini, and Claude on financial document understanding tasks
- 📈 **Embedding Comparison**: Systematic evaluation of retrieval quality across embedding models

---

