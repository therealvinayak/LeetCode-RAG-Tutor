# LeetCode-RAG-Tutor 🤖

Built a small local RAG engine over a weekend to make LeetCode revision less painful. Instead of scrolling through long notes and old solutions, I wanted a way to search my personal DSA knowledge base and quickly retrieve patterns, dry-runs, code snippets, and mnemonics from my markdown notes.

### What it does
* **Indexes structured Markdown/Notion exports**
* **Retrieves relevant problem-solving patterns** using semantic search
* **Preserves code blocks and note structure** for better context
* **Answers queries using an LLM** over the retrieved notes

### Tech Stack
* **Chunking:** `MarkdownHeaderTextSplitter`
* **Embeddings:** `all-MiniLM-L6-v2`
* **Vector Store:** ChromaDB
* **LLM:** Llama 3.3 70B via Groq
* **Framework:** LangChain

### Some things I added
* **MD5-based incremental sync** so only new/updated notes get re-indexed
* **Header-aware chunking** to avoid breaking code blocks
* **Persistent local vector database**
* **Fast inference through Groq** (usually around ~1 second response time)

### Quick Start

1. Install dependencies:
```bash
pip install langchain langchain-chroma langchain-huggingface langchain-groq sentence-transformers python-dotenv

```

2. Create a `/notes/` folder and drop your markdown files inside.
3. Add your API key in a local `.env` file:

```env
GROQ_API_KEY=your_key_here

```

4. Run the interactive tool:

```bash
python app.py

```

Still a small project, but it's already become my go-to tool for interview prep and revising old problem-solving patterns.

```

