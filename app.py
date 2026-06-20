import os
import json
import hashlib
import time
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Core environmental configurations
load_dotenv()
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("❌ Missing Environment Variable: Please check your root level .env file configuration.")

# Deterministic Path Allocations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(BASE_DIR, "notes")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
MANIFEST_FILE = os.path.join(BASE_DIR, "sync_manifest.json")

def compute_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Load state tracker ledger
manifest = {}
if os.path.exists(MANIFEST_FILE):
    try:
        with open(MANIFEST_FILE, 'r') as f:
            manifest = json.load(f)
    except json.JSONDecodeError:
        pass

# Initialize Processing Infrastructure
headers_to_split_on = [("##", "Problem_URL")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Volatile initialization safely wrapping disk directories
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
new_or_updated_chunks = []
manifest_updated = False

print("🔍 Scanning for personal notes update matrices...")
if not os.path.exists(NOTES_DIR):
    os.makedirs(NOTES_DIR)

for filename in os.listdir(NOTES_DIR):
    if filename.endswith(".md"):
        file_path = os.path.join(NOTES_DIR, filename)
        current_hash = compute_md5(file_path)
        
        if filename not in manifest or manifest[filename] != current_hash:
            print(f"🔄 Change Detected: Re-indexing {filename}")
            if filename in manifest:
                # Target purge of historical document slices
                db.delete(where={"source": filename})
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                chunks = markdown_splitter.split_text(text)
                for chunk in chunks:
                    chunk.metadata["source"] = filename
                new_or_updated_chunks.extend(chunks)
                manifest[filename] = current_hash
                manifest_updated = True
            except Exception as e:
                print(f"⚠️ Error parsing file {filename}: {str(e)}")

# Synchronize local Vector Database instances
if new_or_updated_chunks:
    print(f"⚡ Batch Injecting {len(new_or_updated_chunks)} structural blocks into ChromaDB...")
    db.add_documents(new_or_updated_chunks)
    print("✅ Local store updated.")
else:
    print("🎯 Vectors fully synchronized. No reprocessing needed.")

if manifest_updated:
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=4)

# Build Prompt Layout Strategies
prompt_template = """
You are an expert software engineering mentor helping a student review their past LeetCode solutions.
Analyze the provided context notes carefully to answer their question.

CRITICAL INSTRUCTIONS FOR FORMATTING:
1. DO NOT write long paragraphs or dense walls of text. 
2. Use a highly structured, scannable, point-by-point format.
3. Keep sentences short, concise, and easy to memorize for interview preparation.
4. If the answer cannot be found in their notes, answer using general knowledge but explicitly preface it with: "⚠️ [Not found in your personal history notes, but here is the general engineering concept]:"

Structure your response exactly like this:

## 📂 Your C++ Code Reference
(Extract and paste the exact C++ code block found in the retrieved context notes for this problem. Enclose it inside standard ```cpp code blocks)

## 📌 Core Concept Breakdown
- (Short, 1-2 sentence high-level summary of the approach used)
- (Time & Space Complexity analysis of their implementation)

## 💻 Code Pattern & Logic
- (Bullet points breaking down key lines, variables, or data structures like vectors/queues)
- (Explain *why* this specific loop/condition was chosen)

## 🔄 Visual Dry Run (Step-by-Step Trace)
- (Provide a simple, tiny text-based visual trace using a mock mini-input, e.g., a tree with 3 nodes or an array of 3 elements)
- (Show how the variables/structures change state, line-by-line: e.g., "Queue: [root] -> Pop root -> Queue: [left, right]")

## 🧠 Brain Tricks & Tips (Never Forget Again)
- (Give a memorable analogy, mental rule of thumb, or shortcut to immediately recall this pattern in an interview)

Context notes:
{context}

Question: {question}
Answer:"""

# Assemble Pipeline Components
retriever = db.as_retriever(search_kwargs={"k": 4}) # Adjusted down slightly to maximize context concentration
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1) # Kept tight to prevent hallucination shifts
prompt = ChatPromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join([f"Source file: {d.metadata.get('source')}\n{d.page_content}" for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n" + "="*60)
print("🤖 LeetCode RAG Tutor Shell Session Active. [Type 'exit' to close]")
print("="*60 + "\n")

while True:
    user_query = input("You: ")
    if user_query.strip().lower() == 'exit':
        print("Closing tutor engine session. Good luck with prep!")
        break
    if not user_query.strip():
        continue
        
    print("\n🔍 Traversing ChromaDB Indices...")
    start_time = time.time()
    
    retrieved_docs = retriever.invoke(user_query)
    print("📚 Retrieved Vector Matches:")
    for doc in retrieved_docs:
        print(f"   ↳ Source: {doc.metadata.get('source')} | Header: {list(doc.metadata.values())[0] if doc.metadata else 'N/A'}")
    
    print("\n🧠 Querying Llama-3.3 Framework via Groq Core...")
    response = rag_chain.invoke(user_query)
    
    latency = time.time() - start_time
    print(f"\nTutor:\n{response}")
    print("\n" + "-"*60)
    print(f"⏱️ Operational Query Execution Latency: {latency:.2f} seconds")
    print("-"*60 + "\n")