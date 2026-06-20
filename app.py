import os
import json
import hashlib
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load the environment variables from the .env file
load_dotenv() 

# 1. ADD YOUR GROQ API KEY HERE
# os.environ["GROQ_API_KEY"] = "PASTE_YOUR_NEW_GROQ_KEY_HERE"

NOTES_DIR = "./notes"
DB_DIR = "./chroma_db"
MANIFEST_FILE = "./sync_manifest.json"

def compute_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

if os.path.exists(MANIFEST_FILE):
    with open(MANIFEST_FILE, 'r') as f:
        manifest = json.load(f)
else:
    manifest = {}

headers_to_split_on = [("##", "Problem_URL")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
new_or_updated_chunks = []
manifest_updated = False

print("Scanning for note updates...")
if not os.path.exists(NOTES_DIR):
    os.makedirs(NOTES_DIR)

for filename in os.listdir(NOTES_DIR):
    if filename.endswith(".md"):
        file_path = os.path.join(NOTES_DIR, filename)
        current_hash = compute_md5(file_path)
        
        if filename not in manifest or manifest[filename] != current_hash:
            print(f"🔄 Processing changes in: {filename}")
            if filename in manifest:
                db.delete(where={"source": filename})
            
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                chunks = markdown_splitter.split_text(text)
                for chunk in chunks:
                    chunk.metadata["source"] = filename
                new_or_updated_chunks.extend(chunks)
            
            manifest[filename] = current_hash
            manifest_updated = True

if new_or_updated_chunks:
    print(f"Adding {len(new_or_updated_chunks)} chunks to Vector DB...")
    db.add_documents(new_or_updated_chunks)
    print("Database updated successfully!")
else:
    print("Everything is up to date.")

if manifest_updated:
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=4)

# ==================== EXTRA RAG & LLM INTEGRATION ====================

# 2. Setup Retriever
retriever = db.as_retriever(search_kwargs={"k": 6})

# 3. Setup Groq LLM Client (using ultra-fast Llama 3)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

# 4. Define Persona Prompt
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

prompt = ChatPromptTemplate.from_template(prompt_template)

# Helper function to join documents together
def format_docs(docs):
    return "\n\n".join([f"Source file: {d.metadata.get('source')}\n{d.page_content}" for d in docs])

# 5. Build the LangChain RAG pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n" + "="*60)
print("🤖 LeetCode RAG Tutor is Live! Type 'exit' to quit.")
print("="*60 + "\n")

# Interactive Chat Loop
while True:
    user_query = input("You: ")
    if user_query.strip().lower() == 'exit':
        print("Goodbye!")
        break
    if not user_query.strip():
        continue
        
    print("\nSearching notes...")
    
    # DEBUG STEP: Let's see exactly what the database retrieves
    retrieved_docs = retriever.invoke(user_query)
    print("📚 Retrieved Context Sources:")
    for doc in retrieved_docs:
        print(f"   - {doc.metadata.get('source')} (Snippet: {doc.page_content[:40]}...)")
    
    print("\nGenerating tutor response...")
    response = rag_chain.invoke(user_query)
    print(f"\nTutor:\n{response}")
    print("\n" + "-"*60 + "\n")