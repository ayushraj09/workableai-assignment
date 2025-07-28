import re
import sqlite3
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

# 🔐 Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# 🧠 Set up GPT-4.1 model
llm = ChatOpenAI(
    model="gpt-4.1-nano-2025-04-14",  # GPT-4.1
    temperature=0.5
)

# 🧾 System Prompt Template for extraction
SYSTEM_PROMPT = """
You are a LaTeX question extractor working on OCR-processed pages from a scanned math textbook.

The input you receive is a chunk of the full text, which may include partial or complete sections.

Your job is to extract ONLY the question statements.
OMIT any Solution, Hints, Answers.

Instructions:
- Ignore theory, explanations, or headings.
- Clean OCR noise and fix spelling mistakes where needed.
- Understand the context of question.
- Identify question subparts like (i), (ii), etc., and group them under the same question.
- Wrap each question (including subparts) inside triple backticks with LaTeX code.

You may receive incomplete questions at the start or end of the chunk.
Do NOT repeat questions already seen; process only the visible content in this chunk.

Output should be in Latex. Do not wrap in '''latex'''.
"""

# 🧱 LangChain prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{raw_text}")
])

# 🔁 LangChain chain
chain: Runnable = prompt | llm

# 📚 Fetch raw_text from SQLite DB
def fetch_raw_text(chapter: str, topic: str):
    conn = sqlite3.connect("extracted.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT raw_text FROM ocr_raw
        WHERE chapter = ? AND topic = ?
        ORDER BY page_number ASC
    """, (chapter, topic))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise ValueError("❌ No data found for Chapter: {}, Topic: {}".format(chapter, topic))
    
    combined_text = "\n".join([row[0] for row in rows])
    return combined_text

def get_available_chapter_topic_pairs():
    conn = sqlite3.connect("extracted.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT chapter, topic
        FROM ocr_raw
        ORDER BY chapter, topic
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

# 🔁 Recursive Chunking Function
def chunk_text(text, chunk_size=2000, chunk_overlap=75):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

def save_chunks_txt(chunks, chapter, topic, out_dir="saved_chunks"):
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/chapter_{chapter}_topic_{topic}_chunks.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- CHUNK {i + 1} ---\n")
            f.write(chunk.strip() + "\n\n")
    # print(f"💾 Saved all chunks to {filename}\n")

# 🚀 Main Function with Chunked LLM Calls
def load_chunks_txt(chapter, topic, out_dir="saved_chunks"):
    filename = f"{out_dir}/chapter_{chapter}_topic_{topic}_chunks.txt"
    if not os.path.exists(filename):
        return None
    # print(f"📂 Loading existing chunks from {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        chunks = content.split("--- CHUNK")
        # Each chunk starts with " <number> ---\n"
        cleaned_chunks = []
        for chunk in chunks:
            lines = chunk.strip().split("\n", 1)
            if len(lines) == 2:
                cleaned_chunks.append(lines[1].strip())
        return cleaned_chunks

def save_chunks_txt(chunks, chapter, topic, out_dir="saved_chunks"):
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/chapter_{chapter}_topic_{topic}_chunks.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- CHUNK {i + 1} ---\n")
            f.write(chunk.strip() + "\n\n")
    # print(f"💾 Saved all chunks to {filename}\n")


def remove_latex_triple_quotes(text):
    # Removes ```latex ... ``` blocks, keeping only the inner LaTeX
    return re.sub(r"```latex\s*([\s\S]*?)\s*```", r"\1", text)

# 🧾 System Prompt for SECOND LLM Pass (Refiner)
REFINE_PROMPT = """
You are a LaTeX question formatter and cleaner.

Your job is to:
- Identify the questions and number each in order starting from 1 and subparts (if available) from (i).
- Clean any leftover OCR or spelling issues.
- Ensure each question is clearly structured.
- Fix any indentation issues or missing punctuation.

Output should ONLY contain final LaTeX questions numbered and formatted. Do NOT wrap in ``` or any markdown.
Include necessary packages and use begin document and end document.
"""

# Second LLM Chain
refine_prompt = ChatPromptTemplate.from_messages([
    ("system", REFINE_PROMPT),
    ("human", "{latex_questions}")
])

refine_chain: Runnable = refine_prompt | llm

def extract_questions_latex(chapter: str, topic: str):
    # Try loading chunks from local file first
    chunks = load_chunks_txt(chapter, topic)

    if chunks is None:
        # If not found, do the chunking from DB
        raw_text = fetch_raw_text(chapter, topic)
        chunks = chunk_text(raw_text)
        save_chunks_txt(chunks, chapter, topic)
    else:
        pass
        # print(f"🔄 Reusing {len(chunks)} cached chunks")

    # print(f"\n📘 Extracting questions for Chapter {chapter} - Topic: {topic}")
    # print(f"🔹 Total chunks: {len(chunks)}\n")

    all_outputs = []
    for i, chunk in enumerate(chunks):
        # print(f"🧠 Processing chunk {i+1}/{len(chunks)}...")
        result = chain.invoke({"raw_text": chunk})
        cleaned_content = remove_latex_triple_quotes(result.content.strip())
        all_outputs.append(cleaned_content)

    final_output = "\n\n".join(all_outputs)

    # print("🔁 Refining output with second LLM...\n")
    refined_result = refine_chain.invoke({"latex_questions": final_output})
    final_latex_output = refined_result.content.strip()

    # print("✅ Final Numbered & Cleaned LaTeX Questions:\n")
    print(final_latex_output)

# Call the function
# extract_questions_latex("31", "Probability distribution")
