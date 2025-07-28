import json
import sqlite3
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
import os
import torch
import numpy as np
from pix2text import Pix2Text


# --- Pix2Text Setup ---
total_config = {
    'text_formula': {'languages': ('en',)},
}
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

p2t = Pix2Text.from_config(total_config=total_config, device='mps')

# --- Load Chapter Map (limit to 2 chapters) ---
with open("chapter_topic_map.json", "r") as f:
    chapter_data = json.load(f)

# Done - 25, 26, 31
selected_chapters = ["31"]

# --- DB Setup ---
conn = sqlite3.connect("extracted.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS ocr_raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter TEXT,
    topic TEXT,
    page_number INTEGER,
    raw_text TEXT,
    UNIQUE(chapter, topic, page_number)
)
""")
conn.commit()

# --- PDF Setup ---
pdf_path = "rd_sharma.pdf"
doc = fitz.open(pdf_path)

# --- Process Single Page ---
def process_page(chap, topic, page_num):
    try:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        temp_path = "temp_page.png"
        img.save(temp_path)

        # --- Your Pix2Text Config & Call ---
        raw_text = p2t.recognize(temp_path, file_type='text_formula', return_text=True,
                                 auto_line_break=False, save_analysis_res=None)

        # --- Store in DB ---
        cursor.execute("""
            INSERT OR IGNORE INTO ocr_raw (chapter, topic, page_number, raw_text)
            VALUES (?, ?, ?, ?)
        """, (chap, topic, page_num, raw_text))
        conn.commit()

        os.remove(temp_path)
    except Exception as e:
        print(f"❌ Error on Chapter {chap}, Topic {topic}, Page {page_num}:\n{e}")

# --- Preprocess topic pages ---
def get_sorted_topic_pages(chapter_dict):
    topics = chapter_dict["topics"]
    topic_list = []
    for tid, tinfo in topics.items():
        try:
            page = int(float(tinfo["book_page"])) - 1
            topic_list.append((tid, tinfo["title"], page))
        except:
            continue
    # Sort by page number
    topic_list.sort(key=lambda x: x[2])
    return topic_list

# --- Get first page of next chapter ---
def get_next_chapter_start_page(chapter_id):
    chapter_ids = sorted([int(k) for k in chapter_data.keys()])
    current_index = chapter_ids.index(int(chapter_id))
    if current_index + 1 < len(chapter_ids):
        next_chap = str(chapter_ids[current_index + 1])
        next_topics = get_sorted_topic_pages(chapter_data[next_chap])
        return next_topics[0][2] if next_topics else None
    return None


# --- Main Loop ---
for chap in selected_chapters:
    topic_pages = get_sorted_topic_pages(chapter_data[chap])
    for idx, (topic_id, topic_title, start_page) in enumerate(topic_pages):
        # --- Determine end page ---
        if idx + 1 < len(topic_pages):
            next_page = topic_pages[idx + 1][2]
        else:
            # Last topic → go to first topic page of next chapter
            next_page = get_next_chapter_start_page(chap)
            if next_page is None:
                next_page = start_page + 1  # fallback

        # --- Apply your 3-case logic ---
        if next_page == start_page:
            pages_to_process = [start_page]
        else:
            pages_to_process = list(range(start_page, next_page))

        print(f"🔹 Chapter {chap} | Topic {topic_id} ({topic_title}) | Pages {pages_to_process}")
        for page_num in pages_to_process:
            process_page(chap, topic_title, page_num)


print("✅ All done. OCR text stored in extracted.db.")