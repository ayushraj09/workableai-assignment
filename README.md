# WorkableAI: Automated Math Question Extraction from Scanned Textbooks

## 📖 Approach Overview

This project automates the extraction of **math questions** from scanned textbook PDFs using a **Retrieval-Augmented Generation (RAG) pipeline**. The workflow is as follows:

1. **PDF Page Extraction:**  
   - Each chapter/topic is mapped to page numbers using a JSON map.
   - Pages are rendered as images using [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/).

2. **OCR & Formula Recognition:**  
   - [Pix2Text](https://github.com/breezedeus/Pix2Text) is used for OCR, with support for mathematical formula recognition.
   - Preprocessing (binarization, resizing) is applied to improve OCR accuracy.

3. **Data Storage:**  
   - Raw OCR text is stored in a SQLite database, indexed by chapter, topic, and page. 
   (Currently only 3 chapters are stored in database for the sake of time constraint)

4. **RAG Pipeline for Question Extraction:**  
   - Text is chunked and passed to an LLM (OpenAI GPT-4.1 via LangChain) with a carefully crafted prompt to extract only question statements in LaTeX.
   - A second LLM pass refines and numbers the questions, ensuring clean LaTeX output.

5. **Tools Used:**  
   - **OCR & Formula Detection:** Pix2Text, PyMuPDF, PIL, OpenCV
   - **LLM Orchestration:** LangChain, OpenAI GPT-4.1
   - **Database:** SQLite
   - **Utilities:** tqdm, dotenv, numpy

6. **Prompting Strategy:**  
   - System prompts instruct the LLM to extract only questions, ignore solutions/theory, group subparts, and output LaTeX.
   - A second prompt ensures numbering, formatting, and LaTeX document structure.

---

## ⚠️ Challenges Faced & Solutions

- **Scanned Images & OCR Quality:**  
  Scanned textbooks often have noise, skew, and inconsistent lighting, making OCR difficult—especially for mathematical expressions.  
  **Solution:**  
  - Applied image preprocessing (grayscale, adaptive thresholding, resizing) to enhance text and formula clarity.
  - Used Pix2Text for robust formula recognition.

- **Mathematical Expression Extraction:**  
  Standard OCR tools struggle with math notation.  
  **Solution:**  
  - Leveraged Pix2Text, which is specifically trained for math OCR and LaTeX output.

- **High Computation Time:**  
  OCR and LLM processing for each chapter/topic is time-consuming, especially for large textbooks.  
  **Solution:**  
  - Batched processing and chunked text to optimize LLM calls.
  - Used device acceleration (Apple MPS or CUDA) where possible for model inference.
  - Cached intermediate results (chunks, OCR) to avoid redundant computation.

---

## 📌 Assumptions & Limitations

- **Assumptions:**
  - The chapter-topic-page mapping in `chapter_topic_map.json` is accurate.
  - The scanned PDF pages are of sufficient quality for OCR after preprocessing.

- **Limitations:**
  - OCR accuracy may still be affected by poor scan quality, handwriting, or unusual fonts.
  - Mathematical expressions with complex layouts or diagrams may not be perfectly recognized.
  - Processing time is significant for large chapters due to the sequential nature of OCR and LLM calls.
  - The RAG pipeline relies on the LLM's ability to follow prompts; some edge cases may require prompt engineering.

---

## Future Improvements:

- Make a permanent database for OCR raw text and chunks for each chapter/topic for faster retrieval.
- Better prompting.
- Better and cost efficient models.
