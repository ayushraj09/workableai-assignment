import streamlit as st
import sqlite3
import io
import sys
import pandas as pd
from extract_questions import extract_questions_latex, get_available_chapter_topic_pairs
import subprocess
import tempfile
from pathlib import Path
import shutil

def check_pdflatex_installed():
    return shutil.which("pdflatex") is not None

if not check_pdflatex_installed():
    st.warning("⚠️ 'pdflatex' is not available. PDF rendering won't work until it's installed.")

st.set_page_config(page_title="LaTeX Question Extractor", layout="wide")
st.title("📘 RD Sharma LaTeX Question Extractor")

def compile_latex_to_pdf(latex_code: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = f"{tmpdir}/output.tex"
        pdf_path = f"{tmpdir}/output.pdf"
        with open(tex_path, "w") as f:
            f.write(latex_code)

        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path],
            cwd=tmpdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        with open(pdf_path, "rb") as f:
            return f.read()


@st.cache_data
def load_chapter_topic_pairs():
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


# Load chapter-topic pairs
pairs = load_chapter_topic_pairs()
if not pairs:
    st.error("❌ No data found in the database.")
    st.stop()

chapters = sorted(set(ch for ch, _ in pairs))
selected_chapter = st.selectbox("📚 Select Chapter", chapters)

topics = [topic for ch, topic in pairs if ch == selected_chapter]
selected_topic = st.selectbox("🧠 Select Topic", topics)

st.markdown("---")

with st.expander("📋 Show all available Chapter–Topic pairs"):
    df = pd.DataFrame(pairs, columns=["Chapter", "Topic"])
    st.dataframe(df, use_container_width=True)

if st.button("🚀 Extract LaTeX Questions"):
    try:
        with st.spinner("Running LLM pipeline..."):
            buffer = io.StringIO()
            sys.stdout = buffer

            # Run your extraction
            extract_questions_latex(selected_chapter.strip(), selected_topic.strip())

            sys.stdout = sys.__stdout__
            final_output = buffer.getvalue()

            st.success("✅ Extraction complete!")

        # Optional raw LaTeX preview (expandable)
        with st.expander("📜 Raw LaTeX Output"):
            st.code(final_output, language="latex")

        # Compile and show PDF
        try:
            pdf_bytes = compile_latex_to_pdf(final_output)

            # Scrollable PDF view (Streamlit ≥1.35 required)
            st.divider()
            st.subheader("📄 Rendered LaTeX Output (PDF):")
            st.pdf(pdf_bytes, height=800)

            st.download_button(
                label="💾 Download PDF",
                data=pdf_bytes,
                file_name=f"chapter_{selected_chapter}_topic_{selected_topic}.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"❌ Failed to compile LaTeX to PDF: {str(e)}")

        # Optional: download raw .tex
        st.download_button(
            label="💾 Download .tex file",
            data=final_output,
            file_name=f"chapter_{selected_chapter}_topic_{selected_topic}.tex",
            mime="text/plain"
        )

    except Exception as e:
        sys.stdout = sys.__stdout__
        st.error(f"❌ Error: {str(e)}")
