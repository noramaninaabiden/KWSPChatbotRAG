import os
import re

import duckdb
from pypdf import PdfReader

RAW_DIR = os.path.join("data", "raw")
DB_PATH = os.path.join("data", "kwsp.duckdb")

CHUNK_SIZE = 800
OVERLAP = 150

# dividend tables, skipped for now
SKIP_FILES = ["kwsp_bm_02.pdf", "kwsp_en_01.pdf", "kwsp_en_02.pdf"]
JUNK_PHRASES = [
    "Skip to Main Content",
    "Ahli Majikan Korporat BM",
    "WhatsApp UsConnect With Us"
    ]

def clean(text):
    kept_lines = []
    for line in text.split("\n"):
        # drop the repeated page header that contains the website address
        if re.search(r"gov\.?\s*my", line.lower()):
            continue
        kept_lines.append(line.strip())
    text = " ".join(kept_lines)
    text = re.sub(r"\s+", " ", text)

    # page footer like "1 of 52" (OCR sometimes reads it as "10f52")
    text = re.sub(r"\b[\dTIl]{1,3}\s?[o0]f\s?52\b", " ", text)

    # website menu text that came along when copying a page
    for phrase in JUNK_PHRASES:
        text = text.replace(phrase, " ")

    text = re.sub(r"\s+", " ", text)
    return text.strip()

def make_chunks(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            # pull the cut back to the last space so a word isn't split
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP
        # move the start forward to a space so the next chunk begins on a whole word
        next_space = text.find(" ", start)
        if next_space != -1:
            start = next_space + 1
    return chunks

def read_pages(path, file_name):
    if file_name.endswith(".pdf"):
        reader = PdfReader(path)
        return [
            (i, page.extract_text() or "")
            for i, page in enumerate(reader.pages, start=1)
        ]
    with open(path, encoding="utf-8") as f:
        return [(None, f.read())]

rows = []
chunk_id = 0

for file_name in sorted(os.listdir(RAW_DIR)):
    if file_name in SKIP_FILES or not file_name.endswith((".pdf", ".txt")):
        continue

    language = "bm" if "_bm_" in file_name else "en"
    pages = read_pages(os.path.join(RAW_DIR, file_name), file_name)

    for page_number, raw_text in pages:
        text = clean(raw_text)
        if len(text) < 100:
            continue

        for chunk_text in make_chunks(text):
            chunk_id += 1
            rows.append((chunk_id, file_name, language, page_number, chunk_text))

con = duckdb.connect(DB_PATH)
con.execute("DROP TABLE IF EXISTS chunks")
con.execute(
    """
    CREATE TABLE chunks (
        chunk_id INTEGER,
        file_name VARCHAR,
        language VARCHAR,
        page_number INTEGER,
        chunk_text VARCHAR
    )
    """
)
con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", rows)

print("chunks stored:", len(rows))
print(
    con.execute(
        "SELECT file_name, COUNT(*) AS n, AVG(LENGTH(chunk_text)) AS avg_len "
        "FROM chunks GROUP BY file_name"
    ).fetchall()
)
con.close()