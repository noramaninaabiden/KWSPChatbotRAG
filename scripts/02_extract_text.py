import os

from pypdf import PdfReader

RAW_DIR = os.path.join("data", "raw")

for file_name in sorted(os.listdir(RAW_DIR)):
    if not file_name.endswith(".pdf"):
        continue

    reader = PdfReader(os.path.join(RAW_DIR, file_name))
    page_count = len(reader.pages)

    total_chars = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        total_chars += len(text)

    print("=" * 60)
    print(file_name, "- pages:", page_count, "- total characters:", total_chars)

    middle_page = reader.pages[page_count // 2]
    sample = middle_page.extract_text() or ""
    print("Sample from the middle page:")
    print(sample[:400])