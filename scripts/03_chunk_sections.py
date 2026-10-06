import os
import re

import duckdb

RAW_DIR = os.path.join("data", "raw")
DB_PATH = os.path.join("data", "kwsp.duckdb")

CHUNK_SIZE = 800
OVERLAP = 150
MIN_BODY_LENGTH = 60

# Part IX (Sections 75 to 86) is about the old 1951 Act, so it is left out
SKIP_SECTIONS = range(75, 87)

JUNK_PHRASES = [
    "Skip to Main Content",
    "Ahli Majikan Korporat BM",
    "Member Employer Corporate EN",
]

END_MARKERS = ["List of Amendments", "Senarai Pindaan"]

SECTION_RE = re.compile(r"^(Section|Seksyen) (\d+[A-Z]?)$")
PART_RE = re.compile(r"^(PART|BAHAGIAN) [IVXLC]+[A-Z]?$")
SCHEDULE_RE = re.compile(r"^([A-Z]+ SCHEDULE|JADUAL [A-Z]+)$")


def read_sections(path):
    with open(path, encoding="utf-8") as f:
        raw_lines = f.read().replace("\r", "").split("\n")

    lines = []
    for line in raw_lines:
        line = line.strip()
        for phrase in JUNK_PHRASES:
            line = line.replace(phrase, "").strip()
        if line != "":
            lines.append(line)

    sections = []
    current = None
    skip_next = False
    i = 0

    while i < len(lines):
        line = lines[i]
        i += 1

        if line in END_MARKERS:
            break

        if skip_next:
            skip_next = False
            continue

        match = SECTION_RE.match(line)
        if match:
            number = int(re.sub(r"\D", "", match.group(2)))
            title = lines[i] if i < len(lines) else ""
            i += 1
            current = {
                "label": match.group(1) + " " + match.group(2),
                "title": title,
                "body": [],
                "skip": number in SKIP_SECTIONS,
            }
            sections.append(current)
            continue

        if PART_RE.match(line):
            current = None
            skip_next = True
            continue

        if SCHEDULE_RE.match(line):
            title = lines[i] if i < len(lines) else ""
            i += 1
            if title.startswith("[") and i < len(lines):
                title = lines[i]
                i += 1
            current = {"label": line.title(), "title": title, "body": [], "skip": False}
            sections.append(current)
            continue

        if current is not None:
            current["body"].append(line)

    return sections


def clean_body(lines):
    text = " ".join(lines)
    text = re.sub(
        r"Note: The latest Adobe Acrobat Reader.*?Call Management Centre \(CMC\) at [\d-]+ Enquiry",
        " ",
        text,
    )
    text = re.sub(r"Nota: Perisian Adobe Acrobat Reader.*?Kemukakan Pertanyaan", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_chunks(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP
        next_space = text.find(" ", start)
        if next_space != -1:
            start = next_space + 1
    return chunks


rows = []
chunk_id = 0

for file_name in sorted(os.listdir(RAW_DIR)):
    if not file_name.endswith(".txt"):
        continue

    language = "bm" if "_bm_" in file_name else "en"

    for section in read_sections(os.path.join(RAW_DIR, file_name)):
        if section["skip"]:
            continue

        body = clean_body(section["body"])
        if len(body) < MIN_BODY_LENGTH:
            continue

        header = section["label"] + " " + section["title"] + ". "
        for piece in make_chunks(body):
            chunk_id += 1
            rows.append((chunk_id, file_name, language, section["label"], header + piece))

con = duckdb.connect(DB_PATH)
con.execute("DROP TABLE IF EXISTS chunks")
con.execute(
    """
    CREATE TABLE chunks (
        chunk_id INTEGER,
        file_name VARCHAR,
        language VARCHAR,
        location VARCHAR,
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