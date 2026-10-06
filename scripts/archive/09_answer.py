import os
import time

import duckdb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_PATH = os.path.join("data", "kwsp.duckdb")
EMBED_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 3

# best first; if one fails, the next one is tried
GEMINI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

embedder = SentenceTransformer(EMBED_MODEL)
con = duckdb.connect(DB_PATH, read_only=True)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def retrieve(question):
    vector = embedder.encode("query: " + question, normalize_embeddings=True).tolist()
    hits = con.execute(
        f"""
        SELECT c.chunk_id, c.file_name, c.page_number, c.chunk_text,
               list_cosine_similarity(e.embedding, ?::FLOAT[]) AS score
        FROM chunk_embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        ORDER BY score DESC
        LIMIT {TOP_K}
        """,
        [vector],
    ).fetchall()

    results = []
    for chunk_id, file_name, page_number, chunk_text, score in hits:
        neighbours = con.execute(
            """
            SELECT chunk_text
            FROM chunks
            WHERE file_name = ?
              AND chunk_id BETWEEN ? AND ?
            ORDER BY chunk_id
            """,
            [file_name, chunk_id - 1, chunk_id + 1],
        ).fetchall()
        full_text = " ".join(row[0] for row in neighbours)
        results.append((chunk_id, file_name, page_number, full_text, score))
    return results


def where_text(page_number):
    if page_number:
        return f"page {page_number}"
    return "no page number"


def build_prompt(question, results):
    sources = ""
    for number, row in enumerate(results, start=1):
        chunk_id, file_name, page_number, chunk_text, score = row
        sources += f"[{number}] ({file_name}, {where_text(page_number)})\n{chunk_text}\n\n"

    return (
        "You answer questions about the Malaysian EPF (KWSP) Act using only the sources below.\n"
        "Rules:\n"
        "- Use only the information in the sources. Do not use outside knowledge.\n"
        "- If the sources do not contain the answer, say you could not find it in the documents.\n"
        "- Cite the sources you used, like [1] or [2].\n"
        "- Answer in the same language as the question.\n"
        "- Keep the answer short.\n\n"
        f"Sources:\n{sources}"
        f"Question: {question}\n"
    )


def ask_gemini(prompt):
    for model_name in GEMINI_MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text, model_name
            except Exception as error:
                print(f"  ({model_name} failed: {str(error)[:60]})")
                time.sleep(3)
    return None, None


while True:
    question = input("\nAsk a question (or press Enter to quit): ").strip()
    if question == "":
        break

    results = retrieve(question)
    prompt = build_prompt(question, results)
    answer, used_model = ask_gemini(prompt)

    if answer is None:
        print("\nGemini is unavailable right now. Best matching passages instead:")
        for number, row in enumerate(results, start=1):
            print(f"[{number}] {row[3][:300]}")
    else:
        print("\n" + answer)
        print(f"\n(model: {used_model})")

    print("\nSources:")
    for number, row in enumerate(results, start=1):
        chunk_id, file_name, page_number, chunk_text, score = row
        print(f"[{number}] {file_name}, {where_text(page_number)}, chunk {chunk_id}, score {score:.3f}")

con.close()