import os

import duckdb
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join("data", "kwsp.duckdb")
MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 3

model = SentenceTransformer(MODEL_NAME)
con = duckdb.connect(DB_PATH, read_only=True)

while True:
    question = input("\nAsk a question (or press Enter to quit): ").strip()
    if question == "":
        break

    vector = model.encode("query: " + question, normalize_embeddings=True).tolist()

    results = con.execute(
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

    for chunk_id, file_name, page_number, chunk_text, score in results:
        print("-" * 60)
        print(f"score {score:.3f} | {file_name} | page {page_number} | chunk {chunk_id}")
        print(chunk_text)

con.close()