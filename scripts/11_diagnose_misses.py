import csv
import os

import duckdb
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join("data", "kwsp.duckdb")
QUESTIONS_PATH = os.path.join("eval", "questions.csv")
RESULTS_PATH = os.path.join("eval", "results.csv")
EMBED_MODEL = "intfloat/multilingual-e5-small"

embedder = SentenceTransformer(EMBED_MODEL)
con = duckdb.connect(DB_PATH, read_only=True)

with open(QUESTIONS_PATH, newline="", encoding="utf-8") as f:
    questions = {row["id"]: row for row in csv.DictReader(f)}

with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
    results = list(csv.DictReader(f))

missed_ids = [r["id"] for r in results if r["with_neighbours_hit"] == "False"]

for question_id in missed_ids:
    row = questions[question_id]
    phrases = [" ".join(p.lower().split()) for p in row["expected"].split("|")]

    vector = embedder.encode("query: " + row["question"], normalize_embeddings=True).tolist()
    ranked = con.execute(
        """
        SELECT c.chunk_id, c.file_name, c.chunk_text,
               list_cosine_similarity(e.embedding, ?::FLOAT[]) AS score
        FROM chunk_embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        ORDER BY score DESC
        """,
        [vector],
    ).fetchall()

    best_rank = None
    for rank, (chunk_id, file_name, chunk_text, score) in enumerate(ranked, start=1):
        text = " ".join(chunk_text.lower().split())
        if any(p in text for p in phrases):
            best_rank = rank
            break

    print("=" * 70)
    print(f"id {question_id} ({row['language']}): {row['question']}")
    print(f"right chunk first appears at rank {best_rank} of {len(ranked)}")
    print("top 3 returned instead:")
    for chunk_id, file_name, chunk_text, score in ranked[:3]:
        print(f"  chunk {chunk_id} | {file_name} | {score:.3f} | {chunk_text[:110]}")

con.close()