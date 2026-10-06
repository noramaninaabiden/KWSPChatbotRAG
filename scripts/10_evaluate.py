import csv
import os

import duckdb
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join("data", "kwsp.duckdb")
QUESTIONS_PATH = os.path.join("eval", "questions.csv")
RESULTS_PATH = os.path.join("eval", "results.csv")
EMBED_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 5
NEIGHBOUR_WINDOW = 2

embedder = SentenceTransformer(EMBED_MODEL)
con = duckdb.connect(DB_PATH, read_only=True)


def normalise(text):
    return " ".join(text.lower().split())


def phrase_in_corpus(phrases):
    for phrase in phrases:
        count = con.execute(
            "SELECT COUNT(*) FROM chunks WHERE chunk_text ILIKE ?",
            ["%" + phrase + "%"],
        ).fetchone()[0]
        if count > 0:
            return True
    return False


def retrieve(question):
    vector = embedder.encode("query: " + question, normalize_embeddings=True).tolist()
    hits = con.execute(
        f"""
        SELECT c.chunk_id, c.file_name, c.location, c.chunk_text
        FROM chunk_embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        ORDER BY list_cosine_similarity(e.embedding, ?::FLOAT[]) DESC
        LIMIT {TOP_K}
        """,
        [vector],
    ).fetchall()

    results = []
    for chunk_id, file_name, location, chunk_text in hits:
        neighbours = con.execute(
            """
            SELECT chunk_text
            FROM chunks
            WHERE file_name = ?
              AND location = ?
              AND chunk_id BETWEEN ? AND ?
            ORDER BY chunk_id
            """,
            [file_name, location, chunk_id - NEIGHBOUR_WINDOW, chunk_id + NEIGHBOUR_WINDOW],
        ).fetchall()
        with_neighbours = " ".join(row[0] for row in neighbours)
        results.append((chunk_id, chunk_text, with_neighbours))
    return results


with open(QUESTIONS_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

tally = {}
detail = []

for row in rows:
    phrases = [normalise(p) for p in row["expected"].split("|") if p.strip()]

    if not phrase_in_corpus(phrases):
        print(f"SKIPPED id {row['id']}: expected text not found in any chunk, check spelling")
        continue

    results = retrieve(row["question"])
    top3_hit = any(p in normalise(r[1]) for r in results for p in phrases)
    neighbour_hit = any(p in normalise(r[2]) for r in results for p in phrases)

    language = row["language"]
    if language not in tally:
        tally[language] = [0, 0, 0]
    tally[language][0] += 1
    tally[language][1] += int(top3_hit)
    tally[language][2] += int(neighbour_hit)

    chunk_ids = " ".join(str(r[0]) for r in results)
    detail.append([row["id"], language, row["question"], top3_hit, neighbour_hit, chunk_ids])

    if not neighbour_hit:
        print(f"MISS id {row['id']} ({language}): {row['question']}")

with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "language", "question", "top3_hit", "with_neighbours_hit", "chunk_ids"])
    writer.writerows(detail)

print(f"\nRESULTS with TOP_K={TOP_K}, NEIGHBOUR_WINDOW={NEIGHBOUR_WINDOW}")
print(f"{'language':10} {'questions':>9} {'chunks only':>11} {'with neighbours':>16}")
total = [0, 0, 0]
for language, counts in tally.items():
    print(f"{language:10} {counts[0]:>9} {counts[1]:>11} {counts[2]:>16}")
    for i in range(3):
        total[i] += counts[i]
print(f"{'all':10} {total[0]:>9} {total[1]:>11} {total[2]:>16}")