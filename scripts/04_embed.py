import os

import duckdb
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join("data", "kwsp.duckdb")
MODEL_NAME = "intfloat/multilingual-e5-small"

con = duckdb.connect(DB_PATH)
rows = con.execute(
    "SELECT chunk_id, chunk_text FROM chunks ORDER BY chunk_id"
).fetchall()
print("chunks to embed:", len(rows))

model = SentenceTransformer(MODEL_NAME)

ids = [row[0] for row in rows]
texts = ["passage: " + row[1] for row in rows]

vectors = model.encode(
    texts,
    batch_size=16,
    normalize_embeddings=True,
    show_progress_bar=True,
)

con.execute("DROP TABLE IF EXISTS chunk_embeddings")
con.execute("CREATE TABLE chunk_embeddings (chunk_id INTEGER, embedding FLOAT[])")

data = [(chunk_id, vector.tolist()) for chunk_id, vector in zip(ids, vectors)]
con.executemany("INSERT INTO chunk_embeddings VALUES (?, ?)", data)

print("embeddings stored:", len(data))
print("vector length:", len(data[0][1]))
con.close()