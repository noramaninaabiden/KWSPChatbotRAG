import os
import time

import duckdb
import streamlit as st
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

NOT_FOUND_PHRASES = ["could not find", "tidak dapat", "tidak ditemui", "tidak dijumpai"]


def get_api_key():
    if "GEMINI_API_KEY" in os.environ:
        return os.environ["GEMINI_API_KEY"]
    return st.secrets["GEMINI_API_KEY"]


@st.cache_resource
def load_resources():
    embedder = SentenceTransformer(EMBED_MODEL)
    con = duckdb.connect(DB_PATH, read_only=True)
    client = genai.Client(api_key=get_api_key())
    return embedder, con, client


embedder, con, client = load_resources()


def retrieve(question):
    cursor = con.cursor()
    vector = embedder.encode("query: " + question, normalize_embeddings=True).tolist()
    hits = cursor.execute(
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
        neighbours = cursor.execute(
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
            except Exception:
                time.sleep(2)
    return None, None


def looks_like_not_found(answer):
    lowered = answer.lower()
    for phrase in NOT_FOUND_PHRASES:
        if phrase in lowered:
            return True
    return False


st.title("KWSP / EPF Act Assistant")
st.caption(
    "Ask about the EPF Act 1991 in English or Bahasa Melayu. "
    "Answers come only from the documents loaded in this demo. "
    "This is a portfolio project, not official advice. "
    "Please don't enter personal information."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.markdown("**" + source["label"] + "**")
                    st.write(source["text"])

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            results = retrieve(question)
            prompt = build_prompt(question, results)
            answer, used_model = ask_gemini(prompt)

        source_items = []
        for number, row in enumerate(results, start=1):
            chunk_id, file_name, page_number, chunk_text, score = row
            label = f"[{number}] {file_name}, {where_text(page_number)} (match {score:.2f})"
            source_items.append({"label": label, "text": chunk_text})

        if answer is None:
            answer = "The AI model is busy right now, so here are the closest passages from the documents instead."
        elif looks_like_not_found(answer):
            source_items = []

        st.markdown(answer)
        if source_items:
            with st.expander("Sources used"):
                for source in source_items:
                    st.markdown("**" + source["label"] + "**")
                    st.write(source["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": source_items}
    )