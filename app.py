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
TOP_K = 5
NEIGHBOUR_WINDOW = 2

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
        SELECT c.chunk_id, c.file_name, c.location,
               list_cosine_similarity(e.embedding, ?::FLOAT[]) AS score
        FROM chunk_embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        ORDER BY score DESC
        LIMIT {TOP_K}
        """,
        [vector],
    ).fetchall()

    sources = []
    seen = {}
    for chunk_id, file_name, location, score in hits:
        key = (file_name, location)
        if key not in seen:
            seen[key] = {
                "file_name": file_name,
                "location": location,
                "score": score,
                "chunks": {},
            }
            sources.append(seen[key])

        neighbours = cursor.execute(
            """
            SELECT chunk_id, chunk_text
            FROM chunks
            WHERE file_name = ?
              AND location = ?
              AND chunk_id BETWEEN ? AND ?
            ORDER BY chunk_id
            """,
            [file_name, location, chunk_id - NEIGHBOUR_WINDOW, chunk_id + NEIGHBOUR_WINDOW],
        ).fetchall()
        for neighbour_id, neighbour_text in neighbours:
            seen[key]["chunks"][neighbour_id] = neighbour_text

    for source in sources:
        ids = sorted(source["chunks"])
        source["text"] = "\n".join(source["chunks"][i] for i in ids)
    return sources


def build_prompt(question, sources):
    source_text = ""
    for number, source in enumerate(sources, start=1):
        source_text += f"[{number}] ({source['file_name']}, {source['location']})\n"
        source_text += source["text"] + "\n\n"

    return (
        "You answer questions about the Malaysian EPF (KWSP) Act using only the sources below.\n"
        "Rules:\n"
        "- Use only the information in the sources. Do not use outside knowledge.\n"
        "- If the sources do not contain the answer, say you could not find it in the documents.\n"
        "- Cite the sources you used, like [1] or [2].\n"
        "- Answer in the same language as the question.\n"
        "- Keep the answer short.\n\n"
        f"Sources:\n{source_text}"
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
            sources = retrieve(question)
            prompt = build_prompt(question, sources)
            answer, used_model = ask_gemini(prompt)

        source_items = []
        for number, source in enumerate(sources, start=1):
            label = f"[{number}] {source['location']} ({source['file_name']}, match {source['score']:.2f})"
            source_items.append({"label": label, "text": source["text"]})

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