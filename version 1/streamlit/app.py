"""
Streamlit RAG app — Groq + Jina Embeddings + Pinecone
Converted from rag_groq.ipynb

Run with:
    streamlit run app.py

Requires a .env file (same folder) with:
    GROQ_API_KEY=...
    JINA_API_KEY=...
    PINECONE_API_KEY=...
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
load_dotenv()

EMBED_DIM = 768
INDEX_NAME = "jina-try-2"
DEFAULT_NAMESPACE = "my-namespace"
GROQ_MODEL = "llama-3.3-70b-versatile"

st.set_page_config(page_title="RAG Chat (Groq + Pinecone + Jina)", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner=False)
def get_clients():
    """Initialize Groq + Pinecone clients once and cache them."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    jina_api_key = os.getenv("JINA_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    missing = [
        name
        for name, val in [
            ("GROQ_API_KEY", groq_api_key),
            ("JINA_API_KEY", jina_api_key),
            ("PINECONE_API_KEY", pinecone_api_key),
        ]
        if not val
    ]
    if missing:
        st.error(
            "Missing environment variable(s): "
            + ", ".join(missing)
            + ". Add them to a `.env` file next to app.py."
        )
        st.stop()

    groq_client = Groq(api_key=groq_api_key)

    pc = Pinecone(api_key=pinecone_api_key)
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(INDEX_NAME)

    return groq_client, index, jina_api_key


groq_client, index, JINA_API_KEY = get_clients()


# --------------------------------------------------------------------------
# Core RAG functions (ported from the notebook)
# --------------------------------------------------------------------------
def jina_embedding_model(input_data):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    data = {
        "model": "jina-embeddings-v2-base-en",
        "task": "separation",
        "input": input_data,
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response


def retrieve_context(query, top_k=3, namespace=DEFAULT_NAMESPACE):
    """Embeds the query, retrieves top_k matches from Pinecone, and formats
    them into a single context string with page numbers."""
    query_embedding = jina_embedding_model([query]).json()["data"][0]["embedding"]

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )

    context_parts = []
    for i, match in enumerate(results.matches, start=1):
        text = match.metadata.get("text", "")
        page = match.metadata.get("page_number", "N/A")
        score = match.score
        context_parts.append(f"[Chunk {i} | Page {page} | Relevance: {score:.4f}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)
    return context, results.matches


def generate_answer(query, top_k=3, namespace=DEFAULT_NAMESPACE, model=GROQ_MODEL):
    """Retrieves relevant context from Pinecone and asks Groq's LLM to answer
    the query using ONLY that context."""
    context, matches = retrieve_context(query, top_k=top_k, namespace=namespace)

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided context. If the answer is not contained in the context, "
        "say you don't have enough information to answer, rather than guessing. "
        "When relevant, mention which page(s) the information came from."
    )

    user_prompt = f"""Context:
{context}

Question: {query}

Answer the question clearly and concisely based on the context above."""

    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content
    return answer, context, matches


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("💬 RAG Chat — Groq + Pinecone + Jina")
st.caption(f"Index: `{INDEX_NAME}` · Model: `{GROQ_MODEL}` · Embeddings: `jina-embeddings-v2-base-en`")

with st.sidebar:
    st.header("Settings")
    namespace = st.text_input("Pinecone namespace", value=DEFAULT_NAMESPACE)
    top_k = st.slider("Top K chunks to retrieve", min_value=1, max_value=10, value=3)
    show_context = st.checkbox("Show retrieved context", value=True)
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"Chunk {s['i']}: Page {s['page']} (score: {s['score']:.4f})")
        if msg["role"] == "assistant" and msg.get("context") and show_context:
            with st.expander("Retrieved context"):
                st.text(msg["context"])

# Chat input
query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                answer, context, matches = generate_answer(query, top_k=top_k, namespace=namespace)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        st.markdown(answer)

        sources = [
            {"i": i, "page": m.metadata.get("page_number", "N/A"), "score": m.score}
            for i, m in enumerate(matches, start=1)
        ]

        with st.expander("Sources"):
            for s in sources:
                st.write(f"Chunk {s['i']}: Page {s['page']} (score: {s['score']:.4f})")

        if show_context:
            with st.expander("Retrieved context"):
                st.text(context)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "context": context}
    )