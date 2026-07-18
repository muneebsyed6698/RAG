"""
Streamlit RAG app — Groq + Jina Embeddings + Pinecone
Now with an in-app Data Ingestion pipeline (Unstructured -> Jina -> Pinecone)
and a namespace dropdown (pick existing or create new) for both chat and ingestion.

Run with:
    streamlit run app.py

Requires a .env file (same folder) with:
    GROQ_API_KEY=...
    JINA_API_KEY=...
    PINECONE_API_KEY=...
    UNSTRUCTURED_API_KEY=...   # only needed for the "Data Ingestion" tab
"""

import os
import base64
import json
import zlib
import datetime
from io import BytesIO

import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

import unstructured_client
from unstructured_client.models.errors import SDKError

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
load_dotenv()

EMBED_DIM = 768
INDEX_NAME = "rag-with-images"
DEFAULT_NAMESPACE = "my-namespace"
GROQ_MODEL = "llama-3.3-70b-versatile"
CREATE_NEW_LABEL = "➕ Create new namespace..."

st.set_page_config(page_title="RAG Chat (Groq + Pinecone + Jina)", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner=False)
def get_clients():
    """Initialize Groq + Pinecone (+ Unstructured, if configured) clients once and cache them."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    jina_api_key = os.getenv("JINA_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")  # optional, only needed for ingestion

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

    unstructured_client_obj = None
    if unstructured_api_key:
        unstructured_client_obj = unstructured_client.UnstructuredClient(
            api_key_auth=unstructured_api_key
        )

    return groq_client, index, jina_api_key, unstructured_client_obj


groq_client, index, JINA_API_KEY, unstructured_client_obj = get_clients()


# --------------------------------------------------------------------------
# Namespace helpers
# --------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def get_existing_namespaces():
    """List namespaces currently present in the Pinecone index."""
    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {}) or {}
        return sorted(namespaces.keys())
    except Exception:
        return []


def namespace_selector(key_prefix, default=DEFAULT_NAMESPACE, help_text=None):
    """Renders a dropdown of existing namespaces plus a 'create new' option.
    Returns the chosen/typed namespace string."""
    existing = get_existing_namespaces()
    options = existing + [CREATE_NEW_LABEL]

    if default in existing:
        default_index = existing.index(default)
    else:
        default_index = len(options) - 1  # falls back to "create new"

    selected = st.selectbox(
        "Namespace",
        options,
        index=default_index,
        key=f"{key_prefix}_ns_select",
        help=help_text,
    )

    if selected == CREATE_NEW_LABEL:
        namespace = st.text_input(
            "New namespace name",
            value="" if existing else default,
            key=f"{key_prefix}_ns_new",
            placeholder="e.g. product-manuals",
        )
    else:
        namespace = selected

    return namespace.strip()


# --------------------------------------------------------------------------
# Core RAG functions (chat / retrieval side)
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
        context_parts.append(f"[Page {page} | Relevance: {score:.4f}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)
    return context, results.matches


def extract_images(matches):
    """Pulls base64 image strings out of each match's metadata."""
    images = []
    for i, match in enumerate(matches, start=1):
        page = match.metadata.get("page_number", "N/A")
        image_list = match.metadata.get("image", []) or []
        for img_b64 in image_list:
            if img_b64:
                images.append({"i": i, "page": page, "b64": img_b64})
    return images


def decode_base64_image(b64_string):
    """Decodes a base64 string (optionally with a data-URI prefix) to raw bytes."""
    if "," in b64_string and b64_string.strip().lower().startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]
    return base64.b64decode(b64_string)


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


def render_images_expander(images):
    """Renders the 'Images' expander for a list of image dicts (from extract_images)."""
    with st.expander(f"Images ({len(images)})" if images else "Images"):
        if not images:
            st.write("No images found for this answer.")
            return
        for img in images:
            try:
                img_bytes = decode_base64_image(img["b64"])
                st.image(img_bytes, caption=f"Chunk {img['i']} · Page {img['page']}")
            except Exception as e:
                st.warning(f"Could not display image from chunk {img['i']} (Page {img['page']}): {e}")


# --------------------------------------------------------------------------
# Data ingestion pipeline (Unstructured -> chunk -> compress images -> embed -> upsert)
# --------------------------------------------------------------------------
def decode_orig_elements(orig_elements_b64: str) -> list:
    """Reverse the gzip+base64 packing Unstructured applies to orig_elements."""
    decoded = base64.b64decode(orig_elements_b64)
    decompressed = zlib.decompress(decoded)
    return json.loads(decompressed)


def compress_image_b64_to_target(b64_string, target_kb=5, max_size=(512, 512), start_quality=85):
    if not b64_string:
        return b64_string

    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    image_bytes = base64.b64decode(b64_string)
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)

    quality = start_quality
    best_bytes = None

    while quality >= 10:
        out = BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        data = out.getvalue()
        best_bytes = data
        if len(data) <= target_kb * 1024:
            break
        quality -= 5

    return base64.b64encode(best_bytes).decode("utf-8")


def partition_pdf_bytes(
    file_bytes,
    file_name,
    strategy="hi_res",
    max_characters=1500,
    new_after_n_chars=1000,
    combine_under_n_chars=500,
):
    """Calls the Unstructured API and returns the raw element chunks."""
    if unstructured_client_obj is None:
        raise RuntimeError(
            "UNSTRUCTURED_API_KEY is not set. Add it to your .env file to use data ingestion."
        )

    req = {
        "partition_parameters": {
            "files": {
                "content": file_bytes,
                "file_name": file_name,
            },
            "strategy": strategy,
            "chunking_strategy": "by_title",
            "max_characters": max_characters,
            "new_after_n_chars": new_after_n_chars,
            "combine_under_n_chars": combine_under_n_chars,
            "split_pdf_page": True,
            "split_pdf_allow_failed": True,
            "split_pdf_concurrency_level": 15,
            "extract_image_block_types": ["Image", "Table"],
            "extract_image_block_to_payload": True,
            "include_orig_elements": True,
        }
    }

    res = unstructured_client_obj.general.partition(request=req)
    return res.elements


def build_enriched_chunks(raw_chunks, target_kb=10, max_size=(512, 512)):
    """Turns raw Unstructured chunks into {text, page_number, images} dicts,
    with images extracted from orig_elements and compressed."""
    enriched_chunks = []
    for chunk in raw_chunks:
        orig_b64 = chunk.get("metadata", {}).get("orig_elements")
        images = []

        if orig_b64:
            orig_elements = decode_orig_elements(orig_b64)
            for el in orig_elements:
                el_type = el.get("type")
                img_b64 = el.get("metadata", {}).get("image_base64")
                if el_type in ("Image", "Table") and img_b64:
                    compressed = compress_image_b64_to_target(
                        img_b64, target_kb=target_kb, max_size=max_size
                    )
                    images.append(compressed)

        enriched_chunks.append(
            {
                "element_id": chunk.get("element_id"),
                "text": chunk.get("text", ""),
                "page_number": chunk.get("metadata", {}).get("page_number"),
                "images": images,
            }
        )
    return enriched_chunks


def embed_texts_in_batches(texts, batch_size=64, progress_cb=None):
    """Embeds a list of texts via Jina in batches, preserving order."""
    all_embeddings = [None] * len(texts)
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]

        result = jina_embedding_model(batch)
        data = sorted(result.json()["data"], key=lambda x: x["index"])
        for item in data:
            all_embeddings[start + item["index"]] = item["embedding"]

        if progress_cb:
            progress_cb((b + 1) / n_batches, f"Embedding batch {b + 1}/{n_batches}")

    return all_embeddings


def upsert_chunks(chunks, embeddings, namespace, id_prefix="doc", batch_size=100, progress_cb=None):
    """Upserts enriched chunks + embeddings into Pinecone, batched."""
    vectors = [
        {
            "id": f"{id_prefix}_{i}",
            "values": embeddings[i],
            "metadata": {
                "text": chunks[i]["text"],
                "page_number": chunks[i]["page_number"] if chunks[i]["page_number"] is not None else "N/A",
                "image": chunks[i]["images"],
            },
        }
        for i in range(len(chunks))
    ]

    n_batches = (len(vectors) + batch_size - 1) // batch_size
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, len(vectors))
        index.upsert(vectors=vectors[start:end], namespace=namespace)
        if progress_cb:
            progress_cb((b + 1) / n_batches, f"Upserting batch {b + 1}/{n_batches}")

    return len(vectors)


def save_enriched_chunks_to_json(chunks, embeddings, file_name, output_folder=None):
    """Saves enriched chunks with embeddings to a JSON file.
    Returns the path where chunks were saved."""
    if output_folder is None:
        # Default to parent directory of current file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_folder = os.path.dirname(script_dir)  # Go up one level to 'chunking'

    os.makedirs(output_folder, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(file_name)[0]
    output_file = os.path.join(
        output_folder, 
        f"{base_name}_chunks_{timestamp}.json"
    )

    # Prepare chunks with embeddings for serialization
    chunks_with_embeddings = [
        {
            "element_id": chunks[i].get("element_id"),
            "text": chunks[i]["text"],
            "page_number": chunks[i]["page_number"],
            "images": chunks[i]["images"],
            "embedding": embeddings[i] if i < len(embeddings) else None,
        }
        for i in range(len(chunks))
    ]

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_with_embeddings, f, indent=2)

    return output_file


def run_ingestion_pipeline(
    file_bytes,
    file_name,
    namespace,
    strategy="hi_res",
    max_characters=1500,
    new_after_n_chars=1000,
    combine_under_n_chars=500,
    target_kb=10,
    max_image_size=(512, 512),
    id_prefix=None,
):
    """End-to-end: partition -> enrich/compress -> embed -> upsert. Returns a summary dict."""
    status = st.status("Running ingestion pipeline...", expanded=True)

    status.write(f"📄 Partitioning `{file_name}` with Unstructured (strategy=`{strategy}`)...")
    raw_chunks = partition_pdf_bytes(
        file_bytes,
        file_name,
        strategy=strategy,
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_under_n_chars=combine_under_n_chars,
    )
    status.write(f"✅ Got {len(raw_chunks)} chunks from Unstructured.")

    status.write("🖼️ Extracting and compressing images...")
    enriched_chunks = build_enriched_chunks(raw_chunks, target_kb=target_kb, max_size=max_image_size)
    n_with_images = sum(1 for c in enriched_chunks if c["images"])
    status.write(f"✅ {n_with_images} chunk(s) contain image data.")

    status.write("🔢 Embedding chunk text with Jina...")
    progress_bar = st.progress(0.0)

    def embed_progress(frac, label):
        progress_bar.progress(frac, text=label)

    texts = [c["text"] for c in enriched_chunks]
    embeddings = embed_texts_in_batches(texts, progress_cb=embed_progress)
    status.write("✅ Embeddings complete.")

    status.write(f"⬆️ Upserting vectors into Pinecone namespace `{namespace}`...")
    prefix = id_prefix or os.path.splitext(file_name)[0].replace(" ", "_")
    upsert_progress = st.progress(0.0)

    def upsert_progress_cb(frac, label):
        upsert_progress.progress(frac, text=label)

    n_upserted = upsert_chunks(
        enriched_chunks, embeddings, namespace, id_prefix=prefix, progress_cb=upsert_progress_cb
    )
    status.write(f"✅ Upserted {n_upserted} vectors.")

    status.write("💾 Saving chunks to JSON file...")
    chunks_file = save_enriched_chunks_to_json(enriched_chunks, embeddings, file_name)
    status.write(f"✅ Chunks saved to `{os.path.basename(chunks_file)}`")

    status.update(label="Ingestion complete ✅", state="complete", expanded=False)
    get_existing_namespaces.clear()  # refresh the namespace dropdown

    return {
        "file_name": file_name,
        "n_chunks": len(enriched_chunks),
        "n_with_images": n_with_images,
        "n_upserted": n_upserted,
        "namespace": namespace,
        "chunks_file": chunks_file,
    }


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("💬 RAG Chat — Groq + Pinecone + Jina")
st.caption(f"Index: `{INDEX_NAME}` · Model: `{GROQ_MODEL}` · Embeddings: `jina-embeddings-v2-base-en`")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for namespace and settings
if "namespace" not in st.session_state:
    st.session_state.namespace = DEFAULT_NAMESPACE
if "top_k" not in st.session_state:
    st.session_state.top_k = 3
if "show_context" not in st.session_state:
    st.session_state.show_context = True

# Initialize session state for active tab tracking
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"  # default to chat tab

# Use st.radio as a tab selector at the top - this properly persists state
# and allows us to conditionally render the chat input
tab_options = ["💬 Chat", "📥 Data Ingestion"]
tab_mapping = {"💬 Chat": "chat", "📥 Data Ingestion": "ingest"}

# Create columns for the tab selector to make it look like tabs
col1, col2, col3 = st.columns([1, 1, 8])
with col1:
    chat_selected = st.button(
        "💬 Chat", 
        key="btn_chat_tab",
        type="primary" if st.session_state.active_tab == "chat" else "secondary",
        use_container_width=True
    )
with col2:
    ingest_selected = st.button(
        "📥 Data Ingestion",
        key="btn_ingest_tab", 
        type="primary" if st.session_state.active_tab == "ingest" else "secondary",
        use_container_width=True
    )

# Update active tab based on button clicks
if chat_selected:
    st.session_state.active_tab = "chat"
    st.rerun()
if ingest_selected:
    st.session_state.active_tab = "ingest"
    st.rerun()

# Store the boolean for which tab is active
is_chat_tab_active = st.session_state.active_tab == "chat"

st.divider()

# ---------------------------------------------------------------- Chat tab content
if is_chat_tab_active:
    with st.sidebar:
        st.header("Settings")
        st.session_state.namespace = namespace_selector(
            "chat", default=DEFAULT_NAMESPACE, help_text="Which Pinecone namespace to search."
        )
        st.session_state.top_k = st.slider("Top K chunks to retrieve", min_value=1, max_value=10, value=3)
        st.session_state.show_context = st.checkbox("Show retrieved context", value=True)
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    # Create a container for chat messages that will scroll
    chat_container = st.container()

    # Display all messages in the chat container
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("Sources"):
                        for s in msg["sources"]:
                            st.write(f"Chunk {s['i']}: Page {s['page']} (score: {s['score']:.4f})")
                if msg["role"] == "assistant" and msg.get("context") and st.session_state.show_context:
                    with st.expander("Retrieved context"):
                        st.text(msg["context"])
                if msg["role"] == "assistant":
                    render_images_expander(msg.get("images", []))

# ---------------------------------------------------------------- Ingestion tab content
else:
    st.subheader("Ingest a PDF into Pinecone")
    st.caption(
        "Uploads a PDF, partitions & chunks it with Unstructured, extracts + compresses images, "
        "embeds the text with Jina, and upserts everything into the namespace you choose below."
    )

    if unstructured_client_obj is None:
        st.warning(
            "`UNSTRUCTURED_API_KEY` is not set in your `.env` file. Add it to enable data ingestion.",
            icon="⚠️",
        )

    ingest_namespace = namespace_selector(
        "ingest",
        default=DEFAULT_NAMESPACE,
        help_text="Where the ingested chunks will be stored. Pick an existing namespace or type a new one.",
    )

    uploaded_file = st.file_uploader("PDF document", type=["pdf"])

    with st.expander("Advanced chunking / extraction settings"):
        col1, col2 = st.columns(2)
        with col1:
            strategy = st.selectbox("Partition strategy", ["hi_res", "fast", "auto"], index=0)
            max_characters = st.number_input("max_characters", value=1500, min_value=100, step=100)
            new_after_n_chars = st.number_input("new_after_n_chars", value=1000, min_value=100, step=100)
            combine_under_n_chars = st.number_input("combine_under_n_chars", value=500, min_value=0, step=50)
        with col2:
            target_kb = st.number_input("Target image size (KB)", value=10, min_value=1, step=1)
            max_img_dim = st.number_input("Max image dimension (px)", value=512, min_value=64, step=64)

    disabled = uploaded_file is None or not ingest_namespace or unstructured_client_obj is None

    if st.button("🚀 Start Ingestion", disabled=disabled, type="primary"):
        try:
            summary = run_ingestion_pipeline(
                uploaded_file.getvalue(),
                uploaded_file.name,
                ingest_namespace,
                strategy=strategy,
                max_characters=max_characters,
                new_after_n_chars=new_after_n_chars,
                combine_under_n_chars=combine_under_n_chars,
                target_kb=target_kb,
                max_image_size=(max_img_dim, max_img_dim),
            )
            st.success(
                f"Ingested **{summary['file_name']}** into namespace `{summary['namespace']}`: "
                f"{summary['n_upserted']} chunks upserted "
                f"({summary['n_with_images']} contained images). "
                f"Chunks saved to: `{os.path.basename(summary['chunks_file'])}`"
            )
        except SDKError as e:
            st.error(f"Unstructured API error: {e.message}")
        except Exception as e:
            st.error(f"Something went wrong during ingestion: {e}")

    if uploaded_file is None:
        st.info("Upload a PDF above to get started.")

# --------------------------------------------------------------------------
# Chat input - ONLY VISIBLE WHEN CHAT TAB IS ACTIVE
# --------------------------------------------------------------------------
if is_chat_tab_active:
    query = st.chat_input("Ask a question about your documents...")

    if query:
        if not st.session_state.namespace:
            st.warning("Please select or create a namespace in the sidebar before asking a question.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    answer, context, matches = generate_answer(
                        query, 
                        top_k=st.session_state.top_k, 
                        namespace=st.session_state.namespace
                    )
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

            if st.session_state.show_context:
                with st.expander("Retrieved context"):
                    st.text(context)

            images = extract_images(matches)
            render_images_expander(images)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "context": context,
                "images": images,
            }
        )

        st.rerun()