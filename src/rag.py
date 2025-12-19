import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def build_or_load_vectorstore():
    # Local embeddings (δεν χρειάζεται άλλο API key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    docs = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if fname.lower().endswith(".txt"):
            docs.extend(TextLoader(fpath, encoding="utf-8").load())
        elif fname.lower().endswith(".pdf"):
            # ΜΟΝΟ text-based pdf. Αν είναι scanned, θα βγάλει φτωχό κείμενο.
            docs.extend(PyPDFLoader(fpath).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(DB_PATH)
    return vs

_vectorstore = None

def retrieve_context(query: str, k: int = 4) -> str:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_or_load_vectorstore()

    results = _vectorstore.similarity_search(query, k=k)
    if not results:
        return ""

    context_parts = []
    for i, doc in enumerate(results, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        loc = f"{src}" + (f" (page {page})" if page is not None else "")
        context_parts.append(f"[{i}] {loc}\n{doc.page_content}")

    return "\n\n".join(context_parts)
