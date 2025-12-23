"""

 RAG (Retrieval-Augmented Generation)

-Κρατάμε τα έγγραφα (TXT/PDF) τοπικά τα μετατρέπουμε σε embeddings
  και τα αποθηκεύουμε σε vectorstore (FAISS)
- Σε κάθε ερώτηση:
  1) κάνουμε similarity search πάνω στα embeddings
  2) παίρνουμε τα πιο σχετικά αποσπάσματα (top-k)
  3) τα δίνουμε στο LLM ως CONTEXT, ώστε να απαντήσει βασισμένο σε αυτά

! Τα PDF πρέπει να είναι text-based (για scanned PDF θα ήθελε OCR)
"""

import os
import hashlib
from typing import List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Φάκελος που αποθηκεύουμε το FAISS index
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_db")

# Φάκελος με τα δεδομένα (TXT/PDF) που “τρέφουν” το RAG
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Αρχείο “σφραγίδα” για να ξέρουμε αν άλλαξαν τα δεδομένα
FINGERPRINT_PATH = os.path.join(DB_PATH, "_data_fingerprint.txt")


def _compute_data_fingerprint(data_dir: str) -> str:
    """
    Δημιουργεί ένα fingerprint (hash) για το περιεχόμενο του data ώστε να ξέρουμε
    αν προστέθηκαν/αφαιρέθηκαν/αλλάχτηκαν αρχεία.
    filename + size + mtime.
    """
    if not os.path.exists(data_dir):
        return ""

    items = []
    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stat = os.stat(fpath)
        items.append(f"{fname}|{stat.st_size}|{int(stat.st_mtime)}")

    raw = "\n".join(items).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_documents() -> List:
    """
    Φορτώνει όλα τα έγγραφα από data/:
    - .txt μέσω TextLoader
    - .pdf μέσω PyPDFLoader (text-based)

    Επιστρέφει λίστα από LangChain Document objects.
    """
    docs = []

    # Αν δεν υπάρχει ο DATA_DIR, τον φτιάχνουμε (ώστε να είναι πάντα “έτοιμος”)
    os.makedirs(DATA_DIR, exist_ok=True)

    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)

        if fname.lower().endswith(".txt"):
            docs.extend(TextLoader(fpath, encoding="utf-8").load())

        elif fname.lower().endswith(".pdf"):
            #για scanned PDF, το κείμενο θα είναι φτωχό/κενό χωρίς OCR
            docs.extend(PyPDFLoader(fpath).load())

    return docs


def build_or_load_vectorstore():
    """
    Δημιουργεί ή φορτώνει τον vector store (FAISS).
    - Αν υπάρχει ήδη vector_db/ και το fingerprint ταιριάζει με τα αρχεία στο data,
      τότε φορτώνουμε. Διαφορετικά κάνουμε rebuild index

    """

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Υπολογίζουμε fingerprint των data
    current_fp = _compute_data_fingerprint(DATA_DIR)

    # Αν υπάρχει DB και fingerprint αρχείο, ελέγχουμε αν χρειάζεται rebuild
    if os.path.exists(DB_PATH) and os.path.exists(FINGERPRINT_PATH):
        try:
            with open(FINGERPRINT_PATH, "r", encoding="utf-8") as f:
                saved_fp = f.read().strip()
        except OSError:
            saved_fp = ""

        # Αν δεν άλλαξαν τα αρχεία φορτώνουμε το έτοιμο index
        if saved_fp and saved_fp == current_fp:
            return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    # Αλλιώς rebuild από την αρχή
    docs = _load_documents()

    # Αν δεν υπάρχουν έγγραφα φτιάχνουμε κενό index
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs) if docs else []

    vs = FAISS.from_documents(chunks, embeddings) if chunks else FAISS.from_texts([""], embeddings)

    # Αποθήκευση index + fingerprint
    os.makedirs(DB_PATH, exist_ok=True)
    vs.save_local(DB_PATH)

    try:
        with open(FINGERPRINT_PATH, "w", encoding="utf-8") as f:
            f.write(current_fp)
    except OSError:
        # Αν αποτύχει να γράψει το fingerprint να μην είναι fatal.
        pass

    return vs


# Cache στη μνήμη
_vectorstore = None


def retrieve_context(query: str, k: int = 4) -> str:
    """
    Επιστρέφει context string με τα k πιο σχετικά αποσπάσματα από τον vectorstore

    - query: ερώτημα χρήστη
    - k: πόσα σχετικά chunks θα επιστραφούν

    Output format:
    [1] source (page X)
    <text...>

    [2] source ...
    <text...>
    """
    global _vectorstore

    # δημιουργούμε-φορτώνουμε μόνο στην πρώτη κλήση
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
