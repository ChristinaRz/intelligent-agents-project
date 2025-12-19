# Intelligent Agents Project (UniPi)

Multi-agent system in Python featuring:
- Agentic workflow: Planner → Critic → Executor
- RAG (Retrieval-Augmented Generation) using LangChain + FAISS
- Support for TXT and text-based PDF sources (via data/ folder)
- Simple function calling via a custom tool (estimate_study_time)
- CLI interaction (input/output)

## Project Structure
- `src/main.py`: CLI entry point
- `src/agents.py`: agents + workflow + function calling
- `src/rag.py`: document loading, chunking, embeddings, vector store, retrieval
- `data/`: user documents (TXT/PDF) for RAG (not committed if you prefer)

## Setup & Run
```bash
python -m venv .venv
# activate venv (Windows)
.venv\Scripts\activate

pip install -r requirements.txt

# Create .env (not committed)
# OPENROUTER_API_KEY=...
# OPENROUTER_MODEL=... (use a free model id if needed)

python src/main.py
