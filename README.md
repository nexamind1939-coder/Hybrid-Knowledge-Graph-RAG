# Hybrid-Knowledge-Graph-RAG
Hybrid Knowledge Graph RAG using Streamlit, FastAPI, FAISS and OpenAI

## Docker deployment

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Build and start the services:
   ```powershell
   docker compose up --build -d
   ```
3. Open http://localhost:8501 for the Streamlit UI.
4. Open http://localhost:8000/docs for the FastAPI API.

Stop the deployment with:
```powershell
docker compose down
```
