# Finetune + RAG Quiz Generation System (v1)

This branch contains the **working implementation** of a hybrid quiz generation pipeline that combines:
- Fine-tuned T5 generation
- RAG-style validation/refinement with Ollama
- Gemini primary/secondary generation stages
- A React frontend + Flask backend workflow

> Primary working code lives in `current_working/`.

## Project Structure

- `current_working/backend/app.py` — Flask API for stage-wise and full pipeline execution
- `current_working/frontend/` — React + Vite UI
- `current_working/t5-quiz-finetune/` — tokenizer/config assets for the fine-tuned T5 model (large weights excluded)
- `current_working/.env.example` — environment variable template
- `current_working/requirements.txt` — Python dependencies

## Features

- PDF upload and text extraction
- Configurable chunking for long documents
- Stage-by-stage execution:
  - Extract
  - Chunk
  - Finetune stage
  - RAG stage
  - Gemini primary stage
  - Gemini secondary stage
- End-to-end `/api/pipeline/full` execution
- Frontend controls for model and pipeline parameters

## Tech Stack

- **Backend:** Python, Flask, Flask-CORS, Transformers, PyMuPDF
- **Frontend:** React, Vite
- **LLM Services:** Google Gemini API, Ollama, optional ngrok-hosted T5 endpoint

## Quick Start

### 1) Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama installed and running
- (Optional) ngrok URL for remote T5 generation
- Gemini API key

### 2) Backend Setup

```bash
cd current_working
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

Create env file:

```bash
copy .env.example .env
```

Update `.env` values, especially:
- `GEMINI_API_KEY`
- `OLLAMA_MODEL`
- `NGROK_T5_URL` (if using ngrok endpoint)

Start backend:

```bash
cd backend
python app.py
```

Default backend URL: `http://127.0.0.1:5000`

### 3) Frontend Setup

In a new terminal:

```bash
cd current_working/frontend
npm install
npm run dev
```

Default frontend URL: `http://localhost:5173`

## Ollama Setup

Ensure Ollama is running and a model is available:

```bash
ollama list
ollama run llama2-uncensored
```

If you use a different model, set `OLLAMA_MODEL` in `.env`.

## Environment Variables

See `current_working/.env.example` for all keys.

Important variables:
- `GEMINI_API_KEY`
- `CORS_ORIGINS`
- `GEMINI_MODEL_PRIMARY`
- `GEMINI_MODEL_SECONDARY`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `NGROK_T5_URL`
- `NGROK_T5_ENDPOINT`
- `NGROK_T5_MAX_LENGTH`

## API Endpoints (Backend)

- `GET /api/health`
- `POST /api/stage/extract`
- `POST /api/stage/chunk`
- `POST /api/stage/finetune`
- `POST /api/stage/rag`
- `POST /api/stage/gemini-primary`
- `POST /api/stage/gemini-secondary`
- `POST /api/pipeline/full`

## Notes

- Large model weights and virtual environments are intentionally excluded from Git tracking for GitHub compatibility.
- Keep secrets in `.env` only (do not commit real keys).

## Branch Purpose

This branch (`Finetune+Rag_System-v1`) is maintained as a backup snapshot of the working finetune + RAG pipeline implementation.
