# Karl Setup Guide

Complete setup instructions for getting Karl running on your machine.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Step-by-Step Setup

### 1. Clone or Download the Repository

If you're getting this from GitHub:
```bash
git clone <repository-url>
cd Knowledge-Assistant-Rights-Labor
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs the backend, retrieval, embedding, and evaluation dependencies used by KARL.

### 3. Set Up Your API Key (Optional)

**Karl works without an API key**, but you'll get better answers with one.

1. Get a free Gemini API key:
   - Go to https://aistudio.google.com/app/apikey
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the key

2. Create a `.env` file:
   ```bash
   # On Windows (PowerShell)
   Set-Content -Path .env -Value "GEMINI_API_KEY=your_actual_key_here"
   
   # On Linux/Mac
   printf "GEMINI_API_KEY=your_actual_key_here\n" > .env
   ```

3. Confirm `.env` contains:
   ```bash
   GEMINI_API_KEY=your_actual_key_here
   ```

**What happens without a key?**
- ✅ Wage lookups work perfectly
- ✅ Contract search works perfectly  
- ✅ You'll see relevant contract sections
- ⚠️ Answers will be raw chunks instead of synthesized responses

### 4. Process the Contract Data

If the data files don't already exist, run:

```bash
# Parse contract into chunks
python -m backend.ingest.smart_chunker

# Optional: enrich chunks with LLM metadata
python -u -m backend.ingest.enricher --batch-size 15 --delay 1.0

# Build/rebuild vector index
python -m backend.ingest.rebuild_index
```

**Note:** If you see files in `data/chunks/` and `data/wages/`, you can skip this step - the data is already processed!

### 5. Start the Server

```bash
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

If your environment blocks HuggingFace/model downloads, start in offline BM25 mode:

```bash
# Windows (PowerShell)
$env:KARL_HYBRID_VECTOR_WEIGHT="0"
$env:KARL_HYBRID_KEYWORD_WEIGHT="1"
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 6. Open the Frontend

Open `frontend/index.html` in your web browser, or go to `http://127.0.0.1:8000`.

## Troubleshooting

### "Module not found" errors
- Make sure you ran `pip install -r requirements.txt`
- Check that you're using a supported Python version for current dependencies

### Server won't start
- Check if port 8000 is already in use
- Try a different port: `--port 8001`
- Make sure you're in the project directory
- If startup hangs on `huggingface.co`, run with:
  - `KARL_HYBRID_VECTOR_WEIGHT=0`
  - `KARL_HYBRID_KEYWORD_WEIGHT=1`

### "No chunks found" or empty responses
- Run the data processing scripts (step 4)
- Check that `data/chunks/contract_chunks.json` exists

### API key not working
- Make sure your `.env` file is in the project root (same folder as `requirements.txt`)
- Check that the key doesn't have extra spaces or quotes
- Verify the key is valid at https://aistudio.google.com/app/apikey

### Frontend can't connect
- Make sure the server is running (step 5)
- Check that the frontend is trying to connect to `http://127.0.0.1:8000`
- Open browser developer tools (F12) to see error messages

## Testing the Setup

Once everything is running, try asking Karl:

- "What is my overtime rate?"
- "How much vacation do I get?"
- "What are my Weingarten rights?"

If you see answers with citations, everything is working! 🎉

## Next Steps

- Read the main [README.md](README.md) for API documentation
- Check out the evaluation scripts in `backend/evaluate.py`
- Customize prompts in `backend/generation/prompts.py`

## Getting Help

If you run into issues:
1. Check the troubleshooting section above
2. Look at the server logs for error messages
3. Make sure all dependencies are installed correctly
4. Verify your Python version is 3.8+

