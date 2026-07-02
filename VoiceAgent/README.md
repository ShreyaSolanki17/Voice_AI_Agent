# BrightBox Voice Agent

A phone-callable voice AI agent that answers customer questions via Twilio Voice and RAG over a local knowledge base.

## Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Telephony | Twilio Voice | Required by spec; free trial with verified-number calling |
| Voice pipeline | Pipecat | Native Twilio Media Streams support; built-in VAD/interruption handling |
| STT | Deepgram | Low-latency streaming STT with Pipecat integration |
| LLM | OpenAI (gpt-4o-mini) | Fast, cheap, good tool-calling for RAG |
| TTS | Cartesia | Low-latency streaming TTS, Pipecat-native |
| Vector DB | ChromaDB (local) | Self-hosted, zero infrastructure, file-based persistence |
| Server | FastAPI + uvicorn | Handles Twilio webhook + WebSocket media stream |
| Tunneling (dev) | ngrok | Exposes local server to Twilio |

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in API keys

3. Ingest the knowledge base. This project now uses Gemini embeddings for RAG, so set `GEMINI_API_KEY` in `.env`:
   ```
   python -m rag.ingest
   ```

4. Start tunnel and server:
   ```
   ngrok http 8000
   # Set NGROK_DOMAIN to the HTTPS URL (without wss://)
   uvicorn server:app --port 8000
   ```

5. Configure Twilio:
   - Set your Twilio phone number's voice webhook to `https://<ngrok>/incoming-call` (POST)
   - Add `+E164` number to verified callers list

## Directory Structure

```
brightbox-voice-agent/
├── bot.py           # Pipecat pipeline definition
├── server.py        # FastAPI webhook + WebSocket endpoint
├── rag/
│   ├── ingest.py    # KB ingestion into ChromaDB
│   ├── retriever.py # Query ChromaDB for relevant chunks
│   └── kb/          # Markdown knowledge base documents
├── prompts/         # System prompts
└── templates/       # TwiML templates
```

## Known Limitations

- Simple cosine similarity threshold for fallback (tune as needed)
- No auth on webhook endpoint (add for production)
- Single-threaded calls (one customer at a time)
- No call recording/analytics (add for production)

## Production Improvements

- Twilio `<Dial>` to real phone number for human handoff
- Hosted vector DB (Pinecone/Weaviate) at scale
- Call analytics/logging with structured metadata
- Proper telephony error handling and retries
- Streaming barge-in tuning for better interruption handling