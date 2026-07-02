# BrightBox Voice Agent

A phone-callable voice AI agent that answers customer questions via Twilio Voice and RAG over a local knowledge base.

## Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Telephony | Twilio Voice | Required by spec; free trial with verified-number calling |
| Voice pipeline | Pipecat | Native Twilio Media Streams support; built-in VAD/interruption handling |
| STT | Deepgram | Low-latency streaming STT with Pipecat integration |
| LLM | Gemini, OpenAI, Groq fallback | Gemini is primary; OpenAI and Groq are fallback providers |
| TTS | Cartesia | Low-latency streaming TTS, Pipecat-native |
| Vector DB | ChromaDB (local) | Self-hosted, zero infrastructure, file-based persistence |
| Server | FastAPI + uvicorn | Handles Twilio webhook + WebSocket media stream |
| Tunneling (dev) | ngrok | Exposes local server to Twilio |

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in API keys. 
   The example file includes Twilio, ngrok, Deepgram, Cartesia, LLM provider, and Gemini embedding settings used by this project.
   For the chat LLM, set `LLM_PROVIDER=auto` to use the fallback chain `Gemini -> OpenAI -> Groq`, or set it to a specific provider (`gemini`, `openai`, or `groq`) to force one model.

3. Add BrightBox knowledge-base markdown files under `rag/kb/`, then ingest them. This project uses Gemini embeddings for RAG, so set `GEMINI_API_KEY` in `.env` before running:
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

## Twilio Trial Setup Notes

If using a Twilio trial account:
- Twilio provides a trial phone number
- Calls to the Twilio number must be made from a verified phone number
- Add your personal phone number under Twilio Verified Caller IDs before testing
- Set the Twilio phone number Voice webhook to:
  `https://<ngrok-domain>/incoming-call`
- Use HTTP method `POST`

## End-to-End Call Flow

When the setup is correct and a verified phone calls the Twilio number:

- Twilio sends a `POST /incoming-call` request to the FastAPI server
- The server returns TwiML that connects the call to the `/ws` media stream
- Caller audio is streamed over WebSocket to the Pipecat pipeline
- The agent retrieves relevant BrightBox knowledge-base chunks from ChromaDB
- The LLM generates a response and Cartesia streams audio back to the caller
