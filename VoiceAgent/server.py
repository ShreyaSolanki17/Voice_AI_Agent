"""FastAPI server for BrightBox voice agent."""

import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from loguru import logger
import uvicorn

# Load .env from the parent directory (VoiceAgent/.env)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

app = FastAPI()

TWIML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{{ ngrok_domain }}/ws?CallSid={{ call_sid }}"/>
  </Connect>
</Response>"""


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """Handle Twilio webhook for incoming calls."""
    form_data = await request.form()
    caller = form_data.get("From", "Unknown")
    call_sid = form_data.get("CallSid", "")
    logger.info(f"Incoming call from {caller} (CallSid: {call_sid})")

    ngrok_domain = os.getenv("NGROK_DOMAIN", "localhost:8000")
    twiml = TWIML.replace("{{ ngrok_domain }}", ngrok_domain).replace("{{ call_sid }}", call_sid)
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle Twilio media stream WebSocket."""
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from bot import run_bot

    await websocket.accept()
    logger.info("WebSocket connection established")

    # Extract parameters
    stream_sid = websocket.query_params.get("StreamSid", "")
    call_sid = websocket.query_params.get("CallSid", "")

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if account_sid and auth_token and call_sid:
        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            account_sid=account_sid,
            auth_token=auth_token,
        )
        logger.info("Twilio auto-hangup enabled")
    else:
        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
        )
        logger.info("Twilio auto-hangup disabled (missing credentials or CallSid)")

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            serializer=serializer,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer()
        )
    )

    try:
        await run_bot(transport, stream_sid=stream_sid)
    except Exception as e:
        logger.error(f"Bot error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)