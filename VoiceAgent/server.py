"""FastAPI server for BrightBox voice agent."""

import os
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from loguru import logger
import uvicorn

app = FastAPI()

TWIML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{{ ngrok_domain }}/ws"/>
  </Connect>
</Response>"""


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """Handle Twilio webhook for incoming calls."""
    form_data = await request.form()
    caller = form_data.get("From", "Unknown")
    logger.info(f"Incoming call from {caller}")

    ngrok_domain = os.getenv("NGROK_DOMAIN", "localhost:8000")
    twiml = TWIML.replace("{{ ngrok_domain }}", ngrok_domain)
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle Twilio media stream WebSocket."""
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.transports.network.websocket_server import WebSocketServerTransport, WebSocketServerParams
    from bot import run_bot

    await websocket.accept()
    logger.info("WebSocket connection established")

    # Extract Twilio stream SID from query params
    stream_sid = websocket.query_params.get("StreamSid", "")

    serializer = TwilioFrameSerializer(stream_sid=stream_sid)

    transport = WebSocketServerTransport(
        params=WebSocketServerParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            serializer=serializer,
        )
    )

    try:
        await run_bot(transport, stream_sid=stream_sid)
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await transport.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)