"""BrightBox voice agent Pipecat pipeline."""

import os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import Task
from pipecat.frames.frames import EndFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.llm_response_filter import LLMResponseFilter
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.network.websocket_server import WebSocketServerTransport
from pipecat.vad.silero import SileroVAD
from loguru import logger
from prompts.system_prompt import SYSTEM_PROMPT, SIMILARITY_THRESHOLD
from rag.retriever import retrieve


async def get_rag_context(user_transcript: str) -> str:
    """Retrieve KB context or determine if fallback needed."""
    context, _, scores = retrieve(user_transcript)
    if context and max(scores) >= SIMILARITY_THRESHOLD:
        return context
    return ""


async def run_bot(transport: WebSocketServerTransport, stream_sid: str = ""):
    """Run the voice bot pipeline."""
    # Services
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="7b238312-3f0d-4540-9858-b63768491424"
    )
    vad = SileroVAD()

    # LLM context with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    llm_context = OpenAILLMContext(messages)

    # RAG processor - injects context on each user turn
    class RAGContextProcessor(LLMResponseFilter):
        async def process_frame(self, frame, direction):
            if hasattr(frame, "text") and frame.text:
                # Get RAG context for this user message
                context = await get_rag_context(frame.text)
                if context:
                    # Inject retrieved context into system message
                    messages = self._llm_context.get_messages()
                    messages[0]["content"] = f"{SYSTEM_PROMPT}\n\nRetrieved KB context:\n{context}"
                    self._llm_context.set_messages(messages)
            await self.push_frame(frame, direction)

    rag_processor = RAGContextProcessor(llm_context=llm_context)

    pipeline = Pipeline([
        transport.input(),
        vad,
        stt,
        rag_processor,
        llm,
        tts,
        transport.output(),
    ])

    task = Task(pipeline=pipeline)

    # Handle call ending - any empty or goodbye message triggers end
    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, event):
        logger.info("Client disconnected")
        await task.queue_frame(EndFrame())

    # Fallback detection in LLM responses
    @llm.event_handler("on_first_llm_token")
    async def on_llm_response(service, text):
        # If user asked for account-specific info, offer handoff
        if "order #" in text.lower() or "tracking" in text.lower():
            # LLM should handle this via system prompt
            pass

    await task.run()