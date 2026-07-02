"""BrightBox voice agent Pipecat pipeline."""

import os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import EndFrame, TranscriptionFrame, LLMMessagesAppendFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport
from loguru import logger
from prompts.system_prompt import SYSTEM_PROMPT, SIMILARITY_THRESHOLD
from rag.retriever import retrieve


async def get_rag_context(user_transcript: str) -> str:
    """Retrieve KB context or determine if fallback needed."""
    context, _, scores = retrieve(user_transcript)
    if context and max(scores) >= SIMILARITY_THRESHOLD:
        return context
    return ""


async def run_bot(transport: FastAPIWebsocketTransport, stream_sid: str = ""):
    """Run the voice bot pipeline."""
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(endpointing="500")
    )
    llm = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        settings=GoogleLLMService.Settings(model="gemini-2.5-flash")
    )
    # voice and model both in settings; sample_rate=8000 for Twilio's 8 kHz mulaw requirement
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        sample_rate=8000,
        settings=CartesiaTTSService.Settings(
            voice=os.getenv("CARTESIA_VOICE_ID", "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"),
            model=os.getenv("CARTESIA_MODEL", "sonic-3.5"),
        ),
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    llm_context = LLMContext(messages)

    class RAGContextProcessor(FrameProcessor):
        def __init__(self, llm_context: LLMContext):
            super().__init__()
            self._llm_context = llm_context

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame) and frame.text:
                rag_ctx = await get_rag_context(frame.text)
                if rag_ctx:
                    msgs = self._llm_context.get_messages()
                    msgs[0]["content"] = f"{SYSTEM_PROMPT}\n\nRetrieved KB context:\n{rag_ctx}"
                    self._llm_context.set_messages(msgs)
                    logger.debug(f"RAG injected context for: {frame.text[:60]}")
            await self.push_frame(frame, direction)

    rag_processor = RAGContextProcessor(llm_context=llm_context)

    # Use default turn strategies — no custom configuration to avoid blocking StartFrame
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(llm_context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        rag_processor,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline=pipeline, enable_rtvi=False, idle_timeout_secs=120)

    # Queue the greeting once the pipeline is fully started
    @task.event_handler("on_pipeline_started")
    async def on_pipeline_started(task, frame):
        logger.info("Pipeline started — queuing greeting")
        greeting = [{"role": "user", "content": "(call just started, greet the customer warmly)"}]
        await task.queue_frame(LLMMessagesAppendFrame(messages=greeting, run_llm=True))

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, event):
        logger.info("Client disconnected")
        await task.queue_frame(EndFrame())

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
