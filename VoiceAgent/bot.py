"""BrightBox voice agent Pipecat pipeline."""

import os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import EndFrame, TranscriptionFrame, LLMMessagesAppendFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    VADUserTurnStartStrategy,
    LLMTurnCompletionUserTurnStopStrategy,
)
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
    # Services
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(endpointing="500")
    )
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(model="gpt-4o-mini")
    )
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="7b238312-3f0d-4540-9858-b63768491424",
            model="sonic-english",
        )
    )

    # LLM context with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    llm_context = LLMContext(messages)

    # RAG processor - injects KB context on each finalized user transcription
    class RAGContextProcessor(FrameProcessor):
        def __init__(self, llm_context: LLMContext):
            super().__init__()
            self._llm_context = llm_context

        async def process_frame(self, frame, direction: FrameDirection):
            if isinstance(frame, TranscriptionFrame) and frame.text:
                rag_ctx = await get_rag_context(frame.text)
                if rag_ctx:
                    msgs = self._llm_context.get_messages()
                    msgs[0]["content"] = f"{SYSTEM_PROMPT}\n\nRetrieved KB context:\n{rag_ctx}"
                    self._llm_context.set_messages(msgs)
                    logger.debug(f"RAG injected context for: {frame.text[:60]}")
            # Always call super so StartFrame/EndFrame lifecycle is handled by the base class
            await super().process_frame(frame, direction)

    rag_processor = RAGContextProcessor(llm_context=llm_context)

    class DebugProcessor(FrameProcessor):
        def __init__(self, label):
            super().__init__()
            self._label = label
        async def process_frame(self, frame, direction: FrameDirection):
            logger.info(f"{self._label} saw frame: {type(frame).__name__} (direction: {direction})")
            await super().process_frame(frame, direction)

    d1 = DebugProcessor("Debug1(BeforeUserAgg)")
    d2 = DebugProcessor("Debug2(BeforeLLM)")
    d3 = DebugProcessor("Debug3(BeforeAsstAgg)")
    d4 = DebugProcessor("Debug4(BeforeTTS)")

    # Use VAD-based turn strategies so Deepgram's cloud endpointing drives turn detection
    # (The default SmartTurn strategy needs raw audio frames which DeepgramSTTService consumes)
    turn_strategies = UserTurnStrategies(
        start=VADUserTurnStartStrategy(),
        stop=LLMTurnCompletionUserTurnStopStrategy(),
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        llm_context, user_params=LLMUserAggregatorParams(user_turn_strategies=turn_strategies)
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        rag_processor,
        d1,
        user_aggregator,
        d2,
        llm,
        d3,
        assistant_aggregator,
        d4,
        tts,
        transport.output(),
    ])

    task = PipelineTask(pipeline=pipeline, enable_rtvi=False, idle_timeout_secs=60)

    # Send an initial greeting when the caller connects
    @transport.event_handler("on_client_connected")
    async def on_connected(transport, event):
        logger.info("Client connected \u2014 queuing greeting")
        greeting = [{"role": "user", "content": "(call just started, greet the customer warmly)"}]
        await task.queue_frame(LLMMessagesAppendFrame(messages=greeting, run_llm=True))

    # Queue an EndFrame when the caller hangs up
    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, event):
        logger.info("Client disconnected")
        await task.queue_frame(EndFrame())

    # Run the pipeline using PipelineRunner (Pipecat 1.4.0 API)
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)