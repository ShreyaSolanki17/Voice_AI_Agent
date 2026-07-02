"""BrightBox voice agent Pipecat pipeline."""

import os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    LLMContextFrame,
    TranscriptionFrame,
    LLMMessagesAppendFrame,
)
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.service_switcher import ServiceSwitcherStrategyManual
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport
from loguru import logger
from prompts.system_prompt import SYSTEM_PROMPT, SIMILARITY_THRESHOLD
from rag.retriever import retrieve


class ServiceSwitcherStrategyOneWayFailover(ServiceSwitcherStrategyManual):
    """Fail over to the next service without cycling back to earlier services."""

    async def handle_error(self, error: ErrorFrame) -> FrameProcessor | None:
        service_name = error.processor.name if error.processor else self.active_service.name
        logger.warning(f"Service {service_name} reported an error: {error.error}")

        current_idx = self.services.index(self.active_service)
        next_idx = current_idx + 1
        if next_idx >= len(self.services):
            # We've run out of services in the fallback chain.
            logger.error("No fallback service available")
            return None

        return await self._set_active_if_available(self.services[next_idx])


class LLMFailoverRetryProcessor(FrameProcessor):
    """Retry the latest LLM context after the switcher fails over."""

    def __init__(self, retryable_llms: list[FrameProcessor]):
        super().__init__()
        # We only want to retry if one of the initial LLMs fails, not the last one.
        self._retryable_llm_ids = {id(llm) for llm in retryable_llms}
        self._last_context_frame: LLMContextFrame | None = None
        self._retried_failures: set[tuple[int, int]] = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Keep track of the last set of messages sent to the LLM.
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            self._last_context_frame = frame

        # If we get an error from a retryable LLM, resubmit the last context.
        if (
            isinstance(frame, ErrorFrame)
            and direction == FrameDirection.UPSTREAM
            and not frame.fatal
            and frame.processor
            and id(frame.processor) in self._retryable_llm_ids
            and self._last_context_frame
        ):
            # Create a unique key for this specific failure to avoid infinite retry loops.
            context_id = id(self._last_context_frame.context)
            retry_key = (context_id, id(frame.processor))
            # Only retry this specific failure once.
            if retry_key not in self._retried_failures:
                self._retried_failures.add(retry_key)
                logger.warning(
                    f"{frame.processor.name} failed; retrying the current turn with the next LLM fallback"
                )
                await self.push_frame(self._last_context_frame, FrameDirection.DOWNSTREAM)
                return

        await self.push_frame(frame, direction)


def create_llm_service() -> tuple[FrameProcessor, FrameProcessor | None]:
    """Create the configured LLM service and optional failover retry processor."""
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    if provider not in {"auto", "gemini", "openai", "groq"}:
        raise ValueError("LLM_PROVIDER must be one of: auto, gemini, openai, groq")

    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    gemini_llm = None
    if provider in {"auto", "gemini"} and gemini_api_key:
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        logger.info(f"Using Gemini LLM model: {gemini_model}")
        gemini_llm = GoogleLLMService(
            api_key=gemini_api_key,
            settings=GoogleLLMService.Settings(model=gemini_model),
        )

    openai_llm = None
    if provider in {"auto", "openai"} and openai_api_key:
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info(f"Using OpenAI LLM model: {openai_model}")
        openai_llm = OpenAILLMService(
            api_key=openai_api_key,
            settings=OpenAILLMService.Settings(model=openai_model),
        )

    groq_llm = None
    if provider in {"auto", "groq"} and groq_api_key:
        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        logger.info(f"Using Groq LLM model: {groq_model}")
        groq_llm = GroqLLMService(
            api_key=groq_api_key,
            settings=GroqLLMService.Settings(model=groq_model),
        )

    if provider == "auto":
        # Create the fallback chain in order: Gemini -> OpenAI -> Groq
        llms = [llm for llm in [gemini_llm, openai_llm, groq_llm] if llm]
        if len(llms) > 1:
            logger.info(
                "Using LLM runtime fallback chain: "
                + " -> ".join(llm.__class__.__name__ for llm in llms)
            )
            llm = LLMSwitcher(
                llms=llms,
                strategy_type=ServiceSwitcherStrategyOneWayFailover,
            )
            # The retry processor should not try to failover from the *last* LLM.
            return llm, LLMFailoverRetryProcessor(retryable_llms=llms[:-1])
        if llms:
            return llms[0], None

    if provider == "gemini" and gemini_llm:
        return gemini_llm, None
    if provider == "openai" and openai_llm:
        return openai_llm, None
    if provider == "groq" and groq_llm:
        return groq_llm, None

    if provider == "gemini":
        raise ValueError("LLM_PROVIDER=gemini requires GEMINI_API_KEY or GOOGLE_API_KEY")
    if provider == "openai":
        raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
    if provider == "groq":
        raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY")

    raise ValueError(
        "Set GEMINI_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY for the LLM service"
    )


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
    llm, failover_retry_processor = create_llm_service()
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
                # On each user turn, retrieve relevant context from the knowledge base.
                rag_ctx = await get_rag_context(frame.text)
                msgs = self._llm_context.get_messages()
                if rag_ctx:
                    # If context is found, inject it into the system prompt.
                    msgs[0]["content"] = f"{SYSTEM_PROMPT}\n\nRetrieved KB context:\n{rag_ctx}"
                    logger.debug(f"RAG injected context for: {frame.text[:60]}")
                else:
                    # If no context is found, instruct the LLM to use its fallback response.
                    msgs[0]["content"] = (
                        f"{SYSTEM_PROMPT}\n\nRetrieved KB context:\n"
                        "No relevant knowledge base context was found for this question. "
                        "Use the fallback response."
                    )
                    logger.debug(f"RAG fallback context for: {frame.text[:60]}")
                self._llm_context.set_messages(msgs)
            await self.push_frame(frame, direction)

    rag_processor = RAGContextProcessor(llm_context=llm_context)

    # Use default turn strategies — no custom configuration to avoid blocking StartFrame
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(llm_context)

    pipeline_processors = [
        transport.input(),
        stt,
        rag_processor,
        user_aggregator,
    ]
    if failover_retry_processor:
        pipeline_processors.append(failover_retry_processor)
    pipeline_processors.extend([
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    pipeline = Pipeline(pipeline_processors)

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
