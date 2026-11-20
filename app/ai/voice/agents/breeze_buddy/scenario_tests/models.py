"""The models that write an agent's scenarios, play the caller and judge the
call: Bedrock, through the app's own LLM factory, from a different model family
than the luna agent."""

from typing import Any

from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.evals.judge import EvalJudge
from pipecat.services.llm_service import LLMService

from app.ai.voice.agents.breeze_buddy.llm import (
    DEFAULT_LLM_CONFIGURATION,
    get_llm_service,
)
from app.ai.voice.llm.types import ThinkingConfiguration
from app.core.config.static import (
    AGENT_TESTS_CALLER_MODEL,
    AGENT_TESTS_JUDGE_MODEL,
    AGENT_TESTS_WRITER_MAX_TOKENS,
    AGENT_TESTS_WRITER_MODEL,
)


async def _bedrock_llm(
    model: str, max_tokens: int, temperature: float
) -> LLMService[Any]:
    config = DEFAULT_LLM_CONFIGURATION.model_copy(
        update={
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": ThinkingConfiguration(enabled=False),
        }
    )
    return await get_llm_service(config)


async def build_judge() -> EvalJudge:
    llm = await _bedrock_llm(AGENT_TESTS_JUDGE_MODEL, max_tokens=1024, temperature=0.0)
    return EvalJudge(LLMClassifier(llm=llm), explainer=llm)


async def build_caller() -> LLMService[Any]:
    return await _bedrock_llm(AGENT_TESTS_CALLER_MODEL, max_tokens=300, temperature=0.8)


async def build_writer() -> LLMService[Any]:
    return await _bedrock_llm(
        AGENT_TESTS_WRITER_MODEL,
        max_tokens=AGENT_TESTS_WRITER_MAX_TOKENS,
        temperature=0.4,
    )
