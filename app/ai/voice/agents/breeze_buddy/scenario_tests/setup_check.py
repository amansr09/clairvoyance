"""Before any test call: can the agent's own providers start?

Every provider account the agent names (LLM, realtime, speech-to-text,
text-to-speech, observers) is resolved the way a call resolves it, and the
agent's LLM answers one short prompt. A removed provider or a missing key fails
here with its reason, instead of as test calls that never connect.

A Gemini Live agent's LLM is a realtime session, which a one-shot prompt cannot
open: its account is checked, its answer is not.
"""

import asyncio
from typing import List

from pipecat.processors.aggregators.llm_context import LLMContext

from app.ai.voice.agents.breeze_buddy.accounts import (
    account_blocks,
    accounts_for_template,
)
from app.ai.voice.agents.breeze_buddy.llm import get_llm_service
from app.ai.voice.agents.breeze_buddy.template.types import TemplateModel

SETUP_CHECK = "setup-check"
LLM_PROBE = "Reply with the single word OK."
LLM_PROBE_TIMEOUT_S = 30.0


async def setup_problems(template: TemplateModel) -> List[str]:
    """Why the agent cannot start a call; empty when it can."""
    configurations = template.configurations
    accounts = accounts_for_template(template)
    problems = []
    for name, block in account_blocks(configurations):
        try:
            await accounts.get(block)
        except ValueError as e:
            problems.append(f"{name}: {e}")

    llm_config = configurations.llm_configurations if configurations else None
    if llm_config and llm_config.realtime:
        return problems
    try:
        llm = await get_llm_service(llm_config, accounts=accounts)
        reply = await asyncio.wait_for(
            llm.run_inference(LLMContext([{"role": "user", "content": LLM_PROBE}])),
            LLM_PROBE_TIMEOUT_S,
        )
        if not (reply or "").strip():
            problems.append("LLM: the model returned no reply")
    except Exception as e:
        problems.append(f"LLM: {type(e).__name__}: {e}")
    return problems
