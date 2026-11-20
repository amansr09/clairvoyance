"""One scenario run: an EVAL lead, a harness session on the eval websocket, and the
checks on the lead the call left behind.

The harness judges the conversation; the lead answers what only Buddy knows: the
outcome it recorded, the nodes it walked, and any request a fixture did not cover.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from uuid import uuid4

from pipecat.evals.results import EvalScriptResult, EvalSimulationResult
from pipecat.evals.session import EvalSession, EvalSessionParams
from pipecat.evals.simulation import EvalSimulationScenario

from app.ai.voice.agents.breeze_buddy.scenario_tests.access import session_token
from app.ai.voice.agents.breeze_buddy.scenario_tests.models import (
    build_caller,
    build_judge,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.scenario import (
    MAX_DURATION_S,
    BuddyScenario,
)
from app.ai.voice.agents.breeze_buddy.template.types import TemplateModel
from app.core.config.static import PORT
from app.core.logger import logger
from app.database.accessor.breeze_buddy.lead_call_tracker import (
    create_lead_call_tracker,
    get_lead_by_id,
    update_lead_call_completion_details,
)
from app.schemas.breeze_buddy.core import (
    ExecutionMode,
    LeadCallStatus,
    LeadCallTracker,
)

EVAL_PATH = "/agent/voice/breeze-buddy/eval/{lead_id}/ws"
# The call finalizes after the harness hangs up; give it time to land.
FINISH_TIMEOUT_S = 30.0
FINISH_POLL_S = 0.5
# A session still running past its scenario's limits (and the bot's start) is cut off.
CALL_TIMEOUT_S = MAX_DURATION_S + 60


# Failures of the harness itself, not of the agent.
HARNESS_FAILURES = ("judge call failed", "run error:")
# Pipecat's failure kinds for a call the harness ended: it never connected or
# started, raised, or had no judge.
HARNESS_KINDS = frozenset(
    {
        "connect_failed",
        "closed_before_ready",
        "handshake_timeout",
        "error",
        "no_judge",
        "judge_no_verdict",
    }
)


@dataclass
class RunResult:
    lead_id: Optional[str]
    passed: bool
    failures: List[str] = field(default_factory=list)
    outcome: Optional[str] = None
    duration_ms: int = 0
    harness_broke: bool = False

    @property
    def harness_failed(self) -> bool:
        """Failed only because the harness broke: the session, its judge or its caller."""
        return self.harness_broke or (
            bool(self.failures)
            and all(f.startswith(HARNESS_FAILURES) for f in self.failures)
        )


def session_url(lead_id: str) -> str:
    path = EVAL_PATH.format(lead_id=lead_id)
    token = quote(session_token(lead_id), safe="")
    # The test call is served by this same pod: only the agent-tests worker's
    # pods mount the eval route.
    return f"ws://127.0.0.1:{PORT}{path}?token={token}"


async def run_once(item: BuddyScenario, run: int, template: TemplateModel) -> RunResult:
    """One run of a scenario. A run that raises fails on its own, so the suite's
    other runs finish closing their leads."""
    try:
        return await _run_once(item, run, template)
    except Exception as e:
        logger.opt(exception=True).warning(f"Scenario {item.name} run {run} broke: {e}")
        return RunResult(None, False, [f"run error: {e!r}"])


async def _run_once(
    item: BuddyScenario, run: int, template: TemplateModel
) -> RunResult:
    started = time.monotonic()
    # Everything that can fail before the session starts runs before the lead
    # exists; the bot only marks it started, so the stuck-call sweeper never sees
    # a lead the session never reached.
    caller = (
        await build_caller()
        if isinstance(item.scenario, EvalSimulationScenario)
        else None
    )
    judge = await build_judge()
    lead = await _create_eval_lead(item, run, template)
    if not lead:
        return RunResult(None, False, ["lead not created"])

    finished = None
    try:
        session = EvalSession.from_scenario(
            item.scenario,
            session_url(lead.id),
            params=EvalSessionParams(bot_ready_timeout_s=30.0),
            judge=judge,
            persona_llm=caller,
        )
        harness = None
        try:
            async with asyncio.timeout(CALL_TIMEOUT_S):
                harness = await session.run()
        except TimeoutError:
            pass
        finished = await _wait_finished(lead.id)
    finally:
        if not finished or finished.status != LeadCallStatus.FINISHED:
            await _close_unfinished(lead.id)

    # A call the harness broke says nothing of the agent: its lead checks only
    # repeat that the call did not finish.
    broke = _harness_broke(harness)
    failures = _harness_failures(harness)
    if not broke:
        failures += _lead_failures(item, finished)
    return RunResult(
        lead_id=lead.id,
        passed=not failures,
        failures=failures,
        outcome=finished.outcome if finished else None,
        duration_ms=int((time.monotonic() - started) * 1000),
        harness_broke=broke,
    )


async def _create_eval_lead(
    item: BuddyScenario, run: int, template: TemplateModel
) -> Optional[LeadCallTracker]:
    lead_id = str(uuid4())
    return await create_lead_call_tracker(
        id=lead_id,
        reseller_id=template.reseller_id,
        template=template.name,
        template_id=str(template.id),
        merchant_id=template.merchant_id,
        next_attempt_at=None,
        payload={**_example_payload(template), **item.spec.payload},
        meta_data={
            "eval": {
                "scenario": item.name,
                "run": run,
                "http": [f.model_dump() for f in item.spec.http],
            }
        },
        execution_mode=ExecutionMode.EVAL,
        status=LeadCallStatus.PROCESSING,
        call_id=f"eval-{lead_id}",
    )


def _example_payload(template: TemplateModel) -> Dict[str, Any]:
    """The template's own payload examples, so a scenario only states what differs."""
    schema = template.expected_payload_schema or {}
    return {
        key: spec["example"]
        for key, spec in schema.items()
        if isinstance(spec, dict) and "example" in spec
    }


async def _wait_finished(lead_id: str) -> Optional[LeadCallTracker]:
    deadline = time.monotonic() + FINISH_TIMEOUT_S
    lead = await get_lead_by_id(lead_id)
    while lead and lead.status != LeadCallStatus.FINISHED:
        if time.monotonic() > deadline:
            break
        await asyncio.sleep(FINISH_POLL_S)
        lead = await get_lead_by_id(lead_id)
    return lead


async def _close_unfinished(lead_id: str) -> None:
    """Close a lead its session left open: still PROCESSING, it counts as an
    active lead against its template and is later swept as a stuck call."""
    lead = await get_lead_by_id(lead_id)
    if not lead or lead.status == LeadCallStatus.FINISHED:
        return
    await update_lead_call_completion_details(
        id=lead.id,
        status=LeadCallStatus.FINISHED,
        outcome=lead.outcome or "UNKNOWN",
        meta_data={**(lead.metaData or {}), "cleanup": "eval_unfinished"},
        call_end_time=datetime.now(timezone.utc),
        expected_status=LeadCallStatus.PROCESSING,
    )


def _harness_broke(result: Any) -> bool:
    """The harness, not the agent, ended the call. Pipecat sets a simulation's
    error only when it did not finish or its judge gave no verdict."""
    if result is None:
        return True
    if isinstance(result, EvalSimulationResult):
        return result.error is not None
    if isinstance(result, EvalScriptResult):
        return any(
            f.kind in HARNESS_KINDS or f.reason.endswith(HARNESS_FAILURES[0])
            for f in result.failures
        )
    return False


def _harness_failures(result: Any) -> List[str]:
    if result is None:
        return [f"the test call did not end within {CALL_TIMEOUT_S} s"]
    if isinstance(result, EvalSimulationResult):
        return [] if result.passed else [result.failure or "simulation failed"]
    if isinstance(result, EvalScriptResult):
        if result.skipped:
            return [f"skipped: {result.skipped}"]
        return [
            f"turn {f.turn_index} {f.event_name}: {f.reason}" for f in result.failures
        ]
    return [f"unexpected harness result {type(result).__name__}"]


def _lead_failures(item: BuddyScenario, lead: Optional[LeadCallTracker]) -> List[str]:
    if not lead or lead.status != LeadCallStatus.FINISHED:
        return ["the call did not finish"]
    spec = item.spec
    meta = lead.metaData or {}
    failures = []
    if spec.outcomes and lead.outcome not in spec.outcomes:
        failures.append(
            f"outcome {lead.outcome!r}, expected {' or '.join(spec.outcomes)}"
        )
    visited = {n.get("node_name") for n in meta.get("node_traversal") or []}
    failures += [f"never reached node {n!r}" for n in spec.visits if n not in visited]
    failures += [f"entered node {n!r}" for n in spec.avoids if n in visited]
    failures += [
        f"no fixture for {c['method']} {c['url']}"
        for c in (meta.get("eval") or {}).get("http_calls") or []
        if not c.get("matched")
    ]
    return failures
