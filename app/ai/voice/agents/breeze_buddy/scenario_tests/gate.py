"""The go-live gate: content (by fingerprint) goes live only on a passing
verdict; until then it is a draft and real calls run ``published_version``.
A save is gathered, decided (``decide_save``, pure) and applied."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from app.ai.voice.agents.breeze_buddy.managers.utils import (
    spawn_realtime_opening_line_regeneration,
)
from app.ai.voice.agents.breeze_buddy.template.cache import invalidate_template
from app.ai.voice.agents.breeze_buddy.template.types import (
    ConfigurationModel,
    FlowMode,
    TemplateModel,
)
from app.core.config.dynamic import AGENT_TESTS_ENABLED
from app.core.logger import logger
from app.database.accessor.breeze_buddy.agent_tests import (
    get_agent_test_version_runs,
    queue_agent_test_run,
    set_pending_agent_test_runs_go_live,
)
from app.database.accessor.breeze_buddy.template import (
    activate_template_version,
    get_live_template_by_id,
    get_template_by_id,
)
from app.schemas.breeze_buddy.agent_tests import (
    AgentTestRun,
    AgentTestRunStatus,
    AgentTestState,
    AgentTestTrigger,
)

GREEN = (AgentTestRunStatus.PASSED, AgentTestRunStatus.OVERRIDDEN)


def fingerprint(
    flow: Any,
    configurations: Optional[dict],
    expected_payload_schema: Optional[dict],
    expected_callback_response_schema: Optional[dict],
) -> str:
    """What the tests cover: everything that changes how a call goes. The
    template handlers fingerprint content as they write it, and it reads back
    the same: configurations are dumped the way ``template_fingerprint`` does."""
    content = {
        "flow": flow,
        "configurations": configurations,
        "expected_payload_schema": expected_payload_schema,
        "expected_callback_response_schema": expected_callback_response_schema,
    }
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _dump(configurations: Optional[ConfigurationModel]) -> Optional[dict]:
    if not configurations:
        return None
    return configurations.model_dump(
        exclude_none=True, mode="json", context={"reveal_secrets": True}
    )


def template_fingerprint(template: TemplateModel) -> str:
    return fingerprint(
        template.flow,
        _dump(template.configurations),
        template.expected_payload_schema,
        template.expected_callback_response_schema,
    )


def testable(flow: Any, supported_channels: Sequence[str], configurations: Any) -> bool:
    """Scenario tests drive a phone call: not for a chat (widget) agent, whose
    voice is in-widget; an IVR flow, which has no conversation; or an agent
    with MCP servers, whose tool calls would reach them for real."""
    mode = flow.get("mode") if isinstance(flow, dict) else None
    if isinstance(configurations, ConfigurationModel):
        configurations = _dump(configurations)
    mcp = (configurations or {}).get("mcp") or {}
    return (
        "chat" not in (supported_channels or ["voice"])
        and mode != FlowMode.IVR.value
        and not mcp.get("servers")
    )


async def gated(
    flow: Any, supported_channels: Sequence[str], configurations: Any
) -> bool:
    return (
        testable(flow, supported_channels, configurations)
        and await AGENT_TESTS_ENABLED()
    )


@dataclass(frozen=True)
class Verdict:
    """A version's latest pending run and latest finished run."""

    pending: Optional[AgentTestRun] = None
    finished: Optional[AgentTestRun] = None

    @property
    def green(self) -> bool:
        return bool(self.finished and self.finished.status in GREEN)

    @property
    def needs_run(self) -> bool:
        """Nothing is testing this version and nothing has judged it."""
        return not self.pending and (
            not self.finished or self.finished.status == AgentTestRunStatus.ERROR
        )

    @property
    def state(self) -> AgentTestState:
        if self.pending:
            return AgentTestState(self.pending.status.value.lower())
        if self.finished:
            return AgentTestState(self.finished.status.value.lower())
        return AgentTestState.UNTESTED


@dataclass(frozen=True)
class SavePlan:
    is_active: bool
    # The version real calls run while the saved content is a draft; None
    # runs the saved content.
    published_version: Optional[int] = None
    # Queue a run of the saved content.
    queue: bool = False
    # Whether the queued run, or the pending one, turns the agent on when it
    # passes. None leaves the pending runs as they are.
    go_live: Optional[bool] = None
    # Why going live is refused; the save is refused with it.
    refused: Optional[str] = None


def decide_save(
    wants_live: bool,
    was_live: bool,
    verdict: Verdict,
    keep_serving: Optional[int] = None,
) -> SavePlan:
    """``keep_serving`` is the version real calls run now, when the save would
    change what they run; None when they would run the same content anyway, or
    nothing has been live."""
    published = None if verdict.green else keep_serving
    if not wants_live:
        return SavePlan(
            is_active=False,
            published_version=published,
            queue=verdict.needs_run,
            go_live=False,
        )
    if verdict.green:
        return SavePlan(is_active=True)
    if published is not None:
        return SavePlan(
            is_active=True,
            published_version=published,
            queue=verdict.needs_run,
            go_live=True,
        )
    if was_live:
        return SavePlan(is_active=True, queue=verdict.needs_run)
    if verdict.pending:
        return SavePlan(is_active=False, go_live=True)
    if verdict.needs_run:
        return SavePlan(is_active=False, queue=True, go_live=True)
    return SavePlan(
        is_active=False,
        refused="This version of the agent failed its tests. Fix it, or run the "
        "tests again, before it goes live.",
    )


async def plan_save(
    existing: TemplateModel, content_fingerprint: str, wants_live: bool
) -> SavePlan:
    """The plan for saving gated content over an existing template."""
    template_id = str(existing.id)
    verdict = Verdict(
        *await get_agent_test_version_runs(template_id, content_fingerprint)
    )
    keep = await _serving_version(existing, content_fingerprint)
    return decide_save(wants_live, existing.is_active, verdict, keep)


async def _serving_version(
    existing: TemplateModel, content_fingerprint: str
) -> Optional[int]:
    """The version real calls run now, unless the saved content is what they
    run, or nothing has been live."""
    served = (
        await get_live_template_by_id(str(existing.id))
        if existing.is_draft
        else existing
    )
    if not served or template_fingerprint(served) == content_fingerprint:
        return None
    if existing.is_draft:
        return existing.published_version
    return existing.current_version if existing.is_active else None


async def apply_save(
    template: TemplateModel,
    plan: SavePlan,
    trigger: AgentTestTrigger,
    requested_by: Optional[str],
) -> Optional[AgentTestRun]:
    """Queue the saved content's run, or point the pending runs at going live.
    Returns the queued run."""
    template_id = str(template.id)
    if plan.go_live is False:
        await set_pending_agent_test_runs_go_live(template_id, None, False)
    if not plan.queue:
        if plan.go_live:
            await set_pending_agent_test_runs_go_live(
                template_id, template_fingerprint(template), True
            )
        return None
    return await queue_agent_test_run(
        template_id,
        template_fingerprint(template),
        template.current_version,
        bool(plan.go_live),
        trigger,
        requested_by,
    )


async def go_live(template_id: str, fingerprint: str) -> bool:
    """Put the head live (turned on, and the version real calls run) unless it
    changed since it passed, or already is."""
    template = await get_template_by_id(template_id)
    if not template or (template.is_active and not template.is_draft):
        return False
    if template_fingerprint(template) != fingerprint:
        return False
    if not await activate_template_version(template_id, template.current_version):
        return False
    logger.info(f"Agent {template_id} v{template.current_version} is live")
    try:
        await invalidate_template(template_id)
    except Exception as e:
        logger.warning(f"Template cache invalidation failed for {template_id}: {e}")
    spawn_realtime_opening_line_regeneration(
        template.model_copy(update={"is_active": True, "published_version": None})
    )
    return True


def serving_note(template: TemplateModel) -> str:
    """What real calls run now, for an alert."""
    if template.is_draft:
        return f"Keep running v{template.published_version}"
    if template.is_active:
        return f"Run v{template.current_version}"
    return "None: the agent is paused"
