"""Handlers for an agent's tests: the overview the Tests tab polls, runs,
scenarios and the admin override."""

import json

from fastapi import HTTPException, status

from app.ai.voice.agents.breeze_buddy.dispatch.alerts import emit_agent_tests_alert
from app.ai.voice.agents.breeze_buddy.scenario_tests.gate import (
    Verdict,
    gated,
    go_live,
    serving_note,
    template_fingerprint,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.scenario import from_mapping
from app.ai.voice.agents.breeze_buddy.template.types import TemplateModel
from app.core.concurrency import spawn_background_task
from app.core.logger import logger
from app.core.security.authorization import require_admin
from app.database.accessor.breeze_buddy.agent_tests import (
    create_agent_test_scenario,
    delete_agent_test_scenario,
    get_agent_test_version_runs,
    list_agent_test_results,
    list_agent_test_runs,
    list_agent_test_scenarios,
    queue_agent_test_run,
    record_agent_test_override,
    update_agent_test_scenario,
)
from app.database.accessor.breeze_buddy.template import get_template_by_id
from app.schemas import UserInfo
from app.schemas.breeze_buddy.agent_tests import (
    AgentTestRun,
    AgentTestRunDetail,
    AgentTestScenario,
    AgentTestScenarioRequest,
    AgentTestScenarioSource,
    AgentTestsOverview,
    AgentTestTrigger,
    OverrideAgentTestsRequest,
)

from ..version.handlers import _load_and_authorize

RECENT_RUNS = 10
# Every agent's suite shares one worker.
MAX_SCENARIOS = 40
MAX_SCENARIO_CHARS = 32_000


async def _load_gated(
    template_id: str, current_user: UserInfo, operation: str
) -> TemplateModel:
    template = await _load_and_authorize(template_id, current_user, operation)
    if not await gated(
        template.flow, template.supported_channels, template.configurations
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tests are not available for this agent",
        )
    return template


async def _queue(
    template: TemplateModel,
    trigger: AgentTestTrigger,
    current_user: UserInfo,
) -> AgentTestRun:
    """A live agent's run goes live on a pass, which publishes its draft; a
    paused agent stays paused, whatever changed, until it is put live."""
    return await queue_agent_test_run(
        str(template.id),
        template_fingerprint(template),
        template.current_version,
        template.is_active,
        trigger,
        current_user.username,
    )


async def _with_results(run: AgentTestRun) -> AgentTestRunDetail:
    return AgentTestRunDetail(
        **run.model_dump(), results=await list_agent_test_results(run.id)
    )


def _checked(body: AgentTestScenarioRequest) -> AgentTestScenarioRequest:
    """The scenario under its row's name, refused (400) if the harness cannot
    load it or it is too large."""
    scenario = {**body.scenario, "name": body.name}
    if len(json.dumps(scenario)) > MAX_SCENARIO_CHARS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"A scenario is at most {MAX_SCENARIO_CHARS} characters",
        )
    try:
        from_mapping(scenario)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid scenario: {e}"
        )
    return body.model_copy(update={"scenario": scenario})


async def get_agent_tests_handler(
    template_id: str, current_user: UserInfo
) -> AgentTestsOverview:
    template = await _load_and_authorize(template_id, current_user, "read agent tests")
    fingerprint = template_fingerprint(template)
    pending, finished = await get_agent_test_version_runs(template_id, fingerprint)
    current = pending or finished
    return AgentTestsOverview(
        template_id=template_id,
        enabled=await gated(
            template.flow, template.supported_channels, template.configurations
        ),
        is_active=template.is_active,
        current_version=template.current_version,
        published_version=template.published_version if template.is_draft else None,
        fingerprint=fingerprint,
        state=Verdict(pending, finished).state,
        run=await _with_results(current) if current else None,
        runs=await list_agent_test_runs(template_id, RECENT_RUNS),
        scenarios=await list_agent_test_scenarios(template_id),
    )


async def run_agent_tests_handler(
    template_id: str, current_user: UserInfo
) -> AgentTestRun:
    template = await _load_gated(template_id, current_user, "run agent tests")
    return await _queue(template, AgentTestTrigger.MANUAL, current_user)


async def generate_agent_test_scenarios_handler(
    template_id: str, current_user: UserInfo
) -> AgentTestRun:
    """Queue a run that rewrites the generated scenarios (custom ones stay),
    then runs them."""
    template = await _load_gated(template_id, current_user, "generate agent tests")
    return await _queue(template, AgentTestTrigger.REWRITE, current_user)


async def create_agent_test_scenario_handler(
    template_id: str, body: AgentTestScenarioRequest, current_user: UserInfo
) -> AgentTestScenario:
    template = await _load_gated(template_id, current_user, "edit agent tests")
    body = _checked(body)
    if len(await list_agent_test_scenarios(template_id)) >= MAX_SCENARIOS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An agent has at most {MAX_SCENARIOS} scenarios",
        )
    scenario = await create_agent_test_scenario(
        template_id,
        body.name,
        body.scenario,
        AgentTestScenarioSource.CUSTOM,
        current_user.username,
        body.is_enabled,
    )
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A scenario named {body.name!r} already exists",
        )
    await _queue(template, AgentTestTrigger.SCENARIOS, current_user)
    return scenario


async def update_agent_test_scenario_handler(
    template_id: str,
    scenario_id: str,
    body: AgentTestScenarioRequest,
    current_user: UserInfo,
) -> AgentTestScenario:
    template = await _load_gated(template_id, current_user, "edit agent tests")
    body = _checked(body)
    if any(
        s.name == body.name and s.id != scenario_id
        for s in await list_agent_test_scenarios(template_id)
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A scenario named {body.name!r} already exists",
        )
    scenario = await update_agent_test_scenario(
        scenario_id, template_id, body.name, body.scenario, body.is_enabled
    )
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found"
        )
    await _queue(template, AgentTestTrigger.SCENARIOS, current_user)
    return scenario


async def delete_agent_test_scenario_handler(
    template_id: str, scenario_id: str, current_user: UserInfo
) -> None:
    template = await _load_gated(template_id, current_user, "edit agent tests")
    if not await delete_agent_test_scenario(scenario_id, template_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Scenario not found"
        )
    await _queue(template, AgentTestTrigger.SCENARIOS, current_user)


async def override_agent_tests_handler(
    template_id: str, body: OverrideAgentTestsRequest, current_user: UserInfo
) -> AgentTestRun:
    """An admin marks the current version green, with the reason on record.
    A live agent's draft is published at once; a paused agent can then go live."""
    require_admin(current_user)
    template = await _load_gated(template_id, current_user, "override agent tests")
    fingerprint = template_fingerprint(template)
    reason = body.reason.strip()
    run = await record_agent_test_override(
        template_id,
        fingerprint,
        template.current_version,
        current_user.username,
        reason,
    )
    logger.warning(
        f"Admin {current_user.username} overrode the tests of template "
        f"{template_id} v{template.current_version}: {reason}"
    )
    if template.is_draft and template.is_active:
        await go_live(template_id, fingerprint)
    published = await get_template_by_id(template_id)
    spawn_background_task(
        emit_agent_tests_alert(
            "Breeze Buddy Agent Tests Passed by Admin",
            template_id,
            template.name,
            template.current_version,
            template.merchant_id,
            reason,
            serving_note(published or template),
            current_user.username,
        ),
        name=f"agent-tests-override-alert:{template_id}",
    )
    return run
