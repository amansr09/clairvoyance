"""One run of an agent's suite: setup check, scenarios, verdict, go-live."""

import asyncio
import time
from typing import Any, Dict, List, Optional

from app.ai.voice.agents.breeze_buddy.dispatch.alerts import emit_agent_tests_alert
from app.ai.voice.agents.breeze_buddy.scenario_tests.checklist import MISSING_NAME
from app.ai.voice.agents.breeze_buddy.scenario_tests.gate import (
    go_live,
    serving_note,
    template_fingerprint,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.generate import (
    generate_scenarios,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.runner import RunResult, run_once
from app.ai.voice.agents.breeze_buddy.scenario_tests.scenario import (
    BuddyScenario,
    from_mapping,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.setup_check import (
    SETUP_CHECK,
    setup_problems,
)
from app.ai.voice.agents.breeze_buddy.template.types import TemplateModel
from app.core.config.dynamic import AGENT_TESTS_CONCURRENCY
from app.core.config.static import AGENT_TESTS_VERSION_RETENTION
from app.core.logger import logger
from app.core.logger.context import clear_log_context, set_log_context
from app.database.accessor.breeze_buddy.agent_tests import (
    create_agent_test_results,
    finish_agent_test_result,
    finish_agent_test_run,
    get_agent_test_version_runs,
    list_agent_test_scenarios,
    prune_agent_test_history,
    queue_agent_test_run,
    replace_generated_agent_test_scenarios,
    start_agent_test_result,
)
from app.database.accessor.breeze_buddy.template import (
    get_live_template_by_id,
    get_template_by_id,
)
from app.schemas.breeze_buddy.agent_tests import (
    AgentTestResult,
    AgentTestRun,
    AgentTestRunStatus,
    AgentTestScenario,
    AgentTestScenarioSource,
    AgentTestTrigger,
)

LOG_COMPONENT = "buddy.agent_tests"


async def run_suite(run: AgentTestRun) -> None:
    """Never raises: a run that breaks is closed as ERROR with the reason."""
    # The previous run's ids must not leak onto this one's lines.
    clear_log_context()
    set_log_context(component=LOG_COMPONENT, run_id=run.id, template_id=run.template_id)
    try:
        finished = await _run_suite(run)
    except Exception as e:
        logger.opt(exception=True).error(f"Agent test run {run.id} failed: {e}")
        finished = await finish_agent_test_run(
            run.id, AgentTestRunStatus.ERROR, f"The test run failed: {e}"
        )
    if finished and finished.status in (
        AgentTestRunStatus.FAILED,
        AgentTestRunStatus.ERROR,
    ):
        await _alert_failed(finished)
    # Retention rides on the runs themselves, no cron; it never fails a run.
    try:
        live = await get_live_template_by_id(run.template_id)
        await prune_agent_test_history(
            run.template_id,
            AGENT_TESTS_VERSION_RETENTION,
            template_fingerprint(live) if live else None,
        )
    except Exception as e:
        logger.warning(f"Pruning the test history of {run.template_id} failed: {e}")


async def _run_suite(run: AgentTestRun) -> Optional[AgentTestRun]:
    """The finished run; None when it was closed meanwhile (stale)."""
    template = await get_template_by_id(run.template_id)
    if not template or template_fingerprint(template) != run.fingerprint:
        return await finish_agent_test_run(
            run.id,
            AgentTestRunStatus.CANCELLED,
            "The agent changed before its tests started",
        )

    problems = await _setup_check(run, template)
    if problems:
        return await finish_agent_test_run(
            run.id,
            AgentTestRunStatus.FAILED,
            f"The agent cannot start a call: {problems[0]}",
        )
    stored = await _scenarios(template, run)
    lead = _lead_data(stored)
    scenarios = [_with_lead_data(s, lead) for s in stored if s.is_enabled]
    if not scenarios:
        return await finish_agent_test_run(
            run.id, AgentTestRunStatus.ERROR, "The agent has no enabled scenarios"
        )
    outcomes = await _run_scenarios(run, template, scenarios)
    head = await get_template_by_id(run.template_id)
    if not head or head.current_version != template.current_version:
        return await _changed_during(run, head)
    broken = [o for o in outcomes if o.harness_failed]
    if broken:
        return await finish_agent_test_run(
            run.id,
            AgentTestRunStatus.ERROR,
            f"{len(broken)} of {len(outcomes)} test calls did not run: "
            f"{(broken[0].failures or ['the harness broke'])[0]}",
        )
    passed = all(o.passed for o in outcomes)
    failed = sum(not o.passed for o in outcomes)
    finished = await finish_agent_test_run(
        run.id,
        AgentTestRunStatus.PASSED if passed else AgentTestRunStatus.FAILED,
        f"{failed} of {len(outcomes)} scenario runs failed" if not passed else None,
    )
    if finished and passed and finished.go_live:
        await go_live(run.template_id, run.fingerprint)
    return finished


async def _changed_during(
    run: AgentTestRun, head: Optional[TemplateModel]
) -> Optional[AgentTestRun]:
    """A save during the run: its later calls ran other content, so it is no
    verdict. The same content (a pause, a rename, an undo) is run again."""
    cancelled = await finish_agent_test_run(
        run.id, AgentTestRunStatus.CANCELLED, "The agent changed during its tests"
    )
    if cancelled and head and template_fingerprint(head) == run.fingerprint:
        await queue_agent_test_run(
            run.template_id,
            run.fingerprint,
            head.current_version,
            cancelled.go_live,
            AgentTestTrigger.MANUAL,
            run.requested_by,
        )
    return cancelled


async def _setup_check(run: AgentTestRun, template: TemplateModel) -> List[str]:
    """The setup check's problems, recorded as the run's first result. An
    agent that cannot start a call fails the run without placing any."""
    [result] = await create_agent_test_results(run.id, [(None, SETUP_CHECK, 1)])
    await start_agent_test_result(result.id)
    started = time.monotonic()
    problems = await setup_problems(template)
    await finish_agent_test_result(
        result.id,
        not problems,
        None,
        None,
        problems,
        int((time.monotonic() - started) * 1000),
    )
    return problems


def _lead_data(scenarios: List[AgentTestScenario]) -> Dict[str, Any]:
    """The lead data Buddy wrote for the agent's own tests, enabled or not;
    never the missing-name test's, whose name is blank on purpose."""
    for s in scenarios:
        if (
            s.source == AgentTestScenarioSource.GENERATED
            and s.scenario.get("kind") != MISSING_NAME.key
        ):
            payload = (s.scenario.get("buddy") or {}).get("payload")
            if payload:
                return payload
    return {}


def _with_lead_data(
    scenario: AgentTestScenario, lead: Dict[str, Any]
) -> AgentTestScenario:
    """A scenario that states no lead data runs on Buddy's: an agent whose
    schema has no examples would otherwise call about an empty order."""
    buddy = scenario.scenario.get("buddy") or {}
    if not lead or buddy.get("payload"):
        return scenario
    mapping = {**scenario.scenario, "buddy": {**buddy, "payload": lead}}
    return scenario.model_copy(update={"scenario": mapping})


async def _run_scenarios(
    run: AgentTestRun, template: TemplateModel, scenarios: List[AgentTestScenario]
) -> List[RunResult]:
    """Every attempt of every scenario, each result row updated as it goes."""
    items = {s.id: from_mapping(s.scenario) for s in scenarios}
    results = await create_agent_test_results(
        run.id,
        [
            (scenario_id, item.name, attempt)
            for scenario_id, item in items.items()
            for attempt in range(1, item.runs + 1)
        ],
    )
    limit = asyncio.Semaphore(await AGENT_TESTS_CONCURRENCY())

    async def one(result: AgentTestResult) -> RunResult:
        async with limit:
            await start_agent_test_result(result.id)
            outcome = await _call(items[result.scenario_id or ""], result, template)
            await finish_agent_test_result(
                result.id,
                outcome.passed,
                outcome.lead_id,
                outcome.outcome,
                outcome.failures,
                outcome.duration_ms,
            )
            return outcome

    # A failed result write ends the run: its other calls must not run on.
    tasks = [asyncio.create_task(one(r)) for r in results]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _call(
    item: BuddyScenario, result: AgentTestResult, template: TemplateModel
) -> RunResult:
    """One test call. A call the agent failed fails: an agent that gets it
    wrong one time in two is wrong. Only a call the harness broke (the judge or
    the session erroring) is made once more, and counts by its retry."""
    first = await run_once(item, result.attempt, template)
    if first.passed or not first.harness_failed:
        return first
    return await run_once(item, result.attempt, template)


async def _scenarios(
    template: TemplateModel, run: AgentTestRun
) -> List[AgentTestScenario]:
    scenarios = await list_agent_test_scenarios(str(template.id))
    written = any(s.source == AgentTestScenarioSource.GENERATED for s in scenarios)
    if written and run.trigger != AgentTestTrigger.REWRITE:
        return scenarios
    generated = await generate_scenarios(template)
    await replace_generated_agent_test_scenarios(
        str(template.id), generated, run.requested_by
    )
    return await list_agent_test_scenarios(str(template.id))


async def _alert_failed(run: AgentTestRun) -> None:
    """A failed run is otherwise seen only on the Tests tab. Not when this
    content has a newer verdict (an admin passed it meanwhile)."""
    try:
        _, verdict = await get_agent_test_version_runs(run.template_id, run.fingerprint)
        template = await get_template_by_id(run.template_id)
        if template and (verdict is None or verdict.id == run.id):
            await emit_agent_tests_alert(
                "Breeze Buddy Agent Tests Failed",
                run.template_id,
                template.name,
                run.template_version,
                template.merchant_id,
                run.note or run.status.value,
                serving_note(template),
                run.requested_by,
            )
    except Exception as e:
        logger.warning(f"Agent test run {run.id} failure alert failed: {e}")
