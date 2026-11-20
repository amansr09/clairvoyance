"""Accessors for agent tests: stored scenarios, suite runs and their results."""

import json
from typing import Any, Dict, List, Optional, Tuple

from app.core.logger import logger
from app.database import db_connection
from app.database.decoder.breeze_buddy.agent_tests import (
    decode_agent_test_results,
    decode_agent_test_run,
    decode_agent_test_runs,
    decode_agent_test_scenario,
    decode_agent_test_scenarios,
)
from app.database.queries import run_parameterized_query
from app.database.queries.breeze_buddy.agent_tests import (
    claim_next_agent_test_run_query,
    delete_agent_test_scenario_query,
    delete_generated_agent_test_scenarios_query,
    fail_stale_agent_test_runs_query,
    finish_agent_test_result_query,
    finish_agent_test_run_query,
    get_agent_test_version_runs_query,
    insert_agent_test_override_query,
    insert_agent_test_results_query,
    insert_agent_test_scenario_query,
    list_agent_test_results_query,
    list_agent_test_runs_query,
    list_agent_test_scenarios_query,
    prune_agent_test_history_query,
    queue_agent_test_run_query,
    requeue_agent_test_run_query,
    set_pending_agent_test_runs_go_live_query,
    start_agent_test_result_query,
    update_agent_test_scenario_query,
)
from app.schemas.breeze_buddy.agent_tests import (
    AgentTestResult,
    AgentTestResultStatus,
    AgentTestRun,
    AgentTestRunStatus,
    AgentTestScenario,
    AgentTestScenarioSource,
    AgentTestTrigger,
)

PENDING_RUN_STATUSES = (AgentTestRunStatus.QUEUED, AgentTestRunStatus.RUNNING)


async def list_agent_test_scenarios(template_id: str) -> List[AgentTestScenario]:
    try:
        query, values = list_agent_test_scenarios_query(template_id)
        return decode_agent_test_scenarios(await run_parameterized_query(query, values))
    except Exception as e:
        logger.error(f"Error listing agent test scenarios for {template_id}: {e}")
        raise


async def create_agent_test_scenario(
    template_id: str,
    name: str,
    scenario: Dict[str, Any],
    source: AgentTestScenarioSource,
    created_by: Optional[str],
    is_enabled: bool = True,
) -> Optional[AgentTestScenario]:
    """The new scenario, or None when the template already has one of that name."""
    try:
        query, values = insert_agent_test_scenario_query(
            template_id,
            name,
            json.dumps(scenario),
            source.value,
            is_enabled,
            created_by,
        )
        rows = await run_parameterized_query(query, values)
        return decode_agent_test_scenario(rows[0]) if rows else None
    except Exception as e:
        logger.error(f"Error creating agent test scenario {name}: {e}")
        raise


async def replace_generated_agent_test_scenarios(
    template_id: str, scenarios: List[Dict[str, Any]], created_by: Optional[str]
) -> None:
    """Swap the template's generated scenarios for these in one transaction. A
    generated scenario whose name a custom one already has is skipped."""
    try:
        delete_query, delete_values = delete_generated_agent_test_scenarios_query(
            template_id
        )
        async with db_connection() as conn:
            async with conn.transaction():
                await conn.execute(delete_query, *delete_values)
                for scenario in scenarios:
                    query, values = insert_agent_test_scenario_query(
                        template_id,
                        scenario["name"],
                        json.dumps(scenario),
                        AgentTestScenarioSource.GENERATED.value,
                        True,
                        created_by,
                    )
                    await conn.execute(query, *values)
    except Exception as e:
        logger.error(f"Error storing generated scenarios for {template_id}: {e}")
        raise


async def update_agent_test_scenario(
    scenario_id: str,
    template_id: str,
    name: str,
    scenario: Dict[str, Any],
    is_enabled: bool,
) -> Optional[AgentTestScenario]:
    try:
        query, values = update_agent_test_scenario_query(
            scenario_id, template_id, name, json.dumps(scenario), is_enabled
        )
        rows = await run_parameterized_query(query, values)
        return decode_agent_test_scenario(rows[0]) if rows else None
    except Exception as e:
        logger.error(f"Error updating agent test scenario {scenario_id}: {e}")
        raise


async def delete_agent_test_scenario(scenario_id: str, template_id: str) -> bool:
    try:
        query, values = delete_agent_test_scenario_query(scenario_id, template_id)
        return bool(await run_parameterized_query(query, values))
    except Exception as e:
        logger.error(f"Error deleting agent test scenario {scenario_id}: {e}")
        raise


async def queue_agent_test_run(
    template_id: str,
    fingerprint: str,
    template_version: int,
    go_live: bool,
    trigger: AgentTestTrigger,
    requested_by: Optional[str],
) -> AgentTestRun:
    """Queue a run, cancelling the template's older queued runs with it."""
    try:
        query, values = queue_agent_test_run_query(
            template_id,
            fingerprint,
            template_version,
            go_live,
            trigger.value,
            requested_by,
        )
        rows = await run_parameterized_query(query, values)
        if not rows:
            raise RuntimeError("The run was not queued")
        return decode_agent_test_run(rows[0])
    except Exception as e:
        logger.error(f"Error queueing agent tests for {template_id}: {e}")
        raise


async def record_agent_test_override(
    template_id: str,
    fingerprint: str,
    template_version: int,
    requested_by: str,
    reason: str,
) -> AgentTestRun:
    try:
        query, values = insert_agent_test_override_query(
            template_id, fingerprint, template_version, requested_by, reason
        )
        rows = await run_parameterized_query(query, values)
        return decode_agent_test_run(rows[0])
    except Exception as e:
        logger.error(f"Error recording agent test override for {template_id}: {e}")
        raise


async def claim_next_agent_test_run() -> Optional[AgentTestRun]:
    try:
        query, values = claim_next_agent_test_run_query()
        rows = await run_parameterized_query(query, values)
        return decode_agent_test_run(rows[0]) if rows else None
    except Exception as e:
        logger.error(f"Error claiming an agent test run: {e}")
        raise


async def finish_agent_test_run(
    run_id: str, status: AgentTestRunStatus, note: Optional[str] = None
) -> Optional[AgentTestRun]:
    """The finished run, or None when it was no longer RUNNING."""
    try:
        query, values = finish_agent_test_run_query(run_id, status.value, note)
        rows = await run_parameterized_query(query, values)
        return decode_agent_test_run(rows[0]) if rows else None
    except Exception as e:
        logger.error(f"Error finishing agent test run {run_id}: {e}")
        raise


async def requeue_agent_test_run(run_id: str) -> None:
    try:
        query, values = requeue_agent_test_run_query(run_id)
        await run_parameterized_query(query, values)
    except Exception as e:
        logger.error(f"Error requeuing agent test run {run_id}: {e}")
        raise


async def prune_agent_test_history(
    template_id: str, keep: int, live_fingerprint: Optional[str]
) -> None:
    try:
        query, values = prune_agent_test_history_query(
            template_id, keep, live_fingerprint
        )
        rows = await run_parameterized_query(query, values)
        if rows and (rows[0]["runs"] or rows[0]["leads"]):
            logger.info(
                f"Pruned {rows[0]['runs']} agent test run(s) and "
                f"{rows[0]['leads']} test-call lead(s) of template {template_id}"
            )
    except Exception as e:
        logger.error(f"Error pruning agent test history of {template_id}: {e}")
        raise


async def fail_stale_agent_test_runs(stale_minutes: int) -> None:
    try:
        query, values = fail_stale_agent_test_runs_query(stale_minutes)
        rows = await run_parameterized_query(query, values)
        if rows:
            logger.warning(f"Closed {len(rows)} stale agent test run(s) as ERROR")
    except Exception as e:
        logger.error(f"Error closing stale agent test runs: {e}")
        raise


async def set_pending_agent_test_runs_go_live(
    template_id: str, fingerprint: Optional[str], go_live: bool
) -> None:
    try:
        query, values = set_pending_agent_test_runs_go_live_query(
            template_id, fingerprint, go_live
        )
        await run_parameterized_query(query, values)
    except Exception as e:
        logger.error(f"Error setting go_live on runs of {template_id}: {e}")
        raise


async def list_agent_test_runs(template_id: str, limit: int) -> List[AgentTestRun]:
    try:
        query, values = list_agent_test_runs_query(template_id, limit)
        return decode_agent_test_runs(await run_parameterized_query(query, values))
    except Exception as e:
        logger.error(f"Error listing agent test runs for {template_id}: {e}")
        raise


async def get_agent_test_version_runs(
    template_id: str, fingerprint: str
) -> Tuple[Optional[AgentTestRun], Optional[AgentTestRun]]:
    """A version's latest pending run and latest finished run."""
    try:
        query, values = get_agent_test_version_runs_query(template_id, fingerprint)
        runs = decode_agent_test_runs(await run_parameterized_query(query, values))
        pending = next((r for r in runs if r.status in PENDING_RUN_STATUSES), None)
        finished = next((r for r in runs if r.status not in PENDING_RUN_STATUSES), None)
        return pending, finished
    except Exception as e:
        logger.error(f"Error reading agent test runs for {template_id}: {e}")
        raise


async def create_agent_test_results(
    run_id: str, rows: List[Tuple[Optional[str], str, int]]
) -> List[AgentTestResult]:
    """One QUEUED result per (scenario id, scenario name, attempt)."""
    try:
        query, values = insert_agent_test_results_query(
            run_id,
            [r[0] for r in rows],
            [r[1] for r in rows],
            [r[2] for r in rows],
        )
        return decode_agent_test_results(await run_parameterized_query(query, values))
    except Exception as e:
        logger.error(f"Error creating results for agent test run {run_id}: {e}")
        raise


async def start_agent_test_result(result_id: str) -> None:
    try:
        query, values = start_agent_test_result_query(result_id)
        await run_parameterized_query(query, values)
    except Exception as e:
        logger.error(f"Error starting agent test result {result_id}: {e}")
        raise


async def finish_agent_test_result(
    result_id: str,
    passed: bool,
    lead_id: Optional[str],
    outcome: Optional[str],
    failures: List[str],
    duration_ms: int,
) -> None:
    status = AgentTestResultStatus.PASSED if passed else AgentTestResultStatus.FAILED
    try:
        query, values = finish_agent_test_result_query(
            result_id,
            status.value,
            lead_id,
            outcome,
            json.dumps(failures),
            duration_ms,
        )
        await run_parameterized_query(query, values)
    except Exception as e:
        logger.error(f"Error finishing agent test result {result_id}: {e}")
        raise


async def list_agent_test_results(run_id: str) -> List[AgentTestResult]:
    try:
        query, values = list_agent_test_results_query(run_id)
        return decode_agent_test_results(await run_parameterized_query(query, values))
    except Exception as e:
        logger.error(f"Error listing results of agent test run {run_id}: {e}")
        raise
