"""Decoders for agent_test_scenario, agent_test_run and agent_test_result rows."""

import json
from typing import Any, List, Optional

import asyncpg

from app.schemas.breeze_buddy.agent_tests import (
    AgentTestResult,
    AgentTestRun,
    AgentTestScenario,
)


def _jsonb(value: Any) -> Any:
    """asyncpg returns JSONB as str: no codec is registered on the pool."""
    return json.loads(value) if isinstance(value, str) else value


def _str(value: Any) -> Optional[str]:
    return str(value) if value is not None else None


def decode_agent_test_scenario(row: asyncpg.Record) -> AgentTestScenario:
    return AgentTestScenario(
        id=str(row["id"]),
        template_id=str(row["template_id"]),
        name=row["name"],
        scenario=_jsonb(row["scenario"]) or {},
        source=row["source"],
        is_enabled=row["is_enabled"],
        created_by=row["created_by"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def decode_agent_test_run(row: asyncpg.Record) -> AgentTestRun:
    return AgentTestRun(
        id=str(row["id"]),
        template_id=str(row["template_id"]),
        fingerprint=row["fingerprint"],
        template_version=row["template_version"],
        status=row["status"],
        go_live=row["go_live"],
        trigger=row["trigger"],
        requested_by=row["requested_by"],
        note=row["note"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def decode_agent_test_result(row: asyncpg.Record) -> AgentTestResult:
    return AgentTestResult(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        scenario_id=_str(row["scenario_id"]),
        scenario_name=row["scenario_name"],
        attempt=row["attempt"],
        status=row["status"],
        lead_id=_str(row["lead_id"]),
        outcome=row["outcome"],
        failures=_jsonb(row["failures"]) or [],
        duration_ms=row["duration_ms"],
    )


def decode_agent_test_scenarios(
    rows: Optional[List[asyncpg.Record]],
) -> List[AgentTestScenario]:
    return [decode_agent_test_scenario(row) for row in rows or []]


def decode_agent_test_runs(rows: Optional[List[asyncpg.Record]]) -> List[AgentTestRun]:
    return [decode_agent_test_run(row) for row in rows or []]


def decode_agent_test_results(
    rows: Optional[List[asyncpg.Record]],
) -> List[AgentTestResult]:
    return [decode_agent_test_result(row) for row in rows or []]
