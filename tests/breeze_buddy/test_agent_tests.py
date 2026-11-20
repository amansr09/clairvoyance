"""Automatic agent tests: the go-live gate, scenario writing, the suite and its
worker, the scenario runner and the checks on each test call's lead, the eval
route's token, and the template saves that queue runs. The live suite runs end
to end against a server; these pin what decides a verdict and when an agent
goes live."""

from __future__ import annotations

import asyncio
import glob
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, WebSocket
from pipecat.evals.results import (
    EvalAssertionFailure,
    EvalScriptResult,
    EvalSimulationResult,
)
from pipecat.evals.script import EvalScriptScenario, EvalSendAfter
from pipecat.evals.simulation import EvalSimulationScenario

from app.ai.voice.agents.breeze_buddy.agent.flow import load_template_config
from app.ai.voice.agents.breeze_buddy.dispatch import alerts as dispatch_alerts
from app.ai.voice.agents.breeze_buddy.handlers.internal import (
    agent_transfer as transfer_handler,
)
from app.ai.voice.agents.breeze_buddy.managers import utils as manager_utils
from app.ai.voice.agents.breeze_buddy.managers.utils import (
    ensure_realtime_opening_line_cached,
    realtime_opening_line_applies,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests import (
    access,
    checklist,
    gate,
    generate,
    runner,
    scenario as scenario_module,
    suite,
    worker,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.gate import (
    SavePlan,
    Verdict,
    decide_save,
    fingerprint,
    template_fingerprint,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.runner import (
    RunResult,
    _example_payload,
    _lead_failures,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.scenario import (
    MAX_DURATION_S,
    MAX_RUNS,
    MAX_TURNS,
    BuddyScenario,
    BuddySpec,
    from_mapping,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.setup_check import SETUP_CHECK
from app.ai.voice.agents.breeze_buddy.template import loader as loader_module
from app.ai.voice.agents.breeze_buddy.template.types import (
    CreateTemplateRequest,
    ReplaceTemplateRequest,
    TemplateModel,
)
from app.ai.voice.agents.breeze_buddy.types.models import PushLeadRequest
from app.ai.voice.llm.types import RealtimeLLMProvider
from app.api.routers.breeze_buddy import eval as eval_route
from app.api.routers.breeze_buddy.leads import handlers as lead_handlers
from app.api.routers.breeze_buddy.templates import handlers as template_handlers
from app.api.routers.breeze_buddy.templates.agent_tests import (
    handlers as test_handlers,
)
from app.api.routers.breeze_buddy.templates.version import (
    handlers as version_handlers,
)
from app.database.accessor.breeze_buddy import template as template_accessor
from app.database.decoder.breeze_buddy.template import decode_template
from app.database.queries.breeze_buddy.agent_tests import (
    prune_agent_test_history_query,
)
from app.database.queries.breeze_buddy.template import restore_template_head_query
from app.database.queries.breeze_buddy.template_version import (
    prune_template_versions_query,
)
from app.schemas.breeze_buddy.agent_tests import (
    AgentTestResult,
    AgentTestRun,
    AgentTestRunStatus,
    AgentTestScenario,
    AgentTestScenarioRequest,
    AgentTestScenarioSource,
    AgentTestState,
    AgentTestTrigger,
    OverrideAgentTestsRequest,
)
from app.schemas.breeze_buddy.auth import UserInfo, UserRole
from app.schemas.breeze_buddy.core import ExecutionMode, LeadCallStatus, LeadCallTracker

TEMPLATE_ID = "11111111-1111-1111-1111-111111111111"
EXAMPLES = "app/ai/voice/agents/breeze_buddy/examples/templates/*.json"

FLOW: Dict[str, Any] = {
    "initial_node": "greet",
    "nodes": [
        {
            "node_name": "greet",
            "task_messages": [{"role": "system", "content": "Confirm the order."}],
            "functions": [
                {
                    "function_name": "confirm_order",
                    "description": "The customer confirms.",
                    "transition_to": "end",
                    "hooks": [
                        {
                            "name": "update_outcome_in_database",
                            "expected_fields": {
                                "outcome": {"value": "CONFIRM", "source": "static"}
                            },
                        }
                    ],
                },
                {
                    "function_name": "user_busy",
                    "description": "The customer is busy.",
                    "transition_to": "end",
                    "hooks": [
                        {
                            "name": "update_outcome_in_database",
                            "expected_fields": {
                                "outcome": {"value": "BUSY", "source": "static"}
                            },
                        }
                    ],
                },
            ],
        },
        {"node_name": "end", "task_messages": [], "functions": []},
    ],
}


def _template(**overrides: Any) -> TemplateModel:
    values: Dict[str, Any] = {
        "id": TEMPLATE_ID,
        "reseller_id": "breeze",
        "merchant_id": "shop",
        "name": "cod-confirm",
        "flow": json.loads(json.dumps(FLOW)),
        "expected_payload_schema": {
            "customer_name": {"type": "string", "example": "Priya"},
            "shop_name": {"type": "string"},
        },
        "is_active": False,
        "current_version": 3,
    }
    values.update(overrides)
    return TemplateModel(**values)


def _run(status: AgentTestRunStatus, **overrides: Any) -> AgentTestRun:
    values: Dict[str, Any] = {
        "id": "run-1",
        "template_id": TEMPLATE_ID,
        "fingerprint": template_fingerprint(_template()),
        "template_version": 3,
        "status": status,
        "go_live": False,
        "trigger": AgentTestTrigger.CREATED,
        "created_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return AgentTestRun(**values)


def _user(role: UserRole = UserRole.ADMIN) -> UserInfo:
    return UserInfo(id="u1", username="asha", role=role)


# --- the gate ---------------------------------------------------------------


def test_a_paused_agent_stays_paused_and_its_new_content_is_tested():
    plan = decide_save(wants_live=False, was_live=False, verdict=Verdict())
    assert plan == SavePlan(is_active=False, queue=True, go_live=False)


def test_pausing_clears_going_live_but_never_queues_a_judged_version():
    passed = Verdict(finished=_run(AgentTestRunStatus.PASSED))
    assert decide_save(False, True, passed) == SavePlan(
        is_active=False, queue=False, go_live=False
    )


def test_a_draft_asking_to_go_live_waits_for_its_first_run():
    plan = decide_save(wants_live=True, was_live=False, verdict=Verdict())
    assert plan == SavePlan(is_active=False, queue=True, go_live=True)


def test_a_draft_goes_live_at_once_when_this_content_passed():
    for status in (AgentTestRunStatus.PASSED, AgentTestRunStatus.OVERRIDDEN):
        plan = decide_save(True, False, Verdict(finished=_run(status)))
        assert plan == SavePlan(is_active=True)


def test_a_draft_with_a_run_in_flight_points_that_run_at_going_live():
    plan = decide_save(True, False, Verdict(pending=_run(AgentTestRunStatus.RUNNING)))
    assert plan == SavePlan(is_active=False, go_live=True)


def test_going_live_on_a_failed_version_is_refused():
    plan = decide_save(True, False, Verdict(finished=_run(AgentTestRunStatus.FAILED)))
    assert not plan.is_active and plan.refused


def test_a_version_whose_run_broke_is_run_again_rather_than_refused():
    plan = decide_save(True, False, Verdict(finished=_run(AgentTestRunStatus.ERROR)))
    assert plan == SavePlan(is_active=False, queue=True, go_live=True)


def test_a_live_agents_edit_is_a_draft_until_it_passes():
    """Real calls keep the version they run now; the draft goes live on a pass."""
    plan = decide_save(True, True, Verdict(), keep_serving=3)
    assert plan == SavePlan(
        is_active=True, published_version=3, queue=True, go_live=True
    )


def test_a_live_agents_failed_draft_never_reaches_real_calls():
    failed = Verdict(finished=_run(AgentTestRunStatus.FAILED))
    plan = decide_save(True, True, failed, keep_serving=3)
    assert plan == SavePlan(is_active=True, published_version=3, go_live=True)


def test_a_live_agents_edit_that_already_passed_goes_live_at_once():
    passed = Verdict(finished=_run(AgentTestRunStatus.PASSED))
    assert decide_save(True, True, passed, keep_serving=3) == SavePlan(is_active=True)


def test_saving_what_real_calls_already_run_keeps_it_live():
    """Nothing older to fall back to: a legacy live agent's rename, say."""
    assert decide_save(True, True, Verdict()) == SavePlan(is_active=True, queue=True)


def test_resuming_with_a_failed_draft_goes_live_on_the_published_version():
    failed = Verdict(finished=_run(AgentTestRunStatus.FAILED))
    plan = decide_save(True, False, failed, keep_serving=3)
    assert plan == SavePlan(is_active=True, published_version=3, go_live=True)


def test_pausing_keeps_the_published_version_for_when_it_resumes():
    plan = decide_save(False, True, Verdict(), keep_serving=3)
    assert plan == SavePlan(
        is_active=False, published_version=3, queue=True, go_live=False
    )


async def test_a_live_agents_edit_keeps_real_calls_on_its_live_version(monkeypatch):
    monkeypatch.setattr(gate, "get_live_template_by_id", AsyncMock())
    live = _template(is_active=True, current_version=3)
    edited = "e" * 64
    assert await gate._serving_version(live, edited) == 3
    assert await gate._serving_version(live, template_fingerprint(live)) is None
    assert await gate._serving_version(_template(), edited) is None


async def test_a_drafts_next_edit_keeps_real_calls_on_the_published_version(
    monkeypatch,
):
    published = _template(is_active=True, current_version=2)
    monkeypatch.setattr(
        gate, "get_live_template_by_id", AsyncMock(return_value=published)
    )
    draft = _template(is_active=True, current_version=4, published_version=2)
    assert await gate._serving_version(draft, "e" * 64) == 2
    # Edited back to what real calls run: nothing is a draft any more.
    assert await gate._serving_version(draft, template_fingerprint(published)) is None


def test_the_state_shown_prefers_the_run_in_flight():
    assert Verdict().state == AgentTestState.UNTESTED
    finished = _run(AgentTestRunStatus.FAILED)
    assert Verdict(finished=finished).state == AgentTestState.FAILED
    pending = _run(AgentTestRunStatus.QUEUED)
    assert Verdict(pending, finished).state == AgentTestState.QUEUED
    overridden = Verdict(finished=_run(AgentTestRunStatus.OVERRIDDEN))
    assert overridden.state == AgentTestState.OVERRIDDEN and overridden.green


def test_only_phone_voice_agents_are_testable():
    assert gate.testable(FLOW, ["voice"], None)
    assert gate.testable(FLOW, [], None)  # no channels is a voice agent
    # A chat agent's voice is in-widget: it never takes a phone call.
    assert not gate.testable(FLOW, ["voice", "chat"], None)
    assert not gate.testable(FLOW, ["chat"], None)
    assert not gate.testable({**FLOW, "mode": "ivr"}, ["voice"], None)
    # A test call would reach an MCP server for real.
    mcp = {"mcp": {"servers": [{"name": "shop", "url": "https://shop.example"}]}}
    assert not gate.testable(FLOW, ["voice"], mcp)
    assert gate.testable(FLOW, ["voice"], {"mcp": {"servers": []}})


async def test_the_gate_follows_the_live_flag(monkeypatch):
    monkeypatch.setattr(gate, "AGENT_TESTS_ENABLED", AsyncMock(return_value=False))
    assert not await gate.gated(FLOW, ["voice"], None)
    monkeypatch.setattr(gate, "AGENT_TESTS_ENABLED", AsyncMock(return_value=True))
    assert await gate.gated(FLOW, ["voice"], None)
    assert not await gate.gated(FLOW, ["chat"], None)


def test_the_fingerprint_follows_content_not_the_pause_state_or_version():
    base = template_fingerprint(_template())
    assert template_fingerprint(_template(is_active=True, current_version=9)) == base
    edited = json.loads(json.dumps(FLOW))
    edited["nodes"][0]["task_messages"][0]["content"] = "Cancel the order."
    assert template_fingerprint(_template(flow=edited)) != base


def test_content_about_to_be_saved_fingerprints_as_it_will_read_back():
    """The PUT gate fingerprints the request; the suite, the saved row. Every
    example template must give both the same value, or a passed agent could
    never go live."""
    checked = 0
    for path in glob.glob(EXAMPLES):
        data = json.load(open(path))
        data.setdefault("reseller_id", "breeze")
        data.setdefault("name", "example")
        try:
            request = CreateTemplateRequest.model_validate(data)
        except Exception:
            continue
        configurations = (
            request.configurations.model_dump(
                exclude_none=True, mode="json", context={"reveal_secrets": True}
            )
            if request.configurations
            else None
        )
        row = {
            "id": TEMPLATE_ID,
            "reseller_id": "breeze",
            "merchant_id": None,
            "name": "example",
            "flow": json.dumps(request.flow),
            "expected_payload_schema": json.dumps(request.expected_payload_schema),
            "expected_callback_response_schema": json.dumps(
                request.expected_callback_response_schema
            ),
            "configurations": json.dumps(configurations) if configurations else None,
            "secrets": None,
            "telephony_number_id": None,
            "is_active": True,
            "supported_channels": ["voice"],
            "current_version": 1,
            "created_at": None,
            "updated_at": None,
        }
        saved = decode_template(cast(Any, row))
        assert saved is not None
        assert template_fingerprint(saved) == fingerprint(
            request.flow,
            configurations,
            request.expected_payload_schema,
            request.expected_callback_response_schema,
        ), path
        checked += 1
    assert checked >= 10


async def test_applying_a_pause_clears_going_live_and_queues_the_content(monkeypatch):
    set_go_live = AsyncMock(return_value=1)
    queue = AsyncMock(return_value=_run(AgentTestRunStatus.QUEUED))
    monkeypatch.setattr(gate, "set_pending_agent_test_runs_go_live", set_go_live)
    monkeypatch.setattr(gate, "queue_agent_test_run", queue)
    template = _template()

    await gate.apply_save(
        template,
        SavePlan(is_active=False, queue=True, go_live=False),
        AgentTestTrigger.EDITED,
        "asha",
    )

    set_go_live.assert_awaited_once_with(TEMPLATE_ID, None, False)
    queue.assert_awaited_once_with(
        TEMPLATE_ID,
        template_fingerprint(template),
        3,
        False,
        AgentTestTrigger.EDITED,
        "asha",
    )


async def test_applying_go_live_marks_this_versions_run_in_flight(monkeypatch):
    set_go_live = AsyncMock(return_value=1)
    queue = AsyncMock()
    monkeypatch.setattr(gate, "set_pending_agent_test_runs_go_live", set_go_live)
    monkeypatch.setattr(gate, "queue_agent_test_run", queue)
    template = _template()

    run = await gate.apply_save(
        template, SavePlan(is_active=False, go_live=True), AgentTestTrigger.EDITED, None
    )

    assert run is None
    queue.assert_not_awaited()
    set_go_live.assert_awaited_once_with(
        TEMPLATE_ID, template_fingerprint(template), True
    )


# --- writing scenarios -------------------------------------------------------


def _written(**overrides: Any) -> Dict[str, Any]:
    test = {
        "kind": "happy_path",
        "name": "Happy Path",
        "persona": "Priya, polite, short Hinglish sentences.",
        "goal": "Confirm the order.",
        "success": "the agent reads the order and records it as confirmed",
        "outcome": "CONFIRM",
        "visits": ["end", "made_up_node"],
    }
    test.update(overrides)
    return test


def test_written_tests_become_scenarios_bound_to_what_the_template_has():
    template = _template()
    scenarios = generate.assemble(
        [_written()],
        {"shop_name": "Zeel Fashion", "not_a_field": "x"},
        template,
        checklist.checklist(template),
    )

    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert (scenario["name"], scenario["kind"]) == ("happy-path", "happy_path")
    assert scenario["goal"].endswith(generate.WAIT_FOR_AGENT)
    assert scenario["buddy"] == {
        "visits": ["end"],
        "http": [generate.ANSWER_EVERY_REQUEST],
        "outcome": "CONFIRM",
        "payload": {"shop_name": "Zeel Fashion"},
    }


def test_an_outcome_the_template_cannot_record_is_not_checked():
    template = _template()
    kinds = checklist.checklist(template)
    [scenario] = generate.assemble(
        [_written(kind="angry", outcome="REFUNDED")], {}, template, kinds
    )
    assert "outcome" not in scenario["buddy"]
    # A happy path is the outcome it reaches: one the call cannot end with is dropped.
    assert generate.assemble([_written(outcome="REFUNDED")], {}, template, kinds) == []


def test_unusable_and_duplicate_tests_are_dropped():
    template = _template()
    scenarios = generate.assemble(
        [
            _written(),
            _written(name="happy path"),
            _written(name="second-confirm"),
            _written(name="no-goal", goal=""),
            _written(name="", persona="x"),
            _written(name="not-on-the-list", kind="human"),
            "not a test",
        ],
        {},
        template,
        checklist.checklist(template),
    )
    assert [s["name"] for s in scenarios] == ["happy-path"]


def test_outcomes_are_read_from_the_outcome_hooks():
    assert checklist.outcomes(FLOW) == {"CONFIRM", "BUSY"}
    described = generate.describe(_template())
    assert described["outcomes"] == ["BUSY", "CONFIRM"]
    assert described["nodes"][0]["functions"][0] == {
        "name": "confirm_order",
        "when": "The customer confirms.",
        "records_outcome": "CONFIRM",
        "next_node": "end",
        "asks_for": [],
    }


def test_only_an_outcome_a_call_can_end_with_is_expected():
    """An address update recorded on the way to the confirm step is overwritten
    by CONFIRM: a scenario expecting it could never pass."""
    flow = json.loads(json.dumps(FLOW))
    flow["nodes"].insert(
        1,
        {
            "node_name": "address",
            "functions": [
                {
                    "function_name": "address_changed",
                    "transition_to": "greet",
                    "hooks": [
                        {
                            "name": "update_outcome_in_database",
                            "expected_fields": {
                                "outcome": {
                                    "value": "ADDRESS_UPDATED",
                                    "source": "static",
                                }
                            },
                        }
                    ],
                }
            ],
        },
    )
    assert checklist.outcomes(flow) == {"CONFIRM", "BUSY", "ADDRESS_UPDATED"}
    assert checklist.final_outcomes(flow) == {"CONFIRM", "BUSY"}
    assert generate.describe(_template(flow=flow))["outcomes"] == ["BUSY", "CONFIRM"]


def test_the_writers_reply_is_read_even_inside_a_code_fence():
    reply = 'Here you go:\n```json\n{"payload": {}, "tests": []}\n```'
    assert generate._parse(reply) == {"payload": {}, "tests": []}
    with pytest.raises(ValueError):
        generate._parse("I cannot help with that.")


async def test_writing_no_usable_scenario_is_an_error(monkeypatch):
    llm = SimpleNamespace(
        run_inference=AsyncMock(return_value='{"payload": {}, "tests": []}')
    )
    monkeypatch.setattr(generate, "build_writer", AsyncMock(return_value=llm))
    with pytest.raises(ValueError):
        await generate.generate_scenarios(_template())


async def test_a_situation_the_writer_leaves_out_fails_the_run(monkeypatch):
    # A suite without its safety tests must not pass as complete.
    reply = json.dumps({"payload": {}, "tests": [_written()]})
    llm = SimpleNamespace(run_inference=AsyncMock(return_value=reply))
    monkeypatch.setattr(generate, "build_writer", AsyncMock(return_value=llm))

    with pytest.raises(ValueError, match="wrong_person") as left_out:
        await generate.generate_scenarios(_template())

    assert "happy path to BUSY" in str(left_out.value)
    assert llm.run_inference.await_count == 2


# --- the suite ----------------------------------------------------------------


def _scenario_row(name: str = "happy-path") -> AgentTestScenario:
    template = _template()
    [mapping] = generate.assemble(
        [_written(name=name)], {}, template, checklist.checklist(template)
    )
    now = datetime.now(timezone.utc)
    return AgentTestScenario(
        id=f"scenario-{name}",
        template_id=TEMPLATE_ID,
        name=name,
        scenario=mapping,
        source="generated",
        is_enabled=True,
        created_at=now,
        updated_at=now,
    )


def _result(scenario: AgentTestScenario) -> AgentTestResult:
    return AgentTestResult(
        id=f"result-{scenario.name}",
        run_id="run-1",
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        attempt=1,
        status="QUEUED",
    )


class SuiteDb:
    """The suite's accessors, recording what it wrote."""

    def __init__(
        self,
        monkeypatch,
        template: Optional[TemplateModel],
        scenarios: List[AgentTestScenario],
        passed: bool,
        go_live: bool = True,
        finish_returns_run: bool = True,
    ):
        self.finished: List[Any] = []
        self.result_writes: List[Any] = []
        self.activate = AsyncMock(return_value=True)
        self.alert = AsyncMock()
        self.queue = AsyncMock()
        self.generate = AsyncMock(return_value=[])
        self.setup = AsyncMock(return_value=[])
        self.run_once = AsyncMock(
            side_effect=lambda item, attempt, *a: RunResult(
                "lead-1", passed, [] if passed else ["x"]
            )
        )

        self.stored = list(scenarios)
        self.listed = AsyncMock(side_effect=lambda _: self.stored)

        async def write(template_id, generated, created_by):
            self.stored = [*scenarios, _scenario_row()]
            return [_scenario_row()]

        async def finish_run(run_id, status, note=None):
            self.finished.append((status, note))
            if not finish_returns_run:
                return None
            return _run(status, go_live=go_live, note=note)

        async def create_results(run_id, rows):
            if rows[0][1] == SETUP_CHECK:
                return [
                    AgentTestResult(
                        id="result-setup",
                        run_id=run_id,
                        scenario_name=SETUP_CHECK,
                        attempt=1,
                        status="QUEUED",
                    )
                ]
            by_id = {s.id: s for s in [*scenarios, _scenario_row()]}
            return [_result(by_id[scenario_id]) for scenario_id, *_ in rows]

        async def finish_result(*args):
            self.result_writes.append(args)

        for name, value in {
            "get_template_by_id": AsyncMock(return_value=template),
            "list_agent_test_scenarios": self.listed,
            "replace_generated_agent_test_scenarios": AsyncMock(side_effect=write),
            "generate_scenarios": self.generate,
            "create_agent_test_results": create_results,
            "start_agent_test_result": AsyncMock(),
            "finish_agent_test_result": finish_result,
            "finish_agent_test_run": finish_run,
            "run_once": self.run_once,
            "setup_problems": self.setup,
            "AGENT_TESTS_CONCURRENCY": AsyncMock(return_value=5),
            "prune_agent_test_history": AsyncMock(),
            "get_live_template_by_id": AsyncMock(return_value=template),
            "queue_agent_test_run": self.queue,
            "emit_agent_tests_alert": self.alert,
            "get_agent_test_version_runs": AsyncMock(return_value=(None, None)),
        }.items():
            monkeypatch.setattr(suite, name, value)
        # Going live is the gate's, shared with the admin override.
        for name, value in {
            "get_template_by_id": AsyncMock(return_value=template),
            "activate_template_version": self.activate,
            "invalidate_template": AsyncMock(),
            "spawn_realtime_opening_line_regeneration": MagicMock(),
        }.items():
            monkeypatch.setattr(gate, name, value)


async def test_a_passing_run_turns_the_unchanged_draft_on(monkeypatch):
    db = SuiteDb(monkeypatch, _template(), [_scenario_row()], passed=True)

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    assert db.finished == [(AgentTestRunStatus.PASSED, None)]
    db.activate.assert_awaited_once_with(TEMPLATE_ID, 3)
    # The setup check, then the scenario: both recorded as passed.
    assert [w[1] for w in db.result_writes] == [True, True]


async def test_a_failing_run_leaves_the_draft_off(monkeypatch):
    db = SuiteDb(monkeypatch, _template(), [_scenario_row()], passed=False)

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    assert db.finished == [(AgentTestRunStatus.FAILED, "1 of 1 scenario runs failed")]
    db.activate.assert_not_awaited()


async def test_a_run_not_asked_to_go_live_only_records_its_verdict(monkeypatch):
    db = SuiteDb(
        monkeypatch, _template(), [_scenario_row()], passed=True, go_live=False
    )
    await suite.run_suite(_run(AgentTestRunStatus.RUNNING))
    db.activate.assert_not_awaited()


async def test_a_run_the_stale_sweep_closed_never_turns_the_agent_on(monkeypatch):
    db = SuiteDb(
        monkeypatch,
        _template(),
        [_scenario_row()],
        passed=True,
        finish_returns_run=False,
    )
    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))
    db.activate.assert_not_awaited()
    assert db.finished == [(AgentTestRunStatus.PASSED, None)]


async def test_an_agent_edited_after_its_run_was_queued_is_not_tested(monkeypatch):
    edited = json.loads(json.dumps(FLOW))
    edited["initial_node"] = "end"
    db = SuiteDb(monkeypatch, _template(flow=edited), [_scenario_row()], passed=True)

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    assert db.finished[0][0] == AgentTestRunStatus.CANCELLED
    db.run_once.assert_not_awaited()


async def test_an_agent_edited_during_its_run_does_not_go_live(monkeypatch):
    db = SuiteDb(monkeypatch, _template(), [_scenario_row()], passed=True)
    edited = json.loads(json.dumps(FLOW))
    edited["initial_node"] = "end"
    # Edited as its last call ended: the content going live is not what passed.
    monkeypatch.setattr(
        gate, "get_template_by_id", AsyncMock(return_value=_template(flow=edited))
    )

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    assert db.finished == [(AgentTestRunStatus.PASSED, None)]
    db.activate.assert_not_awaited()


def _saved_during_the_run(monkeypatch, saved: TemplateModel) -> None:
    reads = iter([_template(), saved])
    monkeypatch.setattr(
        suite, "get_template_by_id", AsyncMock(side_effect=lambda _: next(reads))
    )


async def test_an_agent_saved_during_its_run_gets_no_verdict(monkeypatch):
    # Its later calls ran the new content: passing says nothing of either.
    db = SuiteDb(monkeypatch, _template(), [_scenario_row()], passed=True)
    edited = json.loads(json.dumps(FLOW))
    edited["initial_node"] = "end"
    _saved_during_the_run(monkeypatch, _template(flow=edited, current_version=4))

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    assert db.finished == [
        (AgentTestRunStatus.CANCELLED, "The agent changed during its tests")
    ]
    db.queue.assert_not_awaited()
    db.activate.assert_not_awaited()


async def test_the_same_content_saved_during_its_run_is_run_again(monkeypatch):
    # A pause, a rename or an undo back to it: the content still needs a verdict.
    db = SuiteDb(monkeypatch, _template(), [_scenario_row()], passed=True)
    _saved_during_the_run(monkeypatch, _template(current_version=4))

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    assert db.finished[0][0] == AgentTestRunStatus.CANCELLED
    db.queue.assert_awaited_once_with(
        TEMPLATE_ID,
        template_fingerprint(_template()),
        4,
        True,
        AgentTestTrigger.MANUAL,
        None,
    )
    db.activate.assert_not_awaited()


async def test_a_call_the_harness_broke_twice_leaves_the_run_unfinished(monkeypatch):
    # Not a verdict on the agent: no FAILED, and going live runs it again.
    db = SuiteDb(monkeypatch, _template(), [_scenario_row()], passed=True)
    db.run_once.side_effect = lambda *a: RunResult(
        "lead-1", False, ["<connect> refused"], harness_broke=True
    )

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    assert db.run_once.await_count == 2
    assert db.finished == [
        (AgentTestRunStatus.ERROR, "1 of 1 test calls did not run: <connect> refused")
    ]
    db.activate.assert_not_awaited()


async def test_custom_scenarios_alone_still_get_the_checklist_written(monkeypatch):
    # One custom test must not stand in for every situation the agent meets.
    custom = _scenario_row("my-check").model_copy(
        update={"source": AgentTestScenarioSource.CUSTOM}
    )
    db = SuiteDb(monkeypatch, _template(), [custom], passed=True)

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    db.generate.assert_awaited_once()


def test_the_missing_name_tests_blank_name_is_not_the_agents_lead_data():
    template = _template()
    kinds = [checklist.MISSING_NAME, checklist.HAPPY_PATH]
    rows = []
    for mapping in generate.assemble(
        [_written(kind="missing_name", name="no-name"), _written()], {}, template, kinds
    ):
        now = datetime.now(timezone.utc)
        rows.append(
            AgentTestScenario(
                id=f"scenario-{mapping['name']}",
                template_id=TEMPLATE_ID,
                name=mapping["name"],
                scenario=mapping,
                source=AgentTestScenarioSource.GENERATED,
                is_enabled=True,
                created_at=now,
                updated_at=now,
            )
        )

    assert rows[0].scenario["buddy"]["payload"] == {"customer_name": ""}
    assert suite._lead_data(rows) == {}


async def test_a_failed_result_write_stops_the_runs_other_calls(monkeypatch):
    # Calls left running would write into a closed run, beside the next one.
    db = SuiteDb(
        monkeypatch,
        _template(),
        [_scenario_row("first"), _scenario_row("second")],
        passed=True,
    )
    monkeypatch.setattr(suite, "AGENT_TESTS_CONCURRENCY", AsyncMock(return_value=1))
    completed: List[str] = []

    async def call(item, attempt, *args):
        await asyncio.sleep(0.05)
        completed.append(item.name)
        return RunResult("lead-1", True)

    async def finish_result(result_id, *args):
        if result_id != "result-setup":
            raise RuntimeError("db down")

    db.run_once.side_effect = call
    monkeypatch.setattr(suite, "finish_agent_test_result", finish_result)

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))
    await asyncio.sleep(0.2)

    assert completed == ["first"]
    assert db.finished == [(AgentTestRunStatus.ERROR, "The test run failed: db down")]


async def test_the_first_run_writes_the_agents_scenarios(monkeypatch):
    db = SuiteDb(monkeypatch, _template(), [], passed=True)
    await suite.run_suite(_run(AgentTestRunStatus.RUNNING))
    db.generate.assert_awaited_once()
    assert db.finished == [(AgentTestRunStatus.PASSED, None)]


async def test_a_rewrite_run_rewrites_the_generated_scenarios_and_runs_them_all(
    monkeypatch,
):
    db = SuiteDb(monkeypatch, _template(), [_scenario_row()], passed=True)

    await suite.run_suite(
        _run(AgentTestRunStatus.RUNNING, trigger=AgentTestTrigger.REWRITE)
    )

    db.generate.assert_awaited_once()
    # Custom scenarios stay, so the suite runs the stored set, not just the new.
    assert db.listed.await_count == 2
    assert db.finished == [(AgentTestRunStatus.PASSED, None)]


async def test_a_run_that_breaks_is_closed_as_an_error(monkeypatch):
    db = SuiteDb(monkeypatch, _template(), [], passed=True)
    # Braces, as a writer's JSON carries: logging the error must not break.
    db.generate.side_effect = ValueError('No usable scenarios in {"tests": []}')

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING))

    [(status, note)] = db.finished
    assert status == AgentTestRunStatus.ERROR and "No usable scenarios" in note


# --- the worker ---------------------------------------------------------------


def _patch_worker(monkeypatch, *runs: AgentTestRun, enabled: bool = True):
    db = SimpleNamespace(
        claim=AsyncMock(side_effect=[*runs, None]),
        fail_stale=AsyncMock(return_value=0),
        requeue=AsyncMock(),
        run_suite=AsyncMock(),
    )
    monkeypatch.setattr(worker, "claim_next_agent_test_run", db.claim)
    monkeypatch.setattr(worker, "fail_stale_agent_test_runs", db.fail_stale)
    monkeypatch.setattr(worker, "requeue_agent_test_run", db.requeue)
    monkeypatch.setattr(worker, "run_suite", db.run_suite)
    monkeypatch.setattr(
        worker, "AGENT_TESTS_STALE_RUN_MINUTES", AsyncMock(return_value=20)
    )
    monkeypatch.setattr(
        worker, "AGENT_TESTS_POLL_INTERVAL_SECONDS", AsyncMock(return_value=5)
    )
    monkeypatch.setattr(worker, "AGENT_TESTS_ENABLED", AsyncMock(return_value=enabled))
    return db


async def test_a_worker_pass_closes_stale_runs_then_runs_every_queued_one(
    monkeypatch,
):
    runs = [_run(AgentTestRunStatus.RUNNING, id="a"), _run(AgentTestRunStatus.RUNNING)]
    db = _patch_worker(monkeypatch, *runs)

    await worker.drain_agent_test_runs()

    db.fail_stale.assert_awaited_once_with(20)
    assert [c.args[0].id for c in db.run_suite.await_args_list] == ["a", "run-1"]


async def test_nothing_runs_while_agent_tests_are_off(monkeypatch):
    db = _patch_worker(monkeypatch, _run(AgentTestRunStatus.RUNNING), enabled=False)

    await worker.drain_agent_test_runs()

    db.claim.assert_not_awaited()
    db.fail_stale.assert_not_awaited()


async def test_turning_agent_tests_off_stops_the_backlog_after_the_current_run(
    monkeypatch,
):
    runs = [_run(AgentTestRunStatus.RUNNING, id="a"), _run(AgentTestRunStatus.RUNNING)]
    db = _patch_worker(monkeypatch, *runs)
    monkeypatch.setattr(
        worker, "AGENT_TESTS_ENABLED", AsyncMock(side_effect=[True, False])
    )

    await worker.drain_agent_test_runs()

    assert [c.args[0].id for c in db.run_suite.await_args_list] == ["a"]
    assert db.claim.await_count == 1


async def test_a_run_cut_off_by_a_stopping_pod_goes_back_to_the_queue(monkeypatch):
    db = _patch_worker(monkeypatch, _run(AgentTestRunStatus.RUNNING))
    started = asyncio.Event()

    async def suite_in_progress(run):
        started.set()
        await asyncio.Event().wait()

    db.run_suite.side_effect = suite_in_progress
    await worker.start_agent_tests_worker()
    await asyncio.wait_for(started.wait(), 1)

    await worker.stop_agent_tests_worker()

    db.requeue.assert_awaited_once_with("run-1")
    assert worker._task is None


@pytest.mark.parametrize("worker_pod", [True, False])
async def test_the_pre_stop_drain_stops_the_worker_only_on_its_pods(
    monkeypatch, worker_pod
):
    from app import main

    stop = AsyncMock()
    monkeypatch.setattr(main, "ENABLE_AGENT_TESTS_WORKER", worker_pod)
    monkeypatch.setattr(main, "stop_agent_tests_worker", stop)
    monkeypatch.setattr(main, "_is_draining", False)

    await main.drain()

    assert stop.await_count == int(worker_pod)


# --- the eval route's token ---------------------------------------------------


def test_a_session_token_opens_only_its_own_lead(monkeypatch):
    monkeypatch.setattr(access, "JWT_SECRET_KEY", "secret")
    token = access.session_token("lead-1")
    assert access.token_matches("lead-1", token)
    assert not access.token_matches("lead-2", token)
    assert not access.token_matches("lead-1", None)
    assert not access.token_matches("lead-1", "0" * 64)


def test_a_token_is_refused_once_expired_and_never_raises(monkeypatch):
    monkeypatch.setattr(access, "JWT_SECRET_KEY", "secret")
    expired = access.session_token("lead-1", int(time.time()) - 1)
    assert not access.token_matches("lead-1", expired)
    # Client input: refused, never raised on.
    for token in ["é" * 10, "9" * 5000 + ".x", "x.y", ".", "123"]:
        assert not access.token_matches("lead-1", token)


def test_no_token_matches_without_a_signing_key(monkeypatch):
    monkeypatch.setattr(access, "JWT_SECRET_KEY", "")
    assert not access.token_matches("lead-1", access.session_token("lead-1"))


async def test_a_test_call_cannot_transfer_to_an_agent_it_cannot_test(monkeypatch):
    # An MCP agent's tools would be called for real; the call stays put.
    target = SimpleNamespace(
        flow=FLOW,
        supported_channels=["voice"],
        configurations={"mcp": {"servers": [{"name": "shop"}]}},
    )
    load = AsyncMock(return_value=(target, {}))
    monkeypatch.setattr(transfer_handler.FlowConfigLoader, "load_template", load)
    monkeypatch.setattr(transfer_handler, "validate_template_compat", MagicMock())
    bot = SimpleNamespace(
        pending_transfer=None,
        transfer_count=0,
        is_eval_mode=True,
        lead=_lead(status=LeadCallStatus.PROCESSING),
    )
    config = SimpleNamespace(
        agent_transfer=SimpleNamespace(targets={"support": "tpl-b"}, max_transfers=1)
    )

    result = await transfer_handler.connect_to_agent(
        cast(Any, SimpleNamespace(bot=bot)), {"target": "support"}, cast(Any, config)
    )

    assert result == {"status": "error", "error": "target cannot take a test call"}


async def test_the_eval_route_refuses_a_session_without_its_token(monkeypatch):
    eval_bot = AsyncMock()
    monkeypatch.setattr(eval_route, "eval_bot", eval_bot)
    monkeypatch.setattr(eval_route, "AGENT_TESTS_ENABLED", AsyncMock(return_value=True))
    monkeypatch.setattr(eval_route, "token_matches", lambda lead, token: False)
    websocket = SimpleNamespace(close=AsyncMock())

    await eval_route.eval_websocket_handler(
        "6f1c1c1e-0000-4000-8000-000000000000",
        cast(WebSocket, websocket),
        token="bad",
    )

    eval_bot.assert_not_awaited()
    assert websocket.close.await_args.kwargs["code"] == 4003


# --- template saves ------------------------------------------------------------


def _patch_template_writes(monkeypatch, saved: TemplateModel) -> Dict[str, Any]:
    mocks: Dict[str, Any] = {
        "get_template_in_scope": AsyncMock(return_value=None),
        "refuse_bad_provider_accounts": AsyncMock(),
        "create_template": AsyncMock(return_value=saved),
        "replace_template": AsyncMock(return_value=saved),
        "get_template_by_id": AsyncMock(return_value=saved),
        "invalidate_template": AsyncMock(),
        "spawn_realtime_opening_line_regeneration": MagicMock(),
        "spawn_background_task": MagicMock(),
        "validate_template_access": MagicMock(),
        "apply_save": AsyncMock(return_value=_run(AgentTestRunStatus.QUEUED)),
    }
    for name, value in mocks.items():
        monkeypatch.setattr(template_handlers, name, value)
    monkeypatch.setattr(gate, "AGENT_TESTS_ENABLED", AsyncMock(return_value=True))
    return mocks


def _request(cls, is_active: bool):
    template = _template()
    return cls(
        reseller_id="breeze",
        merchant_id="shop",
        name="cod-confirm",
        flow=template.flow,
        expected_payload_schema=template.expected_payload_schema,
        is_active=is_active,
    )


async def test_a_new_agent_is_saved_as_a_draft_and_its_tests_queued(monkeypatch):
    mocks = _patch_template_writes(monkeypatch, _template())

    await template_handlers.create_template_handler(
        _request(CreateTemplateRequest, is_active=True), _user()
    )

    assert mocks["create_template"].await_args.kwargs["is_active"] is False
    plan = mocks["apply_save"].await_args.args[1]
    assert plan == SavePlan(is_active=False, queue=True, go_live=True)
    assert mocks["apply_save"].await_args.args[2] == AgentTestTrigger.CREATED


async def test_with_agent_tests_off_a_new_agent_is_saved_as_asked(monkeypatch):
    mocks = _patch_template_writes(monkeypatch, _template())
    monkeypatch.setattr(gate, "AGENT_TESTS_ENABLED", AsyncMock(return_value=False))

    await template_handlers.create_template_handler(
        _request(CreateTemplateRequest, is_active=True), _user()
    )

    assert mocks["create_template"].await_args.kwargs["is_active"] is True
    mocks["apply_save"].assert_not_awaited()


async def test_going_live_on_a_failed_version_refuses_the_save(monkeypatch):
    mocks = _patch_template_writes(monkeypatch, _template())
    monkeypatch.setattr(
        gate,
        "get_agent_test_version_runs",
        AsyncMock(return_value=(None, _run(AgentTestRunStatus.FAILED))),
    )

    with pytest.raises(HTTPException) as refused:
        await template_handlers.replace_template_handler(
            TEMPLATE_ID, _request(ReplaceTemplateRequest, is_active=True), _user()
        )

    assert refused.value.status_code == 409
    mocks["replace_template"].assert_not_awaited()


async def test_going_live_on_a_passed_version_saves_it_live(monkeypatch):
    mocks = _patch_template_writes(monkeypatch, _template())
    versions = AsyncMock(return_value=(None, _run(AgentTestRunStatus.PASSED)))
    monkeypatch.setattr(gate, "get_agent_test_version_runs", versions)

    await template_handlers.replace_template_handler(
        TEMPLATE_ID, _request(ReplaceTemplateRequest, is_active=True), _user()
    )

    versions.assert_awaited_once_with(TEMPLATE_ID, template_fingerprint(_template()))
    assert mocks["replace_template"].await_args.kwargs["is_active"] is True
    assert mocks["apply_save"].await_args.args[1] == SavePlan(is_active=True)


async def test_a_failure_to_queue_tests_does_not_fail_the_save(monkeypatch):
    mocks = _patch_template_writes(monkeypatch, _template())
    mocks["apply_save"].side_effect = RuntimeError("db down")

    response = await template_handlers.create_template_handler(
        _request(CreateTemplateRequest, is_active=True), _user()
    )

    assert response["status"] == "success"


async def test_a_pause_clears_going_live_even_with_agent_tests_off(monkeypatch):
    # A run queued earlier must not turn the agent on once tests are back.
    mocks = _patch_template_writes(monkeypatch, _template())
    monkeypatch.setattr(gate, "AGENT_TESTS_ENABLED", AsyncMock(return_value=False))
    order: List[str] = []
    clear = AsyncMock(side_effect=lambda *a: order.append("clear"))
    monkeypatch.setattr(template_handlers, "set_pending_agent_test_runs_go_live", clear)
    mocks["replace_template"].side_effect = lambda **kw: (
        order.append("save") or _template()
    )

    await template_handlers.replace_template_handler(
        TEMPLATE_ID, _request(ReplaceTemplateRequest, is_active=False), _user()
    )

    clear.assert_awaited_once_with(TEMPLATE_ID, None, False)
    assert order == ["clear", "save"]


async def test_a_pause_that_cannot_clear_going_live_is_refused(monkeypatch):
    mocks = _patch_template_writes(monkeypatch, _template())
    clear = AsyncMock(side_effect=RuntimeError("db down"))
    monkeypatch.setattr(template_handlers, "set_pending_agent_test_runs_go_live", clear)

    with pytest.raises(HTTPException):
        await template_handlers.replace_template_handler(
            TEMPLATE_ID, _request(ReplaceTemplateRequest, is_active=False), _user()
        )

    mocks["replace_template"].assert_not_awaited()


# --- the tests API -------------------------------------------------------------


def _patch_template_load(monkeypatch, template: TemplateModel) -> None:
    """The tests API loads and authorizes through the version handlers."""
    monkeypatch.setattr(
        version_handlers, "get_template_by_id", AsyncMock(return_value=template)
    )
    monkeypatch.setattr(version_handlers, "validate_template_access", MagicMock())


async def test_only_an_admin_can_pass_a_version_by_hand(monkeypatch):
    record = AsyncMock()
    _patch_template_load(monkeypatch, _template())
    monkeypatch.setattr(test_handlers, "record_agent_test_override", record)
    monkeypatch.setattr(gate, "AGENT_TESTS_ENABLED", AsyncMock(return_value=True))
    with pytest.raises(HTTPException) as refused:
        await test_handlers.override_agent_tests_handler(
            TEMPLATE_ID,
            OverrideAgentTestsRequest(reason="checked on a real call"),
            _user(UserRole.MERCHANT),
        )
    assert refused.value.status_code == 403
    record.assert_not_awaited()


@pytest.mark.parametrize(
    "template, publishes",
    [
        # A live agent's failed draft: real calls move to it at once.
        (_template(is_active=True, current_version=4, published_version=2), True),
        # A paused agent only counts as passed; going live stays a choice.
        (_template(is_active=False), False),
    ],
)
async def test_an_admin_pass_publishes_a_live_agents_draft(
    monkeypatch, template, publishes
):
    _patch_template_load(monkeypatch, template)
    monkeypatch.setattr(gate, "AGENT_TESTS_ENABLED", AsyncMock(return_value=True))
    monkeypatch.setattr(test_handlers, "record_agent_test_override", AsyncMock())
    publish = AsyncMock(return_value=True)
    monkeypatch.setattr(test_handlers, "go_live", publish)
    monkeypatch.setattr(
        test_handlers, "get_template_by_id", AsyncMock(return_value=template)
    )
    spawned = MagicMock(side_effect=lambda coro, name: coro.close())
    monkeypatch.setattr(test_handlers, "spawn_background_task", spawned)

    await test_handlers.override_agent_tests_handler(
        TEMPLATE_ID, OverrideAgentTestsRequest(reason="checked on a real call"), _user()
    )

    assert publish.await_count == int(publishes)
    # Every pass by hand is announced, like a template update.
    assert spawned.call_args.kwargs["name"].startswith("agent-tests-override-alert")


@pytest.mark.parametrize("passed, alerted", [(False, True), (True, False)])
async def test_a_failed_run_is_announced(monkeypatch, passed, alerted):
    db = SuiteDb(monkeypatch, _template(), [_scenario_row()], passed=passed)

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    assert db.alert.await_count == int(alerted)
    if alerted:
        title, *_, result, real_calls, _by = db.alert.await_args_list[0].args
        assert title == "Breeze Buddy Agent Tests Failed"
        assert result == "1 of 1 scenario runs failed"
        assert real_calls == "None: the agent is paused"


async def test_a_failure_an_admin_already_passed_is_not_announced(monkeypatch):
    db = SuiteDb(monkeypatch, _template(), [_scenario_row()], passed=False)
    passed_by_hand = _run(AgentTestRunStatus.OVERRIDDEN, id="run-override")
    monkeypatch.setattr(
        suite,
        "get_agent_test_version_runs",
        AsyncMock(return_value=(None, passed_by_hand)),
    )

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    db.alert.assert_not_awaited()


async def test_the_alert_keeps_author_text_inert(monkeypatch):
    send = AsyncMock()
    monkeypatch.setattr(dispatch_alerts.slack_alert, "send", send)

    await dispatch_alerts.emit_agent_tests_alert(
        "t", TEMPLATE_ID, "<!channel> agent", 3, None, "<b>", "v2", "editor"
    )

    values = [f["value"] for f in send.await_args_list[0].kwargs["fields"]]
    assert all("<" not in v for v in values)


def _patch_queue(monkeypatch, template: TemplateModel) -> AsyncMock:
    queue = AsyncMock(return_value=_run(AgentTestRunStatus.QUEUED))
    _patch_template_load(monkeypatch, template)
    monkeypatch.setattr(test_handlers, "queue_agent_test_run", queue)
    monkeypatch.setattr(gate, "AGENT_TESTS_ENABLED", AsyncMock(return_value=True))
    return queue


@pytest.mark.parametrize(
    "is_active",
    # A live agent's run publishes its draft on a pass; a paused agent stays
    # paused, whatever changed, until it is put live.
    [True, False],
)
async def test_only_a_live_agents_run_goes_live_on_a_pass(monkeypatch, is_active):
    queue = _patch_queue(monkeypatch, _template(is_active=is_active))

    await test_handlers.run_agent_tests_handler(TEMPLATE_ID, _user())

    [queued] = queue.await_args_list
    assert queued.args[3] is is_active


async def test_rewriting_the_scenarios_queues_a_rewrite_run(monkeypatch):
    queue = _patch_queue(monkeypatch, _template())

    await test_handlers.generate_agent_test_scenarios_handler(TEMPLATE_ID, _user())

    [queued] = queue.await_args_list
    assert queued.args[4] == AgentTestTrigger.REWRITE


async def test_the_overview_shows_the_run_that_judged_this_version(monkeypatch):
    template = _template()
    finished = _run(AgentTestRunStatus.PASSED)
    newer = _run(AgentTestRunStatus.FAILED, id="run-2", fingerprint="f" * 64)
    _patch_template_load(monkeypatch, template)
    for name, value in {
        "get_agent_test_version_runs": AsyncMock(return_value=(None, finished)),
        "list_agent_test_runs": AsyncMock(return_value=[newer, finished]),
        "list_agent_test_results": AsyncMock(return_value=[]),
        "list_agent_test_scenarios": AsyncMock(return_value=[]),
    }.items():
        monkeypatch.setattr(test_handlers, name, value)
    monkeypatch.setattr(gate, "AGENT_TESTS_ENABLED", AsyncMock(return_value=True))

    overview = await test_handlers.get_agent_tests_handler(TEMPLATE_ID, _user())

    assert overview.state == AgentTestState.PASSED
    assert overview.run is not None and overview.run.id == "run-1"
    assert [r.id for r in overview.runs] == ["run-2", "run-1"]
    assert overview.enabled


async def test_a_scenario_is_capped_in_runs_turns_and_size(monkeypatch):
    base = {"name": "s", "persona": "p", "goal": "g", "success": "x"}
    for mapping in [
        {**base, "buddy": {"runs": MAX_RUNS + 1}},
        {**base, "max_turns": MAX_TURNS + 1},
        {**base, "max_duration_s": MAX_DURATION_S + 1},
    ]:
        with pytest.raises(ValueError, match="at most"):
            from_mapping(mapping)

    _patch_template_load(monkeypatch, _template())
    monkeypatch.setattr(gate, "AGENT_TESTS_ENABLED", AsyncMock(return_value=True))
    huge = AgentTestScenarioRequest(
        name="huge", scenario={**base, "persona": "x" * 40_000}, is_enabled=True
    )
    with pytest.raises(HTTPException) as refused:
        await test_handlers.create_agent_test_scenario_handler(
            TEMPLATE_ID, huge, _user()
        )
    assert refused.value.status_code == 400


async def test_an_agent_has_a_bounded_number_of_scenarios(monkeypatch):
    _patch_template_load(monkeypatch, _template())
    monkeypatch.setattr(gate, "AGENT_TESTS_ENABLED", AsyncMock(return_value=True))
    full = [_scenario_row(f"s-{i}") for i in range(test_handlers.MAX_SCENARIOS)]
    monkeypatch.setattr(
        test_handlers, "list_agent_test_scenarios", AsyncMock(return_value=full)
    )
    create = AsyncMock()
    monkeypatch.setattr(test_handlers, "create_agent_test_scenario", create)
    body = AgentTestScenarioRequest(
        name="one-more",
        scenario={"persona": "p", "goal": "g", "success": "x"},
        is_enabled=True,
    )

    with pytest.raises(HTTPException) as refused:
        await test_handlers.create_agent_test_scenario_handler(
            TEMPLATE_ID, body, _user()
        )

    assert refused.value.status_code == 409
    create.assert_not_awaited()


def test_a_blank_outcome_expects_none():
    assert BuddySpec(outcome="").outcomes == []
    assert BuddySpec(outcome=["", "CONFIRM"]).outcomes == ["CONFIRM"]


async def test_an_agent_with_a_draft_cannot_move_to_another_owner(monkeypatch):
    mocks = _patch_template_writes(monkeypatch, _template(is_active=True))
    monkeypatch.setattr(
        gate, "get_agent_test_version_runs", AsyncMock(return_value=(None, None))
    )
    request = _request(ReplaceTemplateRequest, is_active=True)
    request.flow = _edited_flow()
    request.merchant_id = "another-shop"

    with pytest.raises(HTTPException) as refused:
        await template_handlers.replace_template_handler(TEMPLATE_ID, request, _user())

    assert refused.value.status_code == 409
    mocks["replace_template"].assert_not_awaited()


@pytest.mark.parametrize(
    "mode, reads_live",
    [(ExecutionMode.TELEPHONY, True), (ExecutionMode.TELEPHONY_TEST, False)],
)
async def test_a_pushed_lead_is_checked_against_the_version_it_will_run(
    monkeypatch, mode, reads_live
):
    head, live = AsyncMock(return_value=None), AsyncMock(return_value=None)
    monkeypatch.setattr(lead_handlers, "get_template_by_id", head)
    monkeypatch.setattr(lead_handlers, "get_live_template_by_id", live)
    request = PushLeadRequest.model_validate(
        {
            "request_id": "r1",
            "payload": {},
            "template_id": TEMPLATE_ID,
            "reseller_id": "breeze",
            "execution_mode": mode,
        }
    )

    with pytest.raises(HTTPException):
        await lead_handlers.push_lead_handler(request, _user())

    assert (live if reads_live else head).await_count == 1
    assert (head if reads_live else live).await_count == 0


# --- drafts: what real calls run ------------------------------------------------


def _edited_flow() -> Dict[str, Any]:
    flow = json.loads(json.dumps(FLOW))
    flow["nodes"][0]["task_messages"][0]["content"] = "Confirm the order. Be brief."
    return flow


async def test_real_calls_run_the_published_versions_content(monkeypatch):
    head = _template(
        is_active=True,
        current_version=4,
        published_version=2,
        flow=_edited_flow(),
        secrets={"api_key": "rotated"},
    )
    published = _template(current_version=2)
    monkeypatch.setattr(
        template_accessor, "get_template_by_id", AsyncMock(return_value=head)
    )
    versions = AsyncMock(return_value=(published, None))
    monkeypatch.setattr(template_accessor, "get_template_version", versions)

    live = await template_accessor.get_live_template_by_id(TEMPLATE_ID)

    assert live is not None
    assert live.flow == FLOW
    assert live.secrets == {"api_key": "rotated"}
    assert live.is_active and not live.is_draft
    assert (live.current_version, live.published_version) == (2, None)
    versions.assert_awaited_once_with(TEMPLATE_ID, 2, "breeze", "shop")


async def test_without_a_draft_real_calls_run_the_head(monkeypatch):
    head = _template(is_active=True)
    monkeypatch.setattr(
        template_accessor, "get_template_by_id", AsyncMock(return_value=head)
    )
    versions = AsyncMock()
    monkeypatch.setattr(template_accessor, "get_template_version", versions)

    assert await template_accessor.get_live_template_by_id(TEMPLATE_ID) is head
    versions.assert_not_awaited()


async def test_an_unreadable_published_version_serves_the_head(monkeypatch):
    head = _template(current_version=4, published_version=2)
    monkeypatch.setattr(
        template_accessor, "get_template_by_id", AsyncMock(return_value=head)
    )
    monkeypatch.setattr(
        template_accessor, "get_template_version", AsyncMock(return_value=None)
    )
    assert await template_accessor.get_live_template_by_id(TEMPLATE_ID) is head


@pytest.mark.parametrize(
    "mode, draft",
    [
        (ExecutionMode.EVAL, True),
        (ExecutionMode.TELEPHONY_TEST, True),
        (ExecutionMode.DAILY_TEST, True),
        (ExecutionMode.TELEPHONY, False),
        (ExecutionMode.DAILY, False),
    ],
)
async def test_test_calls_run_the_draft_and_real_calls_the_published_version(
    monkeypatch, mode, draft
):
    loaded: List[str] = []

    async def head(template_id):
        loaded.append("head")
        return _template()

    async def live(template_id):
        loaded.append("live")
        return _template()

    monkeypatch.setattr(loader_module, "get_template_by_id", head)
    monkeypatch.setattr(loader_module, "get_live_template_by_id", live)
    lead = LeadCallTracker(
        id="lead-1",
        reseller_id="breeze",
        template="cod-confirm",
        template_id=TEMPLATE_ID,
        execution_mode=mode,
    )

    await load_template_config(lead)

    assert loaded == ["head" if draft else "live"]


async def test_a_live_agents_draft_is_published_when_it_passes(monkeypatch):
    draft = _template(is_active=True, current_version=3, published_version=2)
    db = SuiteDb(monkeypatch, draft, [_scenario_row()], passed=True)

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    db.activate.assert_awaited_once_with(TEMPLATE_ID, 3)


async def test_a_live_agent_without_a_draft_has_nothing_to_publish(monkeypatch):
    db = SuiteDb(monkeypatch, _template(is_active=True), [_scenario_row()], passed=True)
    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))
    db.activate.assert_not_awaited()


async def test_a_live_agents_prompt_edit_is_saved_as_a_draft(monkeypatch):
    live = _template(is_active=True, current_version=3)
    mocks = _patch_template_writes(monkeypatch, live)
    monkeypatch.setattr(
        gate, "get_agent_test_version_runs", AsyncMock(return_value=(None, None))
    )
    request = _request(ReplaceTemplateRequest, is_active=True)
    request.flow = _edited_flow()

    await template_handlers.replace_template_handler(TEMPLATE_ID, request, _user())

    written = mocks["replace_template"].await_args.kwargs
    assert written["is_active"] is True
    assert (written["set_published"], written["published_version"]) == (True, 3)
    plan = mocks["apply_save"].await_args.args[1]
    assert plan.queue and plan.go_live


def test_the_version_real_calls_run_is_never_pruned():
    query, _ = prune_template_versions_query(TEMPLATE_ID, 20, 10)
    assert "published_version" in query


def test_a_restored_draft_keeps_the_secrets_real_calls_use():
    # Real calls run the published flow with the head's secrets.
    query, values = restore_template_head_query(
        TEMPLATE_ID,
        "n",
        "{}",
        None,
        None,
        None,
        '{"k": "v2"}',
        ["voice"],
        None,
        "breeze",
        "shop",
        True,
        5,
    )
    assert "WHEN $12::boolean AND $13::integer IS NOT NULL" in query
    assert "THEN secrets ELSE $6::jsonb END" in query
    assert values[5] == '{"k": "v2"}'


def test_the_verdict_of_what_real_calls_run_is_never_pruned():
    # Saves that change no content still bump the version past retention.
    query, values = prune_agent_test_history_query(TEMPLATE_ID, 4, "fp-live")
    assert "live_verdict" in query
    assert "r.id NOT IN (SELECT id FROM live_verdict)" in query
    assert values == [TEMPLATE_ID, 4, "fp-live"]


def _gemini_greeting_template(is_draft: bool) -> Any:
    realtime = SimpleNamespace(provider=RealtimeLLMProvider.GEMINI)
    configurations = SimpleNamespace(
        initial_greeting="Namaste, this is Buddy.",
        llm_configurations=SimpleNamespace(realtime=realtime),
    )
    return SimpleNamespace(
        id=TEMPLATE_ID, is_draft=is_draft, configurations=configurations
    )


async def test_a_drafts_greeting_never_lands_in_the_shared_greeting_audio(
    monkeypatch,
):
    """The template key holds the opening line real calls play: a draft's
    neither reads nor regenerates it."""
    redis = SimpleNamespace(get=AsyncMock(return_value=b"audio"))
    monkeypatch.setattr(
        manager_utils, "get_redis_service", AsyncMock(return_value=redis)
    )
    published = _gemini_greeting_template(is_draft=False)
    draft = _gemini_greeting_template(is_draft=True)

    assert realtime_opening_line_applies(published)
    assert not realtime_opening_line_applies(draft)
    assert await ensure_realtime_opening_line_cached(published) == (
        "Namaste, this is Buddy."
    )
    assert await ensure_realtime_opening_line_cached(draft) is None
    assert redis.get.await_count == 1


def test_a_head_that_is_the_published_version_is_not_a_draft():
    assert _template(current_version=3, published_version=2).is_draft
    assert not _template(current_version=3, published_version=3).is_draft
    assert not _template(current_version=3).is_draft


# --- the checklist ------------------------------------------------------------


def _rich_flow() -> Dict[str, Any]:
    """A flow that meets every conditional situation."""
    flow = json.loads(json.dumps(FLOW))
    greet = flow["nodes"][0]
    greet["task_messages"][0]["content"] = "[en]: Hi there [hi]: Namaste"
    greet["functions"] += [
        {
            "function_name": "user_asked_callback_later",
            "transition_to": "end",
            "properties": {"callback_time": {"type": "string"}},
            "hooks": [
                {
                    "name": "update_outcome_in_database",
                    "expected_fields": {
                        "outcome": {"value": "CALLBACK", "source": "static"}
                    },
                },
                {"name": "notify", "http_request": {"url": "https://x.example"}},
            ],
        },
        {"function_name": "transfer_to_agent", "handler": "warm_transfer"},
    ]
    return flow


def test_every_agent_gets_the_common_situations():
    keys = [k.key for k in checklist.checklist(_template())]
    assert keys == [
        "happy_path",
        "busy",
        "wrong_person",
        "angry",
        "repeat",
        "off_topic",
        "robot",
        "changes_mind",
        "missing_name",
    ]


def test_an_end_of_call_webhook_alone_is_no_data_to_fetch():
    flow = {**json.loads(json.dumps(FLOW)), "end_conversation_callbacks": ["hook"]}
    keys = [k.key for k in checklist.checklist(_template(flow=flow))]
    assert "systems_down" not in keys


def test_situations_an_agent_can_meet_are_added_and_no_others():
    rich = [k.key for k in checklist.checklist(_template(flow=_rich_flow()))]
    for key in ("callback", "human", "corrects_details", "language_switch"):
        assert key in rich
    assert "systems_down" in rich
    lean = _template(
        flow={"initial_node": "greet", "nodes": [FLOW["nodes"][0]]},
        expected_payload_schema={"shop_name": {"type": "string"}},
    )
    lean_keys = [k.key for k in checklist.checklist(lean)]
    for key in ("callback", "human", "corrects_details", "language_switch"):
        assert key not in lean_keys
    assert "missing_name" not in lean_keys and "systems_down" not in lean_keys


def test_each_situation_gets_its_own_settings():
    template = _template(flow=_rich_flow())
    kinds = checklist.checklist(template)
    scenarios = generate.assemble(
        [
            _written(kind="systems_down", name="systems-down", outcome=None),
            _written(kind="missing_name", name="no-name", outcome=None),
        ],
        {"customer_name": "Priya", "shop_name": "Zeel"},
        template,
        kinds,
    )
    by_kind = {s["kind"]: s for s in scenarios}
    assert by_kind["systems_down"]["buddy"]["http"] == [generate.FAIL_EVERY_REQUEST]
    assert by_kind["missing_name"]["buddy"]["payload"]["customer_name"] == ""
    assert by_kind["systems_down"]["buddy"]["payload"]["customer_name"] == "Priya"


def test_what_is_still_to_write_is_every_uncovered_situation_and_outcome():
    template = _template()
    kinds = checklist.checklist(template)
    written = generate.assemble([_written()], {}, template, kinds)
    lines = generate.asks(kinds, {"CONFIRM", "BUSY"}, written)
    assert lines[0] == "happy_path: a cooperative person whose call ends with BUSY"
    assert not any("ends with CONFIRM" in line for line in lines)
    assert [line.split(":")[0] for line in lines[1:]] == [
        k.key for k in kinds if k.key != "happy_path"
    ]


async def test_the_writer_is_asked_again_for_what_it_left_out(monkeypatch):
    template = _template()
    kinds = checklist.checklist(template)
    first = {"payload": {}, "tests": [_written()]}
    rest = [
        _written(kind=k.key, name=k.key.replace("_", "-"), outcome=None)
        for k in kinds
        if k.key != "happy_path"
    ] + [_written(name="busy-path", outcome="BUSY")]
    replies = AsyncMock(
        side_effect=[json.dumps(first), json.dumps({"payload": {}, "tests": rest})]
    )
    monkeypatch.setattr(
        generate,
        "build_writer",
        AsyncMock(return_value=SimpleNamespace(run_inference=replies)),
    )

    scenarios = await generate.generate_scenarios(template)

    assert replies.await_count == 2
    asked_again = json.loads(replies.await_args_list[1].args[0].messages[0]["content"])
    assert "happy_path: a cooperative person whose call ends with CONFIRM" not in (
        asked_again["write"]
    )
    assert {s["kind"] for s in scenarios} == {k.key for k in kinds}


# --- the setup check -------------------------------------------------------------


async def test_the_setup_check_names_every_provider_that_cannot_start(monkeypatch):
    from app.ai.voice.agents.breeze_buddy.scenario_tests import setup_check

    async def refuse(block):
        raise ValueError("DEEPGRAM_API_KEY is required for deepgram STT")

    monkeypatch.setattr(
        setup_check,
        "account_blocks",
        lambda configurations: iter([("stt_configuration", object())]),
    )
    monkeypatch.setattr(
        setup_check,
        "accounts_for_template",
        lambda template: SimpleNamespace(get=refuse),
    )
    monkeypatch.setattr(
        setup_check,
        "get_llm_service",
        AsyncMock(side_effect=RuntimeError("deployment not found")),
    )

    problems = await setup_check.setup_problems(_template())

    assert problems == [
        "stt_configuration: DEEPGRAM_API_KEY is required for deepgram STT",
        "LLM: RuntimeError: deployment not found",
    ]


async def test_an_agent_that_cannot_start_a_call_fails_before_any_test_call(
    monkeypatch,
):
    db = SuiteDb(monkeypatch, _template(), [], passed=True)
    db.setup.return_value = ["LLM: RuntimeError: deployment not found"]

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    assert db.finished == [
        (
            AgentTestRunStatus.FAILED,
            "The agent cannot start a call: LLM: RuntimeError: deployment not found",
        )
    ]
    db.generate.assert_not_awaited()
    db.run_once.assert_not_awaited()
    db.activate.assert_not_awaited()


async def test_a_test_call_the_harness_broke_is_run_once_more(monkeypatch):
    db = SuiteDb(monkeypatch, _template(), [_scenario_row()], passed=True)
    db.run_once.side_effect = [
        RunResult("lead-1", False, ["judge call failed"]),
        RunResult("lead-2", True),
    ]

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING))

    assert db.run_once.await_count == 2
    assert db.finished == [(AgentTestRunStatus.PASSED, None)]


async def test_a_call_the_agent_fails_fails_its_scenario_with_no_retry(
    monkeypatch,
):
    """An agent that gets it wrong one time in two is wrong: no second call."""
    db = SuiteDb(monkeypatch, _template(), [_scenario_row()], passed=True)
    db.run_once.side_effect = [
        RunResult("lead-1", False, ["never read the price"]),
        RunResult("lead-2", True),
    ]

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING, go_live=True))

    assert db.run_once.await_count == 1
    assert db.finished == [(AgentTestRunStatus.FAILED, "1 of 1 scenario runs failed")]
    db.activate.assert_not_awaited()


def test_only_situations_about_the_ending_check_it():
    """An angry caller the agent wins over still confirms: that is not a failure."""
    template = _template()
    kinds = [checklist.ANGRY, checklist.HAPPY_PATH]
    angry, happy = generate.assemble(
        [_written(kind="angry", name="angry-caller"), _written()], {}, template, kinds
    )
    assert "outcome" not in angry["buddy"] and angry["buddy"]["visits"] == []
    assert happy["buddy"]["outcome"] == "CONFIRM" and happy["buddy"]["visits"] == [
        "end"
    ]


def test_an_ending_check_names_only_the_node_the_call_ends_in():
    """Passing a node on the way depends on how the caller words it: "yes it's
    me, but I'm busy" skips the order read-out."""
    [happy] = generate.assemble(
        [_written(visits=["greet", "end"])], {}, _template(), [checklist.HAPPY_PATH]
    )
    assert happy["buddy"]["visits"] == ["end"]


def test_safety_situations_make_two_calls():
    """A leak or a false promise one time in two is caught three in four."""
    template = _template()
    [wrong_person] = generate.assemble(
        [_written(kind="wrong_person", name="sister-answers", outcome=None)],
        {},
        template,
        checklist.checklist(template),
    )
    assert wrong_person["buddy"]["runs"] == 2
    assert from_mapping(wrong_person).runs == 2
    assert checklist.KINDS["off_topic"].runs == 2
    assert checklist.KINDS["busy"].runs == 1


async def test_a_custom_scenario_without_lead_data_runs_on_buddys(monkeypatch):
    generated = _scenario_row("written-by-buddy")
    generated.scenario["buddy"]["payload"] = {"customer_name": "Priya"}
    generated.is_enabled = False
    custom = _scenario_row("custom-one")
    custom.source = AgentTestScenarioSource.CUSTOM
    custom.scenario["buddy"].pop("payload", None)
    db = SuiteDb(monkeypatch, _template(), [generated, custom], passed=True)

    await suite.run_suite(_run(AgentTestRunStatus.RUNNING))

    [call] = db.run_once.await_args_list
    item = call.args[0]
    assert item.name == "custom-one"
    assert item.spec.payload == {"customer_name": "Priya"}
    assert db.finished == [(AgentTestRunStatus.PASSED, None)]


def test_the_template_api_reports_which_version_real_calls_run():
    from app.ai.voice.agents.breeze_buddy.utils.secrets import mask_template_secrets

    masked = mask_template_secrets(_template(current_version=4, published_version=2))
    assert (masked.current_version, masked.published_version) == (4, 2)


async def test_the_eval_route_refuses_every_session_while_tests_are_off(monkeypatch):
    eval_bot = AsyncMock()
    monkeypatch.setattr(eval_route, "eval_bot", eval_bot)
    monkeypatch.setattr(
        eval_route, "AGENT_TESTS_ENABLED", AsyncMock(return_value=False)
    )
    lead = "6f1c1c1e-0000-4000-8000-000000000000"
    websocket = SimpleNamespace(close=AsyncMock())

    await eval_route.eval_websocket_handler(
        lead, cast(WebSocket, websocket), token=access.session_token(lead)
    )

    eval_bot.assert_not_awaited()
    assert websocket.close.await_args.kwargs["code"] == 4003


# --- the scenario runner: stored scenarios, lead checks, failed runs -------


STORED: Dict[str, Dict[str, Any]] = {
    "busy": {
        "name": "busy",
        "buddy": {
            "payload": {"customer_name": "Asha"},
            "outcome": ["USER_BUSY", "BUSY"],
            "visits": ["end_conversation_node"],
        },
        "turns": [
            {
                "user": "Abhi busy hoon",
                "expect": [
                    {"event": "function_call", "calls": [{"name": "user_busy"}]}
                ],
            }
        ],
    },
    "opening": {
        "name": "opening",
        "turns": [
            {"expect": [{"event": "response"}]},
            {"user": "Haan boliye", "expect": [{"event": "response"}]},
        ],
    },
    "barge-in": {
        "name": "barge-in",
        "turns": [
            {
                "user": "Ruko",
                "send_after": {"event": "llm_started", "delay_ms": 300},
                "expect": [{"event": "response"}],
            }
        ],
    },
    "wants-time": {
        "name": "wants-time",
        "buddy": {"runs": 3},
        "persona": "A polite borrower who needs a week to pay.",
        "goal": "Ask for a week's time to pay the EMI.",
        "success": "the agent records a callback instead of pushing payment today",
    },
}


@pytest.fixture
def scenarios() -> Dict[str, BuddyScenario]:
    return {name: from_mapping(m) for name, m in STORED.items()}


def test_a_stored_scenario_reads_its_buddy_block(scenarios):
    spec = scenarios["busy"].spec

    assert spec.payload == {"customer_name": "Asha"}
    assert spec.outcomes == ["USER_BUSY", "BUSY"]


def test_the_first_caller_turn_waits_for_the_opening_line(scenarios):
    busy = cast(EvalScriptScenario, scenarios["busy"].scenario)

    assert busy.turns[0].send_after == EvalSendAfter(event="llm_response", delay_ms=0)


def test_a_scenario_that_schedules_its_first_turn_keeps_its_schedule(scenarios):
    opening = cast(EvalScriptScenario, scenarios["opening"].scenario)
    barge_in = cast(EvalScriptScenario, scenarios["barge-in"].scenario)

    assert opening.turns[0].send_after is None
    assert opening.turns[1].send_after is None
    assert barge_in.turns[0].send_after == EvalSendAfter(
        event="llm_started", delay_ms=300
    )


def test_a_scenario_makes_its_own_number_of_calls(scenarios):
    assert isinstance(scenarios["wants-time"].scenario, EvalSimulationScenario)
    assert scenarios["wants-time"].runs == 3
    assert scenarios["busy"].runs == 1


def test_a_script_that_could_outlast_the_limit_is_refused():
    # A turn waits up to 60 s for its events unless told less: eleven of them
    # could hold the shared worker past ten minutes.
    slow = [{"user": "Haan", "expect": [{"event": "response"}]}] * 11
    with pytest.raises(ValueError, match="600 s"):
        from_mapping({"name": "slow", "turns": slow})
    quick = [{"user": "Haan", "expect": [{"event": "response", "within_ms": 5000}]}]
    assert from_mapping({"name": "quick", "turns": quick * 11}).name == "quick"


def test_a_script_cannot_wait_a_day_before_a_turn():
    turns = [{"user": "hi", "send_after": {"delay_ms": 86_400_000}}]
    with pytest.raises(ValueError, match="600 s"):
        from_mapping({"name": "asleep", "turns": turns})


def test_an_audio_mode_scenario_is_refused(monkeypatch):
    spoken = SimpleNamespace(name="busy", bot_audio=True, user_audio=False)
    monkeypatch.setattr(
        scenario_module, "_scenario_from_mapping", lambda data, path: spoken
    )

    with pytest.raises(ValueError, match="not a text-mode scenario"):
        from_mapping(STORED["busy"])


def test_an_unknown_buddy_key_is_refused():
    with pytest.raises(ValueError):
        BuddySpec.model_validate({"outcomes": ["BUSY"]})


def test_a_fixture_typo_is_refused():
    with pytest.raises(ValueError):
        BuddySpec.model_validate({"http": [{"url": "*", "response": {"ok": 1}}]})


# ── checks on the finished lead ──────────────────────────────────────────


def _lead(
    status: LeadCallStatus = LeadCallStatus.FINISHED,
    outcome: Optional[str] = "USER_BUSY",
    nodes: Optional[List[str]] = None,
    http_calls: Optional[List[Dict[str, Any]]] = None,
) -> LeadCallTracker:
    nodes = ["initial", "end_conversation_node"] if nodes is None else nodes
    return LeadCallTracker(
        id="lead-1",
        reseller_id="res-1",
        template="flipkart-recovery",
        status=status,
        outcome=outcome,
        execution_mode=ExecutionMode.EVAL,
        metaData={
            "node_traversal": [{"node_name": n} for n in nodes],
            "eval": {"http_calls": http_calls or []},
        },
    )


def test_a_lead_that_ended_as_expected_has_no_failures(scenarios):
    assert _lead_failures(scenarios["busy"], _lead()) == []


def test_the_lead_must_finish(scenarios):
    assert _lead_failures(scenarios["busy"], None) == ["the call did not finish"]
    assert _lead_failures(
        scenarios["busy"], _lead(status=LeadCallStatus.PROCESSING)
    ) == ["the call did not finish"]


# A lead left PROCESSING counts as active against its template and is later
# swept as a stuck call, so every run leaves its lead FINISHED.


class _RefusedSession:
    @staticmethod
    def from_scenario(*args: Any, **kwargs: Any) -> "_RefusedSession":
        return _RefusedSession()

    async def run(self) -> Any:
        raise ConnectionError("refused")


class _UnbuildableSession:
    @staticmethod
    def from_scenario(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("bad scenario")


def _patch_run(monkeypatch, session: Any, judge: AsyncMock) -> SimpleNamespace:
    seen = SimpleNamespace(created=0, closed=[])

    async def create_lead(*args: Any) -> LeadCallTracker:
        seen.created += 1
        return _lead(status=LeadCallStatus.PROCESSING)

    async def close(lead_id: str) -> None:
        seen.closed.append(lead_id)

    monkeypatch.setattr(runner, "_create_eval_lead", create_lead)
    monkeypatch.setattr(runner, "build_judge", judge)
    monkeypatch.setattr(runner, "EvalSession", session)
    monkeypatch.setattr(runner, "_close_unfinished", close)
    return seen


async def _run_busy(scenarios) -> RunResult:
    return await runner.run_once(scenarios["busy"], 1, cast(Any, None))


@pytest.mark.parametrize(
    "session, error", [(_RefusedSession, "refused"), (_UnbuildableSession, "bad")]
)
async def test_a_run_that_fails_closes_its_lead_and_fails_alone(
    monkeypatch, scenarios, session, error
):
    # Raising out of the suite's gather would close the DB pool under the
    # runs still closing their own leads.
    seen = _patch_run(monkeypatch, session, AsyncMock(return_value=None))

    result = await _run_busy(scenarios)

    assert not result.passed
    assert error in result.failures[0]
    assert seen.closed == ["lead-1"]


async def test_a_judge_that_cannot_be_built_creates_no_lead(monkeypatch, scenarios):
    # Nothing closes a lead the session never reached: the sweeper only sees
    # leads the bot marked started.
    judge = AsyncMock(side_effect=ValueError("API key not found"))
    seen = _patch_run(monkeypatch, _RefusedSession, judge)

    result = await _run_busy(scenarios)

    assert not result.passed
    assert "API key not found" in result.failures[0]
    assert seen.created == 0


class _EndedSession:
    """A session that returns its result, as Pipecat does for a broken one."""

    result: Any = None

    @classmethod
    def from_scenario(cls, *args: Any, **kwargs: Any) -> "_EndedSession":
        return cls()

    async def run(self) -> Any:
        return type(self).result


def _ended(monkeypatch, result: Any) -> None:
    session = type("Session", (_EndedSession,), {"result": result})
    _patch_run(monkeypatch, session, AsyncMock(return_value=None))
    still_open = _lead(status=LeadCallStatus.PROCESSING)
    monkeypatch.setattr(runner, "_wait_finished", AsyncMock(return_value=still_open))


@pytest.mark.parametrize(
    "result",
    [
        EvalSimulationResult(
            simulation_name="busy",
            succeeded=False,
            error="failed to connect to ws://bot: ConnectionRefusedError",
        ),
        EvalScriptResult(
            scenario_name="busy",
            passed=False,
            failures=[
                EvalAssertionFailure(
                    -1, -1, "<connect>", "failed to connect", "connect_failed"
                )
            ],
        ),
    ],
)
async def test_a_call_the_harness_broke_is_not_the_agents_failure(
    monkeypatch, scenarios, result
):
    # Pipecat returns these rather than raising: retried, never a verdict.
    _ended(monkeypatch, result)

    outcome = await _run_busy(scenarios)

    assert outcome.harness_failed
    assert "the call did not finish" not in outcome.failures


async def test_a_call_the_agent_got_wrong_is_the_agents_failure(monkeypatch, scenarios):
    missed = EvalAssertionFailure(
        0, 0, "function_call", "user_busy was never called", "missing_function_call"
    )
    _ended(
        monkeypatch,
        EvalScriptResult(scenario_name="busy", passed=False, failures=[missed]),
    )

    outcome = await _run_busy(scenarios)

    assert not outcome.harness_failed
    assert "turn 0 function_call: user_busy was never called" in outcome.failures


async def test_a_test_call_that_never_ends_is_cut_off(monkeypatch, scenarios):
    class _Hung(_EndedSession):
        async def run(self) -> Any:
            await asyncio.sleep(60)

    _patch_run(monkeypatch, _Hung, AsyncMock(return_value=None))
    still_open = _lead(status=LeadCallStatus.PROCESSING)
    monkeypatch.setattr(runner, "_wait_finished", AsyncMock(return_value=still_open))
    monkeypatch.setattr(runner, "CALL_TIMEOUT_S", 0.05)

    outcome = await _run_busy(scenarios)

    assert outcome.harness_failed
    assert outcome.failures == ["the test call did not end within 0.05 s"]


@pytest.mark.parametrize(
    "status, closes",
    [(LeadCallStatus.PROCESSING, True), (LeadCallStatus.FINISHED, False)],
)
async def test_only_a_lead_still_open_is_closed(monkeypatch, status, closes):
    updates: List[Dict[str, Any]] = []

    async def update(**kwargs: Any) -> None:
        updates.append(kwargs)

    monkeypatch.setattr(
        runner, "get_lead_by_id", AsyncMock(return_value=_lead(status, outcome=None))
    )
    monkeypatch.setattr(runner, "update_lead_call_completion_details", update)

    await runner._close_unfinished("lead-1")

    assert bool(updates) is closes
    if closes:
        assert updates[0]["status"] == LeadCallStatus.FINISHED
        assert updates[0]["outcome"] == "UNKNOWN"
        assert updates[0]["meta_data"]["cleanup"] == "eval_unfinished"
        assert updates[0]["expected_status"] == LeadCallStatus.PROCESSING


def test_outcome_nodes_and_uncovered_requests_are_each_reported(scenarios):
    item = scenarios["busy"]
    item.spec.avoids = ["kyc_node"]
    lead = _lead(
        outcome="NOT_INTERESTED",
        nodes=["initial", "kyc_node"],
        http_calls=[
            {"method": "POST", "url": "https://m.example.com/a", "matched": True},
            {"method": "GET", "url": "https://m.example.com/b", "matched": False},
        ],
    )

    assert _lead_failures(item, lead) == [
        "outcome 'NOT_INTERESTED', expected USER_BUSY or BUSY",
        "never reached node 'end_conversation_node'",
        "entered node 'kyc_node'",
        "no fixture for GET https://m.example.com/b",
    ]


def test_the_template_examples_fill_the_payload():
    template = SimpleNamespace(
        expected_payload_schema={
            "customer_name": {"type": "string", "example": "Rohit"},
            "order_id": {"type": "string"},
            "legacy": "string",
        }
    )

    assert _example_payload(cast(Any, template)) == {"customer_name": "Rohit"}
