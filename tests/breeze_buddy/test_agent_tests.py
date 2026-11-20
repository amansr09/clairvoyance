"""Automatic agent tests: scenario writing, the scenario runner and the checks
on each test call's lead, and the eval route's token."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocket
from pipecat.evals.results import (
    EvalAssertionFailure,
    EvalScriptResult,
    EvalSimulationResult,
)
from pipecat.evals.script import EvalScriptScenario, EvalSendAfter
from pipecat.evals.simulation import EvalSimulationScenario

from app.ai.voice.agents.breeze_buddy.scenario_tests import (
    access,
    checklist,
    generate,
    runner,
    scenario as scenario_module,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.runner import (
    RunResult,
    _example_payload,
    _lead_failures,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.scenario import (
    BuddyScenario,
    BuddySpec,
    from_mapping,
)
from app.ai.voice.agents.breeze_buddy.template.types import (
    TemplateModel,
)
from app.api.routers.breeze_buddy import eval as eval_route
from app.schemas.breeze_buddy.core import ExecutionMode, LeadCallStatus, LeadCallTracker

TEMPLATE_ID = "11111111-1111-1111-1111-111111111111"

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


# --- the tests API -------------------------------------------------------------


def test_a_blank_outcome_expects_none():
    assert BuddySpec(outcome="").outcomes == []
    assert BuddySpec(outcome=["", "CONFIRM"]).outcomes == ["CONFIRM"]


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
