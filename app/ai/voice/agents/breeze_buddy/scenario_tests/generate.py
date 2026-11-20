"""Write an agent's scenarios from its template and the checklist.

The LLM only writes callers, for the situations the checklist names: who picks
up, what they want, and what the agent must do. Python assembles each into a
scenario, applies the situation's settings (failing requests, a blank name,
two calls), keeps outcomes and nodes the template really has, drops any the
harness cannot load, and asks once more for any situation the reply left out.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Set

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService

from app.ai.voice.agents.breeze_buddy.scenario_tests.checklist import (
    HAPPY_PATH,
    KINDS,
    MISSING_NAME,
    SYSTEMS_DOWN,
    Kind,
    checklist,
    end_nodes,
    final_outcomes,
    name_field,
    outcome_of,
)
from app.ai.voice.agents.breeze_buddy.scenario_tests.models import build_writer
from app.ai.voice.agents.breeze_buddy.scenario_tests.scenario import from_mapping
from app.ai.voice.agents.breeze_buddy.template.types import TemplateModel
from app.core.logger import logger

PROMPT_CHARS = 2500
# One writer call; a hung one would hold the worker's only run.
WRITE_TIMEOUT_S = 300.0
# Scenarios test the conversation: every request the agent makes succeeds,
# except where the situation is the failure. A scenario that tests an
# integration states its own fixtures.
ANSWER_EVERY_REQUEST = {"method": "*", "url": "*", "status": 200, "body": {}}
FAIL_EVERY_REQUEST = {
    "method": "*",
    "url": "*",
    "status": 500,
    "body": {"error": "internal server error"},
}
# A caller who agrees to the read-out before its move tests another call.
WAIT_FOR_AGENT = (
    " Agree to nothing your goal does not tell you to. Do not hang up yourself:"
    " wait for the agent to end the call."
)

INSTRUCTIONS = """You write test calls for a phone agent. Each test is a simulated \
person on the agent's call, and what the agent must do with them.

Write exactly the tests listed under "write": one per line, and nothing else.

For each test give:
- kind: the situation's key, the word before the colon in its line;
- name: a short lowercase-hyphenated slug, unique;
- persona: who is on the line and how they speak. Use the language and style \
the agent's script uses, in short sentences. Give no cue that points to a \
different outcome. For a machine, give exactly what it says;
- goal: what the person tries to do in the call, as they would see it: never \
what the test checks. They answer what the agent asks and volunteer nothing the \
situation does not need. The person makes the situation's move (for a \
happy_path, the one that leads to its outcome) before agreeing to anything \
else, and the goal says so plainly ("Before you confirm, ask the agent to...");
- success: one sentence a judge can check from the transcript: what the agent \
must say or do in the conversation. Require only what the agent's own \
instructions require, never an exact wording or an order of steps they leave \
open. For any situation but happy_path, check only that situation's point, \
nothing else the agent's script asks for. Never mention outcomes, functions or \
nodes: those are checked apart from the transcript;
- outcome: the outcome the call ends with, exactly as listed in outcomes, or \
null when the agent's instructions do not make it clear. A function that \
records an outcome and moves on to a node that does not end the call is \
overwritten by a later one;
- visits: the end node the call finishes in (a node with no functions), or [] \
when the agent's instructions do not make it clear.

Also give payload: a realistic value for every lead field that has no example.

Reply with JSON only:
{"payload": {...}, "tests": [{"kind": "...", "name": "...", "persona": "...", \
"goal": "...", "success": "...", "outcome": "...", "visits": ["..."]}]}"""


async def generate_scenarios(template: TemplateModel) -> List[Dict[str, Any]]:
    """Scenario mappings for the template, one per checklist line; raises
    when any could not be written, so no situation goes untested unseen."""
    kinds = checklist(template)
    finals = final_outcomes(template.flow)
    llm = await build_writer()
    written = await _write(llm, template, asks(kinds, finals, []))
    tests, payload = written.get("tests") or [], written.get("payload") or {}
    scenarios = assemble(tests, payload, template, kinds)
    missing = asks(kinds, finals, scenarios)
    if missing:
        try:
            again = await _write(llm, template, missing)
            tests = [*tests, *(again.get("tests") or [])]
            payload = {**(again.get("payload") or {}), **payload}
            scenarios = assemble(tests, payload, template, kinds)
        except Exception as e:
            logger.warning(
                f"Writing the missing scenarios for {template.id} failed: {e}"
            )
        missing = asks(kinds, finals, scenarios)
    if missing:
        left_out = ", ".join(_situation(line) for line in missing)
        raise ValueError(f"Buddy could not write the tests for: {left_out}")
    return scenarios


def _situation(line: str) -> str:
    """A checklist line's situation, as a failed run names it."""
    key = line.split(":", 1)[0]
    return f"happy path to {line.rsplit(' ', 1)[-1]}" if key == HAPPY_PATH.key else key


def asks(
    kinds: List[Kind], finals: Set[str], written: List[Dict[str, Any]]
) -> List[str]:
    """The checklist lines still to write: every situation, and a happy path
    per outcome, that ``written`` does not cover."""
    done = {s.get("kind") for s in written}
    covered = {
        s["buddy"].get("outcome") for s in written if s.get("kind") == HAPPY_PATH.key
    }
    lines = []
    for kind in kinds:
        if kind is HAPPY_PATH:
            lines += [
                f"happy_path: a cooperative person whose call ends with {o}"
                for o in sorted(finals - covered)
            ]
        elif kind.key not in done:
            lines.append(f"{kind.key}: {kind.ask}")
    return lines


async def _write(
    llm: LLMService[Any], template: TemplateModel, lines: List[str]
) -> Dict[str, Any]:
    request = {**describe(template), "write": lines}
    context = LLMContext([{"role": "user", "content": json.dumps(request, indent=1)}])
    reply = await asyncio.wait_for(
        llm.run_inference(context, system_instruction=INSTRUCTIONS),
        WRITE_TIMEOUT_S,
    )
    return _parse(reply or "")


def describe(template: TemplateModel) -> Dict[str, Any]:
    """What the writer needs to know about the agent, trimmed to fit."""
    flow = template.flow if isinstance(template.flow, dict) else {}
    agent: Dict[str, Any] = {"name": template.name}
    if flow.get("system_prompt"):
        agent["prompt"] = str(flow["system_prompt"])[:PROMPT_CHARS]
    agent["nodes"] = [
        {
            "name": node.get("node_name"),
            "instructions": _messages(node.get("role_messages"))
            + _messages(node.get("task_messages")),
            "functions": [_function(f) for f in node.get("functions") or []],
        }
        for node in flow.get("nodes") or []
    ]
    if flow.get("functions"):
        agent["functions"] = [_function(f) for f in flow["functions"]]
    agent["outcomes"] = sorted(final_outcomes(flow))
    agent["lead_fields"] = {
        key: {k: v for k, v in spec.items() if k in ("type", "description", "example")}
        for key, spec in (template.expected_payload_schema or {}).items()
        if isinstance(spec, dict)
    }
    return agent


def assemble(
    tests: List[Any],
    payload: Dict[str, Any],
    template: TemplateModel,
    kinds: List[Kind],
) -> List[Dict[str, Any]]:
    """Scenarios for the written tests, one per situation and one happy path per
    outcome, in checklist order. A test for a situation not on the agent's
    checklist, or a happy path to an outcome the call cannot end with, is
    dropped."""
    wanted = {k.key: i for i, k in enumerate(kinds)}
    finals = final_outcomes(template.flow)
    ends = end_nodes(template.flow)
    fields = set((template.expected_payload_schema or {}).keys())
    lead = {k: v for k, v in payload.items() if k in fields} if fields else {}
    scenarios: List[Dict[str, Any]] = []
    names: Set[str] = set()
    seen: Set[Any] = set()
    for test in tests:
        if not isinstance(test, dict) or test.get("kind") not in wanted:
            continue
        kind = KINDS[test["kind"]]
        outcome = test.get("outcome") if test.get("outcome") in finals else None
        slot = (kind.key, outcome) if kind is HAPPY_PATH else kind.key
        name = _slug(str(test.get("name") or ""))
        texts = [str(test.get(k) or "").strip() for k in ("persona", "goal", "success")]
        if kind is HAPPY_PATH and not outcome:
            continue
        if not name or name in names or slot in seen or not all(texts):
            continue
        scenario = _scenario(kind, name, texts, outcome, test, lead, template, ends)
        try:
            from_mapping(scenario)
        except Exception as e:
            logger.warning(f"Dropping written scenario {name!r}: {e}")
            continue
        names.add(name)
        seen.add(slot)
        scenarios.append(scenario)
    return sorted(scenarios, key=lambda s: wanted[s["kind"]])


def _scenario(
    kind: Kind,
    name: str,
    texts: List[str],
    outcome: Any,
    test: Dict[str, Any],
    lead: Dict[str, Any],
    template: TemplateModel,
    ends: Set[str],
) -> Dict[str, Any]:
    visits = test.get("visits") if isinstance(test.get("visits"), list) else []
    buddy: Dict[str, Any] = {
        "visits": [n for n in visits if n in ends] if kind.checks_ending else [],
        "http": [FAIL_EVERY_REQUEST if kind is SYSTEMS_DOWN else ANSWER_EVERY_REQUEST],
    }
    if outcome and kind.checks_ending:
        buddy["outcome"] = outcome
    if kind.runs > 1:
        buddy["runs"] = kind.runs
    blank = name_field(template) if kind is MISSING_NAME else None
    if lead or blank:
        buddy["payload"] = {**lead, **({blank: ""} if blank else {})}
    return {
        "name": name,
        "kind": kind.key,
        "persona": texts[0],
        "goal": texts[1] + WAIT_FOR_AGENT,
        "success": texts[2],
        "max_turns": kind.max_turns,
        "buddy": buddy,
    }


def _parse(reply: str) -> Dict[str, Any]:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The scenario writer did not reply with JSON")
    parsed = json.loads(reply[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The scenario writer did not reply with an object")
    return parsed


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]


def _messages(messages: Any) -> str:
    text = "\n".join(
        str(m.get("content") or "") for m in messages or [] if isinstance(m, dict)
    )
    return text[:PROMPT_CHARS]


def _function(function: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": function.get("function_name") or function.get("name"),
        "when": str(function.get("description") or "")[:400],
        "records_outcome": outcome_of(function),
        "next_node": function.get("transition_to"),
        "asks_for": sorted((function.get("properties") or {}).keys()),
    }
