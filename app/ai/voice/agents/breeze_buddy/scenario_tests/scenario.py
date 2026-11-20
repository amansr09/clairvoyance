"""Buddy scenarios: a Pipecat scenario and the ``buddy:`` block binding it to a template.

Scenarios run in text mode only. Buddy speaks first, so a scripted scenario's
first caller turn waits for the bot's opening line unless it is scheduled.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pipecat.evals.scenario import _scenario_from_mapping
from pipecat.evals.script import EvalScriptScenario, EvalSendAfter
from pipecat.evals.script_driver import SEND_AFTER_MAX_WAIT_S
from pipecat.evals.session import DEFAULT_EVENT_TIMEOUT_MS
from pipecat.evals.simulation import EvalSimulationScenario
from pydantic import BaseModel, ConfigDict, Field

from app.ai.voice.agents.breeze_buddy.handlers.transport.eval_http import HttpFixture

# A stored scenario's limits: every agent's suite shares one worker.
MAX_RUNS = 3
MAX_TURNS = 30
MAX_DURATION_S = 600


class BuddySpec(BaseModel):
    """What a scenario needs from Buddy, and what the finished call must show."""

    model_config = ConfigDict(extra="forbid")

    payload: Dict[str, Any] = Field(default_factory=dict)
    http: List[HttpFixture] = Field(default_factory=list)
    outcome: Optional[Union[str, List[str]]] = None
    visits: List[str] = Field(default_factory=list)
    avoids: List[str] = Field(default_factory=list)
    runs: Optional[int] = None

    @property
    def outcomes(self) -> List[str]:
        """The accepted final outcomes; none given (or blank) checks none."""
        outcomes = [self.outcome] if isinstance(self.outcome, str) else self.outcome
        return [o for o in outcomes or [] if o]


@dataclass
class BuddyScenario:
    scenario: Union[EvalScriptScenario, EvalSimulationScenario]
    spec: BuddySpec

    @property
    def name(self) -> str:
        return self.scenario.name

    @property
    def runs(self) -> int:
        """How many calls the scenario makes: its own count, else one."""
        own = self.spec.runs
        if own is None and isinstance(self.scenario, EvalSimulationScenario):
            own = self.scenario.runs
        return max(own or 1, 1)


def from_mapping(data: Dict[str, Any]) -> BuddyScenario:
    """A stored agent test: one Pipecat scenario mapping with its ``buddy:`` block."""
    source = Path(f"<{data.get('name') or 'scenario'}>")
    body = {key: value for key, value in data.items() if key != "buddy"}
    spec = BuddySpec.model_validate(data.get("buddy") or {})
    scenario = _scenario_from_mapping(body, source)
    if scenario.bot_audio or scenario.user_audio:
        raise ValueError(f"{source}: {scenario.name} is not a text-mode scenario")
    if isinstance(scenario, EvalScriptScenario):
        _wait_for_opening(scenario)
        turns, duration = len(scenario.turns), _longest_s(scenario)
    else:
        turns, duration = scenario.max_turns, scenario.max_duration_s
    item = BuddyScenario(scenario, spec)
    if item.runs > MAX_RUNS or turns > MAX_TURNS or duration > MAX_DURATION_S:
        raise ValueError(
            f"{source}: at most {MAX_RUNS} runs, {MAX_TURNS} turns and "
            f"{MAX_DURATION_S} s per scenario (a scripted turn waits up to "
            f"{DEFAULT_EVENT_TIMEOUT_MS // 1000} s unless its within_ms says less)"
        )
    return item


def _longest_s(scenario: EvalScriptScenario) -> float:
    """The longest a script can run: each turn's send delay, its wait for the
    event it is sent after, and its wait for the events it expects."""
    total_ms = 0.0
    for turn in scenario.turns:
        if turn.send_after:
            total_ms += turn.send_after.delay_ms
            if turn.send_after.event:
                total_ms += SEND_AFTER_MAX_WAIT_S * 1000
        if turn.expect:
            total_ms += max(
                e.within_ms or DEFAULT_EVENT_TIMEOUT_MS for e in turn.expect
            )
    return total_ms / 1000


def _wait_for_opening(scenario: EvalScriptScenario) -> None:
    """Hold the first caller turn until the bot has said its opening line."""
    first = scenario.turns[0] if scenario.turns else None
    if first and (first.user or first.dtmf) and not first.send_after:
        first.send_after = EvalSendAfter(event="llm_response", delay_ms=0)
