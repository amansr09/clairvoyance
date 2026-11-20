"""Agent tests: a voice agent's stored scenarios, the runs of its suite, and
each scenario's result. See docs/SCENARIO_EVALS.md, "Automatic agent tests"."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

SCENARIO_NAME_PATTERN = r"^[a-z0-9][a-z0-9-]*$"


class AgentTestRunStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"
    OVERRIDDEN = "OVERRIDDEN"


class AgentTestResultStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"


class AgentTestScenarioSource(str, Enum):
    GENERATED = "generated"
    CUSTOM = "custom"


class AgentTestTrigger(str, Enum):
    CREATED = "created"
    EDITED = "edited"
    ROLLBACK = "rollback"
    SCENARIOS = "scenarios"
    # Rewrite the generated scenarios, then run them.
    REWRITE = "rewrite"
    MANUAL = "manual"
    OVERRIDE = "override"


class AgentTestState(str, Enum):
    """Where the agent's current content stands."""

    UNTESTED = "untested"
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    OVERRIDDEN = "overridden"


class AgentTestScenario(BaseModel):
    id: str
    template_id: str
    name: str
    scenario: Dict[str, Any]
    source: AgentTestScenarioSource
    is_enabled: bool
    created_by: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AgentTestRun(BaseModel):
    id: str
    template_id: str
    fingerprint: str
    template_version: int
    status: AgentTestRunStatus
    go_live: bool
    trigger: AgentTestTrigger
    requested_by: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class AgentTestResult(BaseModel):
    id: str
    run_id: str
    scenario_id: Optional[str] = None
    scenario_name: str
    attempt: int
    status: AgentTestResultStatus
    lead_id: Optional[str] = None
    outcome: Optional[str] = None
    failures: List[str] = Field(default_factory=list)
    duration_ms: Optional[int] = None


class AgentTestRunDetail(AgentTestRun):
    results: List[AgentTestResult] = Field(default_factory=list)


class AgentTestsOverview(BaseModel):
    template_id: str
    # False when the feature is off or the agent cannot be tested (IVR, chat only).
    enabled: bool
    is_active: bool
    current_version: int
    # The version real calls run while the current one is a draft; None when
    # they run the current one.
    published_version: Optional[int] = None
    fingerprint: str
    state: AgentTestState
    # The current version's run, with its results: the pending one, else the
    # one that judged it. None when this version has not been tested.
    run: Optional[AgentTestRunDetail] = None
    # Recent runs of any version, newest first.
    runs: List[AgentTestRun] = Field(default_factory=list)
    scenarios: List[AgentTestScenario] = Field(default_factory=list)


class OverrideAgentTestsRequest(BaseModel):
    reason: str = Field(min_length=5, max_length=1000)


class AgentTestScenarioRequest(BaseModel):
    name: str = Field(pattern=SCENARIO_NAME_PATTERN, max_length=128)
    scenario: Dict[str, Any]
    is_enabled: bool = True
