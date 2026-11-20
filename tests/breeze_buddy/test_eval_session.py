"""An eval session on the bot side.

The harness attaches like a carrier but speaks RTVI, and only an EVAL lead may
be driven that way. Outbound HTTP is answered from the scenario's fixtures, the
greeting reaches the harness once the flow is set up, and a reply nobody speaks
still ends the bot's turn, so the end node can end the call as in production.
"""

from __future__ import annotations

import copy
import importlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocket
from pipecat.evals.serializer import EvalSerializer
from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVISendTextFrame
from pipecat.runner.types import CallData
from pipecat.tests.utils import run_test
from starlette.websockets import WebSocketState

# isort: off
# template.types must load before the transport modules (reversing the
# order trips a circular import — see test_response_transform.py).
from app.ai.voice.agents.breeze_buddy.template.types import HttpRequestConfig

import app.ai.voice.agents.breeze_buddy.handlers.transport.http_requester as hr
from app.database.queries.breeze_buddy.analytics.analytics import (
    build_analytics_where_clause,
    get_analytics_call_details_query,
    get_analytics_count_query,
)

# isort: on
from app.ai.voice.agents.breeze_buddy import agent as agent_mod
from app.ai.voice.agents.breeze_buddy.agent import (
    pipeline as pipeline_mod,
    utils as agent_utils,
)
from app.ai.voice.agents.breeze_buddy.agent.transport import (
    create_telephony_transport,
)
from app.ai.voice.agents.breeze_buddy.agent.utils import (
    TRANSPORT_TYPE_EVAL,
    GreetingResult,
)
from app.ai.voice.agents.breeze_buddy.handlers.transport.eval_http import (
    NO_FIXTURE_STATUS,
    EvalHttp,
)
from app.ai.voice.agents.breeze_buddy.processors import (
    SentTextObserver,
    UnspokenReplyProcessor,
)
from app.ai.voice.agents.breeze_buddy.template import utils as template_utils
from app.ai.voice.agents.breeze_buddy.template.context import TemplateContext
from app.ai.voice.agents.breeze_buddy.types.models import PushLeadRequest
from app.ai.voice.agents.breeze_buddy.utils.common import send_webhook_with_retry
from app.schemas.breeze_buddy.core import (
    ExecutionMode,
    LeadCallStatus,
    LeadCallTracker,
)

LEAD_ID = "11111111-1111-1111-1111-111111111111"
EVAL_CONNECT = {
    "label": "rtvi-ai",
    "type": "client-message",
    "id": "eval-connect",
    "data": {"t": "eval-connect", "d": {"skip_tts": True}},
}
FIXTURES = [
    {
        "method": "GET",
        "url": "https://merchant.example.com/orders/*",
        "body": {"ok": 1},
    },
    {"url": "https://merchant.example.com/*", "status": 500, "body": "down"},
]


def make_lead(mode: ExecutionMode = ExecutionMode.EVAL) -> LeadCallTracker:
    return LeadCallTracker(
        id=LEAD_ID,
        reseller_id="res-1",
        template="welcome",
        status=LeadCallStatus.PROCESSING,
        call_id=f"eval-{LEAD_ID}",
        execution_mode=mode,
        metaData={"eval": {"scenario": "s/busy", "run": 1, "http": FIXTURES}},
    )


# ── outbound HTTP answered from fixtures ─────────────────────────────────


def test_the_first_matching_fixture_answers():
    http = EvalHttp(FIXTURES)

    assert http.respond("get", "https://merchant.example.com/orders/42") == (
        200,
        '{"ok": 1}',
    )
    assert http.respond("POST", "https://merchant.example.com/orders/42") == (
        500,
        "down",
    )
    assert [c["matched"] for c in http.calls] == [True, True]


def test_a_request_no_fixture_covers_fails_loudly_without_its_secrets():
    # The URL is built after the template's secrets are filled in, and the miss
    # text reaches the LLM while the record lands on the lead and in the report.
    http = EvalHttp(FIXTURES)

    status, text = http.respond(
        "post", "https://bot:hunter2@other.example.com/hook?token=s3cret"
    )

    assert status == NO_FIXTURE_STATUS
    assert json.loads(text)["error"] == (
        "no eval fixture for POST https://REDACTED@other.example.com/hook?REDACTED"
    )
    assert http.calls == [
        {
            "method": "POST",
            "url": "https://REDACTED@other.example.com/hook?REDACTED",
            "status": NO_FIXTURE_STATUS,
            "matched": False,
        }
    ]


def test_a_fixture_matches_the_url_as_sent_but_records_it_redacted():
    http = EvalHttp([{"url": "https://m.example.com/orders?token=*", "body": "ok"}])

    assert http.respond("GET", "https://m.example.com/orders?token=s3cret") == (
        200,
        "ok",
    )
    assert http.calls[0]["url"] == "https://m.example.com/orders?REDACTED"


def test_a_fixture_with_an_unknown_key_is_refused():
    with pytest.raises(ValueError):
        EvalHttp([{"url": "*", "response": {}}])


async def test_a_template_tool_call_never_leaves_the_process():
    http = EvalHttp(FIXTURES)
    executor = hr.HttpRequestExecutor(http)
    config = HttpRequestConfig(
        url="https://merchant.example.com/orders/{order_id}", method="GET"
    )

    result = await executor.execute(
        config=config, resolved_fields={"order_id": "42"}, fire_and_forget=False
    )
    fired = await executor.execute(
        config=config, resolved_fields={"order_id": "7"}, fire_and_forget=True
    )

    assert result == (200, '{"ok": 1}')
    assert fired is None
    assert [c["url"] for c in http.calls] == [
        "https://merchant.example.com/orders/42",
        "https://merchant.example.com/orders/7",
    ]


async def test_the_end_of_call_webhook_is_answered_by_a_fixture():
    http = EvalHttp(FIXTURES)

    assert (
        await send_webhook_with_retry(
            http, "https://merchant.example.com/orders/hook", {"outcome": "BUSY"}
        )
        is False
    )  # a POST matches the 500 fixture
    assert (
        await send_webhook_with_retry(
            EvalHttp([{"method": "POST", "url": "*"}]), "https://x.example.com", {}
        )
        is True
    )
    # The payload is not kept: it can carry the template's secrets.
    assert http.calls == [
        {
            "method": "POST",
            "url": "https://merchant.example.com/orders/hook",
            "status": 500,
            "matched": True,
        }
    ]


# ── a reply nobody speaks still ends the bot's turn ──────────────────────


def _unspoken(frame: Frame) -> Frame:
    setattr(frame, "skip_tts", True)
    return frame


async def test_an_unspoken_reply_is_marked_like_a_spoken_one():
    reply = [
        _unspoken(LLMFullResponseStartFrame()),
        _unspoken(LLMTextFrame("theek hai")),
        _unspoken(LLMFullResponseEndFrame()),
    ]

    down, up = await run_test(UnspokenReplyProcessor(), frames_to_send=reply)

    assert [type(f) for f in down] == [
        BotStartedSpeakingFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
        BotStoppedSpeakingFrame,
    ]
    assert [type(f) for f in up] == [BotStartedSpeakingFrame, BotStoppedSpeakingFrame]


async def test_a_test_call_opens_no_vendor_stt_or_tts(monkeypatch):
    llm = object()
    vendor = AsyncMock(side_effect=AssertionError("a vendor session was built"))
    monkeypatch.setattr(pipeline_mod, "get_stt_service", vendor)
    monkeypatch.setattr(pipeline_mod, "get_tts_service", vendor)
    monkeypatch.setattr(pipeline_mod, "get_llm_service", AsyncMock(return_value=llm))

    stt, got_llm, tts = await pipeline_mod.create_services(None, text_only=True)

    assert isinstance(stt, pipeline_mod.TextOnlySTTService)
    assert isinstance(tts, pipeline_mod.TextOnlyTTSService)
    assert got_llm is llm
    vendor.assert_not_awaited()


async def test_text_only_tts_speaks_fixed_text_and_passes_replies_through():
    reply = LLMTextFrame("Your order is confirmed.")
    reply.skip_tts = True
    frames = [TTSSpeakFrame(text="Please hold."), reply]

    down, _ = await run_test(
        pipeline_mod.TextOnlyTTSService(sample_rate=16000), frames_to_send=frames
    )

    # Fixed speech: its text reaches the transcript, with a moment of audio so
    # the bot starts and stops speaking as on a call.
    assert any(isinstance(f, TTSTextFrame) and "Please hold." in f.text for f in down)
    assert any(isinstance(f, TTSAudioRawFrame) for f in down)
    # A reply is never synthesized: it arrives as text and passes through.
    assert not any(isinstance(f, TTSTextFrame) and "confirmed" in f.text for f in down)


async def test_a_spoken_reply_passes_through_untouched():
    reply = [LLMFullResponseStartFrame(), TextFrame("hi"), LLMFullResponseEndFrame()]

    down, up = await run_test(UnspokenReplyProcessor(), frames_to_send=reply)

    assert [type(f) for f in down] == [
        LLMFullResponseStartFrame,
        TextFrame,
        LLMFullResponseEndFrame,
    ]
    assert up == []


async def test_an_interrupted_unspoken_reply_stops_once():
    frames = [
        _unspoken(LLMFullResponseStartFrame()),
        InterruptionFrame(),
        _unspoken(LLMFullResponseEndFrame()),
    ]

    down, up = await run_test(UnspokenReplyProcessor(), frames_to_send=frames)

    assert sum(isinstance(f, BotStoppedSpeakingFrame) for f in down) == 1
    assert [type(f) for f in up] == [BotStartedSpeakingFrame, BotStoppedSpeakingFrame]


# ── a typed turn counts as the user speaking ─────────────────────────────


def _pushed(frame: Frame, first_push: bool = True) -> FramePushed:
    link = FrameProcessor()
    return FramePushed(
        source=link,
        destination=link,
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
        first_push=first_push,
    )


async def test_a_sent_text_is_reported_once():
    calls: List[str] = []
    observer = SentTextObserver(lambda: calls.append("spoke"))
    sent = RTVISendTextFrame(msg_id="m1", text="busy hoon")

    await observer.on_push_frame(_pushed(sent))
    await observer.on_push_frame(_pushed(sent, first_push=False))
    await observer.on_push_frame(_pushed(TextFrame("not a turn")))

    assert calls == ["spoke"]


# ── the greeting ─────────────────────────────────────────────────────────


class FakeTask:
    def __init__(self) -> None:
        self.frames: List[Frame] = []

    async def queue_frames(self, frames) -> None:
        self.frames.extend(frames)


async def test_the_greeting_reaches_the_harness_as_one_bot_response():
    task = FakeTask()

    await agent_utils.report_greeting_to_eval(cast(Any, task), "Namaste")

    messages = [cast(OutputTransportMessageFrame, f).message for f in task.frames]
    assert [m["type"] for m in messages] == [
        "bot-llm-started",
        "bot-llm-text",
        "bot-llm-stopped",
    ]
    assert messages[1]["data"] == {"text": "Namaste"}


def test_an_eval_greeting_is_resolved_never_synthesized(monkeypatch):
    monkeypatch.setattr(
        agent_utils, "resolve_initial_greeting", lambda template, payload: "Namaste"
    )

    result = agent_utils.resolve_eval_greeting(
        make_lead(), cast(Any, SimpleNamespace())
    )

    assert result == GreetingResult(source="eval", text="Namaste")


async def test_an_eval_session_sets_its_output_then_greets_after_the_flow(
    monkeypatch,
):
    """skip_tts is set from the top of the pipeline, so the RTVI processor
    agrees with the LLM, and the greeting is reported only once the flow's
    context and tools are queued, so a harness that answers it at once
    cannot interrupt the flow's setup."""
    task = FakeTask()
    flow_set_up = TextFrame("flow set up")

    class FakeFlowManager:
        _global_functions: List[Any] = []

        async def initialize(self, node):
            await task.queue_frames([flow_set_up])

    agent = _eval_agent(FakeHarnessSocket(EVAL_CONNECT))
    agent.task = cast(Any, task)
    agent.lead = make_lead()
    agent.template = cast(Any, SimpleNamespace(id="tpl-1", name="welcome"))
    agent.flow_builder = cast(Any, object())
    agent.flow_manager = cast(Any, FakeFlowManager())
    greeting = GreetingResult(source="eval", text="Namaste")
    monkeypatch.setattr(agent_mod, "resolve_eval_greeting", lambda *a: greeting)
    monkeypatch.setattr(
        agent_mod,
        "build_flow_config",
        lambda *a: ({"initial_node": "initial"}, [], None),
    )
    monkeypatch.setattr(agent_mod, "prepare_initial_node", lambda **kwargs: {})
    monkeypatch.setattr(agent_mod, "spawn_prefill", lambda **kwargs: None)

    await agent._handle_client_connected()

    configure, *rest = task.frames
    assert isinstance(configure, LLMConfigureOutputFrame) and configure.skip_tts
    assert rest[0] is flow_set_up
    assert [cast(OutputTransportMessageFrame, f).message["type"] for f in rest[1:]] == [
        "bot-llm-started",
        "bot-llm-text",
        "bot-llm-stopped",
    ]


# ── who may attach ───────────────────────────────────────────────────────


class FakeHarnessSocket:
    """The harness's websocket: an eval-connect, then nothing."""

    client_state = WebSocketState.CONNECTED

    def __init__(self, opening: Dict[str, Any]) -> None:
        self._opening = [json.dumps(opening)]
        self.query_params: Dict[str, str] = {}
        self.close_code: Optional[int] = None

    async def accept(self) -> None:
        return None

    async def iter_text(self):
        for message in self._opening:
            yield message

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code


@pytest.fixture
def eval_setup(monkeypatch):
    """Everything _setup_eval_transport touches, patched where the agent looks
    it up (no DB, Redis or TTS)."""
    touched: List[str] = []
    template = SimpleNamespace(
        id="tpl-1",
        name="welcome",
        flow={},
        configurations=None,
        supported_channels=["voice"],
    )

    def lead_named(lead: LeadCallTracker):
        async def get_lead_by_id(lead_id):
            return lead

        monkeypatch.setattr(agent_mod, "get_lead_by_id", get_lead_by_id)

    async def mark_initiated(lead_id, when):
        touched.append(lead_id)
        return make_lead()

    async def load_template(lead):
        return template, None, {}

    async def no_vad(**kwargs):
        return None, None

    monkeypatch.setattr(
        agent_mod, "update_lead_call_initiated_time_by_id", mark_initiated
    )
    monkeypatch.setattr(agent_mod, "load_template_config", load_template)
    monkeypatch.setattr(agent_mod, "create_vad_analyzer", no_vad)
    return SimpleNamespace(lead_named=lead_named, touched=touched)


def _eval_agent(ws: FakeHarnessSocket) -> agent_mod.Agent:
    return agent_mod.Agent(
        transport_type=TRANSPORT_TYPE_EVAL, ws=cast(WebSocket, ws), eval_lead_id=LEAD_ID
    )


async def test_an_eval_lead_gets_the_eval_transport(eval_setup):
    eval_setup.lead_named(make_lead())
    agent = _eval_agent(FakeHarnessSocket(EVAL_CONNECT))

    assert await agent._setup_eval_transport() is True

    assert eval_setup.touched == [LEAD_ID]
    assert agent.call_sid == f"eval-{LEAD_ID}"
    assert isinstance(agent.aiohttp_session, EvalHttp)
    assert (
        agent.aiohttp_session.respond("GET", "https://merchant.example.com/orders/1")[0]
        == 200
    )
    assert type(agent.transport).__name__ == "EvalFastAPIWebsocketTransport"


@pytest.mark.parametrize(
    "mode", [ExecutionMode.TELEPHONY, ExecutionMode.TELEPHONY_TEST, ExecutionMode.DAILY]
)
async def test_any_other_lead_is_refused_before_it_is_touched(eval_setup, mode):
    eval_setup.lead_named(make_lead(mode))
    ws = FakeHarnessSocket(EVAL_CONNECT)

    assert await _eval_agent(ws)._setup_eval_transport() is False

    assert ws.close_code == 4003
    assert eval_setup.touched == []


@pytest.mark.parametrize(
    "update",
    [
        {"call_initiated_time": datetime(2026, 11, 19, tzinfo=timezone.utc)},
        {"status": LeadCallStatus.FINISHED},
    ],
)
async def test_a_lead_a_session_already_ran_is_refused(eval_setup, update):
    eval_setup.lead_named(make_lead().model_copy(update=update))
    ws = FakeHarnessSocket(EVAL_CONNECT)

    assert await _eval_agent(ws)._setup_eval_transport() is False

    assert ws.close_code == 4003
    assert eval_setup.touched == []


async def test_an_audio_mode_session_is_refused(eval_setup):
    eval_setup.lead_named(make_lead())
    audio = json.loads(json.dumps(EVAL_CONNECT))
    audio["data"]["d"]["skip_tts"] = False
    ws = FakeHarnessSocket(audio)

    assert await _eval_agent(ws)._setup_eval_transport() is False

    assert ws.close_code == 4003
    assert eval_setup.touched == []


async def test_a_carrier_stream_on_the_eval_route_is_refused(eval_setup):
    eval_setup.lead_named(make_lead())
    start = {"event": "start", "start": {"callId": "c1", "streamId": "s1"}}
    ws = FakeHarnessSocket(start)

    assert await _eval_agent(ws)._setup_eval_transport() is False

    assert ws.close_code == 4003
    assert eval_setup.touched == []


# A lead marked started but left PROCESSING is closed by the stuck-call sweeper,
# so every refusal must come before the lead is touched.


async def test_a_bad_fixture_is_refused_before_the_lead_is_touched(eval_setup):
    typo = {"eval": {"http": [{"url": "*", "response": {}}]}}
    eval_setup.lead_named(make_lead().model_copy(update={"metaData": typo}))
    ws = FakeHarnessSocket(EVAL_CONNECT)

    assert await _eval_agent(ws)._setup_eval_transport() is False

    assert ws.close_code == 4000
    assert eval_setup.touched == []


async def _missing_template(lead):
    raise ValueError("no such template")


@pytest.mark.parametrize("load", [_missing_template])
async def test_a_template_it_cannot_run_is_refused_before_the_lead_is_touched(
    eval_setup, monkeypatch, load
):
    eval_setup.lead_named(make_lead())
    monkeypatch.setattr(agent_mod, "load_template_config", load)
    ws = FakeHarnessSocket(EVAL_CONNECT)

    assert await _eval_agent(ws)._setup_eval_transport() is False

    assert ws.close_code == 4000
    assert eval_setup.touched == []


@pytest.mark.parametrize(
    "started, status, closed",
    [
        (True, LeadCallStatus.PROCESSING, ["EARLY_HANGUP"]),
        (True, LeadCallStatus.FINISHED, []),
        (False, LeadCallStatus.PROCESSING, []),
    ],
)
async def test_a_crash_closes_the_lead_only_if_it_started_and_is_still_open(
    monkeypatch, started, status, closed
):
    completed: List[Dict[str, Any]] = []

    async def complete(**kwargs):
        completed.append(kwargs)

    async def crash(self, runner_args=None):
        if started:
            self.lead = make_lead()
            self.call_sid = self.lead.call_id
        raise RuntimeError("pipeline died")

    async def get_lead_by_id(lead_id):
        return make_lead().model_copy(update={"status": status})

    monkeypatch.setattr(agent_mod.Agent, "run", crash)
    monkeypatch.setattr(agent_mod, "get_lead_by_id", get_lead_by_id)

    with pytest.raises(RuntimeError, match="pipeline died"):
        await agent_mod.eval_bot(
            cast(WebSocket, FakeHarnessSocket(EVAL_CONNECT)), LEAD_ID, complete
        )

    assert [c["outcome"] for c in completed] == closed
    assert all(c["call_id"] == f"eval-{LEAD_ID}" for c in completed)


async def test_the_eval_transport_carries_the_connection_flags():
    params = EvalTransportParams()
    transport = await create_telephony_transport(
        cast(WebSocket, FakeHarnessSocket(EVAL_CONNECT)),
        params,
        TRANSPORT_TYPE_EVAL,
        CallData(body={"skip_tts": True}),
    )

    assert isinstance(params.serializer, EvalSerializer)
    assert type(transport).__name__ == "EvalFastAPIWebsocketTransport"


# ── side effects that must stay home ─────────────────────────────────────


async def test_an_eval_call_sends_no_alert(monkeypatch):
    send = AsyncMock()
    monkeypatch.setattr(template_utils.slack_alert, "send", send)
    bot = SimpleNamespace(lead=make_lead())

    result = await template_utils.send_alert(
        TemplateContext(cast(Any, bot)), {"title": "Escalation"}
    )

    assert result == {"status": "success"}
    send.assert_not_awaited()


async def test_an_eval_call_records_its_requests_and_skips_the_evaluation_queue(
    monkeypatch,
):
    from importlib import import_module

    end_conversation_module = import_module(
        "app.ai.voice.agents.breeze_buddy.handlers.internal.end_conversation"
    )
    enqueue = AsyncMock()
    monkeypatch.setattr(
        end_conversation_module, "enqueue_conversation_evaluation", enqueue
    )
    monkeypatch.setattr(
        end_conversation_module, "update_span_with_evaluation_data", lambda _c: None
    )

    async def webhook(context, args):
        context.aiohttp_session.respond("POST", "https://hooks.example.com/end")

    # In production the callbacks run after the lead is saved; an eval runs them
    # first, so the end-of-call webhook is among the requests it records.
    monkeypatch.setitem(end_conversation_module.callback_map, "webhook", webhook)
    http = EvalHttp(FIXTURES)
    http.respond("GET", "https://merchant.example.com/orders/1")
    lead = make_lead()
    saved: Dict[str, Any] = {}

    async def save(**kwargs):
        # A DB write keeps what the lead held at that moment.
        saved.update(copy.deepcopy(kwargs))
        return lead.model_copy(update={"template_id": "tpl-1"})

    completion = AsyncMock(side_effect=save)
    bot = SimpleNamespace(
        aiohttp_session=http,
        approval_manager=None,
        call_sid=lead.call_id,
        completion_function=completion,
        configurations=SimpleNamespace(knowledge_base=None),
        context=SimpleNamespace(messages=[{"role": "user", "content": "busy hoon"}]),
        conversation_ended=False,
        end_conversation_callbacks=["webhook"],
        errors=[],
        lead=lead,
        metrics_collector=None,
        pending_transfer=None,
        prior_generation_messages=[],
        task=None,
        template=SimpleNamespace(current_version=3),
        transport_type=TRANSPORT_TYPE_EVAL,
    )

    await end_conversation_module.end_conversation(TemplateContext(cast(Any, bot)), {})

    completion.assert_awaited_once()
    meta = saved["meta_data"]
    assert meta["template_version"] == 3
    assert meta["eval"]["http_calls"] == http.calls
    assert [c["url"] for c in meta["eval"]["http_calls"]] == [
        "https://merchant.example.com/orders/1",
        "https://hooks.example.com/end",
    ]
    assert meta["eval"]["scenario"] == "s/busy"
    enqueue.assert_not_awaited()


# The call list a merchant browses shows test calls, never eval sessions;
# aggregates count production calls only.
NOT_EVAL = "lct.execution_mode <> 'EVAL'"


def test_aggregates_count_only_production_calls():
    conditions, _ = build_analytics_where_clause({})

    assert conditions == ["lct.execution_mode IN ('TELEPHONY', 'HOLD_TRANSFER')"]


def test_the_call_list_and_its_count_leave_out_eval_sessions():
    listed, _ = get_analytics_call_details_query({})
    counted, _ = get_analytics_count_query({}, filter_execution_mode=False)

    assert NOT_EVAL in listed
    assert NOT_EVAL in counted
    assert "'TELEPHONY', 'HOLD_TRANSFER'" not in listed


def test_a_pushed_lead_cannot_be_an_eval_lead():
    request = {
        "request_id": "r1",
        "payload": {},
        "template_id": "2fe25d1b-eced-4caa-bf64-de0300000000",
        "reseller_id": "res-1",
    }

    with pytest.raises(ValueError, match="eval runner only"):
        PushLeadRequest.model_validate({**request, "execution_mode": "EVAL"})
    assert (
        PushLeadRequest.model_validate(
            {**request, "execution_mode": "TELEPHONY_TEST"}
        ).execution_mode
        == ExecutionMode.TELEPHONY_TEST
    )


@pytest.mark.parametrize("worker_pod", [True, False])
def test_the_eval_route_is_mounted_only_on_agent_tests_worker_pods(
    monkeypatch, worker_pod
):
    """Test calls must never reach a pod that sets up or dials real calls."""
    from app.api.routers import breeze_buddy
    from app.core.config import static

    try:
        monkeypatch.setattr(static, "ENABLE_AGENT_TESTS_WORKER", worker_pod)
        paths = {r.path for r in importlib.reload(breeze_buddy).router.routes}
        assert any("/eval/" in path for path in paths) is worker_pod
    finally:
        monkeypatch.undo()
        importlib.reload(breeze_buddy)
