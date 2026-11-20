"""The pieces an eval session runs on, and what an EVAL lead never reaches.

Outbound HTTP is answered from the scenario's fixtures, a test call opens no
vendor STT or TTS, and a reply nobody speaks still ends the bot's turn. An EVAL
lead sends no alert, never shows in the call list or aggregates, and no pushed
lead can claim the mode.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, List, cast
from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVISendTextFrame
from pipecat.tests.utils import run_test

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
from app.ai.voice.agents.breeze_buddy.agent import (
    pipeline as pipeline_mod,
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
