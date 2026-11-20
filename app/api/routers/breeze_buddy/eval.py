"""Eval harness websocket for agent tests, mounted only on the agent-tests
worker's pods. It admits a session only while agent tests are on (dynamic
AGENT_TESTS_ENABLED), and only with its lead's signed token."""

from typing import Optional

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from app.ai.voice.agents.breeze_buddy.agent import eval_bot
from app.ai.voice.agents.breeze_buddy.managers.calls import handle_call_completion
from app.ai.voice.agents.breeze_buddy.scenario_tests.access import token_matches
from app.ai.voice.agents.breeze_buddy.utils.transport.websockets import (
    is_disconnected,
)
from app.core.config.dynamic import AGENT_TESTS_ENABLED
from app.core.logger import logger

router = APIRouter()


@router.websocket("/eval/{lead_id}/ws")
async def eval_websocket_handler(
    lead_id: str, websocket: WebSocket, token: Optional[str] = None
):
    if not await AGENT_TESTS_ENABLED():
        await websocket.close(code=4003, reason="Eval sessions are off")
        return
    if not token_matches(lead_id, token):
        logger.warning(f"Eval session for lead {lead_id} refused: bad token")
        await websocket.close(code=4003, reason="Not an eval lead")
        return
    try:
        await eval_bot(websocket, lead_id, handle_call_completion)
    except WebSocketDisconnect:
        logger.info(f"Eval session for lead {lead_id} disconnected")
    except Exception:
        logger.opt(exception=True).error(f"Eval session for lead {lead_id} failed")
        if not is_disconnected(websocket):
            await websocket.close(code=1011, reason="Internal Server Error")
