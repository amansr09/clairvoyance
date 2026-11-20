"""The agent-tests worker, on its own deployment (ENABLE_AGENT_TESTS_WORKER):
closes runs a dead pod left RUNNING, then runs queued ones. API and dialler
pods only queue runs."""

import asyncio
from typing import Optional

from app.ai.voice.agents.breeze_buddy.scenario_tests.suite import run_suite
from app.core.config.dynamic import (
    AGENT_TESTS_ENABLED,
    AGENT_TESTS_POLL_INTERVAL_SECONDS,
    AGENT_TESTS_STALE_RUN_MINUTES,
)
from app.core.logger import logger
from app.database.accessor.breeze_buddy.agent_tests import (
    claim_next_agent_test_run,
    fail_stale_agent_test_runs,
    requeue_agent_test_run,
)

_task: Optional[asyncio.Task] = None


async def drain_agent_test_runs() -> None:
    """Run queued runs until none is left, or agent tests are turned off."""
    if not await AGENT_TESTS_ENABLED():
        return
    await fail_stale_agent_test_runs(await AGENT_TESTS_STALE_RUN_MINUTES())
    while run := await claim_next_agent_test_run():
        try:
            await run_suite(run)
        except asyncio.CancelledError:
            # The pod is stopping (a deploy): the next worker runs it again.
            await requeue_agent_test_run(run.id)
            raise
        if not await AGENT_TESTS_ENABLED():
            return


async def _poll() -> None:
    while True:
        try:
            await drain_agent_test_runs()
        except Exception as e:
            logger.opt(exception=True).error(f"Agent tests worker pass failed: {e}")
        await asyncio.sleep(await AGENT_TESTS_POLL_INTERVAL_SECONDS())


async def start_agent_tests_worker() -> None:
    global _task
    if _task and not _task.done():
        return
    _task = asyncio.create_task(_poll(), name="agent-tests-worker")
    logger.info("Agent tests worker started")


async def stop_agent_tests_worker() -> None:
    global _task
    if not _task:
        return
    _task.cancel()
    await asyncio.gather(_task, return_exceptions=True)
    _task = None
    logger.info("Agent tests worker stopped")
