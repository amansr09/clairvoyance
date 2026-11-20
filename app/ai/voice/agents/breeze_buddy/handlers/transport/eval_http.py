"""Outbound HTTP for eval sessions, answered from the lead's fixtures.

Stands in for the aiohttp session, so tool calls, hooks and webhooks never reach
a merchant. An unmatched request gets a 404 and is recorded as unmatched.
"""

import json
from fnmatch import fnmatchcase
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from app.core.logger import logger
from app.core.network import redact_url

NO_FIXTURE_STATUS = 404


class HttpFixture(BaseModel):
    """One canned response: the first fixture whose method and URL glob match wins."""

    model_config = ConfigDict(extra="forbid")

    url: str = "*"
    method: str = "*"
    status: int = 200
    body: Any = None

    def matches(self, method: str, url: str) -> bool:
        return fnmatchcase(method.upper(), self.method.upper()) and fnmatchcase(
            url, self.url
        )

    def text(self) -> str:
        if self.body is None or isinstance(self.body, str):
            return self.body or ""
        return json.dumps(self.body, ensure_ascii=False)


class EvalHttp:
    """Records an eval session's outbound requests and answers them from fixtures."""

    def __init__(self, fixtures: Optional[List[Dict[str, Any]]] = None):
        self._fixtures = [HttpFixture.model_validate(f) for f in fixtures or []]
        self.calls: List[Dict[str, Any]] = []

    def respond(self, method: str, url: str) -> Tuple[int, str]:
        """Answer from the first matching fixture.

        The URL is matched as sent but recorded and echoed redacted, and the body
        is never kept: both carry the template's resolved secrets, and the record
        lands on the lead and in the LLM's context.
        """
        method = method.upper()
        safe_url = redact_url(url)
        fixture = next((f for f in self._fixtures if f.matches(method, url)), None)
        if fixture:
            status, text = fixture.status, fixture.text()
        else:
            status = NO_FIXTURE_STATUS
            text = json.dumps({"error": f"no eval fixture for {method} {safe_url}"})
        self.calls.append(
            {
                "method": method,
                "url": safe_url,
                "status": status,
                "matched": fixture is not None,
            }
        )
        logger.info(
            f"[EVAL] {method} {safe_url} -> {status}"
            f"{'' if fixture else ' (no fixture)'}"
        )
        return status, text
