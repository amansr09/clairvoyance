"""Order webhooks treat a switched-off template as a skip, not a failure.

Intake answers 409 when the template is inactive (the merchant's off switch)
and 403 for a blacklisted number. Both are expected states the order webhook
cannot fix by retrying, so they log at info and return ``skipped``; any other
push error stays ``error``.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status

from app.api.routers.breeze_buddy.webhooks.breeze import utils as breeze_utils
from app.api.routers.breeze_buddy.webhooks.woocommerce import utils as woo_utils

MODULES = [
    pytest.param(woo_utils, id="woocommerce"),
    pytest.param(breeze_utils, id="breeze"),
]


def _push_raises(monkeypatch, module, code: int) -> None:
    async def _no_leads(_request_id):
        return []

    async def _push(_req, _user):
        raise HTTPException(status_code=code, detail=f"refused with {code}")

    monkeypatch.setattr(module, "get_leads_by_request_id", _no_leads)
    monkeypatch.setattr(module, "push_lead_handler", _push)


async def _push(module):
    return await module.push_lead(
        topic="order.created",
        order_id=101,
        phone="+919876543210",
        payload={"customer_mobile_number": "+919876543210"},
        merchant_id="merchant-1",
        reseller_id="reseller-1",
        template_id="6f1c2a9e-4b7d-4e3a-9c2f-1d5e8b7a0c41",
        trigger_reason="order created",
    )


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    "code",
    [status.HTTP_409_CONFLICT, status.HTTP_403_FORBIDDEN],
    ids=["inactive", "blacklisted"],
)
async def test_expected_refusals_are_skips(monkeypatch, module, code):
    _push_raises(monkeypatch, module, code)

    result = await _push(module)

    assert result == {"status": "skipped", "reason": f"refused with {code}"}


@pytest.mark.parametrize("module", MODULES)
async def test_other_push_errors_stay_errors(monkeypatch, module):
    _push_raises(monkeypatch, module, status.HTTP_400_BAD_REQUEST)

    result = await _push(module)

    assert result["status"] == "error"
