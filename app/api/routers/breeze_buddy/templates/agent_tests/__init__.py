"""
An agent's tests: its scenarios, the runs of its suite, and the go-live verdict.

Endpoints:
- GET    /templates/{id}/tests                        - Verdict, latest run, scenarios
- POST   /templates/{id}/tests/run                    - Queue a run
- POST   /templates/{id}/tests/scenarios/generate     - Queue a rewrite of them
- POST   /templates/{id}/tests/scenarios              - Add a custom scenario
- PUT    /templates/{id}/tests/scenarios/{sid}        - Edit or disable a scenario
- DELETE /templates/{id}/tests/scenarios/{sid}        - Remove a scenario
- POST   /templates/{id}/tests/override               - Admin: pass this version
"""

from fastapi import APIRouter, Depends, status

from app.api.security.breeze_buddy.rbac_token import get_current_user_with_rbac
from app.schemas import UserInfo
from app.schemas.breeze_buddy.agent_tests import (
    AgentTestRun,
    AgentTestScenario,
    AgentTestScenarioRequest,
    AgentTestsOverview,
    OverrideAgentTestsRequest,
)

from .handlers import (
    create_agent_test_scenario_handler,
    delete_agent_test_scenario_handler,
    generate_agent_test_scenarios_handler,
    get_agent_tests_handler,
    override_agent_tests_handler,
    run_agent_tests_handler,
    update_agent_test_scenario_handler,
)

router = APIRouter()


@router.get("/templates/{template_id}/tests", response_model=AgentTestsOverview)
async def get_agent_tests(
    template_id: str, current_user: UserInfo = Depends(get_current_user_with_rbac)
):
    """Where the current version stands, its run with results, recent runs and
    the scenarios. The Tests tab polls this."""
    return await get_agent_tests_handler(template_id, current_user)


@router.post(
    "/templates/{template_id}/tests/run",
    response_model=AgentTestRun,
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_agent_tests(
    template_id: str,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    """Queue a run of the current version; the agent-tests worker runs it."""
    return await run_agent_tests_handler(template_id, current_user)


@router.post(
    "/templates/{template_id}/tests/scenarios/generate",
    response_model=AgentTestRun,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_agent_test_scenarios(
    template_id: str, current_user: UserInfo = Depends(get_current_user_with_rbac)
):
    """Queue a run that rewrites the generated scenarios (custom ones stay),
    then runs them."""
    return await generate_agent_test_scenarios_handler(template_id, current_user)


@router.post(
    "/templates/{template_id}/tests/scenarios",
    response_model=AgentTestScenario,
    status_code=status.HTTP_201_CREATED,
)
async def create_agent_test_scenario(
    template_id: str,
    body: AgentTestScenarioRequest,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    """Add a custom scenario and re-run the suite. 400 if the harness cannot
    load it, 409 if the name is taken."""
    return await create_agent_test_scenario_handler(template_id, body, current_user)


@router.put(
    "/templates/{template_id}/tests/scenarios/{scenario_id}",
    response_model=AgentTestScenario,
)
async def update_agent_test_scenario(
    template_id: str,
    scenario_id: str,
    body: AgentTestScenarioRequest,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    """Edit, enable or disable a scenario and re-run the suite."""
    return await update_agent_test_scenario_handler(
        template_id, scenario_id, body, current_user
    )


@router.delete(
    "/templates/{template_id}/tests/scenarios/{scenario_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_agent_test_scenario(
    template_id: str,
    scenario_id: str,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    """Remove a scenario and re-run the suite."""
    await delete_agent_test_scenario_handler(template_id, scenario_id, current_user)


@router.post(
    "/templates/{template_id}/tests/override",
    response_model=AgentTestRun,
)
async def override_agent_tests(
    template_id: str,
    body: OverrideAgentTestsRequest,
    current_user: UserInfo = Depends(get_current_user_with_rbac),
):
    """Admin only: pass the current version by hand, with the reason on record."""
    return await override_agent_tests_handler(template_id, body, current_user)
