"""SQL for agent tests: stored scenarios, suite runs and per-scenario results."""

from typing import Any, List, Optional, Tuple

SCENARIO_TABLE = "agent_test_scenario"
RUN_TABLE = "agent_test_run"
RESULT_TABLE = "agent_test_result"

_PENDING = "('QUEUED', 'RUNNING')"


def list_agent_test_scenarios_query(template_id: str) -> Tuple[str, List[Any]]:
    text = f"""
        SELECT * FROM {SCENARIO_TABLE}
        WHERE template_id = $1
        ORDER BY created_at, name;
    """
    return text, [template_id]


def insert_agent_test_scenario_query(
    template_id: str,
    name: str,
    scenario_json: str,
    source: str,
    is_enabled: bool,
    created_by: Optional[str],
) -> Tuple[str, List[Any]]:
    """No row when the template already has a scenario of that name."""
    text = f"""
        INSERT INTO {SCENARIO_TABLE}
            (template_id, name, scenario, source, is_enabled, created_by)
        VALUES ($1, $2, $3::jsonb, $4, $5, $6)
        ON CONFLICT (template_id, name) DO NOTHING
        RETURNING *;
    """
    return text, [template_id, name, scenario_json, source, is_enabled, created_by]


def update_agent_test_scenario_query(
    scenario_id: str,
    template_id: str,
    name: str,
    scenario_json: str,
    is_enabled: bool,
) -> Tuple[str, List[Any]]:
    text = f"""
        UPDATE {SCENARIO_TABLE}
        SET name = $3, scenario = $4::jsonb, is_enabled = $5, updated_at = now()
        WHERE id = $1 AND template_id = $2
        RETURNING *;
    """
    return text, [scenario_id, template_id, name, scenario_json, is_enabled]


def delete_agent_test_scenario_query(
    scenario_id: str, template_id: str
) -> Tuple[str, List[Any]]:
    text = f"""
        DELETE FROM {SCENARIO_TABLE}
        WHERE id = $1 AND template_id = $2
        RETURNING id;
    """
    return text, [scenario_id, template_id]


def delete_generated_agent_test_scenarios_query(
    template_id: str,
) -> Tuple[str, List[Any]]:
    text = f"""
        DELETE FROM {SCENARIO_TABLE}
        WHERE template_id = $1 AND source = 'generated';
    """
    return text, [template_id]


def queue_agent_test_run_query(
    template_id: str,
    fingerprint: str,
    template_version: int,
    go_live: bool,
    trigger: str,
    requested_by: Optional[str],
) -> Tuple[str, List[Any]]:
    """Queue a run, superseding the template's queued ones. It goes live when a
    superseded run of the same content was to: a scenario change must not
    drop a waiting agent's go-live."""
    text = f"""
        WITH superseded AS (
            UPDATE {RUN_TABLE}
            SET status = 'CANCELLED', note = 'Superseded by a newer run',
                finished_at = now(), updated_at = now()
            WHERE template_id = $1 AND status = 'QUEUED'
            RETURNING fingerprint, go_live
        )
        INSERT INTO {RUN_TABLE}
            (template_id, fingerprint, template_version, status, go_live,
             trigger, requested_by)
        VALUES (
            $1, $2, $3, 'QUEUED',
            $4 OR EXISTS (
                SELECT 1 FROM superseded WHERE go_live AND fingerprint = $2
            ),
            $5, $6
        )
        RETURNING *;
    """
    return text, [
        template_id,
        fingerprint,
        template_version,
        go_live,
        trigger,
        requested_by,
    ]


def insert_agent_test_override_query(
    template_id: str,
    fingerprint: str,
    template_version: int,
    requested_by: str,
    note: str,
) -> Tuple[str, List[Any]]:
    text = f"""
        INSERT INTO {RUN_TABLE}
            (template_id, fingerprint, template_version, status, trigger,
             requested_by, note, started_at, finished_at)
        VALUES ($1, $2, $3, 'OVERRIDDEN', 'override', $4, $5, now(), now())
        RETURNING *;
    """
    return text, [template_id, fingerprint, template_version, requested_by, note]


def claim_next_agent_test_run_query() -> Tuple[str, List[Any]]:
    """The oldest queued run, claimed as RUNNING. SKIP LOCKED keeps two worker
    pods from claiming the same run."""
    text = f"""
        UPDATE {RUN_TABLE}
        SET status = 'RUNNING', started_at = now(), updated_at = now()
        WHERE id = (
            SELECT id FROM {RUN_TABLE}
            WHERE status = 'QUEUED'
            ORDER BY created_at
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING *;
    """
    return text, []


def finish_agent_test_run_query(
    run_id: str, status: str, note: Optional[str]
) -> Tuple[str, List[Any]]:
    """No row when the run is no longer RUNNING (closed as stale)."""
    text = f"""
        UPDATE {RUN_TABLE}
        SET status = $2, note = $3, finished_at = now(), updated_at = now()
        WHERE id = $1 AND status = 'RUNNING'
        RETURNING *;
    """
    return text, [run_id, status, note]


def requeue_agent_test_run_query(run_id: str) -> Tuple[str, List[Any]]:
    """Put a RUNNING run back in the queue, without the results it recorded."""
    text = f"""
        WITH requeued AS (
            UPDATE {RUN_TABLE}
            SET status = 'QUEUED', started_at = NULL, updated_at = now()
            WHERE id = $1 AND status = 'RUNNING'
            RETURNING id
        )
        DELETE FROM {RESULT_TABLE}
        WHERE run_id IN (SELECT id FROM requeued);
    """
    return text, [run_id]


def prune_agent_test_history_query(
    template_id: str, keep: int, live_fingerprint: Optional[str]
) -> Tuple[str, List[Any]]:
    """Retention: drop a template's finished runs older than its newest
    ``keep`` versions, except the published version's and the verdict of the
    content real calls run (``live_fingerprint``; saves that change no content
    still bump the version), with their results and their test calls' EVAL
    leads, wherever a transfer moved them; also the EVAL leads no kept run
    recorded (a retried call's first lead) older than every kept run, and
    those of deleted templates. Only EVAL leads are ever deleted. (A lead's id
    is text; a result's lead_id is a uuid.)"""
    text = f"""
        WITH head AS (
            SELECT current_version, published_version FROM template WHERE id = $1
        ),
        live_verdict AS (
            SELECT id FROM {RUN_TABLE}
            WHERE template_id = $1 AND fingerprint = $3
              AND status <> 'CANCELLED' AND status NOT IN {_PENDING}
            ORDER BY created_at DESC
            LIMIT 1
        ),
        doomed AS (
            SELECT r.id FROM {RUN_TABLE} r, head
            WHERE r.template_id = $1
              AND r.status NOT IN {_PENDING}
              AND r.template_version <= head.current_version - $2::integer
              AND r.template_version IS DISTINCT FROM head.published_version
              AND r.id NOT IN (SELECT id FROM live_verdict)
        ),
        kept_leads AS (
            SELECT res.lead_id::text AS lead_id FROM {RESULT_TABLE} res
            JOIN {RUN_TABLE} r ON r.id = res.run_id
            WHERE r.template_id = $1
              AND res.lead_id IS NOT NULL
              AND r.id NOT IN (SELECT id FROM doomed)
        ),
        oldest_kept AS (
            SELECT min(created_at) AS at FROM {RUN_TABLE}
            WHERE template_id = $1 AND id NOT IN (SELECT id FROM doomed)
        ),
        dropped_leads AS (
            DELETE FROM lead_call_tracker
            WHERE execution_mode = 'EVAL'
              AND (
                  template_id IS NULL
                  OR id IN (
                      SELECT lead_id::text FROM {RESULT_TABLE}
                      WHERE run_id IN (SELECT id FROM doomed)
                  )
                  OR (
                      template_id = $1
                      AND id NOT IN (SELECT lead_id FROM kept_leads)
                      AND created_at < (SELECT at FROM oldest_kept)
                  )
              )
            RETURNING id
        ),
        dropped_runs AS (
            DELETE FROM {RUN_TABLE} WHERE id IN (SELECT id FROM doomed)
            RETURNING id
        )
        SELECT (SELECT count(*) FROM dropped_runs) AS runs,
               (SELECT count(*) FROM dropped_leads) AS leads;
    """
    return text, [template_id, keep, live_fingerprint]


def fail_stale_agent_test_runs_query(stale_minutes: int) -> Tuple[str, List[Any]]:
    """Close runs whose worker died with them RUNNING."""
    text = f"""
        UPDATE {RUN_TABLE}
        SET status = 'ERROR', note = 'The test run stopped before it finished',
            finished_at = now(), updated_at = now()
        WHERE status = 'RUNNING'
          AND started_at < now() - make_interval(mins => $1)
        RETURNING id;
    """
    return text, [stale_minutes]


def set_pending_agent_test_runs_go_live_query(
    template_id: str, fingerprint: Optional[str], go_live: bool
) -> Tuple[str, List[Any]]:
    """Set go_live on the template's pending runs; of one version only when
    ``fingerprint`` is given."""
    text = f"""
        UPDATE {RUN_TABLE}
        SET go_live = $3, updated_at = now()
        WHERE template_id = $1
          AND status IN {_PENDING}
          AND ($2::text IS NULL OR fingerprint = $2)
        RETURNING id;
    """
    return text, [template_id, fingerprint, go_live]


def list_agent_test_runs_query(template_id: str, limit: int) -> Tuple[str, List[Any]]:
    text = f"""
        SELECT * FROM {RUN_TABLE}
        WHERE template_id = $1
        ORDER BY created_at DESC
        LIMIT $2;
    """
    return text, [template_id, limit]


def get_agent_test_version_runs_query(
    template_id: str, fingerprint: str
) -> Tuple[str, List[Any]]:
    """A version's latest pending run and latest finished run (at most one each)."""
    text = f"""
        SELECT DISTINCT ON (status IN {_PENDING}) *
        FROM {RUN_TABLE}
        WHERE template_id = $1 AND fingerprint = $2 AND status <> 'CANCELLED'
        ORDER BY status IN {_PENDING}, created_at DESC;
    """
    return text, [template_id, fingerprint]


def insert_agent_test_results_query(
    run_id: str,
    scenario_ids: List[Optional[str]],
    scenario_names: List[str],
    attempts: List[int],
) -> Tuple[str, List[Any]]:
    text = f"""
        INSERT INTO {RESULT_TABLE}
            (run_id, scenario_id, scenario_name, attempt, status)
        SELECT $1, scenario_id, scenario_name, attempt, 'QUEUED'
        FROM unnest($2::uuid[], $3::text[], $4::int[])
            AS r(scenario_id, scenario_name, attempt)
        RETURNING *;
    """
    return text, [run_id, scenario_ids, scenario_names, attempts]


def start_agent_test_result_query(result_id: str) -> Tuple[str, List[Any]]:
    text = f"""
        UPDATE {RESULT_TABLE}
        SET status = 'RUNNING', updated_at = now()
        WHERE id = $1;
    """
    return text, [result_id]


def finish_agent_test_result_query(
    result_id: str,
    status: str,
    lead_id: Optional[str],
    outcome: Optional[str],
    failures_json: str,
    duration_ms: int,
) -> Tuple[str, List[Any]]:
    text = f"""
        UPDATE {RESULT_TABLE}
        SET status = $2, lead_id = $3, outcome = $4, failures = $5::jsonb,
            duration_ms = $6, updated_at = now()
        WHERE id = $1;
    """
    return text, [
        result_id,
        status,
        lead_id,
        outcome,
        failures_json,
        duration_ms,
    ]


def list_agent_test_results_query(run_id: str) -> Tuple[str, List[Any]]:
    text = f"""
        SELECT * FROM {RESULT_TABLE}
        WHERE run_id = $1
        ORDER BY created_at, scenario_name, attempt;
    """
    return text, [run_id]
