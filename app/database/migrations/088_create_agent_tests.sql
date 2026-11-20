-- ===========================================================================
-- 088: agent tests — a voice agent's scenario suite, its runs and results
-- ---------------------------------------------------------------------------
-- A template's scenarios run on every change to its content (flow,
-- configurations, both schemas), identified by a sha256 fingerprint of that
-- content. A version is green when its latest verdict PASSED or was
-- OVERRIDDEN by an admin; a draft goes live only on green.
--
-- Status and source values live in code (scenario_tests/), not in CHECKs.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS agent_test_scenario (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id  uuid NOT NULL REFERENCES template(id) ON DELETE CASCADE,
    name         varchar(128) NOT NULL,
    -- A Pipecat scenario mapping (persona/goal/success, or turns) with its
    -- buddy: block; the template is always the owning row's.
    scenario     jsonb NOT NULL,
    -- generated | custom
    source       varchar(20) NOT NULL,
    is_enabled   boolean NOT NULL DEFAULT TRUE,
    created_by   varchar(255),
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT agent_test_scenario_name_format CHECK (name ~ '^[a-z0-9][a-z0-9-]*$')
);

CREATE UNIQUE INDEX IF NOT EXISTS agent_test_scenario_template_name_uq
    ON agent_test_scenario (template_id, name);

CREATE TABLE IF NOT EXISTS agent_test_run (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id       uuid NOT NULL REFERENCES template(id) ON DELETE CASCADE,
    fingerprint       char(64) NOT NULL,
    template_version  integer NOT NULL,
    -- QUEUED | RUNNING | PASSED | FAILED | ERROR | CANCELLED | OVERRIDDEN
    status            varchar(20) NOT NULL,
    -- Turn the agent on when this run passes and the agent is unchanged.
    go_live           boolean NOT NULL DEFAULT FALSE,
    -- created | edited | rollback | scenarios | rewrite | manual | override
    trigger           varchar(20) NOT NULL,
    requested_by      varchar(255),
    -- An override's reason, or why a run could not finish.
    note              text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    started_at        timestamptz,
    finished_at       timestamptz,
    updated_at        timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT agent_test_run_fingerprint_format CHECK (fingerprint ~ '^[0-9a-f]{64}$')
);

-- The Tests tab and the gate: a template's runs, newest first.
CREATE INDEX IF NOT EXISTS agent_test_run_template_created
    ON agent_test_run (template_id, created_at DESC);

-- The worker's claim: oldest queued run first.
CREATE INDEX IF NOT EXISTS agent_test_run_queued
    ON agent_test_run (created_at) WHERE status = 'QUEUED';

CREATE TABLE IF NOT EXISTS agent_test_result (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id         uuid NOT NULL REFERENCES agent_test_run(id) ON DELETE CASCADE,
    scenario_id    uuid REFERENCES agent_test_scenario(id) ON DELETE SET NULL,
    scenario_name  varchar(128) NOT NULL,
    attempt        integer NOT NULL,
    -- QUEUED | RUNNING | PASSED | FAILED
    status         varchar(20) NOT NULL,
    -- The EVAL lead the attempt ran as: its transcript and requests.
    lead_id        uuid,
    outcome        varchar(255),
    failures       jsonb NOT NULL DEFAULT '[]'::jsonb,
    duration_ms    integer,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT agent_test_result_attempt_positive CHECK (attempt > 0)
);

CREATE INDEX IF NOT EXISTS agent_test_result_run
    ON agent_test_result (run_id);
