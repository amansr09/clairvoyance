-- ===========================================================================
-- 087: template.published_version — the version real calls run while a newer
-- edit waits for its tests
-- ---------------------------------------------------------------------------
-- NULL: calls run the head (current_version), as before. Set: calls run that
-- version's snapshot in template_version, and the head is a draft that goes
-- live only once its agent tests pass (scenario_tests/gate.py). Test calls
-- always run the head. Only the agent-tests gate sets it, so with the gate off
-- it stays NULL everywhere.
-- ===========================================================================

ALTER TABLE template ADD COLUMN IF NOT EXISTS published_version integer;

ALTER TABLE template DROP CONSTRAINT IF EXISTS template_published_version_positive;
ALTER TABLE template ADD CONSTRAINT template_published_version_positive
    CHECK (published_version IS NULL OR published_version > 0);
