-- Migration: Add EVAL to execution_mode CHECK constraint
-- Description: Leads created by the scenario eval harness. A distinct mode
--              keeps them out of dispatch, retries, analytics and the CRM.

ALTER TABLE lead_call_tracker
DROP CONSTRAINT IF EXISTS lead_call_tracker_execution_mode_check;

ALTER TABLE lead_call_tracker
ADD CONSTRAINT lead_call_tracker_execution_mode_check
CHECK (execution_mode IN ('TELEPHONY', 'TELEPHONY_TEST', 'DAILY', 'DAILY_TEST', 'DAILY_STREAM', 'HOLD_TRANSFER', 'EVAL'));
