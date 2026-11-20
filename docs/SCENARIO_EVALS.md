# Scenario evals

Scenario tests for Buddy voice templates, run before a template goes live. Each
run drives a real call through Buddy's own pipeline (template flow, LLM, function
hooks, end node, call completion) with Pipecat's eval harness playing the caller.
Code: `app/ai/voice/agents/breeze_buddy/scenario_tests/` (checklist, writer,
runner, gate, suite, worker); bot side: `_setup_eval_transport` in
`app/ai/voice/agents/breeze_buddy/agent/__init__.py`.

## Automatic agent tests

With `AGENT_TESTS_ENABLED` on, every phone voice agent is tested on its own
(not a chat agent, whose voice is in-widget; not an IVR flow; not an agent with
MCP servers, whose tools a test call would reach for real), and goes live only once its tests pass. The loom Tests tab shows each
scenario queued, running, passed or failed.

`AGENT_TESTS_ENABLED` is a live flag (`app/core/config/dynamic.py`): DevCycle /
the Redis `devcycle:flags` blob, else the environment, else off. It is read on
every save, run and test session, so turning it on or off needs no restart.

- **A new agent is a draft.** Creating it with `is_active: true` saves it paused
  and queues a run that turns it on when it passes. The first run writes the
  agent's scenarios from the checklist below.
- **Every edit is a draft until it passes.** Any change to the flow (prompts
  included), configurations or either schema, from the console, the API or a
  rollback, is tested. Until it passes, the edit is the template's head, which
  test calls run (`EVAL`, `TELEPHONY_TEST`, `DAILY_TEST`), while real calls keep
  running the published version (`template.published_version`, migration 087).
  On a pass the head is published by itself; a failed draft never reaches a
  real call. Nothing is rolled back: the draft stays, marked failed, to fix.
  A rollback that is a draft keeps the head's secrets, which real calls use.
- **What a real call runs** comes from `get_live_template_by_id`: the published
  version's content under the head's name, secrets, number and pause state. The
  voice loader (call start and agent transfer), the dispatcher (pre-checks,
  greeting pre-warm, number and credentials), lead intake (payload schema; test
  modes check the head), inbound routing and IVR selection, and the chat cache
  use it. An agent with a draft cannot move to another owner (409). A
  draft's greeting audio is kept per lead, out of the template's shared
  greeting key, and the published version's snapshot is never pruned. Still
  reading the head: the post-call outcome check and CRM mirror, the Daily room's
  noise setting and the Plivo account lookup, so a draft that changes those
  settings reaches them before it passes.
- **Tests key on a fingerprint** of that content, so pausing, resuming or
  renaming never re-runs them, and content that already passed is green again
  at once. With `AGENT_TESTS_ENABLED` off, saves go live as asked and nothing
  is tested; a draft already waiting keeps real calls on its published version
  until its next save.
- **Going live:** a paused agent whose content passed goes live; one with a
  published version goes live on it while its draft waits; one with neither
  waits paused for its run, or is refused (409) if its content failed. Pausing
  is never refused, and from then on no pending run turns the agent back on,
  whether tests are on or not.
- **An admin can pass a version by hand** (`POST /templates/{id}/tests/override`,
  admins only) with a reason, which the run history keeps. A live agent's draft
  is published at once; a paused agent can then go live.
- **Failures are announced**: a failed or broken run, and every pass by hand,
  is posted to the Slack alerts channel like a template update (agent, version,
  result, what real calls run now, who).
- Generated scenarios answer every outbound request with a 200, so they test
  the conversation; a custom scenario that tests an integration states its own
  fixtures.

### What every agent is tested on

`scenario_tests/checklist.py` lists the situations real calls meet that a text
conversation can judge. Buddy writes one test per situation that applies to the
agent (`scenario_tests/generate.py`, Claude Sonnet 4.5), in its language and
with its own outcomes and nodes, and asks once more for any it left out.

| Situation | Applies when |
| --- | --- |
| Happy path, one per way a call can end | always |
| Busy now; angry caller; asks to repeat; "are you a robot?" | always |
| Wrong person or family member (no details shared); off-topic or a promise it cannot make. Two calls, both must pass | always |
| Says one choice and takes it back | the call can end more than one way |
| Busy, call me back at a time | a callback outcome or function |
| Asks for a human | a transfer function |
| Gives a detail and corrects it | the agent collects details |
| Switches language | the prompt speaks more than one |
| No customer name in the lead | the payload has a name field |
| Systems it looks data up in are down | it makes HTTP requests during the call |

A scenario checks how the call ends (its outcome and the end node it finishes
in) only for the situations about the ending: a happy path, changing one's
mind, a callback, a human, a corrected detail. Elsewhere more than one ending is
right (an angry caller won over may still confirm), so the conversation alone is
judged. Nodes on the way are not checked: whether a call passes one depends on
how the caller words it.

Every scenario decides going live. A machine answering (voicemail, a hold
message) is not tested: real calls catch it on the audio (carrier machine
detection, the voicemail observer), which a text test cannot show. Noise,
mis-heard speech and barge-in need audio tests.

Every run starts with a **setup check**: each provider account the agent names
is resolved the way a call resolves it, and its LLM answers one prompt. A
removed provider (Azure) or a missing key fails the run there, with the reason,
before any test call.

A test call the agent fails, fails its scenario: an agent that gets it wrong
one time in two is wrong, so there is no second chance. Only a call the harness
broke is made once more, and counts by its retry: one that never connected or
started, whose judge or simulated caller erred, or that ran past its limits.
Pipecat reports these by failure kind, not by raising. If the retry breaks too,
the run ends `ERROR` (it did not finish), never `FAILED`. The safety situations
make two calls, so a slip that happens one time in two is caught three times in
four.

The writer must cover every situation on the checklist: one it leaves out twice
ends the run `ERROR`, naming it, so no situation goes untested unseen. Custom
scenarios alone never stand in for the checklist. A run is a verdict only on
content nobody saved over while it ran: a save during it ends it `CANCELLED`,
since its later calls ran other content, and when the saved content is the
run's own (a pause, a rename, an undo) it is run again.

### Where tests run

Only on the agent-tests worker: pods with `ENABLE_AGENT_TESTS_WORKER=true`, a
deployment of the same image that takes no traffic (no Service), like the
dialler. API and dialler pods only queue runs; the eval route is mounted on the
worker's pods alone, so no test call shares a pod with call setup, dialing or a
live call. While `AGENT_TESTS_ENABLED` is on, the worker polls every
`AGENT_TESTS_POLL_INTERVAL_SECONDS` (5): it closes a run a dead pod left
`RUNNING` for `AGENT_TESTS_STALE_RUN_MINUTES` (20) as `ERROR`, then runs queued
runs, oldest first, stopping between runs if the flag goes off (`SKIP LOCKED`,
so two worker pods never run the same one). A run a stopping pod cuts off (a
deploy) goes back to the queue, its results dropped, for the next pod: on the
preStop `/drain`, before uvicorn stops taking the run's test calls. Each suite
connects to its own pod's eval route (`ws://127.0.0.1:{PORT}`),
`AGENT_TESTS_CONCURRENCY` (5) sessions at a time. These three are live settings
(`app/core/config/dynamic.py`). A scenario makes at most 3 calls of 30 turns
and 600 s, and an agent has at most 40 scenarios. A scripted scenario counts
each turn's longest wait (60 s per expected event unless `within_ms` says less,
30 s for a `send_after` event, plus its delay), and every test call is cut off
a minute past the limit.

| Worker pod env | |
| --- | --- |
| `ENABLE_AGENT_TESTS_WORKER` | `true` |
| Pod labels | not `app: clairvoyance`, so the API Service never routes to it |
| preStop | `httpGet /drain`: stops the worker while the pod still takes connections, so the run it cuts off is requeued, not failed |

Like a CRM worker pod, it runs only its own loop: the background scheduler, the
dispatcher and the conversation analysis worker never start there, whatever
their flags say. Every release must give it the same image as the API and
voice-agent pods, or it tests agents on old bot code. Without such a pod, runs
stay queued and drafts wait; nothing else breaks.

Test history is pruned after every run, like template versions: an agent keeps
the runs of its newest `AGENT_TESTS_VERSION_RETENTION` (4) versions and of the
published one, and the run that judged the content real calls run (a pause or a
rename bumps the version without changing content), with their results and
test-call leads. Older runs go with their `EVAL` leads, even those a transfer
moved to another agent, as do the leads of a deleted agent and a retried call's
first lead once every kept run is newer. Only `EVAL` leads are ever deleted.

A run supersedes the template's queued ones, keeping their wish to go live for
the same content. A live agent's run always goes live on a pass, which
publishes its draft. A paused agent stays paused, whatever changed: only going
live (Go live, a save that goes live, or creating it live) turns it on once its
tests pass, never Run tests or a scenario edit.

### Models

| Role | Default (`app/core/config/static.py`, env) |
| --- | --- |
| Judge: did the call meet the scenario's success line | `AGENT_TESTS_JUDGE_MODEL`, Claude Haiku 4.5 (Bedrock India profile) |
| Simulated caller | `AGENT_TESTS_CALLER_MODEL`, Claude Haiku 4.5 (Bedrock India profile) |
| Scenario writer, once per agent and on Rewrite | `AGENT_TESTS_WRITER_MODEL`, Claude Sonnet 4.5 (Bedrock global profile; no India one), up to `AGENT_TESTS_WRITER_MAX_TOKENS` (8192), 5 minutes per call |
| The agent under test, and the setup check's one prompt | the agent's own LLM configuration |

All go through the app's own LLM factory, keys and default region; the judge
and caller are a different model family from the luna agent.

| Endpoint | |
| --- | --- |
| `GET /templates/{id}/tests` | The current version's state and run, recent runs, scenarios |
| `POST /templates/{id}/tests/run` | Queue a run of the current version; a paused agent stays paused |
| `POST /templates/{id}/tests/scenarios/generate` | Queue a run that rewrites the generated scenarios (custom ones stay), then runs them |
| `POST`, `PUT`, `DELETE /templates/{id}/tests/scenarios[/{sid}]` | Custom scenarios; each change re-runs the suite |
| `POST /templates/{id}/tests/override` | Admin only, with a reason |

## Custom scenarios

The Tests tab's **Add scenario** stores a Pipecat scenario as JSON with a
`buddy` block; it always runs against its own agent.

```json
{
  "persona": "Priya, in a meeting, answering in short English sentences.",
  "goal": "Say you cannot talk now. Agree to nothing your goal does not tell you to. Do not hang up yourself: wait for the agent to end the call.",
  "success": "the agent ends the call politely without reading the order",
  "max_turns": 6,
  "buddy": {"outcome": "BUSY", "visits": ["end_conversation_node"]}
}
```

| `buddy` key | Meaning |
| --- | --- |
| `payload` | Lead data, over the schema's `example` values; none given, the data Buddy wrote for the agent's own tests |
| `http` | Fixtures for outbound HTTP (below) |
| `outcome` | The lead's final outcome, one value or a list |
| `visits` / `avoids` | Nodes the call must reach / must not enter |
| `runs` | Calls this scenario makes, at most 3; every one must pass |

- Simulated callers (`persona`, `goal`, `success`) suit paths that are brittle
  to script. Keep the persona free of cues for other outcomes: a persona told it
  is "at work" says "busy", and the agent rightly ends as BUSY.
- A scripted scenario (`turns`) checks exact events. Buddy speaks first, so its
  first caller turn waits for the greeting unless it sets `send_after`; end it
  with `event: bot_ended` when the call should end the way production calls do.

## HTTP fixtures

Tool calls, function hooks and the end-of-call webhook never leave the process in
an eval. The first fixture whose method and URL glob match answers; a request no
fixture covers gets a 404 and fails the run. A fixture key other than `method`,
`url`, `status` or `body` refuses the scenario when it is saved. Fixtures match
the URL as sent, but the run records and reports it without its query string or
credentials, and never keeps request bodies: both carry the template's secrets.

```json
{"http": [
  {"method": "GET", "url": "https://merchant.example.com/orders/*", "body": {"status": "shipped"}},
  {"method": "POST", "url": "*", "status": 200}
]}
```

## What an eval session changes in Buddy

An eval runs as an `EVAL` lead. The call takes the production path, with these
differences only:

- Transport: the harness attaches over `/agent/voice/breeze-buddy/eval/{lead_id}/ws`
  with Pipecat's eval transport, carrying the lead's token (an HMAC of its id and
  a 10-minute expiry under the server's signing key, signed like the try-on
  token); without a valid one the handshake is refused. The bot
  refuses an audio-mode session, a lead whose execution mode is not `EVAL`, or
  one a session already ran, with close code 4003 before touching it; while
  agent tests are off it refuses every session. A carrier route refuses an eval
  connection, as it has no call id. Only the runner creates `EVAL` leads:
  `/push/lead/v2` rejects the mode.
- No eval lead is left `PROCESSING`. A template the bot cannot run (IVR, or one
  that fails to load) is refused with 4000, also before the lead is touched; a
  session that crashes after it started closes its lead as `EARLY_HANGUP`; the
  runner closes any lead its run left open. The stuck-call sweeper closes an
  `EVAL` lead without the P1 page it sends for real calls.
- Sessions run in text mode with stand-ins for the template's STT and TTS
  (`TextOnlySTTService`, `TextOnlyTTSService` in `agent/pipeline.py`), so a test
  call opens no Soniox, ElevenLabs or other vendor session. Replies arrive as
  text; fixed speech (`tts_say`, fillers, a pre-TTS message) becomes a moment of
  silence, so its text still reaches the transcript. The greeting is reported to the
  harness once the flow's context and tools are queued, a typed turn counts as
  the caller speaking for idle handling, and `UnspokenReplyProcessor` marks the
  bot's turn so the end node's post-actions run.
- Pipecat 1.14's eval transport turns TTS off below the RTVI processor, which
  then turns it back on after every typed turn. The agent also sets it from the
  top of the pipeline; drop that once Pipecat fixes it.
- The end-of-call callbacks run just before the lead is saved instead of just
  after, so the webhook they send is among the requests the run records.
- The result is recorded even when the agent has no call execution config yet,
  so a brand-new agent can be tested before it is set up to dial.
- Outbound HTTP goes to the fixtures; `send_alert` logs instead of posting to
  Slack; the call is excluded from the CRM mirror, from retries and from the
  post-call conversation evaluation queue.
- Every finished call, eval or not, records `template_version` in its metadata.
- A test call cannot transfer to an agent that cannot be tested: an MCP agent's
  tools would be called for real. The transfer is refused and the call carries
  on with its own agent.

Not covered yet: agents with MCP servers, chat agents and IVR templates are not
tested (the eval route refuses them too), and audio mode (real STT and TTS on
both sides) is refused.
