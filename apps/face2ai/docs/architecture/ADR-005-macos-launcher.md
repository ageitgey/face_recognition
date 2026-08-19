# ADR-005 - macOS launcher: an AppleScript applet over a shell service script

Status: accepted for the launcher; the `.app` bundle is not built yet (see Verification)
Date: 2026-08-19

## Context

Starting Face2AI today means opening a terminal, remembering `uv run --project apps/face2ai face2ai`, and
remembering to stop it again. The user asked for a Mac launcher — a Dock/Spotlight entry that starts the
service, opens the UI and stops the service on quit, the pattern their other launchers on this machine
already use.

**This ADR states plainly what follows from what.** The launcher is the user's explicit request. It is *not*
a consequence of the six-dimension review or of the root cause in `docs/plans/2026-08-19-boundary-contracts.md`
("every artifact in this repository is scoped to one feature"). Of the work committed alongside it, exactly
two items are consequences of that review: the committed lockfile and the documented environment surface.
Presenting the launcher as a root-cause fix would be the same kind of unearned claim the review was written
to remove.

## Decision

Two artifacts, with the failure surface in the testable one:

1. **`scripts/face2ai-service.sh {start|stop|status}`** — a plain bash script that owns the process:
   `start` probes `/healthz` and refuses to start a second instance, launches the app detached with its log
   at `$FACE2AI_DATA_DIR/face2ai.log` and its pid at `$FACE2AI_DATA_DIR/face2ai.pid`, and blocks until
   `/healthz` answers or `FACE2AI_START_TIMEOUT_SECONDS` expires (then it prints the log tail, stops what it
   started, and exits non-zero); `stop` sends SIGTERM to the launched process and its descendants, waits
   `FACE2AI_STOP_TIMEOUT_SECONDS`, escalates to SIGKILL, and is a clean no-op when nothing runs; `status`
   prints the `/healthz` JSON and exits non-zero when the service is down.
2. **`Face2AI.app`** — an AppleScript applet in `~/Applications` built with `osacompile`. Its entire body:

   ```applescript
   on run
       do shell script "<repo>/apps/face2ai/scripts/face2ai-service.sh start"
       do shell script "open http://127.0.0.1:8765"
   end run

   on quit
       do shell script "<repo>/apps/face2ai/scripts/face2ai-service.sh stop"
       continue quit
   end quit
   ```

   Built by `scripts/build-macos-app.sh` (to be added with the bundle, by the session that builds it):
   `osacompile -o "$HOME/Applications/Face2AI.app" launcher.applescript`.

The applet carries no product logic. Everything that can fail — double-start, startup timeout, signal
escalation, "is it actually up" — lives in the shell script, where it is exercised from a terminal on a
spare port. That split is the point of this ADR.

## Why this is not the alternative ADR-001 rejected

`ADR-001-local-modular-monolith.md` lists under **Alternatives**:

> 1. React + Vite frontend plus FastAPI backend.
> 2. Tauri/desktop shell plus React and Python sidecar.

and decides:

> Use one local FastAPI process with explicit domain/port/adapter/service boundaries and a same-origin
> browser UI built from HTML, CSS and ES modules. The UI uses adapted ReactBits-inspired motion patterns
> without taking a React/Vite dependency in S0.

The applet is none of the three things that made alternative 2 expensive:

- **No new UI runtime.** There is no webview, no bundled browser engine, no React. The UI is the same
  same-origin page served by the same FastAPI process, opened in the user's own browser by `open`.
- **No bundled interpreter and no sidecar.** A "Python sidecar" means shipping an interpreter inside the
  desktop shell and managing it as a child of that shell. The applet ships nothing: it shells out to a
  script that runs the project's existing venv (or `uv`) on this machine. Deleting the `.app` leaves the
  product exactly as it was.
- **No second implementation of the product.** The applet is ~6 lines of AppleScript with no knowledge of
  recognition, enrollment, presence or events. It cannot drift from the product because it contains none
  of it.

What it does add is an OS-level start/stop wrapper around the same local process — the "one runtime" of
ADR-001 is preserved, not doubled. None of ADR-001's four **Review triggers** (all about React/Vite:
independent interactive surfaces, component duplication, component testing, Party Mirror composition) is
touched by it, so ADR-001 is not up for review.

## Reversed decision: `apps/face2ai/uv.lock` is now committed

The project `CLAUDE.md` recorded: *"`apps/face2ai/uv.lock` is untracked (never committed); `uv sync`
regenerates it."* That is reversed here, and the reason is specific rather than a general preference for
lockfiles:

- The sibling voice agent (`apps/face2ai-agent/`) tracks its lock; the product did not, so no second party
  could reproduce this environment at all.
- `dlib` — the native library behind the confirmed re-entrancy defect (unlocked, two threads on one frame
  report `[1, 78, 193, 456]` faces and abort the interpreter with returncode 133) — is pinned only as
  `dlib>=19.7` in the fork's `setup.py`. The lock is the *only* artifact in the repository that pins it to
  `20.0.1` with a sha256.

A committed lockfile enforces nothing on its own. The owner is CI: the app-shell job must install with
`uv sync --frozen`, which fails when the lock and `pyproject.toml` disagree instead of silently
re-resolving. **That CI change is not in this commit** (`.github/workflows/face2ai.yml` is owned by another
task in flight) and is the follow-up that turns this from a document into a rule.

## Alternatives

1. **A LaunchAgent that keeps the service running at login.** Rejected: this is a camera-facing product;
   always-on is the wrong default, and a permanently bound port would collide with the manually run
   backend, voice agent and SSH tunnel that already share this machine.
2. **Tauri or Electron shell** — ADR-001 alternative 2, still rejected, for ADR-001's reasons.
3. **A shell alias or a `Makefile` target.** Rejected only because it does not answer the request: no Dock
   icon, no Spotlight entry, no quit hook. The alias case is served by calling the script directly.

## Consequences

- The service script, not the applet, is the thing to test and to fix. It is exercised on a spare port.
- `stop` never kills a process it did not start: if `/healthz` answers while the pid file records nothing
  alive, it refuses and exits non-zero. This is deliberate — port 8765 on this machine also carries a
  manually started backend, a voice agent and a reverse SSH tunnel, and a launcher that killed by port
  would take all of them down.
- The SIGKILL escalation is load-bearing *today*: with one SSE subscriber attached the current process
  ignores SIGTERM (§3 of the boundary-contracts plan). After that task lands, SIGTERM will be enough and
  the escalation becomes the backstop it should be. Measured both ways.
- The applet is unsigned and un-notarized, so the first launch needs the usual Gatekeeper confirmation.
- Quitting the applet stops the backend, which disconnects any voice agent or Hermes plugin subscribed to
  its event stream. That is the honest meaning of "Quit"; it is not a background service.
- The `.app` hard-codes the repository path. Moving the checkout breaks the launcher until it is rebuilt.

## Verification

Measured for the service script (spare port 8815, isolated `FACE2AI_DATA_DIR`, port 8765 untouched):
`start` → `status` (`{"status":"ok"}` / `{"ready":true,"reason":null}`) → `start` again (refused, one pid,
no second process) → `stop` → `status` (down, exit 1) → `stop` (clean no-op, exit 0), pid file gone, no
listener left. Separately: `stop` with one SSE subscriber attached escalated to SIGKILL after 3 s and freed
the port; `stop` against a process started outside the script refused and left it alive.

**Not executed yet, and therefore not claimed:** the bundle itself. The manual checklist stays open until
someone runs it on this Mac — `open -a Face2AI` → `/healthz` answers → the browser opens the UI → Quit →
the process is gone and the port is free. It must be executed, not asserted.
