# AGENTS.md — Observer

Instructions for AI coding agents (Claude, Codex, Cursor, Copilot, or any other
assistant) working in this repository. Read this before making changes.

## Working agreement — read this first

Roles are split deliberately and this is the most important rule in this file:

- **Human is the architect, technical lead, and business analyst.** He decides
  the architecture, the module boundaries, which interfaces exist, and what
  gets built next.
- **The AI assistant is the developer.** It implements what's asked, inside the
  structure that's already there, and writes the tests/docs that go with it.

Concretely:

**Ask before, don't just do:**
- Adding, removing, renaming, or changing the method signature of anything in
  `src/interfaces/`.
- Changing the `SupervisorState` state machine (states or transitions).
- Changing the dependency direction (`implementations → core → interfaces`) or
  coupling `core`/`interfaces` to a specific implementation.
- Adding a new top-level module/package or a new third-party dependency.
- Any change to `main.py` that changes *which components exist*, as opposed to
  registering a new implementation behind an existing interface.

If a task seems to need one of the above, stop and propose it — explain the
trade-off in a sentence or two — rather than deciding and implementing it.

**Free to do without asking:**
- Fix bugs, add/improve tests, improve docstrings and comments.
- Refactor inside a single file/class as long as its public interface doesn't
  change.
- Add any new code that its allowed by above rules.

## Source of truth for architecture

**`README.md`, not this file, is canonical for Architecture and Data flow.**
It has the full interface table, the data-flow diagram, and the repo layout.
Read it before touching `src/`. If a change alters the data flow, adds/changes
an interface, or changes the state machine, update `README.md`'s Architecture
and Data flow sections in the same change — don't let them drift from the code.

The one-line summary, so it isn't missed: interface-driven (Strategy pattern)
throughout. Every component is an ABC in `src/interfaces/`; concrete classes
live in `src/implementations/`; everything is wired together in exactly one
place, `src/main.py`. The `Supervisor` in `src/core/` is the coordinator and
only depends on the interfaces, never on a concrete implementation.

## Platform: developed on Windows 11, deployed on Raspberry Pi

The target device is a Raspberry Pi; all day-to-day development happens on
Windows 11. **Every change must run unmodified on both** — no code path may
assume one OS.

## Multiple clients, one Observer core

The current FastAPI dashboard (`WebController`) is **one client** of Observer,
not *the* client. Desktop, other web UIs, or external systems consuming
Observer are expected later, each as its own `IController` implementation.

- `IController` is the seam. `Supervisor`/`core` must stay ignorant of HTTP,
  FastAPI, JSON shapes, or anything web-specific — it only knows the interface.
- If asked to support another client, implement a new `IController`, don't
  extend `Supervisor`'s public surface to fit that client's shape.
- The `Supervisor` only *pushes* data out (`build_status()`, `latest_frame`)
  and *accepts* requests (`request_start()`, etc.) — it never reaches back into
  a controller. Keep that direction.

## Repo layout

```
src/
  main.py              where all the components are wired together
  config.py            every setting, overridable by env var or .env
  core/                the Supervisor: state machine + per-frame loop
  interfaces/           the interfaces and the data they pass
  implementations/      the concrete components, plus dashboard.html
tests/                 pytest suite — no camera needed
prompts/               the original specification prompts
playground/            scratch scripts, not part of the running system
```

## Commands

```bash
pip install -r requirements.txt
python main.py          # dashboard on http://localhost:8000/
pytest                  # config in pytest.ini; no camera/network needed
pytest -s               # walkthrough tests print their vectors with labels
```

Python 3.12+. New settings go in `src/config.py` (pydantic-settings `Config`
class) with a default and a docstring, not scattered `os.environ` reads.

## Conventions already in the code — follow them

- `from __future__ import annotations` + full type hints on every module.
- Docstrings explain *why*, not just what (see `src/core/supervisor.py` for
  the style: short module docstring on intent, per-method notes on non-obvious
  decisions).
- Side channels (e.g. the feature CSV writer) must never disturb the main
  pipeline: wrap them in `try/except`, log, and continue.
- The processing loop never crashes on a bad frame or a failed step: catch,
  log, set an error message, transition to `ERROR` state, and let `STOP`
  recover it — don't let an exception propagate out of `tick()`/`step()`.
- `Supervisor` state shared with other threads is read/written behind
  `self._lock`; anything a UI thread can read concurrently with the
  processing loop needs the same treatment.

## Known gaps (see README for detail — don't silently "fix" these; ask first)

- Trained models aren't saved/loaded on restart yet, though the code paths
  exist.
- The dashboard is intentionally read-only beyond Start/Stop/camera-select.
- Two open TODOs in the anomaly detector around training-data flattening and
  whether loading a model should restore full system state.

## Before finishing a change

- Run `pytest`.
- If you touched an interface, the state machine, or the data flow, update
  `README.md` accordingly.
- If you made an architectural judgment call instead of asking, say so
  explicitly when reporting back, so it can be reviewed rather than silently
  accepted.
