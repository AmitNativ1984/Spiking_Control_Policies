---
description: Fan out every applicable bug review (RL, numerical, SNN, software) over one target in parallel
argument-hint: "[file or directory — defaults to the open file, else the uncommitted diff]"
---

# /review-all — Parallel Multi-Lens Review

Review one target through every lens that applies, **concurrently**. This is the default review entry point; the single-lens commands exist for when you want just one.

## 1. Resolve the target (do this here — the agents cannot see your IDE)

- If `$ARGUMENTS` names a file or directory, that is the target.
- Else, if a file is open in the IDE, that is the target.
- Else, run `git diff --name-only HEAD` and use the changed `.py` files.
- If that is empty, say so and stop.

State the resolved target in one line before dispatching.

## 2. Choose the lenses

Always include:
- **`review-rl`** — this is a robotics RL repo; coordinate frames and env logic are always in scope.
- **`review-sw`** — always.

Include conditionally:
- **`review-snn`** — if the target touches spiking code: `networks/snn/`, `simple_hover_snn/`, popsan, encoder, decoder, snntorch, norse, surrogate gradients, spike rates, or teacher-student warm-up.
- **`review-num`** — if the target contains tensor math, losses, normalization, gradients, or any `torch` numerics beyond plain plumbing.

When in doubt, include the lens. A skipped lens costs more than a redundant one.

## 3. Dispatch in parallel

Spawn all chosen agents **in a single message** so they run concurrently, each with the same explicit target paths. Run them in the background.

Tell the user which lenses you dispatched and that results will arrive as each finishes.

## 4. Consolidate

As reports arrive, present them under one heading per lens, tables **verbatim**. Then add a short synthesis:

```
### Consolidated
| Severity | Lens | File:Line | Issue |
|----------|------|-----------|-------|
```
- Merge duplicates where two lenses flagged the same line, noting both lenses agreed (agreement raises confidence).
- Where two lenses **disagree** about the same line, show both and say which you find more credible and why. Do not silently pick one.
- Order by severity: all CRITICAL first, then WARNING, then INFO.

End with the single most important thing to fix first.

Do not apply fixes unless the user asks. If they do, fix CRITICAL items first and re-run only the lenses whose findings you touched.
