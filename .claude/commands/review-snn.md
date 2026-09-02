---
description: Spiking neural network bug review (spike health, encoder bounds, LIF dynamics, surrogate gradients)
argument-hint: "[file or directory — defaults to the open file, else the uncommitted diff]"
---

# /review-snn — Spiking Neural Network Bug Review

Run the **`review-snn`** agent. Do not perform the review yourself in this context.

## 1. Resolve the target (do this here — the agent cannot see your IDE)

- If `$ARGUMENTS` names a file or directory, that is the target.
- Else, if a file is open in the IDE, that is the target.
- Else, run `git diff --name-only HEAD` and use the changed `.py` files.
- If that is empty, say so and stop.

## 2. Dispatch

Spawn the `review-snn` agent with the resolved target passed explicitly as absolute paths. Run it in the background unless the next thing you do depends on the result.

## 3. Relay

Report the agent's findings to the user **verbatim** — the full severity tables and the spike-health summary. Do not summarize them away; the tables are the deliverable. Add at most two lines of your own: which target was reviewed, and the single highest-severity item.

Do not apply fixes unless the user asks.
