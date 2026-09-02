---
description: Diagnose a training run — did it learn, diverge, or die, and how does it differ from a baseline
argument-hint: "[run dir] [baseline run dir] — defaults to the newest run"
---

# /run-triage — Diagnose a Training Run

Run the **`run-triage`** agent. Do not parse the metrics yourself in this context — that is the entire point of the agent. TensorBoard event files, wandb scalar dumps and log tails must never enter the main session.

## 1. Resolve the target

- If `$ARGUMENTS` gives one path, that is the run.
- If it gives two, the second is the baseline to compare against.
- If it gives none, pass no target — the agent picks the newest run and states which.

## 2. Dispatch

Spawn the `run-triage` agent with the resolved paths. Run it in the background — triage takes a while and nothing here depends on it immediately.

## 3. Relay

Report the agent's report **verbatim**: status line, verdict, metrics table, red flags, config drift, recommended next step. Do not re-summarize the verdict in your own words.

If the verdict is ❌ failed or ⚠️ suspect, offer to run `/orchestrate` on the recommended next step. Do not start fixing anything unprompted.
