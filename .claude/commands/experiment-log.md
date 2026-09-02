# /experiment-log — Structured Research Journal

You maintain a durable, paper-ready research log for this project. Each entry captures *why* an experiment was run and *what it showed*, so that months later the reasoning trail exists instead of being lost in run-directory timestamps. This compounds directly into the methods/results sections of a paper.

## Log Location

Append to `navigation_with_obstacles/EXPERIMENTS.md` (create it with an `# Experiments` header if missing). Newest entries at the **top** (reverse chronological).

## Modes

- `/experiment-log <free text>` — create a new entry from what the user says + what you can infer from the current git state and recent runs.
- `/experiment-log` (no arg) — infer a draft entry from: current branch, `git diff`/recent commits since last entry, the newest `runs/` dir, and the open file. Then ask the user to confirm/fill the **Hypothesis** and **Result** before writing.
- `/experiment-log close <id>` — fill in the Result/Outcome of an existing open entry (status `🔬 running` → `✅`/`❌`/`➖`).

## Entry Template

```
## <YYYY-MM-DD> · <short title> · <status>
**Branch/commit**: <branch> @ <short-sha>
**ID**: <YYYY-MM-DD-nn>

**Hypothesis** — <what you expected and why, in one or two sentences.>

**Setup** — <the meaningful config: network, num_envs, num_steps, lr, reward weights, teacher/student checkpoint, seed. Link the run dir(s).>

**Change vs previous** — <the one or few deltas from the last comparable run; "first of its kind" if none.>

**Result** — <numbers that answer the hypothesis: final reward, divergence, spike health. Cite the run dir / triage. Empty + status 🔬 if still running.>

**Outcome** — ✅ confirmed / ❌ refuted / ➖ inconclusive — <one-line takeaway.>

**Next** — <the follow-up this implies.>
```

Status legend: `🔬 running` · `✅ confirmed` · `❌ refuted` · `➖ inconclusive`.

## Rules
- Convert relative dates to absolute (today's date is available in context).
- Keep entries terse and factual — this is a lab notebook, not prose. No filler.
- **Never invent results.** If a number isn't available, leave Result empty and mark `🔬 running`; offer to run `/run-triage` on the run dir to fill it.
- Pull the config deltas from the actual config files / `git diff`, not from memory.
- Cross-reference: link related entries by ID and link the run dir paths so `/run-triage` can pick them up.
- After writing, show the user the entry and the file path.
