---
description: Route a task across the domain experts (RL/SNN, flight stack, mechanics), synthesize one plan, then implement it
argument-hint: "<what you want to get done>"
---

# /orchestrate — Route Work Across the Experts

The user's goal: **$ARGUMENTS**

You are the orchestrator. You stay in this session, you own the decomposition and the synthesis, and **you are the only one who writes code.** The experts are read-only advisors.

## The expert roster

| Agent | Owns | Triggers on |
|---|---|---|
| `rl-aerial-gym-planner` | RL, Isaac Gym, Aerial Gym, SNN | rewards, curriculum, obs/action spaces, PPO/rl_games, task/robot/env registries, tensors, PopSAN/snnTorch, distillation |
| `robotics-stack-expert` | Flight stack & middleware | PX4, ArduPilot, ROS 2, Gazebo, MAVLink, uXRCE-DDS, Isaac Sim/Pegasus, launch orchestration, DDS/networking |
| `sim2real-mech-expert` | Mechanics & sim2real | mass, CoM, inertia, motors, props, batteries, thrust curves, drag, materials, URDF/SDF `<inertial>`, real-vs-sim behavioural gaps |
| `review-*` | Correctness | invoked at the end via `/review-all`, not during planning |

## Procedure

### 1. Restate
One line: what "done" means for this goal. If that is genuinely ambiguous in a way that changes the work, ask now — before spending any agent on it.

### 2. Decompose
Break the goal into workstreams and tag each with its owning domain. Note which workstreams depend on another's output.

### 3. Decide whether to fan out at all

**This is the important step. Read it before spawning anything.**

- **One domain → do NOT spawn.** Load that expert's skill inline (`aerial-gym-isaac-gym-expert`, `px4-ros2-gazebo-expert`, `isaac-sim-expert`, `uav-mech-expert`) and do the work yourself. A subagent starts cold and re-derives context you already have; for single-domain work it is strictly slower.
- **Two or more domains → fan out.** Spawn only the tagged experts, **in a single message** so they run in parallel, in the background.
- **A workstream that depends on another's answer** does not go in the first wave. Fan out the independent ones, then dispatch the dependent one with the first wave's answer included in its prompt.

Say out loud which experts you are spawning and why — or, if you are not spawning any, say that and proceed.

### 4. Brief each expert properly
A cold agent knows nothing about this conversation. Each prompt must carry:
- The overall goal, in one sentence, so it can judge relevance.
- Its specific workstream, and explicitly what is **not** its problem.
- The concrete files, configs or run dirs you already know are involved.
- Any decision already made that it must not relitigate.
- What you want back: a plan, a diagnosis, or specific numbers.

### 5. Synthesize
When the reports are in, produce **one** plan — not three pasted together:
- Ordered steps across all domains, with the real file each touches.
- **Contradictions between experts stated explicitly**, with your call and your reasoning. Two experts disagreeing about an interface is the most valuable thing this command produces — never paper over it.
- Open questions that no expert could resolve.

Show the plan and get the user's go-ahead before editing anything.

### 6. Implement
You do the edits, in this session, sequentially. Do not spawn writer agents — concurrent writers conflict, and one writer keeps the diff coherent.

If a variant needs to be explored without disturbing the working tree, use a worktree-isolated agent for that variant only, and say so first.

### 7. Verify
Run `/review-all` on the diff. Then, if the change affects training, propose the run and `/run-triage` it once it finishes.

## Standing rules

- Experts advise; this session writes.
- Cite real `file:line`. If an expert returns something you cannot verify in the code, check it before acting on it — a confident agent report is not evidence.
- Prefer the smallest change that works. This repo follows upstream aerial_gym structure; divergence must sit at the line that forces it.
