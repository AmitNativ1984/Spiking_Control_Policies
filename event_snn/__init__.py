"""Supervised BPTT experiment: what a stateful SNN learns about depth and ego-velocity from events.

Plan: docs/event_snn_experiment_plan.md. Modules that need Isaac Gym (sim, train, evaluate
--rollout) import it themselves; config, data, model and loss are plain torch.
"""
