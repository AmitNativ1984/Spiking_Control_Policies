"""Offline tools: observation-statistics collection and encoder visualization.

Deliberately empty. Both modules import isaacgym or torch at module scope, so importing
them is a decision the caller makes explicitly (and, for collect_obs_stats, after
`import isaacgym` has already happened).
"""
