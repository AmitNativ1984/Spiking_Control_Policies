#!/usr/bin/env bash
# Stop hook: if code changed this session, remind Claude to run the review fan-out.
# Exit 2 (asyncRewake) re-wakes the model with the reminder injected as context.
#
# /review-all dispatches the applicable review-* agents in parallel and picks the
# lenses itself, so this no longer has to enumerate which reviews apply.
cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0
changed=$(git status --porcelain 2>/dev/null | grep -E '\.py$')
[ -z "$changed" ] && exit 0
echo "Code changed this session. Run /review-all on the changed files (it fans out the RL, software, and — where applicable — SNN and numerical reviews in parallel). Changed files:"
echo "$changed"
exit 2
