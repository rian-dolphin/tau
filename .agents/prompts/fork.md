---
description: Manage this fork — sync with upstream, contribute a fix upstream, or use a pending fix locally before it merges.
---

You are helping manage a **forked open-source repo** that carries private local
customizations while still contributing bug fixes upstream. Follow the model and
flows below. First read the request, then pick the matching flow. If the request
is empty or ambiguous, briefly ask which of the flows is intended.

Request: {{ args }}

## Repo model (the invariants — never violate these)

- **Remotes**: `origin` = my fork; `upstream` = the original project.
- **`main`** is a *pristine mirror* of `upstream/main`. **Never commit to `main`.**
  It only ever moves by fast-forward from upstream.
- **`personal`** is my working branch and conceptually equals *`main` + a stack of
  local commits*. That stack holds two kinds of commits: permanent private tweaks
  (tool descriptions, system-prompt edits) and fixes that are pending upstream.
  **This is the branch I run from.**
- Short-lived **`fix/<thing>`** branches are cut fresh from clean `main`, one per
  upstream PR, so PRs never contain my private tweaks.
- `git rerere` is enabled globally, so recurring conflict resolutions replay
  automatically.
- The tool is installed **editable** (`uv tool install --editable .`), so the
  running version is whatever branch is checked out in this directory. Stay on
  `personal` for normal use; when working a `fix/*` branch you are temporarily
  running that branch's code — `git checkout personal` restores my setup.

Before any destructive step (`reset --hard`, force-push, branch delete), inspect
state first and show it. Confirm before force-pushing or deleting anything that
is published.

## Flow A — Sync with upstream (pull in upstream changes)

```bash
git fetch upstream
git checkout main && git merge --ff-only upstream/main   # ff-only asserts main never diverged
git checkout personal && git rebase main                 # replay my stack onto new upstream
```

- If the `--ff-only` merge is refused, `main` has stray commits — stop and move
  them to `personal` before continuing; do not create a merge commit on `main`.
- On rebase conflicts: resolve each, `git add`, `git rebase --continue`. Use
  `git rebase --abort` to bail out safely. `rerere` will assist on repeats.
- Commits that already landed upstream (e.g. a fix that merged) are patch-equivalent
  and get dropped automatically during rebase — expect the stack to shrink; that is
  correct, not data loss.
- Optionally keep the fork's `main` clean too: `git push origin main --force-with-lease`.

## Flow B — Contribute a fix upstream

```bash
git checkout main && git checkout -b fix/<thing>   # branch off pristine main
# ...make ONLY the fix, commit it...
git push -u origin fix/<thing>
```

Then open a PR from `origin:fix/<thing>` → `upstream/main`. Keep the branch to just
the fix — no private tweaks. When it merges upstream it returns to me via Flow A.

## Flow C — Use a fix locally before it merges upstream

Author the fix on a `fix/<thing>` branch (Flow B) for the PR, then bring the same
commit onto `personal` so I can use it immediately:

```bash
git checkout personal
git cherry-pick <fix-commit-sha>
```

When the PR later merges, Flow A's rebase drops the cherry-picked copy automatically
(patch-id match). If reviewers changed the fix or it was squash-merged, the rebase
may pause with a conflict or an "empty commit" — resolve toward upstream's now-canonical
version, or `git rebase --skip` if my copy is redundant.

## Output

State which flow you're running and why, run the steps (pausing for confirmation
before destructive/published actions), and report the resulting branch state.
