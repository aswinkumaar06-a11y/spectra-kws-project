---
description: Auto-commit and push all code changes to GitHub immediately after every modification.
---

# Auto-Commit and Push Rule

After **every code change** (file creation, modification, or deletion) in the `spectra-kws-project` workspace:

1. **Stage** all changes: `git add -A`
2. **Commit** with a descriptive message summarizing the change
3. **Push** to `origin/main` immediately: `git push origin main`

This ensures GitHub stays in sync at all times. Do NOT batch changes or wait for the user to ask — push after every modification.
