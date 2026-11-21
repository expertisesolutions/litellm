# Expertise fork of litellm: the `expertise-main` branch

This file is about our fork only. It is not part of upstream litellm.

## What `expertise-main` is

`expertise-main` is the official branch of our fork. It plays the same role for us that `main` plays for upstream (`BerriAI/litellm`).

- It holds upstream litellm plus every change we made on top of it.
- It is the only branch our litellm installs build from.
- It is the only branch that other work is merged into.

The branch is named after its role, not after what it contains. Its content changes with time. Its role does not.

## Remotes

| Remote | URL | Use |
|---|---|---|
| `origin` | `git@github.com:expertisesolutions/litellm.git` | Our fork. We push here. |
| `upstream` | `https://github.com/BerriAI/litellm.git` | Upstream. We only fetch from it. |

Note: in this repo the fork remote is named `origin` and the upstream remote `upstream`, the reverse of the llama.cpp fork. Check with `git remote -v` before you push. We have no push rights on `BerriAI/litellm`.

`main` in our fork is a plain mirror of upstream `main` at the fork point. It is frozen. Never commit to it. The fork also carries hundreds of branches inherited from upstream (the `assets*` branches and similar). They are all frozen upstream artifacts. None of them are ours, and none of them are used.

This fork is public. Do not commit host names, IP addresses, credentials, or any private data. Keep deploy details for our own hosts in the private repositories that hold them.

## Rules

1. Every change to litellm that we run ends up in `expertise-main`. A change that is not in `expertise-main` is not part of our litellm.
2. Start new work from `expertise-main`. Use a short-lived branch (for example `fix/provider-shape`).
3. When the work is done and tested, merge the branch into `expertise-main` and push it to `origin`.
4. Do not build a production install from a feature branch. Build from a commit that is already in `expertise-main`.
5. Do not rewrite the history of `expertise-main` for normal work (no force push, no rebase of published commits). The one allowed exception is a deliberate move to a new upstream release tag, described in "Taking in upstream changes". It needs a backup tag and `--force-with-lease`.
6. Run the targeted tests for the code you touched before you merge. The full litellm suite is very large. The test tree is organized per provider. Run the area you touched, not the world.
7. Keep our patches small and separate from upstream code where you can. A smaller patch conflicts less when we take a new upstream version.

## Normal workflow

```sh
git fetch origin
git switch -c fix/my-change origin/expertise-main
# work, commit, test
git switch expertise-main
git pull --ff-only origin expertise-main
git merge --no-ff fix/my-change
git push origin expertise-main
```

## Building the production install

Downstream deploys build a pinned Python venv from a commit of this branch. The venv is a copy: pushing to `expertise-main` does not update an existing venv. Rebuild the venv from the new commit and restart the proxy deliberately.

Since 2026-11-19 the branch is based on upstream release `v1.109.1`, so the installed package reports version `1.109.1`.

The package has a Rust extension, built by `maturin`, so an install compiles it. Install it from a git URL pinned to a full commit of this branch. `pip` then records the repository URL and commit in the package metadata (`direct_url.json`).

Our provisioning scripts (in the private ops repo) install from this fork, never from PyPI. They check two things after every run: the recorded install URL points at this fork, and the file `litellm/llms/chatgpt/responses/transformation.py` in the installed package contains `_system_item_as_developer`. A PyPI install has neither the fork URL nor our line. A rebuild can therefore not silently fall back to the vanilla release.

## Taking in upstream changes

We follow upstream on purpose, not automatically. Move to a new upstream release tag in this order:

1. Fetch the upstream tags: `git fetch upstream --tags`.
2. Create a work branch at the new upstream release tag: `git worktree add -b rebase/<tag> <path> <tag>`.
3. Replay our commits on top of it with `git cherry-pick -x`, oldest first. This gives the same history as `git rebase --onto`, and it lets you decide per commit whether upstream already has the change.
4. For each commit that carries code, write or port its regression test first. If the test already passes on the plain upstream tag, keep the test and drop the code.
5. Run the targeted tests, and compare them with a plain upstream baseline so you only look at failures that are new.
6. Push the work branch.
7. Push a backup tag that points at the old tip of `expertise-main`.
8. Replace `expertise-main` with `git push --force-with-lease=expertise-main:<old-tip> origin <work-branch>:expertise-main`. The lease must name the exact old tip.

Use this path only for a move to a new release tag. For all other work, use the normal workflow above.

litellm moves very fast. Expect conflicts in `AGENTS.md` first, because upstream edits it often.

## Where we started

`expertise-main` was created on 2026-10-22 from the branch `fix/anthropic-adapter-cache-read-input-tokens` (commit `c69822ac61`). It was based on the upstream release line through version 1.81.13.

On 2026-11-19, by owner decision, we replayed our commits on top of upstream release tag `v1.109.1`. That release contains the fix that sends `system` input items to the ChatGPT backend as `developer` (upstream pull request 48668). The old branch tip was `16f2ffb1ee`. It is kept as the tag `expertise-main-pre-rebase-2026-11-19`.

What we carry on top of `v1.109.1`:

- Docs: this file, and a one-line header in `AGENTS.md`.
- Test: a regression test that a usage object with only `prompt_tokens_details.cached_tokens` still gives `cache_read_input_tokens` in the Anthropic response, in both non-streaming and streaming form. Upstream now has this behavior, so we dropped the old code patch (`c69822ac61`) and kept the test.
- Test: a regression test that system messages shaped like a Claude Code request, sent through a router, never reach the ChatGPT backend with role `system`. They arrive as `developer` items or as `instructions`. It runs through both `litellm.responses` and `litellm.completion`.

The older branch name `fix/anthropic-adapter-cache-read-input-tokens` is kept so old references still work. Do not add new work to it.
