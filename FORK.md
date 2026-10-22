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
5. Do not rewrite the history of `expertise-main` (no force push, no rebase of published commits).
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

Downstream deploys build a pinned Python venv from a commit of this branch. Our install reports version `1.81.13`, which is the upstream release our line is based on. The venv is a copy: pushing to `expertise-main` does not update an existing venv. Rebuild the venv from the new commit and restart the proxy deliberately.

Known open item: the provisioning script pins the version against PyPI. Rebuilding from PyPI would silently drop our patch. A rebuild must come from a commit of this branch, not from PyPI.

## Taking in upstream changes

We follow upstream on purpose, not automatically.

```sh
git fetch upstream
git switch -c sync/upstream-<date> expertise-main
git merge upstream/main
# resolve conflicts, run the targeted tests
git switch expertise-main
git merge --no-ff sync/upstream-<date>
git push origin expertise-main
```

Merge upstream into our branch. Do not rebase our branch onto upstream.

litellm moves very fast. Our line was only one commit behind `upstream/main` when this branch was created (2026-10-22), but it will fall behind again quickly. The file we patch is the Anthropic experimental pass-through adapter (`litellm/llms/anthropic/experimental_pass_through/adapters/`). Expect conflicts there first.

## Where we started

`expertise-main` was created on 2026-10-22 from the branch `fix/anthropic-adapter-cache-read-input-tokens` (commit `c69822ac61`).

- Base: the upstream release line through the 2026-02-16 release (version 1.81.13), pulled in through two release-branch merges.
- Our patch: a single commit (`c69822ac61`). It derives `cache_read_input_tokens` from `prompt_tokens_details.cached_tokens` in the Anthropic experimental pass-through adapter, in all three translation sites (non-streaming, streaming delta, streaming iterator), so a local llama-server reports cache reads through the router. It adds a regression test for that usage shape.

The older branch name `fix/anthropic-adapter-cache-read-input-tokens` is kept so old references still work. Do not add new work to it.
