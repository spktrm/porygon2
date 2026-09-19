# Porygon2 — agent instructions

`CLAUDE.md` is the one source of project rules: commands, the repository map,
the invariants, the design principles and the conventions. Read it first and
follow it as written. This file deliberately restates none of it — a second
copy drifted within two weeks (2026-09-07 → 2026-09-19).

`LESSONS.md` is the archive: what was tried, what was measured, and the revert
handle for every mechanism deleted. Search it before proposing a mechanism,
sizing a coefficient, deleting or restoring anything, or writing a plan's
"declined" section.

What `CLAUDE.md` assumes that an agent other than Claude Code does not get:

- Its hooks and tools are not available to you. The rule they enforce still
  binds: pytest runs on the GPU, never under `JAX_PLATFORMS=cpu`.
- Check `git status --short` before editing and preserve unrelated
  working-tree changes, including those inside the `data/ps` submodule.
- `docs/` is gitignored and local to one machine: do not commit it, cite it as
  public, or assume a fresh clone has it.
- Executable definitions, configuration and contract tests establish current
  behaviour. Some READMEs and older comments describe retired architectures.
