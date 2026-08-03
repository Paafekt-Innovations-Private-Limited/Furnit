# ADR: organize Furnit documentation as code

- Date: 2026-08-03
- Status: Accepted

## Context

Furnit's current implementation was documented across a large root `CONTEXT.md`,
platform READMEs, code-local notes, release checklists, experiments, and dated audits.
The files contained valuable evidence, but there was no canonical reading order or
lifecycle index. That made stale submission and UI claims easy to mistake for current
behavior.

The sibling `paafekt-seventhd` repository uses a compact read-first file, central
architecture/index documents, ADRs, runbooks, and an explicit deferred area. Furnit
needs the same navigation discipline without breaking links or separating detailed
platform docs from their code owners.

## Decision

- Add `AGENTS.md`, `docs/READ_FIRST.md`, `docs/README.md`, `docs/overview.md`, and
  `docs/architecture.md` as the golden-path spine.
- Keep detailed iOS docs under `Furnit/` and detailed Android docs under `android/`;
  add platform indexes instead of moving them wholesale.
- Reduce `CONTEXT.md` to a compatibility pointer so one large narrative no longer
  competes with the active spine.
- Classify every existing documentation file as active, operational, research,
  historical, or deferred.
- Keep dated/historical files at their existing paths for link stability and add
  clear superseding banners where their claims could mislead.
- Record only source-confirmed stubs, unreachable branches, compatibility parameters,
  cleanup candidates, and tooling follow-ups in a deferred ledger. The ledger does
  not authorize deletion.
- Treat code, build configuration, and verified artifacts as the final source of truth.

## Consequences

- A new contributor has a short deterministic reading path.
- Platform-specific knowledge stays close to code and can evolve independently.
- Historical evidence remains accessible without being presented as live state.
- Adding or changing behavior now carries a documentation-index maintenance cost.
- The repository privacy-policy source still requires a separate website deployment;
  repository organization does not deploy external content.

## Alternatives considered

- Moving every document under one central directory was rejected because it would
  break established links and weaken code ownership.
- Deleting stale audits was rejected because they contain useful release, test, and
  cleanup provenance.
- A broad speculative unused-code list was rejected; only verified findings belong in
  the deferred ledger.
