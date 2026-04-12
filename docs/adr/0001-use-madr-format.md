# Use MADR Format for Architecture Decision Records

- **Status**: Accepted
- **Date**: 2026-04-11
- **Deciders**: Project team

## Context and Problem Statement

We need a standard format for documenting significant architectural decisions in this project.
Without a consistent format, decisions get lost in chat, PRs, or people's heads, and new
contributors have no way to understand why the system is designed the way it is.

## Decision Drivers

- Must be lightweight enough that people actually use it
- Must enforce documentation of alternatives considered (critical for an ML project where
  model selection decisions need rigorous justification)
- Must render cleanly on GitHub without special tooling
- Should support status tracking so stale decisions are visible

## Considered Options

- **Option A**: MADR 3.x (Markdown Architectural Decision Records)
- **Option B**: Nygard-style ADRs
- **Option C**: No standard format — prose docs in `docs/architecture/`

## Decision Outcome

**Chosen option: Option A (MADR 3.x)**, because it enforces a "Considered Options" section
and "Pros and Cons" breakdown, which is critical in an ML project where model selection
decisions (SVD vs NCF vs LightGCN) need to document why alternatives were rejected. It also
has a clean GitHub rendering and lightweight tooling.

### Positive Consequences

- Every ADR captures alternatives, not just the chosen option
- New contributors can understand not just *what* was decided but *why other options were ruled out*
- MADR's explicit Status field makes superseded decisions easy to identify

### Negative Consequences / Risks

- MADR is slightly more verbose than Nygard style — mitigated by providing a template that
  makes filling it in fast

## Pros and Cons of the Options

### Option A: MADR 3.x

- Pro: Mandatory "Considered Options" + "Pros and Cons" sections prevent lazy one-option ADRs
- Pro: Clean GitHub rendering, no special tooling required
- Pro: Active maintenance and community adoption
- Con: More sections to fill in than Nygard style

### Option B: Nygard-style ADRs

- Pro: Extremely minimal — just Context, Decision, Status, Consequences
- Con: No enforced alternatives section; easy to write an ADR that just says "we chose X"
- Con: Consequences section conflates positive and negative outcomes

### Option C: No standard format

- Pro: Zero overhead
- Con: Decisions become undiscoverable; new contributors cannot learn from past choices
- Con: No consistency across docs makes the decision history hard to audit

## Links

- [MADR specification](https://adr.github.io/madr/)
- [Michael Nygard's original ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
