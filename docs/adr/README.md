# Architecture Decision Records (ADRs)

ADRs document significant architectural decisions — what was decided, why, and what alternatives were considered.

## Format

We use [MADR 3.x](https://adr.github.io/madr/) (Markdown Architectural Decision Records). See [template.md](template.md) for the full template.

## How to write a new ADR

1. Copy `template.md` to a new file: `NNNN-short-title.md` (e.g., `0004-choose-graph-library.md`)
2. Fill in all sections. The "Considered Options" section is mandatory — at least two alternatives.
3. Set **Status** to `Proposed` and open a PR for review.
4. Once agreed, update **Status** to `Accepted` and merge.
5. If a decision is later reversed, update **Status** to `Superseded by ADR-XXXX`.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-use-madr-format.md) | Use MADR format for ADRs | Accepted |

<!-- Add new rows here as ADRs are created -->
