# Agent Operating Manual: Broadcast Mind

## Principles

1. **READ** — Read relevant documentation before changing architecture
2. **UNDERSTAND** — Inspect existing implementation before replacing it
3. **DO NOT INVENT APIs** — Use defined contracts, don't make up interfaces
4. **DO NOT SILENTLY CHANGE EPISTEMIC SEMANTICS** — Fact/claim/evidence distinctions are deliberate
5. **PRESERVE PROVENANCE** — Every record must have source attribution
6. **DO NOT MERGE PERSONA OPINION INTO FACTUAL WORLD STATE** — Opinions stay in persona store
7. **DO NOT REMOVE FUNCTIONALITY WITHOUT DOCUMENTING WHY** — Deprecate, don't delete
8. **UPDATE DOCUMENTATION WHEN ARCHITECTURE CHANGES** — Docs are the architectural memory
9. **UPDATE DECISION RECORDS WHEN MAJOR DECISIONS CHANGE** — DECISIONS.md is the source of truth
10. **TEST BEFORE DECLARING COMPLETION** — No unverified claims of "done"
11. **DISTINGUISH CONFIRMED BEHAVIOR FROM ASSUMPTIONS** — Use CONFIRMED/INFERRED/UNKNOWN labels
12. **PREFER SMALL, REVERSIBLE CHANGES** — Avoid big bang rewrites
13. **AVOID UNNECESSARY DEPENDENCIES** — Each dependency must justify its cost
14. **PRESERVE LOCAL-FIRST OPERATION** — Core functionality must work offline
15. **MAINTAIN PROVIDER BOUNDARIES** — TTS, LLM, and source providers are swappable
16. **DO NOT PREMATURELY OPTIMIZE** — Get it working before making it fast
17. **DO NOT BUILD SPECULATIVE ABSTRACTIONS WITHOUT A CONCRETE NEED** — YAGNI applies

## Required workflow

```
READ
  ↓
UNDERSTAND
  ↓
PLAN
  ↓
DOCUMENT
  ↓
IMPLEMENT
  ↓
TEST
  ↓
VERIFY
  ↓
UPDATE DOCUMENTATION
```

### READ

Before touching any code, read:

- `docs/ARCHITECTURE.md` — High-level architecture
- `docs/DOMAIN_MODEL.md` — Core entities
- `docs/EPISTEMIC_MODEL.md` — Epistemic categories
- `docs/CONTRACTS.md` — Interface definitions
- Any relevant existing code

### UNDERSTAND

- Inspect the existing implementation
- Identify what already works
- Identify what is broken or missing
- Verify assumptions against source code
- Label findings as CONFIRMED, INFERRED, or UNKNOWN

### PLAN

- Write a plan to `docs/plans/` if the change is non-trivial
- Identify dependencies and ordering
- Define success criteria
- Define test strategy

### DOCUMENT

- Update relevant docs before or with the code change
- Update DECISIONS.md if a major decision changes
- Update CONTRACTS.md if interfaces change
- Document why, not just what

### IMPLEMENT

- Follow existing code patterns
- Keep changes small and reversible
- Preserve provenance at every step
- Do not merge persona opinion into world state

### TEST

- Write tests before or alongside implementation
- Verify epistemic categories are preserved
- Verify provenance is maintained
- Verify persona isolation

### VERIFY

- Run the full test suite
- Verify the change works end-to-end
- Verify documentation is consistent with implementation
- Verify no unintended side effects

### UPDATE DOCUMENTATION

- Update docs to reflect the change
- Update decision records if architecture changed
- Update contracts if interfaces changed
- Ensure cross-references are valid

## Labels for findings

- **CONFIRMED** — Verified by reading source or running code
- **INFERRED** — Reasonably deduced from code behavior
- **UNKNOWN** — Cannot determine from available information
- **NEEDS VERIFICATION** — Requires runtime testing
- **PROPOSED** — Suggestion, not yet validated

## Anti-patterns

- Do not claim something is "implemented" unless you verified it in the repository
- Do not claim something is "guaranteed" unless there is an actual invariant or test enforcing it
- Do not hide uncertainty — mark it explicitly
- Do not invent APIs — use the defined contracts
- Do not silently change epistemic semantics — the fact/claim distinction is deliberate
- Do not remove functionality without documenting why — deprecate first
- Do not merge persona opinion into factual world state — this is a core constraint
- Do not build speculative abstractions — need must be concrete
- Do not prematurely optimize — working first, fast later
- Do not add unnecessary dependencies — each one must justify its cost

## Escalation

When stuck:

1. Check `docs/OPEN_QUESTIONS.md` — maybe it's a known question
2. Check `docs/DECISIONS.md` — maybe a decision was already made
3. Ask the user for clarification — do not guess
4. Block the task with a clear reason — do not proceed with wrong assumptions