# Multi-Agent Newsroom Architecture: Broadcast Mind

## Overview

The eventual system should support several synthetic minds
participating in the interpretation of the same world. This
document defines the possible roles, communication patterns,
and boundaries.

## Possible roles

These are starting points, not a hard-coded taxonomy. Roles
may emerge as the system evolves.

- **Researcher** — Ingests and normalizes source material
- **Fact Checker** — Verifies claims against evidence
- **Historian** — Provides temporal context and precedent
- **Skeptic** — Challenges claims and identifies weaknesses
- **Analyst** — Evaluates evidence and draws conclusions
- **Persona** — Interprets the world through a specific
  psychological lens
- **Editor** — Determines relevance, importance, and broadcast
  priority
- **Broadcaster** — Generates spoken output scripts

## Role vs. capability

Not every role needs to be a separate service. Some roles are
capabilities that any agent can exercise:

**Actual services/agents (separate processes):**
- Researcher (source ingestion is I/O bound and benefits from
  isolation)
- Fact Checker (verification may require different model/
  credentials)
- Editor (editorial decisions benefit from independence)

**Capabilities (functions within the system):**
- Skeptic (a skepticism mode, not necessarily a separate agent)
- Analyst (analysis is a capability applied by the system)
- Persona (personas are user-configured, not system agents)
- Broadcaster (script generation is a generation capability)

## Communication

Agents communicate through a shared message bus with typed
messages:

- **Evidence discovered** — Researcher → Fact Checker, Analyst
- **Claim verified/contradicted** — Fact Checker → Analyst,
  Editor
- **Analysis complete** — Analyst → Editor, Persona
- **Editorial decision** — Editor → Broadcaster, Queue
- **Persona opinion** — Persona → Editor (as input, not
  directive)
- **Correction issued** — Fact Checker → all agents

All messages carry provenance: source agent, timestamp,
confidence, evidence references.

## Shared state

The world state is the single source of truth. Agents read
from it and write to it through defined interfaces:

- Researchers write: Evidence, Claims
- Fact Checkers write: Claim verdicts, Corrections
- Analysts write: Analysis records
- Editors write: EditorialDecisions
- Personas write: PersonaOpinions (never world facts)

## Disagreement

Disagreement between agents is expected and valuable:

1. Fact Checker and Analyst may disagree on claim verdict
2. Personas may disagree with each other and with the world
   state
3. Skeptic may challenge Analyst conclusions
4. Editor reconciles disagreements using evidence and editorial
   rules

Disagreements are recorded, not suppressed. The broadcast may
note where experts disagree.

## Editorial decisions

The Editor determines:

- What enters the broadcast queue
- Priority ordering
- Segment type (breaking, update, analysis, etc.)
- Which persona(s) should interpret the story
- Whether a story needs more evidence before broadcasting

Editorial decisions are transparent — the system can explain
why a story was chosen or excluded.

## Conflict resolution

When agents disagree:

1. Evidence is re-evaluated
2. Confidence scores are compared
3. Source reliability is checked
4. If unresolved, the disagreement is noted and both
   positions may be broadcast with attribution
5. The user sees "Sources disagree" rather than a forced
   consensus

## Provenance

Every agent action carries:

- Agent identity
- Input evidence references
- Model used
- Confidence score
- Timestamp
- Correction chain (if applicable)

## Agent memory

Agents may retain memory across sessions for:

- Pattern recognition (recurring themes, unreliable sources)
- Relationship tracking (which sources tend to contradict
  each other)
- Editorial preference calibration

Agent memory is separate from world state and persona memory.

## Simplification principle

Avoid unnecessary multi-agent complexity. Use agents where
separation of responsibility provides actual value. A single
agent handling research + fact-checking + analysis may be
sufficient for a small system. Add agents when the boundaries
prove necessary, not before.

## Minimal viable newsroom

For initial implementation, a single NewsProcessor with
modular capabilities is sufficient:

- Ingestion capability (researcher)
- Analysis capability (analyst + fact checker combined)
- Editorial capability (editor)
- Generation capability (broadcaster)

Personas operate on top of this — they are not agents in the
system, they are interpreters of the system's output.