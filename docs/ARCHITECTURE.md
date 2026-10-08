# Canonical Architecture: Broadcast Mind

> Starting hypothesis — subject to revision as implementation
> proceeds.

## High-level data flow

```
WORLD STATE
    |
    +-- External Sources (RSS, web search, web pages, APIs)
    +-- User Sources (documents, notes, repos, saved research)
    +-- Historical Knowledge (previous broadcasts, claims, evidence)
    +-- Saved World Snapshots
    |
    v
ANALYST NETWORK
    |
    +-- Fact extraction
    +-- Claim extraction
    +-- Contradiction detection
    +-- Evidence evaluation
    +-- Analysis
    +-- Speculation
    +-- Historical context
    |
    v
PERSONA NETWORK
    |
    +-- Identity
    +-- Psyche
    +-- Values
    +-- Curiosity
    +-- Interests
    +-- Humor
    +-- Social behavior
    +-- Blind spots
    +-- Persistent memory
    +-- Previous opinions
    |
    v
NEWSROOM / EDITORIAL LAYER
    |
    +-- Relevance
    +-- Importance
    +-- Novelty
    +-- Continuity
    +-- Personal relevance
    +-- Persona relevance
    +-- Story follow-up
    |
    v
BROADCAST QUEUE
    |
    +-- Breaking
    +-- Update
    +-- Analysis
    +-- Retrospective
    +-- Explainer
    +-- Opinion
    +-- Discovery
    +-- Deep Dive
    +-- Humor
    +-- Synthetic
    |
    v
BROADCAST
    |
    +-- Script
    +-- Transcript
    +-- Voice
    +-- Evidence
    +-- Analysis
    +-- Archive
```

## Layer responsibilities

### World State

The persistent model of reality. Contains entities, claims,
evidence, relationships, timelines, and confidence levels. This
layer is factual — it does not contain opinions.

**Immutable core rule:** World state is append-only. Corrections
create new evidence records; they do not mutate old ones.

### Analyst Network

Extracts facts, claims, contradictions, and evidence from source
material. Each analyst is a capability, not necessarily a separate
service. Analysts evaluate provenance and confidence.

### Persona Network

Each persona has identity, psyche, values, cognitive style,
blind spots, memory, and opinions. Opinions belong to personas,
not the world state. Personas may disagree. Personas may be
deliberately absurd. A simulation mode runs multiple radically
different personas on the same world state.

### Newsroom / Editorial Layer

Determines what matters. Applies relevance, importance, novelty,
continuity, personal relevance, and persona relevance scoring.
Makes editorial decisions about what enters the broadcast queue
and in what order.

### Broadcast Queue

Ordered segments with types. Breaking stories queue for the next
available slot; they do not interrupt currently speaking segments.
The queue always has something meaningful to discuss — when there
is no breaking news, it generates retrospectives, analyses,
explainers, discoveries, deep dives, or synthetic thought
experiments.

### Broadcast

The spoken output. Script → transcript → voice → evidence →
analysis → archive. Every segment is archived with full provenance.

## Architecture decisions (initial)

| Decision | Status | Rationale |
|---|---|---|
| Local-first inference | PROPOSED | Privacy, sovereignty, offline operation |
| SQLite for persistence | PROPOSED | Proven by news08, simple, local |
| Ollama for LLM | PROPOSED | Already integrated in news08 |
| Edge-TTS for voice | PROPOSED | Already integrated in news08 |
| Append-only world state | PROPOSED | Enables correction without losing history |
| Persona opinions separate from world facts | PROPOSED | Core epistemic requirement |
| Non-preemptive broadcast | PROPOSED | Preserves broadcast coherence |
| Pluggable source providers | PROPOSED | Avoids coupling to one source type |
| TF-IDF → semantic embeddings | PROPOSED | news08 uses TF-IDF; semantic is better |
| Monolith → modular | PROPOSED | Single-file won't scale |

## What the architecture improves over news08

- Adds world state persistence (news08 has none)
- Adds epistemic categories (news08 treats all LLM output as fact)
- Adds persona system (news08 has no persona concept)
- Adds story tracking across time (news08 treats each cycle
  independently)
- Adds correction/revision mechanism (news08 cannot retract)
- Adds personal knowledge integration (news08 uses only RSS)
- Adds relevance explanation (news08 has opaque importance scores)
- Adds broadcast archive with provenance (news08 saves flat Markdown)

## What this architecture does NOT yet specify

- Exact database technology (SQLite confirmed, but ORM TBD)
- Exact embedding model (nomic-embed-text is a candidate)
- Exact agent runtime (subprocess? threading? async tasks?)
- Exact queue implementation (in-memory? persistent queue?)
- Exact graph implementation (networkx? custom? graph DB?)
- Exact deployment strategy (systemd? Docker? bare?)
- Retention/compression policy for old broadcasts
- Model selection criteria

See `docs/OPEN_QUESTIONS.md` for the full list.