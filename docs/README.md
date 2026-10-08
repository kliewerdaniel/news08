# Documentation Index: Broadcast Mind

## Map of the documentation system

```
docs/ARCHAEOLOGY.md        Repository archaeology — what exists, what doesn't
docs/PROJECT_VISION.md     The "why" — core concept and design principles
docs/ARCHITECTURE.md       The "what" — canonical high-level architecture
docs/DOMAIN_MODEL.md       Core entities and their relationships
docs/EPISTEMIC_MODEL.md    Fact/claim/evidence/analysis distinctions
docs/PERSONA_SYSTEM.md     Persona architecture and multi-persona support
docs/NEWSROOM.md           Multi-agent newsroom design
docs/BROADCAST_PIPELINE.md Complete broadcast lifecycle
docs/SOURCES.md            Source providers and configuration
docs/WORLD_SNAPSHOTS.md    Point-in-time world captures and comparison
docs/PERSONAL_KNOWLEDGE.md User knowledge integration
docs/RELEVANCE_ENGINE.md   Relevance scoring and personalization
docs/ARTIFACTS.md          Broadcast artifact archival
docs/VOICE.md              TTS architecture and provider interface
docs/UI_ARCHITECTURE.md    Control room UI design
docs/PORTABILITY.md        Clone → configure → run
docs/ROADMAP.md            Implementation phases
docs/DECISIONS.md          ADR-style decision log
docs/OPEN_QUESTIONS.md     Unresolved questions
docs/CONTRACTS.md          Subsystem interface definitions
AGENTS.md                  Agent operating manual
```

## Cross-references

### Vision → Architecture

- PROJECT_VISION.md defines the concept; ARCHITECTURE.md defines the structure
- Architecture is a hypothesis — see DECISIONS.md for rationale

### Architecture → Domain

- ARCHITECTURE.md layers map to DOMAIN_MODEL.md entities
- World State layer ↔ World, WorldSnapshot, Story, Event
- Analyst Network ↔ Analyst, Analysis, Evidence
- Persona Network ↔ Persona, PersonaMemory, PersonaOpinion
- Newsroom ↔ EditorialDecision, BroadcastQueue
- Broadcast ↔ BroadcastSegment, Transcript, AudioArtifact

### Domain → Epistemics

- DOMAIN_MODEL.md entities carry epistemic categories
- EPISTEMIC_MODEL.md defines the categories and flow rules
- Every claim in the domain model has a provenance and confidence

### Epistemics → Personas

- EPISTEMIC_MODEL.md separates world facts from persona opinions
- PERSONA_SYSTEM.md defines how personas hold opinions without contaminating facts
- Simulation mode requires explicit labeling as simulated

### Newsroom → Pipeline

- NEWSROOM.md defines agent roles
- BROADCAST_PIPELINE.md defines the lifecycle those agents participate in
- EditorialEngine contract in CONTRACTS.md implements newsroom decisions

### Sources → Pipeline

- SOURCES.md defines source providers
- Pipeline COLLECT phase uses SourceProvider interface
- Source reliability affects evidence confidence (EPISTEMIC_MODEL.md)

### Voice → UI

- VOICE.md defines TTS provider interface
- UI_ARCHITECTURE.md includes voice configuration controls
- Per-persona voice assignment defined in PERSONA_SYSTEM.md

### Roadmap → All

- ROADMAP.md phases follow dependency chain
- Each phase builds on previous phase's deliverables
- Phase 0 (this documentation) gates all other phases

### Decisions → Architecture

- DECISIONS.md explains why architecture choices were made
- Architecture changes require DECISIONS.md updates
- Open questions in OPEN_QUESTIONS.md may change decisions

### Contracts → Implementation

- CONTRACTS.md defines interfaces before implementation
- Each contract maps to a module in the codebase
- Contracts are conceptual — schemas TBD

### Agents → All

- AGENTS.md is the entry point for any coding agent
- Agents must read relevant docs before changing code
- AGENTS.md workflow applies to all implementation phases

## Document status

All documents in this index are PROPOSED unless marked otherwise.
None represent implemented functionality — they represent the
planning foundation for future implementation.

## Related documents by topic

### What exists
- ARCHAEOLOGY.md

### Why we're building this
- PROJECT_VISION.md

### What we're building
- ARCHITECTURE.md
- DOMAIN_MODEL.md
- EPISTEMIC_MODEL.md
- PERSONA_SYSTEM.md
- NEWSROOM.md

### How it works
- BROADCAST_PIPELINE.md
- SOURCES.md
- WORLD_SNAPSHOTS.md
- PERSONAL_KNOWLEDGE.md
- RELEVANCE_ENGINE.md
- ARTIFACTS.md
- VOICE.md
- UI_ARCHITECTURE.md

### How we get there
- PORTABILITY.md
- ROADMAP.md
- DECISIONS.md
- OPEN_QUESTIONS.md
- CONTRACTS.md

### How agents work here
- AGENTS.md