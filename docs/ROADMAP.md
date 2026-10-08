# Implementation Phases: Broadcast Mind

## Phase 0 — Repository archaeology and architecture freeze

**Status:** IN PROGRESS (this documentation)

- Complete repository archaeology
- Freeze architecture documents
- Get user approval before implementation

## Phase 1 — World model and persistence

- Define SQLite schema for world state
- Implement entity, claim, evidence tables
- Implement append-only correction mechanism
- Basic CRUD for world state operations
- World snapshot creation and comparison

## Phase 2 — Source ingestion and story tracking

- Pluggable source provider interface
- RSS provider (retained from news08)
- Web search provider
- User document provider
- Story tracking (cluster → story matching)
- Deduplication across sources

## Phase 3 — Evidence and epistemic model

- Epistemic category enforcement
- Provenance tracking
- Confidence scoring
- Contradiction detection
- Correction workflow

## Phase 4 — Persona system

- Persona configuration (YAML/JSON)
- Trait vector (52-dimension, from news08)
- Persona memory store
- Opinion tracking
- Persona evolution (evidence-driven)

## Phase 5 — Editorial/newsroom system

- Relevance scoring engine
- Editorial decision logic
- Broadcast queue with segment types
- Non-preemptive queue semantics
- "Why this story?" explanation

## Phase 6 — Broadcast queue

- Priority-ordered queue
- Segment generation pipeline
- Multiple segment types
- Breaking story queuing (non-preemptive)
- Idle content generation (retrospective,
  analysis, explainer)

## Phase 7 — Script generation

- Persona-aware script generation
- Evidence-grounded scripts
- Uncertainty communication
- Script archival with provenance

## Phase 8 — TTS

- Provider-pluggable TTS interface
- Edge-TTS provider (retained)
- Voice selection per persona
- Hardware-aware provider selection
- Audio artifact archival

## Phase 9 — UI/control room

- Live broadcast display
- Real-time transcript
- Persona management UI
- Source configuration UI
- Queue display
- Analysis panel (SHOW ANALYSIS)
- World snapshot viewer

## Phase 10 — World snapshots and historical comparison

- Snapshot creation (on-demand, periodic)
- Snapshot comparison (diff)
- Restoration capability
- Change detection broadcasts

## Phase 11 — Personal knowledge integration

- Document ingestion
- Note ingestion
- Repository discovery
- Personal vs. external fact separation

## Phase 12 — Multi-persona simulation

- God Mode simulation
- Persona disagreement display
- Comparative analysis view

## Phase 13 — Portable deployment

- Configuration externalization
- Environment variable support
- Docker deployment
- Clone → configure → run

## Ordering rationale

The ordering follows dependency chains:

1. World model first (everything else depends on it)
2. Source ingestion (feeds the world model)
3. Epistemic model (governs how evidence enters world)
4. Personas (interpret world state)
5. Editorial (decides what to broadcast)
6. Queue (orders segments)
7. Script + TTS (produces output)
8. UI (user interaction)
9. Snapshots (audit/history)
10. Personal knowledge (user context)
11. Simulation (advanced persona use)
12. Portability (deployment)

This ordering may change based on implementation
discoveries during Phases 1–3.