# UI Architecture: Broadcast Mind

## Overview

The UI should feel like a dark spaceship/control-room
interface, not a conventional SaaS dashboard.

## Core experience

The default broadcast experience is visually clean.
Evidence and reasoning are available through a
deliberate "SHOW ANALYSIS" interaction — they do
not clutter the default view.

## UI components

### Live broadcast

The primary view — currently playing segment
with transcript, persona indicator, and progress.

### Live transcript

Real-time transcript of the current segment.
Scrollable, searchable.

### Current story

The story being discussed — with context,
related entities, and evidence summary.

### Current persona

Which persona is speaking — with trait summary
and current opinion.

### Next queue

Upcoming segments in the broadcast queue —
topic, type, priority, estimated time.

### Evidence panel

Evidence supporting the current claim —
sources, confidence, provenance. Accessible
via "SHOW ANALYSIS" interaction.

### Analysis panel

Full analyst evaluation — fact extraction,
claim extraction, contradiction detection,
confidence scores. Accessible via "SHOW
ANALYSIS" interaction.

### Story graph

Visual graph of related stories, entities,
and claims. Interactive — click to explore.

### World graph

Visual graph of the world state — entities,
relationships, timelines.

### Timeline / archive

Browse past broadcasts, segments, and world
state changes.

### Persona management

Create, edit, configure personas. View
trait vectors, memories, opinions.

### Source configuration

Manage sources — RSS feeds, search queries,
user documents. Configure reliability and
priority.

### World snapshots

View, compare, and restore world snapshots.

### Simulation / God Mode

Run multi-persona simulation. Compare how
different personas interpret the same world.

## Interaction patterns

### SHOW ANALYSIS

Evidence and reasoning are hidden by default.
The user taps "SHOW ANALYSIS" to reveal:

- Source citations
- Confidence scores
- Contradiction flags
- Analyst notes
- Persona opinions

This keeps the default view clean while
making reasoning available when needed.

### Component boundaries

- Each component owns its data
- Components communicate through a shared
  state store
- WebSocket provides live updates
- No component directly calls another
  component's API

## Design principles

- Dark theme (spaceship/control-room aesthetic)
- Minimal default view, rich on demand
- Live updates via WebSocket
- Responsive layout
- Accessible (keyboard navigation, screen reader
  support)
- Local-first (no cloud-dependent UI)

## Comparison to news08

news08 has no UI — it's a CLI script. The new
system adds a full web-based control room with:

- Live broadcast display
- Real-time transcript
- Persona management
- Source configuration
- World snapshot visualization
- Analysis panels
- Simulation mode UI
- Story and world graphs

## Component boundaries (without premature pixel design)

- **Broadcast display** — Current segment, transcript,
  persona indicator
- **Queue display** — Next segments, priority, type
- **Evidence panel** — Sources, confidence, provenance
- **Analysis panel** — Full analyst evaluation
- **Story graph** — Interactive story/entity graph
- **World graph** — Entity relationship graph
- **Timeline** — Chronological broadcast history
- **Persona manager** — Create/edit personas
- **Source manager** — Configure sources
- **Snapshot viewer** — Compare world states
- **Simulation runner** — Multi-persona simulation