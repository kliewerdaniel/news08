# World Snapshots: Broadcast Mind

## Overview

The system should be able to save the current
understanding of the world at a point in time.
Snapshots enable comparison, audit, and restoration.

## Snapshot contents

A snapshot conceptually includes:

- Stories (active and recent)
- Entities (tracked real-world objects)
- Claims (all tracked claims with verdicts)
- Evidence (supporting and contradicting)
- Timelines (chronological ordering)
- Confidence levels (per claim/entity)
- Unresolved questions
- Recent broadcasts
- Relevant persona state (opinions at snapshot time)
- Important changes since last snapshot

## Snapshot creation

Snapshots are created:

- On demand (user request)
- Periodically (e.g., daily, configurable)
- After major world state changes (configurable
  threshold)
- Before and after editorial decisions (audit trail)

## Snapshot comparison

The system should compare two snapshots:

```
WORLD SNAPSHOT A
        vs
WORLD SNAPSHOT B
        ↓
WHAT CHANGED?
```

Comparison produces:

- New entities
- Changed entities (attribute deltas)
- New claims
- Changed claim verdicts
- New evidence
- New stories
- Resolved stories
- Confidence changes
- Persona opinion shifts

## Serialization

Snapshots are serialized as:

- JSON for machine-readable archive
- Markdown for human-readable export
- Both include full provenance

## Versioning

Each snapshot has:

- Unique ID
- Timestamp
- World state version reference
- Creator (system or user)
- Description (why snapshot was taken)

## Provenance

Snapshots carry full provenance:

- Which world state version they reference
- Which sources were available
- Which personas were active
- Which editorial decisions were in effect

## Restoration

Snapshots can be restored to a previous world state:

1. Snapshot identifies the world state version
2. System reverts to that version
3. All changes after the snapshot are preserved as
   corrections
4. The system continues from the restored state

Restoration is logged and auditable.

## Storage

Snapshots are stored in SQLite with:

- Snapshot metadata table
- Snapshot content (JSON blob or structured tables)
- Diff records between consecutive snapshots

Retention policy: configurable. Default retains
all snapshots for 90 days, then compresses to
weekly summaries.

## Comparison to news08

news08 has no snapshot capability. It processes
articles and broadcasts them with no memory of
previous state. The snapshot system adds:

- Point-in-time capture of world understanding
- Change detection between snapshots
- Audit trail for world state evolution
- Restoration capability
- Basis for "what changed?" broadcast segments