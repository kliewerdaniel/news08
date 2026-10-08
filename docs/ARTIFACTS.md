# Broadcast Artifacts: Broadcast Mind

## Overview

Every spoken segment becomes an archival artifact
containing full provenance.

## Artifact contents

Each artifact includes, where applicable:

- **Segment metadata** — ID, timestamp, duration,
  segment type
- **Persona** — Which persona generated this segment
- **Script** — The generated broadcast script
- **Transcript** — The spoken text (if transcribed)
- **Sources** — All source references used
- **Claims** — Claims made in the segment
- **Evidence** — Evidence supporting each claim
- **Confidence** — Confidence per claim
- **Analysis** — Analyst evaluation results
- **Timestamp** — When the segment was generated
- **World snapshot** — World state reference at
  generation time
- **Audio reference** — Path to audio file
- **Editorial decision** — Why this segment was chosen

## Artifact storage

Artifacts are stored as:

- SQLite records for metadata
- Files for audio and transcripts
- JSON for structured data (claims, evidence,
  analysis)

## Correction support

Because artifacts carry full provenance, they
support later correction:

1. A claim in an old artifact is contradicted by
   new evidence
2. A Correction record is created
3. The old artifact is not deleted — it remains
   with a correction note
4. The broadcast system may replay the correction
   in a future segment
5. The user sees both the original and the
   correction

## Comparison

Artifacts enable:

- "What did the system say about X last week?"
- "How has the system's view of X changed?"
- "What evidence supports claim Y?"
- "Which persona said what about Z?"

## Archive retention

Retention policy is configurable:

- Audio: 30 days default, then compressed to
  summary
- Transcripts: retained indefinitely
- Metadata: retained indefinitely
- Evidence: retained indefinitely (world state
  append-only)

## Comparison to news08

news08 saves broadcast logs as flat Markdown files.
The artifact system adds:

- Structured metadata
- Full provenance per segment
- Claim and evidence tracking
- Correction support
- Queryable archive
- Audio reference with retention policy