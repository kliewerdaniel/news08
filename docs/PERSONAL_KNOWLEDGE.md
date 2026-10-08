# Personal Knowledge Integration: Broadcast Mind

## Overview

The system should eventually be able to ingest the
user's own knowledge: documents, notes, research,
GitHub repositories, project documentation, and saved
material. This personal knowledge is distinct from
external-world facts but queryable by the editorial
system.

## Knowledge categories

### Personal documents

- User-written files (markdown, text, PDF, DOCX)
- Research notes
- Project documentation
- Saved articles

### Repositories

- GitHub repos (code + README + docs)
- Local projects
- Configuration files
- Code comments and docstrings

### Notes

- Structured notes (obsidian, plain text)
- Meeting notes
- Decision logs
- Research summaries

### Generated artifacts

- Previous broadcast transcripts
- Previous analysis outputs
- Persona opinions (from previous sessions)

## Separation from external facts

**Critical rule:** Personal knowledge must remain
separate from external-world facts while being
queryable by the editorial system.

- Personal knowledge is tagged with `source_type:
  personal`
- Personal knowledge does not contaminate world state
  facts
- Personal knowledge can inform editorial decisions
  (e.g., "this story relates to the user's project")
- Personal knowledge can be used as evidence for
  user-specific claims
- The system must distinguish "the user believes X"
  from "X is a fact"

## Ingestion pipeline

1. **Discovery** — Find new/changed personal files
   and repos
2. **Extraction** — Extract text, structure, and
   entities from documents
3. **Normalization** — Convert to standard article
   format
4. **Linking** — Link to world state entities and
   claims where appropriate
5. **Tagging** — Tag with source_type: personal
   and provenance

## Querying by editorial system

The editorial system can query personal knowledge for:

- Relevance to user's current projects
- Personal context for broadcast segments
- User's own research on a topic
- Previously generated insights

## Privacy

- All personal knowledge stays local
- No personal data leaves the user's machine
- Personal knowledge is not shared with personas
  unless explicitly configured
- User can delete personal knowledge at any time

## Provenance

Personal knowledge records carry:

- File path or repo URL
- Extraction timestamp
- Extraction method
- User identity (if multi-user)
- Source type (document, repo, note, artifact)

## Comparison to news08

news08 has no personal knowledge integration. It only
processes RSS feeds. The new system adds:

- Multiple personal source types
- Separation from external facts
- Editorial queryability
- Privacy preservation
- User-controlled deletion