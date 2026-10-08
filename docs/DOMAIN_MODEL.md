# Domain Model: Broadcast Mind

Core entities for the Broadcast Mind system. Conceptual definitions
before any schema implementation.

## World

**Purpose:** The persistent model of reality — everything the
system knows to be true or uncertain about the world.

**Ownership:** System-wide, shared by all personas and analysts.

**Lifecycle:** Append-only. New evidence is added; old evidence is
never deleted or modified. Corrections create new records that
supersede old conclusions without erasing them.

**Relationships:** Contains Entities, Claims, Evidence, Timelines,
and HistoricalSummaries.

**Persistence:** SQLite (confirmed from news08). Append-only tables
with versioned records.

**Provenance:** Every world state entry must cite its source(s).

**Mutable:** No — append-only.

**Versioned:** Yes — every change creates a new version; old
versions remain accessible.

**User-visible:** Yes — via timeline, snapshot comparison, and
correction history.

## WorldSnapshot

**Purpose:** A point-in-time capture of the world state, used for
comparison, audit, and restoration.

**Ownership:** System.

**Lifecycle:** Created on demand, retained per policy.

**Relationships:** References Stories, Entities, Claims, Evidence,
Timelines.

**Persistence:** Serialized snapshot + metadata.

**Provenance:** Timestamp, source world state version, creator
(persona or system).

**Mutable:** No — snapshots are immutable once created.

**Versioned:** N/A (immutable).

**User-visible:** Yes — diff between snapshots.

## Story

**Purpose:** A narrative thread that tracks a developing event or
topic over time, composed of multiple Events.

**Ownership:** World State.

**Lifecycle:** Created when first evidence emerges; updated as new
evidence arrives; may be closed when resolved or abandoned.

**Relationships:** Contains Events, References Claims, Evidence,
Entities.

**Persistence:** SQLite.

**Provenance:** First article/source that established the story.

**Mutable:** Yes — stories evolve with new evidence.

**Versioned:** Yes — story versions track narrative evolution.

**User-visible:** Yes — story tracking is a core user experience.

## Event

**Purpose:** A single factual occurrence or publication that
contributes to a Story.

**Ownership:** World State.

**Lifecycle:** Created when source material is ingested; linked to
Story; may be superseded by newer evidence.

**Relationships:** Belongs to Story, has Evidence, may contradict
other Events.

**Persistence:** SQLite.

**Provenance:** Source article URL, fetch timestamp, extractor.

**Mutable:** No — events are immutable; corrections create new
events.

**Versioned:** No (immutable).

**User-visible:** Yes — event timeline.

## Claim

**Purpose:** A statement asserted in source material that may or
may not be factual.

**Ownership:** World State (as a tracked claim), not as fact.

**Lifecycle:** Created when extracted from source; may be confirmed,
contradicted, or remain uncertain.

**Relationships:** Supported by Evidence, may contradict other
Claims, attributed to Source/Entity.

**Persistence:** SQLite.

**Provenance:** Source URL, extractor (which analyst), confidence.

**Mutable:** Confidence/verdict may update; claim text immutable.

**Versioned:** Yes — verdict history tracked.

**User-visible:** Yes — claim verification status shown to user.

## Evidence

**Purpose:** A piece of source material or analysis that supports
or contradicts a Claim.

**Ownership:** World State.

**Lifecycle:** Created when source ingested; linked to Claim(s);
may be superseded by stronger evidence.

**Relationships:** Supports/Contradicts Claims, from Source.

**Persistence:** SQLite (reference to source + extracted content).

**Provenance:** Source URL, extraction method, timestamp.

**Mutable:** No — evidence records are immutable.

**Versioned:** No (immutable).

**User-visible:** Yes — evidence panel in UI.

## Source

**Purpose:** The origin of a piece of information (RSS feed, web
page, API, user document).

**Ownership:** System configuration.

**Lifecycle:** Registered by user/admin; may be enabled/disabled.

**Relationships:** Provides Evidence, Claims.

**Persistence:** Configuration store.

**Provenance:** Feed URL, API endpoint, file path.

**Mutable:** Yes — configuration changes.

**Versioned:** No.

**User-visible:** Yes — source management UI.

## Entity

**Purpose:** A real-world object (person, organization, location,
concept) tracked by the system.

**Ownership:** World State.

**Lifecycle:** Created on first mention; updated as more information
arrives.

**Relationships:** Mentioned in Events/Claims/Evidence, linked to
Timeline entries.

**Persistence:** SQLite.

**Provenance:** First source mentioning the entity.

**Mutable:** Yes — entity attributes update with new information.

**Versioned:** Yes — attribute change history.

**User-visible:** Yes — entity graph.

## Timeline

**Purpose:** Chronological ordering of Events for a Story or Entity.

**Ownership:** World State (derived from Events).

**Lifecycle:** Derived, not independently created.

**Relationships:** Ordered Events.

**Persistence:** Computed from Events table.

**Provenance:** Event timestamps.

**Mutable:** No — timeline is derived from immutable Events.

**Versioned:** No (derived).

**User-visible:** Yes — timeline view.

## HistoricalSummary

**Purpose:** A summary of what happened over a time period, used
for retrospectives and context.

**Ownership:** System-generated.

**Lifecycle:** Created on demand or periodically.

**Relationships:** References Events, Stories, WorldSnapshots.

**Persistence:** SQLite or generated on demand.

**Provenance:** Generation timestamp, source period.

**Mutable:** May be regenerated.

**Versioned:** Yes — each generation is a new version.

**User-visible:** Yes — broadcast segments, archive.

## Persona

**Purpose:** A synthetic mind with identity, psyche, values,
cognitive style, and opinions.

**Ownership:** User-configured.

**Lifecycle:** Created by user; may be edited, duplicated, or
archived.

**Relationships:** Holds PersonaMemories, PersonaOpinions,
PersonaExperiences; participates in Broadcasts.

**Persistence:** SQLite (configuration + memory).

**Provenance:** User creation or system default.

**Mutable:** Yes — traits evolve, memories accumulate.

**Versioned:** Yes — trait evolution history.

**User-visible:** Yes — persona management UI.

## PersonaMemory

**Purpose:** A remembered experience or observation that shaped a
persona's perspective.

**Ownership:** Persona.

**Lifecycle:** Created when persona processes an event or
broadcast; persists across sessions.

**Relationships:** Belongs to Persona, references World State
(events, claims).

**Persistence:** SQLite.

**Provenance:** Source event/broadcast, timestamp.

**Mutable:** No — memories are immutable records.

**Versioned:** No (immutable).

**User-visible:** Yes — persona memory browser.

## PersonaExperience

**Purpose:** A record of a persona's direct interaction with a
story, claim, or event — including emotional response and
interpretation.

**Ownership:** Persona.

**Lifecycle:** Created during broadcast generation; retained for
future reference.

**Relationships:** Belongs to Persona, references World State.

**Persistence:** SQLite.

**Provenance:** Generation timestamp, persona ID, world state
version.

**Mutable:** No — experiences are immutable.

**Versioned:** No (immutable).

**User-visible:** Yes — persona insight panel.

## PersonaOpinion

**Purpose:** A persona's interpreted judgment on a Claim or Story,
explicitly marked as opinion, not fact.

**Ownership:** Persona.

**Lifecycle:** Created when persona analyzes a claim; may be
revised by new evidence.

**Relationships:** About Claim/Story, held by Persona.

**Persistence:** SQLite.

**Provenance:** Persona ID, source claim/story, analysis timestamp.

**Mutable:** Yes — opinions may be revised.

**Versioned:** Yes — opinion revision history.

**User-visible:** Yes — opinion panel, persona disagreement view.

## Analyst

**Purpose:** A capability that extracts, evaluates, or interprets
information from source material.

**Ownership:** System.

**Lifecycle:** Configured by architecture, not user-facing.

**Relationships:** Produces Analysis, evaluates Evidence.

**Persistence:** Configuration.

**Provenance:** N/A (system component).

**Mutable:** No — analyst capabilities are architectural.

**Versioned:** No.

**User-visible:** No — internal system component.

## Analysis

**Purpose:** The output of an analyst's evaluation — fact
extraction, claim extraction, contradiction detection, evidence
evaluation, or interpretation.

**Ownership:** World State (for factual analysis) or Persona (for
opinion analysis).

**Lifecycle:** Created during ingestion/analysis phase; retained
for audit.

**Relationships:** References Evidence, Claims, Sources.

**Persistence:** SQLite.

**Provenance:** Analyst type, source material, timestamp.

**Mutable:** No — analysis is immutable once recorded.

**Versioned:** No (immutable).

**User-visible:** Yes — show analysis on demand.

## EditorialDecision

**Purpose:** The newsroom's decision about what to broadcast, in
what order, and with what framing.

**Ownership:** Editorial Layer (system).

**Lifecycle:** Created during editorial evaluation; executed when
segment reaches front of queue.

**Relationships:** References Story, Claim, Evidence, Persona,
BroadcastQueue.

**Persistence:** SQLite.

**Provenance:** Scoring inputs, persona relevance weights, editorial
rules.

**Mutable:** No — decisions are immutable once executed.

**Versioned:** Yes — decision history.

**User-visible:** Yes — "why this story?" explanation.

## BroadcastQueue

**Purpose:** Ordered list of segments waiting to be broadcast.

**Ownership:** System.

**Lifecycle:** Segments added by editorial layer; consumed by
broadcast layer; cleared after playback.

**Relationships:** Contains BroadcastSegments.

**Persistence:** In-memory + SQLite fallback.

**Provenance:** Editorial decisions.

**Mutable:** Yes — queue order changes as new segments are added or
priorities shift.

**Versioned:** No.

**User-visible:** Yes — next-up display.

## BroadcastSegment

**Purpose:** A single unit of broadcast content — script, voice,
evidence, analysis.

**Ownership:** System.

**Lifecycle:** Created by editorial layer; spoken by broadcast
layer; archived after playback.

**Relationships:** Contains Script, Transcript, Voice, Evidence,
Analysis; references EditorialDecision.

**Persistence:** SQLite + audio files.

**Provenance:** Editorial decision ID, persona ID, world state
version.

**Mutable:** No — segments are immutable once created.

**Versioned:** Yes — segment archive versioning.

**User-visible:** Yes — broadcast history, replay.

## Transcript

**Purpose:** The text that was spoken in a broadcast segment.

**Ownership:** System.

**Lifecycle:** Created during TTS generation; retained for archive.

**Relationships:** Belongs to BroadcastSegment.

**Persistence:** SQLite + text files.

**Provenance:** Script generation timestamp.

**Mutable:** No — transcripts are immutable.

**Versioned:** No (immutable).

**User-visible:** Yes — transcript display.

## AudioArtifact

**Purpose:** The generated audio file for a broadcast segment.

**Ownership:** System.

**Lifecycle:** Created by TTS; deleted per retention policy.

**Relationships:** Belongs to BroadcastSegment.

**Persistence:** File system + SQLite reference.

**Provenance:** TTS provider, voice settings, timestamp.

**Mutable:** No — audio artifacts are immutable.

**Versioned:** No (immutable, but may be regenerated).

**User-visible:** Yes — audio replay.

## BroadcastHistory

**Purpose:** Record of all broadcasts, for recall, comparison, and
learning.

**Ownership:** System.

**Lifecycle:** Created after each broadcast cycle; retained per
policy.

**Relationships:** References BroadcastSegments, WorldSnapshots,
PersonaStates.

**Persistence:** SQLite.

**Provenance:** Broadcast timestamp, persona IDs, world state
version.

**Mutable:** No — broadcast history is immutable.

**Versioned:** N/A (immutable).

**User-visible:** Yes — broadcast archive.

## Correction

**Purpose:** A record that a previous claim, story, or conclusion
was wrong or incomplete, with the corrected information.

**Ownership:** System (triggered by analyst or user).

**Lifecycle:** Created when contradiction or new evidence
invalidates prior conclusion; applied to world state.

**Relationships:** References original Claim/Story, provides
correcting Evidence.

**Persistence:** SQLite.

**Provenance:** Detection method (analyst, user, contradiction),
timestamp, correcting source.

**Mutable:** No — corrections are immutable.

**Versioned:** N/A (immutable).

**User-visible:** Yes — correction notifications, archive.

## Simulation

**Purpose:** A parallel execution context where multiple personas
interpret the same world state independently, used for comparison
and insight.

**Ownership:** System.

**Lifecycle:** Created on demand; results retained for comparison.

**Relationships:** Contains PersonaSessions, references WorldState.

**Persistence:** SQLite for results.

**Provenance:** World state version, persona IDs, simulation
parameters.

**Mutable:** No — simulation results are immutable.

**Versioned:** Yes — simulation runs are versioned.

**User-visible:** Yes — simulation comparison UI.

## Scenario

**Purpose:** A hypothetical world state used for "what if"
analysis or stress-testing the system's responses.

**Ownership:** User/system.

**Lifecycle:** Created on demand; may be saved or discarded.

**Relationships:** Modifies WorldState temporarily for simulation.

**Persistence:** Optional SQLite.

**Provenance:** User input or system-generated.

**Mutable:** No — scenario definitions are immutable; results may
vary.

**Versioned:** Yes.

**User-visible:** Yes — scenario manager.

## PersonaSession

**Purpose:** A single execution context for a persona during a
broadcast cycle or simulation.

**Ownership:** System.

**Lifecycle:** Created at broadcast/simulation start; destroyed
after completion.

**Relationships:** Belongs to Persona, references WorldState
snapshot.

**Persistence:** SQLite (for continuity across cycles).

**Provenance:** Broadcast/simulation start timestamp.

**Mutable:** Yes — session state evolves during execution.

**Versioned:** No.

**User-visible:** No — internal system component.