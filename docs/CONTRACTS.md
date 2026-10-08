# Implementation Contracts: Broadcast Mind

## SourceProvider

```
interface SourceProvider:
    fetch() → List[RawArticle]
    normalize(raw: RawArticle) → NormalizedArticle
    get_source_identity() → SourceID
    get_reliability() → float
    is_available() → bool
```

**Responsibility:** Fetch and normalize source material.

**Inputs:** None (configuration-driven).

**Outputs:** List of normalized articles.

**Errors:** Logs and returns empty list on failure;
circuit breaker prevents hammering.

## WorldStore

```
interface WorldStore:
    add_evidence(evidence: Evidence) → EvidenceID
    add_claim(claim: Claim) → ClaimID
    get_claim(claim_id: ClaimID) → Claim
    get_evidence_for_claim(claim_id: ClaimID) → List[Evidence]
    get_story(story_id: StoryID) → Story
    list_stories(filter: StoryFilter) → List[Story]
    add_correction(correction: Correction) → CorrectionID
    create_snapshot() → SnapshotID
    compare_snapshots(a: SnapshotID, b: SnapshotID) → Diff
```

**Responsibility:** Persistent world state with append-only
semantics.

**Inputs:** Evidence, claims, corrections, snapshot requests.

**Outputs:** Stored records with IDs and timestamps.

**Errors:** Transaction failures rolled back; corruption
detected at startup.

## StoryTracker

```
interface StoryTracker:
    create_story(events: List[Event]) → StoryID
    update_story(story_id: StoryID, events: List[Event]) → Story
    match_to_existing(article: Article) → Optional[StoryID]
    close_story(story_id: StoryID, reason: str) → Story
    list_active() → List[Story]
```

**Responsibility:** Track stories across time.

**Inputs:** New events, article-cluster matches.

**Outputs:** Story IDs and updated story state.

**Errors:** Unmatched articles become new stories; orphan
events logged.

## EvidenceStore

```
interface EvidenceStore:
    add_evidence(source: Source, content: str, extractor: str) → EvidenceID
    get_evidence(evidence_id: EvidenceID) → Evidence
    list_evidence_for_claim(claim_id: ClaimID) → List[Evidence]
    contradict_evidence(evidence_id: EvidenceID, new_evidence: Evidence) → Contradiction
```

**Responsibility:** Store and retrieve evidence records.

**Inputs:** Source material, extractor identity.

**Outputs:** Evidence records with provenance.

**Errors:** Duplicate evidence detected and skipped.

## PersonaStore

```
interface PersonaStore:
    create_persona(config: PersonaConfig) → PersonaID
    get_persona(persona_id: PersonaID) → Persona
    update_persona(persona_id: PersonaID, updates: PersonaUpdate) → Persona
    list_personas() → List[Persona]
    add_memory(persona_id: PersonaID, memory: PersonaMemory) → MemoryID
    add_opinion(persona_id: PersonaID, opinion: PersonaOpinion) → OpinionID
    get_opinions(persona_id: PersonaID) → List[PersonaOpinion]
```

**Responsibility:** Persona configuration, memory, and
opinion management.

**Inputs:** Persona configuration, memories, opinions.

**Outputs:** Persona records with IDs.

**Errors:** Trait vector validation on update; invalid
traits rejected.

## PersonaRuntime

```
interface PersonaRuntime:
    interpret(world_state: WorldState, persona_id: PersonaID) → PersonaInterpretation
    evolve(persona_id: PersonaID, evidence: Evidence) → TraitDeltas
    get_current_opinions(persona_id: PersonaID) → List[PersonaOpinion]
```

**Responsibility:** Persona interpretation and evolution.

**Inputs:** World state, evidence for evolution.

**Outputs:** Persona interpretations, trait deltas.

**Errors:** Evolution bounded by trait limits (0.0–1.0).

## Analyst

```
interface Analyst:
    extract_facts(article: Article) → List[Fact]
    extract_claims(article: Article) → List[Claim]
    detect_contradictions(claims: List[Claim]) → List[Contradiction]
    evaluate_evidence(evidence: List[Evidence]) → Evaluation
    analyze(article: Article) → Analysis
```

**Responsibility:** Extract, evaluate, and interpret source
material.

**Inputs:** Articles and evidence.

**Outputs:** Facts, claims, contradictions, evaluations.

**Errors:** Failed extraction logged; partial results returned.

## EditorialEngine

```
interface EditorialEngine:
    evaluate_story(story: Story, context: EditorialContext) → EditorialScore
    make_decision(stories: List[Story], queue: BroadcastQueue) → EditorialDecision
    explain_decision(decision: EditorialDecision) → Explanation
```

**Responsibility:** Determine what matters and why.

**Inputs:** Stories, user relevance context, current queue.

**Outputs:** Editorial decisions with explanations.

**Errors:** No stories meet threshold → generate idle content.

## BroadcastQueue

```
interface BroadcastQueue:
    enqueue(segment: BroadcastSegment) → QueuePosition
    dequeue() → Optional[BroadcastSegment]
    peek() → Optional[BroadcastSegment]
    list_queue() → List[BroadcastSegment]
    remove(segment_id: SegmentID) → bool
```

**Responsibility:** Priority-ordered segment queue.

**Inputs:** Segments to enqueue; dequeue requests.

**Outputs:** Next segment for broadcast.

**Errors:** Empty queue returns None; queue persists across
restarts.

## ScriptGenerator

```
interface ScriptGenerator:
    generate_script(segment: BroadcastSegment, persona: Persona) → Script
    generate_transition(previous: str, current: str) → str
    refine_script(script: str, guidance: str) → Script
```

**Responsibility:** Generate broadcast scripts with persona voice.

**Inputs:** Segment content, persona traits, optional guidance.

**Outputs:** Script text with tone indicators.

**Errors:** Fallback script on failure; logged.

## VoiceProvider

```
interface VoiceProvider:
    synthesize(text: str, voice_settings: VoiceSettings) → AudioData
    list_voices() → List[Voice]
    is_available() → bool
```

**Responsibility:** Convert text to speech.

**Inputs:** Script text, voice settings.

**Outputs:** Audio data (format TBD).

**Errors:** Falls back to next provider; logs failure.

## Archive

```
interface Archive:
    store_artifact(artifact: BroadcastArtifact) → ArtifactID
    get_artifact(artifact_id: ArtifactID) → BroadcastArtifact
    search_artifacts(query: ArtifactQuery) → List[BroadcastArtifact]
    list_artifacts(filter: ArtifactFilter) → List[BroadcastArtifact]
```

**Responsibility:** Store and retrieve broadcast artifacts.

**Inputs:** Artifacts (script, transcript, audio, metadata).

**Outputs:** Artifact IDs and search results.

**Errors:** Storage failures logged; partial writes rolled back.

## SnapshotStore

```
interface SnapshotStore:
    create_snapshot(world_state: WorldState) → SnapshotID
    get_snapshot(snapshot_id: SnapshotID) → WorldSnapshot
    compare_snapshots(a: SnapshotID, b: SnapshotID) → Diff
    restore_snapshot(snapshot_id: SnapshotID) → WorldState
    list_snapshots() → List[WorldSnapshot]
```

**Responsibility:** World snapshot lifecycle.

**Inputs:** World state at point in time.

**Outputs:** Snapshots with IDs and diffs.

**Errors:** Snapshot creation failure logged; system continues.