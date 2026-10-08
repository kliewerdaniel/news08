# Project Vision: Broadcast Mind

> A local-first artificial newsroom that maintains a persistent model of the
> world, allows multiple synthetic minds to interpret it, remembers what they
> previously believed, tracks how evidence changes those beliefs, and
> continuously turns that evolving world model into an audible broadcast.

## Core experience

The user wakes up. They turn on the system and their soundbar. The system
already knows what happened while they were asleep. It identifies what
actually matters. It follows stories over time rather than treating every
article as an isolated event. It distinguishes facts from claims, analysis,
speculation, satire, and synthetic material. It admits uncertainty. It can
remember what it previously believed. It can recognize when newer evidence
changes an earlier conclusion. It can occasionally surprise the user with
something fascinating even when there is no major breaking news. It can
incorporate the user's own research, notes, projects, documents, and
repositories. It should make the user feel connected to the world without
requiring them to stare at a screen. The system should remain useful even
when there is no breaking news.

## Design principles

1. **Local-first** — All inference, storage, and processing runs on the
   user's hardware. No cloud dependencies for core functionality.

2. **Epistemic honesty** — The system must never collapse "LLM generated
   text" into "truth." Facts, claims, evidence, analysis, interpretation,
   speculation, fiction, satire, synthetic material, and unknown must
   remain explicitly distinguished.

3. **World/persona separation** — The world model contains facts and
   evidence. Personas hold opinions, interpretations, and biases. Opinions
   never contaminate factual state.

4. **Evidence magnitude ≠ sufficiency** — Strong evidence for one claim
   does not prove an unrelated claim. Invariants over thresholds.

5. **Negative architecture** — The system is defined as much by what it
   cannot do (cannot authorize, cannot verify, cannot confirm) as by what
   it can.

6. **Persistence** — The system remembers previous broadcasts, previous
   beliefs, previous corrections, and evolves over time.

7. **Continuity** — Stories are tracked across time. Broadcasts reference
   prior coverage. The system knows what it already said.

8. **Pluggability** — Sources, models, TTS providers, and storage backends
   are swappable without changing core logic.

## Key capabilities

- Persistent world model with entity, claim, and evidence tracking
- Multiple synthetic personas with independent memories and opinions
- Editorial decision layer that determines what matters and why
- Broadcast queue with segment types (breaking, update, analysis,
  retrospective, explainer, opinion, discovery, deep dive, humor, synthetic)
- Non-preemptive broadcast — breaking stories queue for next slot, not
  mid-segment interruption
- Personal knowledge ingestion (documents, notes, repos, research)
- World snapshots with diff/change detection
- Voice output with provider-pluggable TTS
- Transparent relevance scoring with explainability ("why did you choose
  this story?")

## Non-goals (for now)

- Real-time breaking news interruption (breaking stories queue for next
  segment)
- Multi-agent live conversation (personas interpret separately, editorial
  layer decides what to broadcast)
- Mobile app (web UI first)
- Social media integration
- Video generation

## Success criteria

1. System operates continuously without human intervention
2. Broadcasts are coherent, fact-grounded, and honest about uncertainty
3. Personas evolve based on evidence, not arbitrary triggers
4. User can ask "why did you choose this story?" and get a transparent
   answer
5. System can recall and correct previous conclusions
6. All personal knowledge stays local
7. Another engineer can clone the repo, configure, and run it

## Relationship to news08

news08 is the current implementation — a single-file RSS→LLM→TTS
pipeline. Broadcast Mind is the conceptual evolution. The existing code
provides useful primitives (RSS fetching, Ollama integration, circuit
breaker, dedup, TTS) but lacks the architectural foundation for the
longer-term vision. Documentation and planning come first; implementation
follows only after the architecture is frozen.