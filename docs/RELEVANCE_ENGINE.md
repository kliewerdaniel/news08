# Relevance Engine: Broadcast Mind

## Overview

The system determines "what matters to me" through both
explicit configuration and learned behavior.

## Signal sources

### Explicit configuration

- User-defined interests (topics, entities, domains)
- Current projects (user tells the system what they're
  working on)
- Source preferences (which sources to prioritize)
- Persona relevance weights (which persona's perspective
  matters most)

### Learned behavior

- Historical interests (what the user has listened to)
- Recent behavior (what the user has queried or
  explored)
- Story importance (evidence-based scoring)
- Novelty (new information vs. repeated coverage)
- Unresolved stories (stories with open questions)
- Persona interests (which personas find the story
  relevant)
- Relationship to previous broadcasts (continuity,
  follow-up)

## Scoring architecture

Each story receives a composite relevance score:

```
relevance = w1 * explicit_interest
          + w2 * learned_interest
          + w3 * story_importance
          + w4 * novelty
          + w5 * continuity
          + w6 * persona_relevance
```

Weights are configurable per user. Default weights
favor explicit interests and story importance.

## Transparency

The system must be able to explain its relevance
decisions:

> "Why did you choose this story?"

The explanation includes:

- Which signals fired
- Their individual scores
- The composite score
- What threshold was crossed
- What alternative stories were considered and why
  they scored lower

## Personal relevance

Personal relevance is distinct from general importance:

- A story about the user's employer is personally
  relevant even if not globally important
- A story about the user's research area is relevant
  even if niche
- A story about the user's interests is relevant even
  if low-impact

Personal relevance is labeled as such in broadcasts:

> "This story is personally relevant because..."

## Editorial integration

The relevance engine feeds into the editorial
decision layer:

1. Relevance score computed for each story
2. Stories above threshold enter the broadcast queue
3. Queue ordering considers relevance + importance
   + novelty
4. Editor can override relevance scoring
5. Overrides are logged for learned behavior

## Continuous learning

The system learns from user behavior:

- User listens to a segment → increase relevance
  weight for that topic
- User skips a segment → decrease weight (but not
  zero — one skip doesn't prove disinterest)
- User queries a topic → increase relevance for that
  topic
- User saves a story → high positive signal
- User corrects the system → strong signal for
  that domain

Learning is gradual and conservative — single events
don't dramatically shift weights.

## Comparison to news08

news08 has a simple `--topic` flag and a hard-coded
`relevancy_threshold: 5`. The new system adds:

- Multiple explicit signal sources
- Learned behavior from user interaction
- Composite scoring with configurable weights
- Transparent explanation of relevance decisions
- Personal vs. general relevance distinction
- Editorial override capability