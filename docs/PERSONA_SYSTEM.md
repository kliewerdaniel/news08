# Persona System: Broadcast Mind

## Overview

The persona system is significantly deeper than a simple tone
configuration. Personas are synthetic minds with identity,
psyche, values, drives, cognitive styles, biases, memories,
experiences, and opinions. They interpret the world through
their own lens, and they may disagree with each other and with
the factual world state.

## Persona dimensions

### Identity

- Name
- Role description
- Background / origin story
- Age (if applicable)
- Expertise domains

### Psyche

- Big Five personality traits (confirmed from news08:
  extraversion, agreeableness, conscientiousness, neuroticism,
  openness)
- Dark Triad traits (machiavellianism, narcissism, psychopathy)
- Moral Foundations (care/harm, fairness/cheating,
  loyalty/betrayal, authority/subversion, sanctity/degradation)
- Political orientation (liberalism/conservatism, economic)
- Emotional traits (optimism, pessimism, empathy, cynicism,
  hope, despair)

### Cognitive Style

- Analytical vs. intuitive
- Creative vs. concrete
- Logical vs. abstract
- Skepticism level
- Gullibility level
- Objectivity vs. subjectivity

### Communication

- Verbosity
- Formality
- Sarcasm tendency
- Humor style
- Directness

### Values & Drives

- Core values (justice, truth, loyalty, freedom, etc.)
- Curiosity level
- Ambition
- Risk tolerance
- Contentment vs. restlessness

### Blind Spots

- Domains where the persona is likely to miss things
- Biases the persona is prone to
- Topics the persona avoids or struggles with

### Memory

- Persistent memories of past broadcasts
- Remembered interactions with other personas
- Learned experiences from corrections
- Historical opinions and how they changed

### Opinions

- Held opinions on entities, claims, stories
- Opinions are explicitly tagged as persona-held
- Opinions may be wrong — the system tracks this
- Personas may revise opinions based on new evidence

## Persona representation

### Visual (UI)

Personas are represented visually in the control room:

- Avatar or visual indicator
- Trait radar chart (52-dimension visualization)
- Current opinion summary
- Memory timeline
- Evolution graph

### Structural (data)

Personas are stored as configuration + memory:

- Identity and trait vector (JSON, 52 dimensions)
- Memory records (append-only)
- Opinion records (mutable, versioned)
- Experience log (immutable)

## Multi-persona support

Multiple personas must be supported simultaneously. Each
persona:

- Has its own trait vector
- Has its own memory store
- Holds its own opinions
- Interprets the same world state independently
- May disagree with other personas
- May criticize other personas' interpretations
- May criticize the user's interpretation
- May be deliberately absurd

## Persona isolation

**Critical rule:** Persona memory and opinions must remain
separate from world memory. World state contains facts and
evidence. Personas hold interpretations and opinions.

- Persona memory never contaminates world state
- Persona opinions are always labeled as such
- World state provides evidence; personas interpret it
- A persona's wrong opinion does not change world facts

## Simulation / God Mode

A simulation mode runs multiple radically different personas
on the same world state simultaneously. The output shows how
different minds interpret the same facts differently.

**Critical requirement:** Simulation output must always be
labeled as simulated interpretation, never as factual reporting.
The system must make it explicit that simulated personas are
not factual authorities.

Example broadcast framing:

> "In a simulation, a skeptical analyst said X, while an
> optimistic analyst said Y. The actual evidence is Z."

## Persona lifecycle

1. **Creation** — User defines or system generates a persona
   with initial trait vector
2. **Operation** — Persona processes world state, forms
   opinions, generates broadcast content
3. **Evolution** — Persona traits shift based on evidence (not
   arbitrary triggers)
4. **Memory accumulation** — Persona retains experiences and
   learned patterns
5. **Opinion revision** — Persona updates opinions when new
   evidence contradicts old ones
6. **Archival** — Inactive personas may be archived but not
   deleted (preserves history)

## Trait evolution

Trait evolution is driven by evidence, not by keyword matching.
When a persona processes a claim or event:

1. The analyst extracts the factual content
2. The persona evaluates it through its cognitive style
3. If the evidence is strong enough, the persona's opinion
   may shift
4. Trait deltas are calculated and applied with bounds
   (0.0–1.0 per trait)
5. Evolution is logged for audit

## Anti-patterns to avoid

- Persona as "tone slider" — personas are minds, not filters
- Trait changes without evidence — evolution must be evidence-driven
- Persona opinion as world fact — opinions never enter world
  state without evidence
- Simulated persona as factual authority — simulation is
  always labeled
- Persona memory merging with world memory — they are separate
  stores
- Hard-coded personas — personas should be configurable and
  extensible