# Migration Map: news08 → Broadcast Mind

## Overview

Map the current news08 system to the proposed
Broadcast Mind architecture. Categories:

- **KEEP** — Retain as-is or with minimal changes
- **REFACTOR** — Keep concept, change implementation
- **REPLACE** — Replace with better approach
- **REMOVE** — Obsolete, remove without replacement
- **UNKNOWN** — Unclear, needs investigation
- **NEW** — Doesn't exist in news08, must be built

## Source ingestion

| Subsystem | news08 | Category | Notes |
|---|---|---|---|
| RSS fetching (aiohttp + feedparser) | `fetch_feeds_batch` | KEEP | Proven, working |
| HTML content extraction | `extract_content` (regex) | REPLACE | Use newspaper3k (already in requirements) |
| Feed configuration | `feeds.yaml` | KEEP | Extend for multi-source |
| Deduplication (content hash) | `is_duplicate` / `cache_article` | KEEP | Proven approach |
| Web search sources | None | NEW | Must be added |
| User document sources | None | NEW | Must be added |
| Source reliability tracking | None | NEW | Must be added |
| Source failure handling | Basic error logging | REPLACE | Circuit breaker exists, needs integration |

## Data model

| Subsystem | news08 | Category | Notes |
|---|---|---|---|
| Article dataclass | `Article` | KEEP | Clean, extendable |
| BroadcastSegment dataclass | `BroadcastSegment` | KEEP | Good seed for segment model |
| SQLite article cache | `news_cache.db` | REPLACE | Replace with world state schema |
| World state | None | NEW | Core addition |
| Story tracking | None | NEW | Core addition |
| Entity tracking | None | NEW | Core addition |
| Claim tracking | None | NEW | Core addition |
| Evidence tracking | None | NEW | Core addition |
| Epistemic categories | None | NEW | Core addition |
| Persona system | None | NEW | Core addition |
| Persona memory | None | NEW | Core addition |
| Persona opinions | None | NEW | Core addition |
| World snapshots | None | NEW | Core addition |
| Correction records | None | NEW | Core addition |

## Analysis pipeline

| Subsystem | news08 | Category | Notes |
|---|---|---|---|
| Summarization (Ollama) | `generate_summary_safe` | KEEP | Proven, extend |
| Relevancy scoring | `calculate_relevancy_score` | REPLACE | Replace with relevance engine |
| TF-IDF clustering | `cluster_articles_tfidf` | REPLACE | Use semantic embeddings |
| Importance scoring | `calculate_importance_scores` | REPLACE | Use editorial decision engine |
| Sentiment analysis | NLTK VADER | REPLACE | More robust analysis needed |
| Fact extraction | None | NEW | Must be added |
| Claim extraction | None | NEW | Must be added |
| Contradiction detection | None | NEW | Core addition |
| Evidence evaluation | None | NEW | Core addition |

## Script generation

| Subsystem | news08 | Category | Notes |
|---|---|---|---|
| Script generation (Ollama) | `generate_segment_script` | KEEP | Proven, extend with persona |
| Transition phrases | `generate_transition_phrase` | KEEP | Useful, extend |
| Script refinement | `refine_script` | KEEP | Extend with guidance |
| Persona-aware generation | None | NEW | Core addition |
| Evidence-grounded scripts | None | NEW | Core addition |
| Uncertainty communication | None | NEW | Core addition |

## Audio / TTS

| Subsystem | news08 | Category | Notes |
|---|---|---|---|
| Edge-TTS integration | `generate_and_queue_audio` | KEEP | Proven |
| Audio playback thread | `play_audio_from_queue` | KEEP | Proven |
| Audio artifact archival | None | NEW | Must be added |
| Voice per persona | None | NEW | Core addition |
| Provider pluggability | None | NEW | Core addition |

## State management

| Subsystem | news08 | Category | Notes |
|---|---|---|---|
| Circuit breaker | `CircuitBreaker` | KEEP | Proven pattern |
| Performance monitor | `PerformanceMonitor` | KEEP | Proven |
| Logging | file + console | KEEP | Extend with structured logging |
| Broadcast logs | Markdown files | REPLACE | Replace with artifact archive |
| Persistent state | SQLite cache only | REPLACE | Full world state needed |
| Broadcast history | None | NEW | Core addition |

## Broadcast semantics

| Subsystem | news08 | Category | Notes |
|---|---|---|---|
| Continuous loop | `run_continuous` | KEEP | Core behavior |
| Fetch interval | `--fetch_interval` | KEEP | Configurable |
| Topic filter | `--topic` | REFACTOR | Extend to relevance engine |
| Guidance | `--guidance` | REFACTOR | Extend to persona system |
| Non-preemptive queue | None | NEW | Core addition |
| Idle content generation | None | NEW | Core addition |
| Breaking story queuing | None | NEW | Core addition |

## UI / Monitoring

| Subsystem | news08 | Category | Notes |
|---|---|---|---|
| CLI output | logging | KEEP | Extend with UI |
| No UI | None | NEW | Control room required |
| WebSocket updates | None | NEW | Live monitoring |
| Persona management UI | None | NEW | Core addition |
| Source configuration UI | None | NEW | Core addition |
| Analysis display | None | NEW | SHOW ANALYSIS interaction |

## Personal knowledge

| Subsystem | news08 | Category | Notes |
|---|---|---|---|
| User document ingestion | None | NEW | Core addition |
| Note ingestion | None | NEW | Core addition |
| Repo discovery | None | NEW | Core addition |
| Personal vs. external separation | None | NEW | Core addition |

## Multi-persona

| Subsystem | news08 | Category | Notes |
|---|---|---|---|
| Single persona | None (no persona concept) | NEW | Core addition |
| Multiple personas | None | NEW | Core addition |
| Persona disagreement | None | NEW | Core addition |
| Simulation mode | None | NEW | Core addition |

## Summary

| Category | Count |
|---|---|
| KEEP | 13 |
| REFACTOR | 3 |
| REPLACE | 8 |
| REMOVE | 0 |
| UNKNOWN | 0 |
| NEW | 28 |

## Key insight

news08 is a working RSS→LLM→TTS pipeline. The
Broadcast Mind architecture preserves the proven
components (RSS fetching, Ollama integration, circuit
breaker, dedup, TTS) while adding the architectural
foundation (world state, epistemic model, personas,
editorial layer, artifact archive) that news08
completely lacks.

Nothing useful is being thrown away. The migration
is additive — new capabilities layer on top of the
existing pipeline rather than replacing it.