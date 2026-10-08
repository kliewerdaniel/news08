# Repository Archaeology: news08

## What news08 currently is

A single Python script (`main.py`, ~772 lines) that implements an automated
continuous news broadcast generator. It fetches RSS feeds, summarizes articles
via Ollama LLM, clusters them with TF-IDF/K-Means, generates broadcast scripts,
converts them to speech with edge-tts, and plays audio continuously.

## How data moves through the system

```
feeds.yaml → fetch_feeds_batch → Article objects
    → process_articles_smart → summarization (Ollama)
    → cluster_articles_tfidf → importance scoring
    → create_broadcast_segments → script generation (Ollama)
    → generate_and_queue_audio → edge-tts → audio queue → playback thread
```

Each cycle runs in `run_continuous`, sleeps for the configured interval, then
repeats. No persistence of broadcast state between cycles beyond the article
cache.

## Components currently exist (CONFIRMED)

- **RSS ingestion** — `fetch_feeds_batch`, `fetch_single_feed`, feedparser-based
- **HTML cleaning** — `extract_content` strips tags with regex
- **Deduplication** — content-hash against SQLite `articles` table (3-day TTL)
- **Summarization** — Ollama `mistral-small:24b-instruct-2501-q8_0` via REST API
- **Relevancy scoring** — LLM-based 0–10 score against `--topic` filter
- **Clustering** — TF-IDF + K-Means on titles/summaries
- **Importance scoring** — freshness (40%), content quality (30%), sentiment (20%), readability (10%)
- **Script generation** — Ollama broadcast model, anchor-style 5-6 sentence segments
- **TTS** — edge-tts with `en-US-JennyNeural`, queue-based async playback
- **Circuit breaker** — 3-failure threshold, 30s recovery on Ollama calls
- **Performance monitoring** — articles processed, timing, error rate
- **Logging** — file + console, broadcast logs saved as Markdown

## What the system does well (CONFIRMED)

- Simple architecture, easy to understand and modify
- Local-first (Ollama + edge-tts, no cloud dependencies)
- Deduplication prevents re-broadcasting identical articles
- Circuit breaker protects against Ollama failures
- Content-hash caching is correct
- Multi-feed async fetching with batching
- Graceful degradation on summary failures (falls back to content truncation)

## What it does poorly (CONFIRMED)

- No world model — articles are ephemeral, no story tracking
- No entity extraction or relationship mapping
- No distinction between fact and opinion
- TF-IDF clustering is keyword-based, not semantic
- Importance scoring is heuristic, not evidence-based
- No persistence of persona, opinion, or belief state
- No correction or revision mechanism
- Broadcast logs are flat Markdown, not queryable
- Single-file architecture, no modularity
- No tests
- Hard-coded models and voice in CONFIG dict
- No source reliability tracking
- No user personalization beyond `--topic` flag
- The script generates "news" from raw RSS without any editorial judgment about
  what matters or what's changing

## What should probably be retained (PROPOSED)

- RSS fetching architecture (aiohttp + feedparser)
- Ollama integration pattern (REST API calls)
- Circuit breaker pattern
- Content hash deduplication approach
- Edge-tts for TTS
- Performance monitoring skeleton
- Logging infrastructure
- Command-line argument parsing

## What should probably be replaced (PROPOSED)

- TF-IDF clustering → semantic embeddings with proper clustering
- Heuristic importance scoring → editorial decision engine
- Flat script generation → segmented pipeline with persona awareness
- No state → persistent world model
- SQLite article cache → proper source-of-truth architecture
- Single-file → modular architecture
- No epistemic categories → explicit fact/claim/evidence model

## What is unknown (UNKNOWN)

- How many RSS feeds are actually reachable from the user's network
- Ollama model availability on the user's hardware (mistral-small:24b requires ~16GB RAM)
- Whether edge-tts works on the user's macOS system
- User's actual interest profile (the `--topic` flag is one-shot, not persistent)
- Whether the user wants multiple personas
- Whether the user wants real-time or scheduled broadcasts
- Audio playback preferences (speaker, headphones, soundbar)
- Target hardware capabilities (the user has a primary machine and a secondary
  machine with limited resources per memory)

## Needs verification (NEEDS VERIFICATION)

- Can Ollama serve mistral-small:24b on the user's hardware?
- Does edge-tts produce satisfactory audio quality?
- Do the RSS feeds in feeds.yaml actually resolve?
- Does the circuit breaker pattern work correctly in async context?

## Architectural debt

- All state is ephemeral — no broadcast history, no story continuity
- No separation between world facts and persona opinions
- No evidence provenance — summaries are opaque LLM outputs
- TF-IDF clustering conflates topical similarity with story identity
- Importance scoring mixes freshness with quality without calibration
- CONFIG dict is global mutable state
- No dependency injection
- No interface boundaries between components

## Implementation risks

- Single-file monolith will become unmaintainable as features are added
- Ollama REST API is synchronous-blocking under the hood (run_in_executor)
- No rate limiting on feed fetching
- No error recovery for partial broadcast failures
- Audio queue can grow unbounded if TTS is slower than generation

## Useful existing abstractions

- Article dataclass is clean
- BroadcastSegment dataclass is a good seed for the segment model
- CircuitBreaker is functional
- PerformanceMonitor provides a metrics pattern
- sqlite3 caching shows understanding of dedup needs

## Obsolete abstractions

- `target_segments: 2500` in CONFIG — impossibly high, likely a placeholder
- `max_broadcast_length: 900000000` — effectively unbounded
- `relevancy_threshold: 5` — hard-coded magic number
- `extract_content` regex-based HTML stripping — fragile, better to use newspaper3k
  (already in requirements but unused)

## Unanswered questions

- Should broadcasts be continuous or on-demand?
- Should the system remember what it broadcast yesterday/last week?
- Should multiple listeners have different persona profiles?
- Is TTS needed at all for the MVP, or is text-first acceptable?
- Does the user want live playback or batch-generated audio files?
- Should the system wake the user, or is it background?

## Labels used

- **CONFIRMED** — verified by reading source code
- **INFERRED** — reasonably deduced from code behavior
- **UNKNOWN** — cannot determine from code alone
- **NEEDS VERIFICATION** — requires runtime testing
- **PROPOSED** — suggestion for future architecture, not yet validated