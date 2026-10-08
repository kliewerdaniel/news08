# Source Architecture: Broadcast Mind

## Overview

The original news08 used a large RSS source collection.
The new architecture supports multiple source types with
pluggable providers.

## Supported source types

- **RSS** — RSS/Atom feeds (retained from news08)
- **Web search** — Search engine queries for active
  research
- **Web pages** — Direct page fetch and extraction
- **APIs** — Structured data sources where appropriate
- **User documents** — User-provided files
- **Notes** — User's own notes and observations
- **Repositories** — GitHub repos, code documentation
- **Saved research** — Previously saved articles, papers
- **Future providers** — Pluggable for new source types

## Source provider interface

All sources implement a common interface:

```
SourceProvider
  - fetch() → List[RawArticle]
  - normalize(raw) → NormalizedArticle
  - get_source_identity() → SourceID
  - get_reliability() → float (0.0–1.0)
  - is_available() → bool
```

## Source normalization

All sources produce normalized articles with:

- Title
- Content (cleaned text)
- URL (source URL, if web-based)
- Published timestamp
- Source identity
- Source reliability score
- Fetch timestamp
- Language

## Deduplication

Articles are deduplicated across all sources using:

1. Content hash (primary — detects identical/similar content)
2. URL matching (detects same article across feeds)
3. Title similarity (detects reprinted articles)

Deduplication window: configurable (default: 3 days,
matching news08).

## Source identity

Each source has a stable identity:

- RSS feed: feed URL + feed title
- Web search: search engine + query
- Web page: page URL
- API: API endpoint + provider name
- User document: file path + document hash

Source identity is immutable — if a source changes its
URL or content, it gets a new identity.

## Source reliability

Each source carries a reliability score (0.0–1.0):

- Known reputable outlets: 0.8–1.0
- Mixed reliability: 0.5–0.8
- Unverified/unknown: 0.2–0.5
- Known unreliable: 0.0–0.2

Reliability affects evidence confidence and editorial
decisions. Low-reliability sources require corroboration
before their claims enter the world state as facts.

## Provenance

Every piece of evidence carries:

- Source identity
- Fetch timestamp
- Extraction method
- Extractor identity
- Confidence in extraction

## Source failures

When a source fails:

1. Error is logged with source identity
2. Circuit breaker pattern prevents hammering failed
   sources
3. After N failures, source is marked unavailable
4. System continues with remaining sources
5. User is notified of persistent failures

## Source configuration

Sources are configured in a YAML file (similar to
news08's feeds.yaml) but extended for multiple source
types:

```yaml
sources:
  rss:
    - url: "https://feeds.bbci.co.uk/news/world/rss.xml"
      name: "BBC World"
      reliability: 0.85
      enabled: true
      fetch_interval_minutes: 15

  search:
    - engine: "duckduckgo"
      query: "breaking news"
      reliability: 0.6
      enabled: true

  user_docs:
    - path: "~/notes/research.md"
      reliability: 0.7
      enabled: true
```

## Source prioritization

Sources are prioritized by:

1. Reliability score
2. Recency of last successful fetch
3. User-configured priority
4. Source type (user docs > RSS > search)

Higher priority sources are fetched first; lower priority
sources are fetched if time permits.

## Comparison to news08

news08 had a flat list of RSS feeds in `feeds.yaml`. The
new architecture:

- Supports multiple source types (not just RSS)
- Each source has a reliability score
- Source identity is stable and tracked
- Failures are handled with circuit breakers
- Configuration is typed and validated
- Sources are pluggable — new types can be added
  without changing core logic
- User documents are first-class sources