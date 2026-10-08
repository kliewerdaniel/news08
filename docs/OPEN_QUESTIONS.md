# Open Questions: Broadcast Mind

## Technology choices (require experimentation)

1. **Exact database technology** — SQLite confirmed, but ORM
   choice (SQLAlchemy async? Tortoise? Raw SQL?) needs
   evaluation.

2. **Exact source providers** — RSS confirmed (news08). Web
   search provider needs selection (DuckDuckGo? SearXNG?
   Custom?). User document ingestion needs implementation.

3. **Exact TTS implementation** — Edge-TTS is the starting
   point. Piper, Coqui, or other local TTS needs research.
   See `docs/VOICE.md` for the research task.

4. **Exact queue implementation** — In-memory queue with
   SQLite persistence? Redis? Simple SQLite-backed queue?

5. **Exact agent runtime** — Subprocess? Threading? Async
   tasks? Celery? Needs evaluation based on concurrency
   requirements.

6. **Exact graph implementation** — NetworkX? Custom in-memory
   graph? Graph database (Neo4j)? Needs evaluation.

7. **Exact deployment strategy** — Docker? systemd? Manual?
   Needs target environment definition.

8. **Retention/compression policy** — How long to keep audio?
   When to compress old broadcasts? How many snapshots to
   retain?

9. **Model selection** — mistral-small:24b is the starting
   point. Does it run on user hardware? What about summary
   vs. broadcast model differentiation?

## Architectural questions

10. **Multi-persona concurrency** — Can multiple personas
    process the same world state simultaneously? What are
    the consistency guarantees?

11. **World state update timing** — When does the world state
    update relative to broadcast? Before or after script
    generation?

12. **Breaking story definition** — What constitutes a
    "breaking" story vs. an update? Who decides?

13. **Persona disagreement handling** — When personas
    disagree, does the broadcast present both views or
    pick one?

14. **Source reliability calibration** — How is source
    reliability determined? Static config or learned?

15. **Evidence threshold for facts** — What confidence level
    upgrades a claim to a fact? Who sets this threshold?

## Operational questions

16. **Monitoring and alerting** — What does the system do
    when it detects a problem? Who gets notified?

17. **Backup and recovery** — How is user data backed up?
    What is the recovery procedure?

18. **Updates and versioning** — How does the system update
    without losing state? How are schema migrations handled?

19. **Multi-user support** — Is the system single-user or
    multi-user? If multi-user, how is data隔离ed?

20. **Offline behavior** — What happens when Ollama is
    unavailable? Does the system queue requests or skip?

## Unresolved decisions

21. **Exact embedding model** — nomic-embed-text is the
    candidate. Needs verification of quality and resource
    usage.

22. **Clustering algorithm** — K-Means is the starting point.
    DBSCAN? Hierarchical? Needs comparison.

23. **Voice cloning** — Should personas have unique cloned
    voices? This is a future research topic, not a current
    decision.

24. **"God Mode" simulation** — How many personas in
    simulation? What's the output format? Needs design.

## Marked as requiring experimentation

- Database ORM choice → requires prototype
- TTS provider comparison → requires testing on user hardware
- Source provider selection → requires network testing
- Agent runtime model → requires concurrency testing
- Graph implementation → requires scale testing