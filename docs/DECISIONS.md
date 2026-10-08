# Decision Log: Broadcast Mind

## Decision 1: Local-first architecture

**Context:** The user values sovereignty, privacy, and offline operation.
The system must run without cloud dependencies.

**Alternatives:**
- Cloud-hosted LLM APIs (OpenAI, Anthropic)
- Hybrid cloud/local
- Fully cloud-based

**Chosen approach:** Local-first with Ollama for LLM inference and
edge-tts for TTS. All data stays on user hardware.

**Reason:** Privacy, sovereignty, offline operation, no recurring costs,
no API dependency.

**Consequences:**
- Hardware requirements for local inference
- Model quality may be lower than cloud APIs
- No automatic model updates

**Status:** PROPOSED — needs verification on user hardware

---

## Decision 2: SQLite for persistence

**Context:** news08 already uses SQLite. The world model needs
persistent storage.

**Alternatives:**
- PostgreSQL (more robust, requires server)
- MongoDB (document model, more complex)
- File-based JSON (simple, not queryable)

**Chosen approach:** SQLite with async SQLAlchemy. Proven by
news08, simple, local, zero-config.

**Reason:** Local-first, zero infrastructure, proven in news08,
sufficient for single-user workload.

**Consequences:**
- Not suitable for multi-user concurrent access
- Scaling limited (acceptable for single user)

**Status:** PROPOSED

---

## Decision 3: Append-only world state

**Context:** The system must remember past beliefs and corrections.
Deleting or mutating old records would lose audit history.

**Alternatives:**
- Mutable world state (overwrite old records)
- Soft-delete with retention policy
- Full snapshot every cycle

**Chosen approach:** Append-only with corrections referencing
previous records. Old records preserved for audit.

**Reason:** Audit trail, correction history, "what did the system
believe yesterday?" capability.

**Consequences:**
- Storage grows over time (mitigated by retention policy)
- Queries must filter for current vs. historical

**Status:** PROPOSED

---

## Decision 4: World state / persona opinion separation

**Context:** The core epistemic requirement is that opinions never
contaminate factual world state.

**Alternatives:**
- Unified store (facts and opinions together)
- Tag-based separation (same table, different tags)
- Separate databases

**Chosen approach:** Separate tables with strict interface
boundaries. World state never accepts persona opinions.
Persona opinions reference world state claims but never
modify them.

**Reason:** Epistemic integrity, auditability, persona isolation.

**Consequences:**
- More complex queries (joins across stores)
- Requires discipline in implementation

**Status:** PROPOSED — critical architectural constraint

---

## Decision 5: YAML source configuration

**Context:** news08 uses feeds.yaml for RSS feeds. The new system
needs multi-source configuration.

**Alternatives:**
- JSON config files
- Database-stored configuration
- Environment variables
- UI-only configuration (no files)

**Chosen approach:** YAML files for human readability, with
environment variable overrides for sensitive values.

**Reason:** Human-editable, versionable, diffable, familiar from
news08.

**Consequences:**
- YAML parsing dependency
- File-based config doesn't support real-time updates without
  reload

**Status:** PROPOSED

---

## Decision 6: CLI-first with web UI

**Context:** news08 is a CLI script. The new system needs a UI
for persona management, source configuration, and live monitoring.

**Alternatives:**
- CLI only (no UI)
- Web UI only (no CLI)
- TUI (terminal UI)

**Chosen approach:** Web UI (Next.js) for rich interaction,
CLI for automation and headless operation.

**Reason:** User experience for control room, real-time monitoring,
persona visualization. CLI for scripting and server use.

**Consequences:**
- Frontend dependency (Node.js)
- More complex deployment

**Status:** PROPOSED

---

## Decision 7: Edge-TTS as default voice provider

**Context:** news08 uses edge-tts. It's free, local, and produces
reasonable quality.

**Alternatives:**
- Piper (lighter weight)
- Coqui (more features, complex)
- Cloud TTS (higher quality, requires internet)

**Chosen approach:** Edge-TTS as default, pluggable interface
for alternatives.

**Reason:** Already integrated, free, local, good enough for MVP.

**Consequences:**
- Limited voice selection
- macOS dependency for edge-tts
- Not suitable for all languages

**Status:** PROPOSED — voice research needed (see VOICE.md)

---

## Decision 8: Semantic embeddings over TF-IDF

**Context:** news08 uses TF-IDF + K-Means for clustering. Semantic
embeddings produce better clusters.

**Alternatives:**
- TF-IDF (current, simpler)
- Universal Sentence Encoder
- nomic-embed-text (Ollama, local)

**Chosen approach:** nomic-embed-text via Ollama for local
semantic embeddings. TF-IDF as fallback.

**Reason:** Better clustering quality, local, consistent with
Ollama-based architecture.

**Consequences:**
- Requires nomic-embed-text model
- More resource-intensive than TF-IDF
- Ollama must be running for embeddings

**Status:** PROPOSED

---

## Decision 9: Single-file → modular architecture

**Context:** news08 is a single 772-line file. The new system
requires modularity for maintainability.

**Alternatives:**
- Keep single file (simple, no modularity)
- Split into modules (more files, better organization)
- Full microservice architecture (overkill)

**Chosen approach:** Python package with clear module boundaries:
core, agents, sources, store, voice, api, ui.

**Reason:** Maintainability, testability, extensibility. Single
file doesn't scale beyond proof-of-concept.

**Consequences:**
- More files to manage
- Import complexity
- Need tests to prevent regression

**Status:** PROPOSED