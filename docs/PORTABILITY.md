# Portability: Broadcast Mind

## Overview

The repository should eventually be cloneable by
another person. They should be able to configure
and run the system without modifying core
application code.

## Separation

```
CORE ENGINE     — Application logic, never modified
                 by users
CONFIGURATION   — Environment variables, config files
PERSONAS        — Persona definitions and trait vectors
SOURCE CONFIG   — feeds.yaml, sources config
USER DATA       — Personal knowledge, notes, research
LOCAL RUNTIME   — Ollama, TTS, database files
```

## Clone → Configure → Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/kliewerdaniel/news08.git
   cd news08
   ```

2. **Configure environment**
   - Copy `.env.example` to `.env`
   - Set `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
   - Set `DATABASE_URL` (default: SQLite local)

3. **Create/select personas**
   - Edit `personas/` directory or use UI
   - Each persona is a YAML/JSON file
   - Default personas provided

4. **Configure sources**
   - Edit `feeds.yaml` or use UI
   - Add RSS feeds, search queries, user docs

5. **Configure TTS**
   - Set `TTS_PROVIDER`, `TTS_VOICE` in `.env`
   - Install provider if needed

6. **Add personal knowledge**
   - Place files in `user_data/`
   - System discovers and ingests on next cycle

7. **Run the system**
   ```bash
   python main.py
   ```

## What users configure

- Sources (RSS feeds, search, documents)
- Personas (traits, identities, voices)
- TTS (provider, voice, speed)
- Relevance weights
- Broadcast interval
- Retention policy

## What users never modify

- Core engine code
- Domain model schemas
- Epistemic category definitions
- Editorial decision logic
- Analyst capabilities

## Comparison to news08

news08 requires editing CONFIG dict in
main.py for model selection and processing
parameters. The new system moves all
configuration to external files and environment
variables.

## Directory layout (target)

```
news08/
├── core/              # Core engine (never modified by users)
├── config/            # Configuration files
│   ├── .env.example
│   ├── feeds.yaml
│   └── sources.yaml
├── personas/          # Persona definitions
│   ├── default/
│   └── custom/
├── user_data/         # User-provided knowledge
│   ├── documents/
│   ├── notes/
│   └── research/
├── docs/              # This documentation
├── main.py            # Entry point
└── README.md
```

## Comparison to news08 structure

news08 is a single file with hard-coded
configuration. The new system separates:

- Config → YAML files + environment variables
- Personas → YAML/JSON files
- User data → Dedicated directory
- Core engine → Python package (not a script)