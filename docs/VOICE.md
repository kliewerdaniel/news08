# Voice Architecture: Broadcast Mind

## Overview

Voice must be provider-pluggable. The architecture
supports multiple TTS backends without changing
core broadcast logic.

## Provider interface

```
VoiceProvider
  - synthesize(text, voice_settings) → AudioData
  - list_voices() → List[Voice]
  - get_voice_settings(voice_id) → VoiceSettings
  - is_available() → bool
```

## Supported providers (initial candidates)

### Edge-TTS (current, news08)

- Free, local
- Good quality neural voices
- Limited voice selection
- No custom voice cloning

### Lightweight local TTS

- Piper, Coqui, or similar
- Very low resource usage
- Lower quality than neural TTS
- Good for secondary/low-power machine

### Higher quality local TTS

- XTTS, VALL-E, or similar
- Higher quality
- More resource intensive
- Suitable for primary machine

### Qwen-based TTS

- Qwen3-TTS MLX 4-bit (where hardware permits)
- Good quality
- Local inference only

## Hardware considerations

The user has two machines:

- **Primary machine:** Can support more advanced
  models (XTTS, higher-quality local TTS)
- **Secondary machine:** Limited resources —
  lightweight TTS (Piper, edge-tts)

The system should auto-detect available hardware
and select the best available provider.

## Voice per persona

Each persona can have a distinct voice:

- Persona A → voice 1 (authoritative, deep)
- Persona B → voice 2 (conversational, bright)
- Persona C → voice 3 (analytical, measured)

Voices are configured per persona, not hard-coded.

## Synthetic voices only

The system uses synthetic voices. Impersonation of
real people is not part of the core system.

## Voice selection logic

1. Check persona voice preference
2. Check hardware capabilities
3. Select best available provider
4. Fall back to next provider if unavailable
5. Log voice selection for reproducibility

## TTS research task (explicit)

The current TTS implementation (edge-tts with
en-US-JennyNeural) is a starting point. A proper
voice architecture requires research into:

- Local TTS options on macOS (piper, coqui, etc.)
- Quality comparisons between providers
- Resource usage on secondary machine
- Multi-voice support per TTS engine
- Voice cloning (if desired, separate from core)

This research is a future task, not a current
implementation requirement.

## Comparison to news08

news08 uses edge-tts exclusively with a hard-coded
voice. The new architecture:

- Provider-pluggable interface
- Multiple backend candidates
- Per-persona voice assignment
- Hardware-aware selection
- Fallback chain
- Explicit research gap for voice comparison