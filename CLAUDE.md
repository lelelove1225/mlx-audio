# mlx-audio — ReFelicia TTS backend (fork)

A fork of `Blaizzy/mlx-audio` used as the **text-to-speech backend for ReFeliciaV2**, a Unity
character-conversation app in the sibling repo `~/Documents/GitHub/ReFelicia/ReFeliciaV2`.

Remotes: `origin` = this fork (`github.com/lelelove1225/mlx-audio`), `upstream` = `Blaizzy/mlx-audio`.
Sync upstream with `git fetch upstream && git rebase upstream/main`.

## Role & architecture

- The Unity client drives each turn and calls a TTS server over the **`/ws/tts` WebSocket**
  (ReFelicia/SBV2-compatible). Wire-format spec: **`docs/ws-tts-protocol.md`** — read it before
  touching the server protocol or building a compatible server. The server loops on one connection
  (many requests per connection); the client keeps the socket persistent across turns.
- Two interchangeable backends speak this protocol:
  - **This MLX server** (`mlx_audio.server`) — Mac / Apple Silicon.
  - **A CUDA `Irodori-TTS-Server`** (Aratako/Irodori-TTS) — current production backend, on a
    ROG Flow Z13 (Intel / Ubuntu), migrating to an NVIDIA **GB10 (DGX Spark)** ~2026-06-23.

## Run the MLX server

```bash
MLX_AUDIO_REFELICIA_TTS_MODEL=mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit \
MLX_AUDIO_REFELICIA_TTS_REF_AUDIO=<10–30s clean single-speaker clip> \
MLX_AUDIO_REFELICIA_TTS_NUM_STEPS=6 \
MLX_AUDIO_REFELICIA_TTS_CFG_GUIDANCE_MODE=independent \
uv run --extra server mlx_audio.server --host 127.0.0.1 --port 8880
```

## Gotchas (hard-won — don't relearn these)

- **Sample rate must be truthful.** Irodori outputs 48 kHz; the `start` message's `sample_rate`
  must equal the PCM rate. A client-side 44.1 kHz override once made clones sound "not similar"
  (pitch ≈ −1.5 semitones). Unity `SampleRateOverride` must be `0`.
- `CFG_GUIDANCE_MODE=independent`, **not** `alternating` (alternating halves speaker guidance at
  low steps → worse clone). `NUM_STEPS` 6–8 (6 is the aggressive floor).
- On CUDA use **bf16** (fp16 is unsupported and no faster on Ampere/Ada). `torch.compile` is
  deliberately off (recompile-on-shape-miss freeze is worse than the ~100 ms it saves).
- Voice (clone) and personality (StyleRAG) are **orthogonal**: clone = timbre, StyleRAG = wording.
- Reference audio / speaker embedding is precomputed/cached, not re-encoded per request.

## Sibling repo & design doc

- Unity client: `~/Documents/GitHub/ReFelicia/ReFeliciaV2` (the "Character OS" — TurnCoordinator,
  PromptPipeline, StyleRAG, Working Memory, chunker, the `/ws/tts` client).
- Design doc: `~/Documents/GitHub/ReFelicia/docs/ReFeliciaMobileV2_CompleteDesign.md`.
  **Keep code and this doc in sync — explicit user rule.** Relevant: §8.3 trace, §11.2 chunking,
  §11.3 TTS backend.

## Current state (2026-06-20)

- TTFA ~725 ms / p95 ~891 ms on CUDA (bf16, 6 steps, voice clone + StyleRAG persona,
  sentence-boundary incremental). WebSocket is persistent across turns with reconnect-and-retry
  on a stale socket.
- Pending: **GB10 migration** (Tuesday) — see memory `gb10-migration-plan`. Open decisions:
  full-audio vs incremental chunking, inter-sentence playback gap (fix = server concurrent
  synthesis on GB10), a possible *thin* VoiceGateway. **Measure the network share of TTFA before
  any refactor.**
- Deeper, session-spanning context lives in the project memory files (`refelicia-tts-setup`,
  `gb10-migration-plan`).
