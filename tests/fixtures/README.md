# Test fixtures

## `reference-zh.wav`

A short Mandarin speech clip copied from the upstream
[MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) repository
(`assets/audio/zh_1.wav`, Apache-2.0, © OpenMOSS) — the same sample the project's
own documentation uses for voice cloning.

`scripts/e2e.mjs` uploads it through the Voices page to exercise the real
codec-encode path (reference audio → audio codes → stored voice → synthesis).

Override the fixture with `MOCKBIRD_REFERENCE_WAV=/path/to/clip.wav`.
