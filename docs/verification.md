# Jak and Daxter verification

Candidate `0.3.0-rc.1` targets Windows x64 and [Apple Silicon macOS](macos.md) with official OpenGOAL **v0.3.9**. The launcher protocol was checked with **v2.11.2**. Intel Macs and Linux are unsupported.

## Current source checks, 2026-11-11

The current mouth correction passes a fresh native ARM64 model reconstruction. Two Mac runs produced identical files. The full Windows comparison matches tooth weights, joint indices, geometry, topology, morphs, bind matrices, materials and image bytes. Two other float weight arrays differ across platforms: the maximum face difference is `0.0000001192`; the maximum brow difference is `0.0001001358`. The original full builder produces exactly the same Mac brow weights as the split builder. This numerical difference predates the source split and tooth correction.

The game-specific suite completed 60 tests on Apple Silicon: 56 passed and four Windows-only cases were skipped. Twelve portable filesystem fixtures also passed on this Mac. All 13 Windows filesystem fixtures remain configured for native Windows checks.

Coverage includes wrong-game inputs, corrupt packages, source inventories, staged writes, rollback, interrupted transactions, hardlinks and lock ownership. Native Windows junction, path and child-process checks require the Windows run. These source checks are separate from player testing.

[`builder/source-files.json`](../builder/source-files.json) separates the complete `sourceFiles` developer ZIP from the smaller `runtimeFiles` installer source set. Package checks verify each manifest against its own file set. They also check pinned dependencies, runtime versions, Windows adapter provenance and Mac executable modes. Game models and textures are excluded from distributable source and installer archives.

The fresh per-game layout has not had a new full player session. Historical coverage follows.

## Model checks

The builder produces replacements for Jak, Daxter, Samos and Keira. It checks geometry, facial buffers, textures, material flags and the scoped bones-object change in GAME.CGO.

The builder checks original tools, disc revisions and consumed assets before conversion. It validates the generated model data and records payload checksums. Earlier conversions covered the supported EU and USA profiles, including the Jak 1 Black Label edition setting.

## Historical player coverage

These checks were recorded before the 2026-11-11 repository split.

- **Windows:** Earlier Windows EU checks covered village and beach loading, intro rendering, gaze, blinking, and matched jump, attack, power-up, uppercut and swim poses. USA conversion and model equivalence were checked offline.
- **Apple Silicon:** On 2026-10-31, an earlier Apple Silicon build was checked in the village and intro. Power-up, uppercut and landing motion showed mouth, brow and trouser deformation. Matched original and mod power-up frames were also checked. These results apply to that earlier build.

## Comparisons and image policy

The [gallery](comparisons/jak1.md) contains 10 matched comparisons for Jak and Daxter. Lossless compression checks preserved the decoded RGB pixels and image dimensions. The [comparison manifest](comparisons/manifest.json) records file hashes and capture details.

Refreshed comparison images use a content hash in the filename and plain relative Markdown links for GitHub mobile compatibility.

Before-and-after images, including launcher thumbnails, must use unchanged screenshot pixels. Only cropping, placement and lossless compression are allowed. Do not use image generation, redraw details, retouch colours or upscale captures. Thumbnail source hashes and crop rectangles are recorded in [`mod-source-provenance.json`](../mod-source-provenance.json).

## Limits

- Shadows retain the original gameplay hulls and can differ from cinematic contours.
- NPCs keep gameplay expression and gaze behavior. The mod adds no gaze-tracking AI or cinematic dialogue lip-sync.
- Player checks cover representative scenes and animations, not a full campaign or every NPC.
- The source split and mouth correction were checked with real model data. A new full player session was not run. The historical visual checks remain limited to their recorded builds.
