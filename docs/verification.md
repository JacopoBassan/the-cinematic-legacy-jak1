# Jak and Daxter verification

Release `0.3.8` targets Windows x64, [Apple Silicon macOS](macos.md) and [experimental Linux x64](linux.md) with official OpenGOAL **v0.3.9**. The launcher protocol was checked with **v2.11.2**. Intel Macs are unsupported. Real Linux launcher import, model conversion and gameplay are not yet validated.

## Jak’s tongue and brows

On Windows PAL, Jak’s mouth and brows were checked against the same frames in the original Misty Island opening and one Geyser Rock power-cell celebration. The small fold behind his upper teeth belongs to the roof of the mouth. [In-game comparisons](comparisons/jak1.md#jak)

## Refreshed HUD

Refreshed HUD 1.0 texture preservation was verified with the supported PAL and USA discs on Windows using OpenGOAL v0.3.9. The same texture handling runs in the macOS and Linux installers.

## Source checks

The tooth correction tested on Apple Silicon on 2026-11-11 passes native ARM64 model reconstruction. Two Mac runs produced identical files. The full Windows comparison matches tooth weights, joint indices, geometry, topology, morphs, bind matrices, materials and image bytes. Two other float weight arrays differ across platforms: the maximum face difference is `0.0000001192`; the maximum brow difference is `0.0001001358`. The original full builder produces exactly the same Mac brow weights as the split builder. This numerical difference predates the source split and tooth correction.

On 2026-11-11, the game-specific suite completed 60 tests on Apple Silicon: 56 passed and four Windows-only cases were skipped. Twelve portable filesystem fixtures also passed on this Mac.

On Windows, all 60 tests and 13 installer filesystem checks passed on 2026-11-15. The repository verifier also passed for all 52 source files, 42 runtime files and 10 comparisons.

Coverage includes wrong-game inputs, corrupt packages, source inventories, staged writes, rollback, interrupted transactions, hardlinks and lock ownership. The Windows checks also cover junctions, paths and child-process cleanup. These source checks are separate from player testing.

[`builder/source-files.json`](../builder/source-files.json) separates the complete `sourceFiles` developer ZIP from the smaller `runtimeFiles` installer source set. Package checks verify each manifest against its own file set. They also check pinned dependencies, runtime versions, Windows adapter provenance and Mac/Linux executable modes. Game models and textures are excluded from distributable source and installer archives.

The latest model corrections were checked on Windows. Mac model and player checks below apply to their dated builds; the latest clothing, cloth-fit and Samos corrections have not had a new native Mac model conversion or player session.

## Linux package and installer checks

Original input checks passed for the supported EU and USA discs with the pinned Linux x64 tools in Ubuntu 24.04 under Rosetta. The required native input fingerprints are recorded in [the Linux profile](../builder/platforms/linux-x64.json). These checks did not run cinematic conversion or gameplay.

The Linux workflow uses Ubuntu 24.04 x64. It builds the final TAR.GZ and checks archive hashes, executable modes, source files, the bundled Python runtime and native dependency operations.

Launcher builds prefer the source shipped with the pinned OpenGOAL tools. The launcher merges source files into the active game folder, where older files can remain. The builder checks the selected source, the player’s disc and generated game files.

Asset-free installer fixtures cover first install, updates, input and save preservation, wrong-game and running-game refusal, cancellation, rollback, child-process cleanup and retry. They use synthetic files and do not drive the real launcher or convert real game assets. They also check transaction marker ownership and inherited runtime settings. The workflow stores a `verification-linux-x64` receipt.

Release assembly passed with both builder and standard archive names. Checks covered one download per platform, metadata, source files and release checksums. Missing Linux input and a changed Linux runtime pin were rejected before output was written. These archive fixtures use synthetic runtime files.

## Model checks

The builder produces replacements for Jak, Daxter, Samos and Keira. It checks geometry, facial buffers, textures, material flags and the scoped bones-object change in GAME.CGO.

For both discs, the builder uses OpenGOAL’s `buildinfo.json` to select the revision. For older extractions, it reads the `BOOT2` entry in `SYSTEM.CNF`. It checks the original tools, selected executables and required archives before conversion. It validates the generated model data and records payload checksums.

Complete Windows EU and USA Black Label conversions passed on 2026-11-15 with the clothing, cloth-fit and Samos corrections. All four generated character prototypes match between regions. GAME.fr3 decodes identically, and the village character models, vertex and index data, morph buffers and textures match. Regional level data outside those character sections can differ. USA gameplay was not tested.

Windows EU checks covered sampled opening-cinematic frames, matched character views, facial expressions and representative idle, walk, run, jump, landing and turning motion. The complete generated character data matches the data inspected in those captures. The Samos socket correction preserves eye controls and facial animation buffers; its forced blink screenshot did not independently demonstrate a blink.

## Historical player coverage

These checks were recorded before the 2026-11-11 repository split.

- **Windows:** Earlier Windows EU checks covered village and beach loading, intro rendering, gaze, blinking, and matched jump, attack, power-up, uppercut and swim poses. USA conversion and model equivalence were checked offline.
- **Apple Silicon:** On 2026-10-31, an earlier Apple Silicon build was checked in the village and intro. Power-up, uppercut and landing motion showed mouth, brow and trouser deformation. Matched original and mod power-up frames were also checked. These results apply to that earlier build.

## Comparisons and image policy

The [gallery](comparisons/jak1.md) shows the original and cinematic models side by side. The [comparison manifest](comparisons/manifest.json) records file hashes, capture details, and the source revision for each converted image.

Refreshed comparison images use a content hash in the filename and plain relative Markdown links for GitHub mobile compatibility.

Use real captures for comparisons and launcher thumbnails. Keep the same framing and scale for both panels. The converted gallery images use progressive JPEG at quality 94 with full color resolution. Keep launcher thumbnails lossless. Do not use image generation, redraw details, retouch colors or upscale captures. Thumbnail source hashes and crop rectangles are recorded in [`mod-source-provenance.json`](../mod-source-provenance.json).

## Limits

- Shadows retain the original gameplay hulls and can differ from cinematic contours.
- NPCs keep gameplay expression and gaze behavior. The mod adds no gaze-tracking AI or cinematic dialogue lip-sync.
- Player checks cover representative scenes and animations, not a full campaign or every NPC.
- A full campaign has not been tested. Fine texture seams and animated metal reflections were not fully verified by the representative movement captures.
- The latest Windows EU checks and USA conversion are recorded separately from the earlier Mac player coverage.
