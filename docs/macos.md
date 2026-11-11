# Jak and Daxter on macOS

Use the Apple Silicon TAR.GZ installer with official OpenGOAL tooling **v0.3.9**. The launcher flow was checked with **v2.11.2**. Intel Macs are unsupported.

Install a supported local Jak and Daxter game first. Vanilla Jak II must also be installed in the same launcher for the cinematic donors. Then select the TAR.GZ in **Jak and Daxter > Features > Mods > Add from File**. Leave the archive packed. Setup includes Python and its dependencies.

The checked launcher requires TAR.GZ for local Mac imports. Use the Mac package; Windows executables and generated game code are platform-specific.

## Current validation

The current mouth correction passes native ARM64 model reconstruction. Its tooth weights match the Windows reference. Geometry, joint routing, morphs, materials and image bytes also match. Small face and brow weight differences remain between platforms. The original full builder produces the same Mac brow weights as the split builder. See [verification](verification.md) for the measured differences.

The 2026-10-31 mouth, brow and trouser motion results came from an earlier build. No new full player session was run for the source split. The current model checks and historical visual results are recorded separately.

## Developer checks

Use a Python 3.11 environment with the dependencies in `builder/requirements.txt`. Run from the repository root:

```sh
python tools/verify_mod_repository.py
TMPDIR=/private/tmp python tools/run_tests.py
python tools/build_mod_installer.py --platform macos-arm64 --output dist/input
```

The package helper obtains the pinned native runtime. Packaging verifies source hashes, dependency operations, the launcher entry point and executable permissions. Archive validation rejects links, special files and privileged modes.

See [building](building.md) for the Jak and Daxter conversion command and the separate developer and installer source lists. Conversion work and payload directories must stay outside the installed game.
