"""Asset-free checks for donor validation, scoped code packing and portability."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'builder'))
from jak1 import build_release as builder
from jak1 import eye_alignment


def archive(records):
    result = bytearray(struct.pack('<I', len(records))+b'archive'+b'\0'*53)
    assert len(result) == 64
    for name, payload in records:
        result += struct.pack('<I', len(payload))+name.encode().ljust(60, b'\0')
        result += payload+b'\xcc'*(-len(payload)%16)
    return bytes(result)


class BuilderTests(unittest.TestCase):
    def test_jak_eye_projection_uses_native_binocular_reference_for_live_gaze(self):
        # Synthetic eye surfaces contain no extracted game geometry. Their
        # inward/outward directions are known independently of UV calibration.
        high, low, data = [], [], {}
        x, y = np.meshgrid(np.linspace(-.5, .5, 9), np.linspace(-.5, .5, 9))
        normal = np.c_[x.ravel(), y.ravel(), np.sqrt(1-x*x-y*y).ravel()]
        triangles = []
        for row in range(8):
            for column in range(8):
                at = row*9+column
                triangles.extend([[at, at+1, at+9], [at+1, at+10, at+9]])
        indices = np.array(triangles)
        for side, center in enumerate([1., -1.]):
            position = normal*.05+[center, 2., 0.]
            authored = np.c_[.5+.7*normal[:, 0], .5-.75*normal[:, 1]]
            native = authored+[.2 if side == 0 else -.2, 0]
            for primitives, uv, prefix in [(high, authored, 'h'), (low, native, 'l')]:
                keys = {kind: f'{prefix}{side}-{kind}' for kind in ['POSITION', 'NORMAL', 'TEXCOORD_0']}
                data.update({keys['POSITION']: position, keys['NORMAL']: normal,
                             keys['TEXCOORD_0']: uv, f'{prefix}{side}-indices': indices})
                primitives.append(dict(attributes=keys, indices=f'{prefix}{side}-indices'))
        uv, report = eye_alignment.calibrated_jak_eye_uvs(
            low, data.__getitem__, high, data.__getitem__, {0: 0, 1: 1}, {0: 0, 1: 1})
        offsets = [-15/128, 15/128]
        self.assertEqual(report['nativeHorizontalIdleOffsets'], offsets)
        for sample in report['correctedSamples']:
            delta = np.array(sample['compositorOffset'])
            for side, center in enumerate([1., -1.]):
                control = sample['nativeCompositorControls'][side]
                np.testing.assert_allclose(control, delta+[offsets[side], 0])
                point = sample['eyes'][side]['position']
                # Horizontal pupils move in the same direction, from each
                # native reference, rather than converging during idle.
                self.assertAlmostEqual(point[0]-center, .05*delta[0]/.7, places=6)
                if sample['gaze'] == 'neutral':
                    np.testing.assert_allclose(sample['eyes'][side]['direction'], [0, 0, 1], atol=1e-6)
            self.assertLess(sample['binocularDirectionDifferenceDegrees'], .2)
        # Both gameplay idle and the opening intro use the same binocular X
        # references, with different authored vertical controls.
        for vertical_control in [6/128, 21/128]:
            for side, center in enumerate([1., -1.]):
                sample = eye_alignment.sample_eye(high[side], data.__getitem__, uv[side],
                    [.5+offsets[side], .5+vertical_control])
                self.assertAlmostEqual(sample['position'][0], center, places=6)
                self.assertAlmostEqual(sample['direction'][0], 0, places=6)
        for side in [0, 1]:
            self.assertGreaterEqual(float(uv[side].min()), 0)
            self.assertLessEqual(float(uv[side].max()), 1)
            # The fix changes only horizontal UVs, retaining the old V values.
            np.testing.assert_array_equal(uv[side][:, 1],
                (.5-report['sharedUVScale'][1]*normal[:, 1]).astype('<f4'))

    def test_missing_extraction_metadata_is_normalized_without_modifying_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root/'source'
            original.mkdir()
            (original/'SYSTEM.CNF').write_bytes(b'untouched input')
            profile = dict(inputs={'SYSTEM.CNF': 'already verified'}, elfHash=123456789)
            staged = builder.stage_disc(original, root/'staged', profile, 'SCUS-97265')
            self.assertEqual((staged/'SYSTEM.CNF').read_bytes(), b'untouched input')
            self.assertEqual(json.loads((staged/'buildinfo.json').read_text()),
                             [dict(serial='SCUS-97265', elf_hash=123456789)])
            self.assertFalse((original/'buildinfo.json').exists())

    def test_disc_flags_reach_both_compiler_and_make_without_editing_native_gp(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            project = root/'project'
            gp = project/'goal_src/jak1/game.gp'
            gp.parent.mkdir(parents=True)
            original = b';; upstream project\r\n(if *jak1-full-game* 1 0)\r\n'
            gp.write_bytes(original)
            for territory, full in [(1, True), (0, False)]:
                config = dict(targetSerial='selected', regionalProfiles={
                    'selected': dict(territory=territory, fullGame=full)})
                command = builder.compile_command(root, project, '(make "target")', config)
                flag = '#t' if full else '#f'
                self.assertIn(f'(set! *default-territory* {territory})', command[-1])
                self.assertIn(f'(set! *jak1-full-game* {flag})', command[-1])
                wrapper = (root/'regional-game.gp').read_bytes()
                self.assertTrue(wrapper.endswith(original))
                self.assertIn(f'(define *jak1-full-game* {flag})'.encode(), wrapper)
                self.assertEqual(gp.read_bytes(), original)
                self.assertTrue(json.loads((root/'compiler-edition.json').read_text())[
                    'nativeProjectBodyByteExact'])

    def test_archive_changes_only_the_bones_code_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = archive([('unchanged', b'GOAL\x01'),
                                ('bones', b'art record'),
                                ('bones', b'GOALnative')])
            (root/'native').write_bytes(original)
            (root/'new').write_bytes(b'GOALnew code')
            report = builder.replace_bones(root/'native', root/'new', root/'out')
            result = (root/'out').read_bytes()
            untouched_end = 64+80+80
            self.assertEqual(result[:untouched_end], original[:untouched_end])
            self.assertEqual(report['changedObjects'], ['bones'])
            self.assertTrue(report['unrelatedObjectRecordsByteExact'])

    def test_duplicate_code_or_truncation_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'new').write_bytes(b'GOALnew')
            for payload in [archive([('bones', b'GOALone'), ('bones', b'GOALtwo')]),
                            archive([('bones', b'GOALone')])[:-1]]:
                (root/'native').write_bytes(payload)
                with self.assertRaises(ValueError):
                    builder.replace_bones(root/'native', root/'new', root/'out')
                self.assertFalse((root/'out').exists())

    def test_modified_or_unsupported_donor_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for role, serial in [('target', 'SCUS_971.24'), ('donor', 'SCES_516.08')]:
                (root/role).mkdir()
                (root/role/'SYSTEM.CNF').write_text('BOOT2=cdrom0:\\'+serial+';1')
                (root/role/'input').write_bytes(role.encode())
            (root/'tool').write_bytes(b'tool')
            config = dict(game='jak1', openGoalVersion='test',
                sourceData=str(root), discData=str(root/'target'),
                donorData=str(root/'donor'), sourceTrees={},
                tools={'tool': builder.sha(root/'tool')},
                payloadPaths=builder.PAYLOAD_PATHS,
                regionalProfiles={'SCUS-97124': {'elfHash': 1, 'inputs': {'input': builder.sha(root/'target/input')}}},
                donorProfiles={'SCES-51608': {'elfHash': 2, 'inputs': {'input': builder.sha(root/'donor/input')}}})
            selected = builder.preflight(root, root, config)
            self.assertEqual(selected['donorSerial'], 'SCES-51608')
            (root/'donor/input').write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'modified donor'):
                builder.preflight(root, root, config)
            (root/'donor/SYSTEM.CNF').write_text('BOOT2=cdrom0:\\SCUS_000.00;1')
            with self.assertRaisesRegex(ValueError, 'Unsupported donor'):
                builder.preflight(root, root, config)

    def test_work_and_output_cannot_overlap_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, disc, donor, tools = [root/name for name in ['source', 'disc', 'donor', 'tools']]
            config = dict(sourceRoot=str(source), targetDisc=str(disc), donorDisc=str(donor))
            with patch.object(builder, 'preflight', return_value=config):
                for work, output in [(source/'work', root/'payload'),
                                     (root/'work', root/'work/payload')]:
                    with self.assertRaises(ValueError):
                        builder.build(source, tools, work, output, config)
                    self.assertFalse(output.exists())

    def test_source_closure_imports_after_relocation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'builder'
            for name in builder.SOURCE_CLOSURE:
                destination = root/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(builder.RELEASE/name, destination)
            env = dict(os.environ, PYTHONPATH=str(root),
                       CINEMATIC_JAK1_DATA=str(root.parent/'local-data'),
                       CINEMATIC_JAK1_WORK=str(root.parent/'local-work'),
                       CINEMATIC_BUILD_WORK=str(root.parent/'local-work'))
            code = ('from jak1 import build_release; '
                    'import sys; sys.path.insert(0, str(build_release.HERE)); '
                    'import build_prototypes, build_materials, fit_faces, build_faces, build_textures; '
                    'assert str(build_faces.DATA).endswith("local-data"); '
                    'assert str(build_faces.WORK).endswith("local-work")')
            result = subprocess.run([sys.executable, '-c', code], cwd=root.parent,
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
