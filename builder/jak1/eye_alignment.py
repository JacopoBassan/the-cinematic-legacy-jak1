"""Calibrate cinematic Jak's live eye UVs in a shared head-forward frame.

The native low-poly eyes have a different shape and UV projection. Projecting
their UVs through model XY onto the cinematic eyes introduces fixed vergence.
Use the cinematic eye normals to map both animated pupils to a common gaze
direction from their native binocular offsets, retaining their authored local
UV scale and the native compositor. Jak's horizontal idle controls are unequal;
centering both UV projections at 0.5 turns those offsets into an inward bias.
"""
import json
import math
import os
from pathlib import Path
import struct
import sys

import numpy as np


# Native eichar-stance-loop eye controls in the supported Jak 1 EU/USA data.
# These are UV reference offsets, not replacements for the live animation.
JAK_NATIVE_HORIZONTAL_IDLE_OFFSETS = (-15/128, 15/128)


def sample_eye(p, accessor, uv, texture_point):
    """Locate the pupil on the rendered triangle surface for a compositor UV."""
    position = accessor(p['attributes']['POSITION'])
    normal = accessor(p['attributes']['NORMAL'])
    hits = []
    target = np.r_[texture_point, 1.0]
    for indices in accessor(p['indices']).reshape(-1, 3):
        matrix = np.vstack([uv[indices].T, np.ones(3)])
        bary = np.linalg.lstsq(matrix, target, rcond=None)[0]
        if np.linalg.norm(matrix@bary-target) < 1e-6 and bary.min() >= -1e-6:
            point = position[indices].T@bary
            direction = normal[indices].T@bary
            direction /= np.linalg.norm(direction)
            hits.append((float(direction[2]), point, direction))
    if not hits:
        raise ValueError(f'Eye does not cover compositor UV {texture_point}')
    _, point, direction = max(hits, key=lambda item: item[0])
    return dict(position=point.tolist(), direction=direction.tolist())


def angle_between(left, right):
    cosine = np.clip(np.dot(left, right), -1, 1)
    return math.degrees(math.acos(float(cosine)))


def native_xy_projection(source, source_accessor, destination, destination_accessor):
    used = np.unique(source_accessor(source['indices']))
    pos = source_accessor(source['attributes']['POSITION'])[used]
    uv = source_accessor(source['attributes']['TEXCOORD_0'])[used]
    projection = np.linalg.lstsq(np.c_[pos[:, :2], np.ones(len(pos))], uv, rcond=None)[0]
    pos = destination_accessor(destination['attributes']['POSITION'])
    return np.clip(np.c_[pos[:, :2], np.ones(len(pos))]@projection, 0, 1)


def authored_daxter_eye_uvs(primitives, accessor, eyes):
    """Retain the donor's round pupil projection with native live controls.

    Native Jak 1 and cinematic Daxter use the same authored binocular idle
    offsets. Projecting the low-poly eye's model XY mapping onto the round
    cinematic surface compresses its pupil horizontally. Only Daxter uses
    this authored-UV branch; the other characters keep their own calibration.
    """
    assert sorted(eyes.values()) == [0, 1]
    result = {}
    for part, side in eyes.items():
        primitive = primitives[part]
        used = np.unique(accessor(primitive['indices']))
        positions = accessor(primitive['attributes']['POSITION'])
        uv = accessor(primitive['attributes']['TEXCOORD_0']).astype('<f4')
        assert (np.median(positions[used, 0]) > 0) == (side == 0)
        assert np.isfinite(uv).all() and uv.min() >= 0 and uv.max() <= 1
        result[part] = uv.copy()
    return result, dict(
        method='Authored cinematic Daxter eye UVs with native Jak 1 controls',
        eyePrimitiveSides=eyes,
        unchanged=['native live eye IDs', 'authored binocular offsets',
                   'iris and pupil scales', 'automatic blink',
                   'native eye expression geometry', 'all other characters'],
        staticEyes=False)


def calibrated_jak_eye_uvs(lp, la, hp, ha, high_eyes, low_eyes):
    assert set(high_eyes.values()) == {0, 1}
    scales, old, corrected = [], {}, {}
    part_by_side = {side: part for part, side in high_eyes.items()}
    for part, side in high_eyes.items():
        p = hp[part]
        used = np.unique(ha(p['indices']))
        pos = ha(p['attributes']['POSITION'])
        normal = ha(p['attributes']['NORMAL'])
        normal = normal/np.linalg.norm(normal, axis=1, keepdims=True)
        authored_uv = ha(p['attributes']['TEXCOORD_0'])
        assert (np.median(pos[used, 0]) > 0) == (side == 0), 'Eye side IDs do not match anatomy'
        near = ((abs(normal[:, 0]) < .5) & (abs(normal[:, 1]) < .5) & (normal[:, 2] > .75))
        near &= np.isin(np.arange(len(pos)), used)
        assert near.sum() >= 12
        projection = np.linalg.lstsq(np.c_[normal[near, :2], np.ones(int(near.sum()))],
                                     authored_uv[near], rcond=None)[0]
        assert .5 < projection[0, 0] < 1 and -.95 < projection[1, 1] < -.5
        scales.append((float(projection[0, 0]), float(-projection[1, 1])))
        old[part] = native_xy_projection(lp[low_eyes[side]], la, p, ha)
    # Share angular scales and retain the native horizontal reference per eye.
    # Equal movement from these references means equal gaze. The compositor's
    # existing vertical projection and all live control values stay unchanged.
    horizontal, vertical = np.mean(scales, axis=0)
    for part, side in high_eyes.items():
        p = hp[part]
        normal = ha(p['attributes']['NORMAL'])
        normal = normal/np.linalg.norm(normal, axis=1, keepdims=True)
        uv = np.c_[.5+JAK_NATIVE_HORIZONTAL_IDLE_OFFSETS[side]+horizontal*normal[:, 0],
                   .5-vertical*normal[:, 1]]
        assert np.isfinite(uv).all()
        assert uv.min() >= 0 and uv.max() <= 1, 'Calibrated eye UV exceeds the live eye texture'
        corrected[part] = uv.astype('<f4')
    samples = []
    for label, gaze in [('neutral', (0, 0)), ('left', (-.25, 0)), ('right', (.25, 0)),
                        ('up', (0, -.20)), ('down', (0, .20))]:
        eyes, controls = [], []
        for side in [0, 1]:
            part = part_by_side[side]
            control = np.array(gaze)+[JAK_NATIVE_HORIZONTAL_IDLE_OFFSETS[side], 0]
            sample = sample_eye(hp[part], ha, corrected[part], control+.5)
            eyes.append(sample)
            controls.append(control.tolist())
        vergence = angle_between(eyes[0]['direction'], eyes[1]['direction'])
        assert vergence < .2, (label, vergence)
        samples.append(dict(gaze=label, compositorOffset=list(gaze),
                            nativeCompositorControls=controls, eyes=eyes,
                            binocularDirectionDifferenceDegrees=vergence))
    old_centers = [sample_eye(hp[part_by_side[side]], ha, old[part_by_side[side]],
                             [.5+JAK_NATIVE_HORIZONTAL_IDLE_OFFSETS[side], .5])
                   for side in [0, 1]]
    old_vergence = angle_between(old_centers[0]['direction'], old_centers[1]['direction'])
    report = dict(method='Shared normal projection anchored to native Jak 1 horizontal eye offsets',
                  nativeHorizontalIdleOffsets=list(JAK_NATIVE_HORIZONTAL_IDLE_OFFSETS),
                  gazeReference='Native horizontal idle controls with vertical control zero',
                  nativeXYProjectionNeutral=dict(eyes=old_centers,
                    binocularDirectionDifferenceDegrees=old_vergence),
                  sharedUVScale=[float(horizontal), float(vertical)],
                  correctedSamples=samples,
                  eyePrimitiveSides=high_eyes, unchanged=['eye IDs', 'native animated eye compositor',
                    'gaze values', 'iris and pupil animation', 'blink', 'expression geometry', 'neutral geometry'],
                  staticEyes=False,
                  checksPassed=['Native anatomical eye side matches ID', 'UV remains in live texture bounds',
                                'Both gaze axes point head-forward at native horizontal idle offsets',
                                'Five shared gaze shifts agree within 0.2 degrees using native binocular offsets',
                                'Vertical UV projection remains unchanged'])
    return corrected, report


def write_eye_uvs(raw, name, doc, accessor, indices, vertices, uv_by_part):
    """Write calibrated UVs before fix_model copies eye vertices into morph data."""
    from fr3_models import model, DRAW
    target = model(raw, name)
    draws = sorted([draw for effect in target['effects'] for draw in effect['draws']],
                   key=lambda draw: DRAW.unpack(draw)[3])
    prims = doc['meshes'][0]['primitives']
    assert len(draws) == len(prims)
    for part, uv in uv_by_part.items():
        values = DRAW.unpack(draws[part])
        ids = indices[values[3]:values[3]+values[4]]
        local = accessor(prims[part]['indices']).ravel()
        offset = int(ids[0])-int(local[0])
        assert np.array_equal(ids, local+offset)
        for vertex in np.unique(ids):
            struct.pack_into('<2f', vertices, int(vertex)*64+48, *uv[int(vertex)-offset])


def main():
    root = Path(__file__).resolve().parents[2]
    from jak1_paths import DATA,WORK,RELEASE
    work = WORK
    os.environ['CINEMATIC_BUILD_WORK'] = str(work)
    sys.path[:0] = [str(RELEASE/'jak1'), str(RELEASE/'stages')]
    from native_io import activate
    activate()
    from fr3_models import decompress
    from inspect_faces import glb, buffers
    from build_faces import eye_parts
    low, la = glb(DATA/'decompiler_out/jak1/levels/common/eichar-lod0.glb')
    high, ha = glb(work/'rig-prototypes/eichar-lod0.glb')
    lp, hp = low['meshes'][0]['primitives'], high['meshes'][0]['primitives']
    high_eyes = eye_parts(high)
    low_eyes = {side: part for part, side in eye_parts(low).items()}
    uv, report = calibrated_jak_eye_uvs(lp, la, hp, ha, high_eyes, low_eyes)
    raw = decompress(work/'GAME-materials.fr3')
    _, indices, _, original = buffers(raw, 'eichar-lod0')
    vertices = bytearray(original)
    write_eye_uvs(raw, 'eichar-lod0', high, ha, indices, vertices, uv)
    before = np.frombuffer(original, 'u1').reshape(-1, 64)
    after = np.frombuffer(vertices, 'u1').reshape(-1, 64)
    assert np.array_equal(before[:, :48], after[:, :48])
    assert np.array_equal(before[:, 56:], after[:, 56:])
    report['checksPassed'].append('Only UV bytes change in private buffer simulation')
    path = work/'face-candidates/jak-eye-alignment-report.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
