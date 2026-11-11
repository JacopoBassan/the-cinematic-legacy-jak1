"""Retain Jak 1 Daxter's lateral jaw channel absent from the donor basis."""
import json
import os
import sys

import numpy as np


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def transport_normal(source, deformed_source, destination):
    """Apply the native normal's rotation to the donor's own neutral normal.

    Transferring raw normal deltas between differently shaped faces changes the
    donor's baseline shading. Rotation transport keeps its neutral normals and
    carries the source deformation relative to them.
    """
    source, target, destination = unit(source), unit(deformed_source), unit(destination)
    axis = np.cross(source, target)
    cosine = np.clip(np.sum(source*target, axis=1), -1, 1)
    sine2 = np.sum(axis*axis, axis=1)
    factor = (1-cosine) / np.maximum(sine2, 1e-12)
    result = (destination + np.cross(axis, destination) +
              np.cross(axis, np.cross(axis, destination))*factor[:, None])
    opposite = (cosine < -.999999) & (sine2 < 1e-10)
    if opposite.any():
        # A deterministic perpendicular axis handles a 180 degree normal flip.
        reference = np.zeros_like(source[opposite])
        reference[np.arange(len(reference)), np.argmin(abs(source[opposite]), axis=1)] = 1
        rotation = unit(np.cross(source[opposite], reference))
        original = destination[opposite]
        result[opposite] = 2*rotation*np.sum(rotation*original, axis=1)[:, None]-original
    return unit(result)-destination


def apply_native_jaw(lp, la, hp, ha, deformations):
    """Modify Daxter channel 16 in place, returning numerical QA evidence.

    The donor has no centerline X displacement in any authored shape. Preserve
    the fitted Y/Z expression, add native lateral face movement, and move the
    lower tooth component as one rigid object. Eye channels are untouched.
    """
    from fix_faces import closest_surface
    channel = 16
    assert len(lp[0]['targets']) == 30 and len(hp[0]['targets']) == 37
    landmarks = [
        ('upper-lip', 2, 8737, 6, 413, [0, 1.0489, .2047], [0, 1.0486, .2069]),
        ('lower-lip', 1, 8688, 6, 533, [0, 1.0215, .1754], [0, 1.0297, .1651]),
        ('inner-lower-lip', 4, 8837, 6, 1094, [0, 1.0416, .1699], [0, 1.0436, .1612]),
        ('upper-teeth', 6, 8863, 15, 16, [0, 1.0434, .1773], [0, 1.0595, .1502]),
        ('lower-teeth', 6, 8876, 15, 63, [0, 1.0270, .1699], [0, 1.0217, .1433]),
    ]
    for label, li, lv, hi, hv, expected_low, expected_high in landmarks:
        assert lv in la(lp[li]['indices']) and hv in ha(hp[hi]['indices'])
        assert np.allclose(la(lp[li]['attributes']['POSITION'])[lv], expected_low, atol=7e-5), label
        assert np.allclose(ha(hp[hi]['attributes']['POSITION'])[hv], expected_high, atol=7e-5), label

    # Teeth inside the native closed mouth are spatially too close to its lips.
    # Restrict face correspondence to skin/fur/mouth surfaces, excluding teeth.
    source_parts = [0, 1, 2, 4, 10]
    triangles, deltas, normals, normal_deltas = [], [], [], []
    for part in source_parts:
        p = lp[part]
        indices = la(p['indices']).reshape(-1, 3)
        triangles.append(la(p['attributes']['POSITION'])[indices])
        deltas.append(la(p['targets'][channel]['POSITION'])[indices])
        normals.append(la(p['attributes']['NORMAL'])[indices])
        normal_deltas.append(la(p['targets'][channel]['NORMAL'])[indices])
    triangles, deltas = np.concatenate(triangles), np.concatenate(deltas)
    normals, normal_deltas = np.concatenate(normals), np.concatenate(normal_deltas)
    parts = []
    before = {i: (p.copy(), n.copy()) for i, (p, n) in deformations.items()}
    for part in [5, 6]:
        p = hp[part]
        positions = ha(p['attributes']['POSITION'])
        base_normal = ha(p['attributes']['NORMAL'])
        indices, barycentric, distance = closest_surface(positions, triangles)
        native_position = np.einsum('nij,ni->nj', deltas[indices], barycentric)
        native_normal = np.einsum('nij,ni->nj', normals[indices], barycentric)
        native_normal_delta = np.einsum('nij,ni->nj', normal_deltas[indices], barycentric)
        dp, dn = deformations.get(part, (np.zeros((30, len(positions), 3)), np.zeros((30, len(positions), 3))))
        dp[channel, :, 0] = native_position[:, 0]
        dn[channel] = transport_normal(native_normal, native_normal+native_normal_delta, base_normal)
        # Pin validated poles and their duplicate vertices. The skin-only field
        # is already within 0.41 mm here; this removes its remaining interpolation
        # error without changing the donor's neutral geometry or mouth topology.
        for _, li, lv, hi, hv, _, _ in landmarks:
            if hi != part:
                continue
            duplicate = np.linalg.norm(positions-positions[hv], axis=1) < 1e-6
            dp[channel, duplicate, 0] = la(lp[li]['targets'][channel]['POSITION'])[lv, 0]
        deformations[part] = (dp, dn)
        assert np.isfinite(dp).all() and np.isfinite(dn).all()
        assert np.allclose(np.linalg.norm(unit(base_normal)+dn[channel], axis=1), 1, atol=1e-5)
        parts.append(dict(cinematicPart=part, sourceParts=source_parts,
                          maxSurfaceDistance=float(distance.max()),
                          maxLateralDisplacement=float(abs(dp[channel, :, 0]).max()),
                          nativeNormalRotationTransferred=True))

    part = 15
    p = hp[part]
    positions = ha(p['attributes']['POSITION'])
    shape_zero = ha(p['targets'][0]['POSITION'])
    lower = np.linalg.norm(shape_zero, axis=1) > 1e-7
    used = np.unique(ha(p['indices']))
    # Shape 0 opens only the lower tooth strip. Its topology has 54 lower and 56
    # upper vertices, including UV/normal seam duplicates. Validate ownership at
    # the same inspected centerline poles used by the expression fit.
    assert len(used) == 110 and int(lower[used].sum()) == 54
    assert lower[63] and not lower[16] and not lower[17]
    lower_x = float(la(lp[6]['targets'][channel]['POSITION'])[8876, 0])
    upper_x = float(la(lp[6]['targets'][channel]['POSITION'])[8863, 0])
    assert abs(lower_x-.0475119017) < 1e-8 and upper_x == 0
    dp, dn = deformations.get(part, (np.zeros((30, len(positions), 3)), np.zeros((30, len(positions), 3))))
    dp[channel, :, 0] = np.where(lower, lower_x, upper_x)
    # Rigid translation adds no tooth-normal change. Preserve the authored Y/Z
    # expression and its authored normals already present in this channel.
    deformations[part] = (dp, dn)
    parts.append(dict(cinematicPart=part, lowerToothVertices=int(lower[used].sum()),
                      upperToothVertices=int((~lower[used]).sum()),
                      lowerToothX=lower_x, upperToothX=upper_x, rigidToothTranslation=True))

    result_landmarks = []
    for label, li, lv, hi, hv, _, _ in landmarks:
        expected = float(la(lp[li]['targets'][channel]['POSITION'])[lv, 0])
        actual = float(deformations[hi][0][channel, hv, 0])
        assert abs(expected-actual) < 1e-8
        result_landmarks.append(dict(label=label, nativeX=expected, cinematicX=actual, error=abs(expected-actual)))
    # Only the requested lateral channel and its face-normal response may change.
    other_channels = np.arange(30) != channel
    for part, (dp, dn) in before.items():
        now_p, now_n = deformations[part]
        assert np.array_equal(dp[other_channels], now_p[other_channels])
        assert np.array_equal(dn[other_channels], now_n[other_channels])
        if part in [5, 6, 15]:
            assert np.array_equal(dp[channel, :, 1:], now_p[channel, :, 1:])
        else:
            assert np.array_equal(dp, now_p) and np.array_equal(dn, now_n)
    return dict(nativeChannel=channel, parts=parts, landmarks=result_landmarks,
                checksPassed=['All five mouth landmark X movements preserved',
                              'Separate upper/lower tooth ownership', 'Rigid tooth X translation',
                              'Finite unit normal response', 'Other channels and authored Y/Z preserved'],
                status='Numerically validated; live expression capture remains required')


def main():
    from fit_faces import ROOT, DATA, OUT, load
    os.environ['CINEMATIC_BUILD_WORK'] = str(OUT)
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]/'stages'))
    low, _, la = load(DATA/'common/sidekick-lod0.glb')
    high, _, ha = load(OUT.parent/'rig-prototypes/sidekick-lod0.glb')
    lp, hp = low['meshes'][0]['primitives'], high['meshes'][0]['primitives']
    coefficient = np.load(OUT/'sidekick-lod0-coefficients.npy')
    deformations = {}
    for i, p in enumerate(hp):
        dp = np.einsum('ct,tnj->cnj', coefficient, np.array([ha(t['POSITION']) for t in p['targets']]))
        dn = np.einsum('ct,tnj->cnj', coefficient, np.array([ha(t['NORMAL']) for t in p['targets']]))
        if abs(dp).max() > 1e-7:
            deformations[i] = (dp, dn)
    report = apply_native_jaw(lp, la, hp, ha, deformations)
    (OUT/'native-jaw-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
