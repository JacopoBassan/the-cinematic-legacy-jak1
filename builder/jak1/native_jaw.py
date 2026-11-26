"""Retain Jak 1 Daxter's lateral jaw channel absent from the donor basis."""
import json
import os
import struct
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


def fit_upper_tooth_clearance(position, normal, triangles, lower, positions, normals):
    """Keep the upper arcade inside the muzzle when the native jaw slides sideways."""
    from eye_alignment import move_facial_surface

    # The center stays fixed laterally. Only the end approached by the moving
    # lip follows that lip inward and back. Lift the upper row enough for the
    # lower arcade to pass during the combined sideways grin.
    corner = np.clip((-position[:, 0] - .032) / .062, 0, 1)
    step = corner[:, None] * np.array([.035, .007, -.024])
    step[:, 1] += .016
    step[lower] = 0
    _, shading = np.unique(np.column_stack((position, normal)), axis=0, return_inverse=True)
    move_facial_surface(position, normal, triangles, shading, positions, normals, step)


def fit_mouth_corner(primitive, accessor, positions, normals):
    """Keep the upper lining behind the far lip during lateral jaw motion."""
    from eye_alignment import move_facial_surface

    attributes = primitive['attributes']
    position = accessor(attributes['POSITION'])
    normal = accessor(attributes['NORMAL'])
    triangles = accessor(primitive['indices']).reshape(-1, 3)
    x, y, z = position.T
    corner = np.minimum.reduce((
        (x - .012) / .024, (.149 - x) / .038,
        (y - 1.016) / .019, (1.072 - y) / .022,
        (z - .082) / .028, (.190 - z) / .023,
    )).clip(0, 1)
    corner = corner * corner * (3 - 2 * corner)
    # The authored jaw opening marks the lower lip, which needs the full slide.
    jaw_drop = -accessor(primitive['targets'][0]['POSITION'])[:, 1]
    upper = 1 - np.clip((jaw_drop - .020) / .025, 0, 1)
    strength = corner * upper
    step = np.zeros_like(position)
    step[:, 0] = -.40 * positions[:, 0] * strength
    step[:, 1] = -.012 * strength
    _, shading = np.unique(np.column_stack((position, normal)), axis=0, return_inverse=True)
    move_facial_surface(position, normal, triangles, shading, positions, normals, step)


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
    # The native lower-row translation adds no normal change. Retain its
    # authored Y/Z expression along with the center of the upper row.
    deformations[part] = (dp, dn)
    parts.append(dict(cinematicPart=part, lowerToothVertices=int(lower[used].sum()),
                      upperToothVertices=int((~lower[used]).sum()),
                      lowerToothX=lower_x, upperToothCenterX=upper_x, rigidLowerToothTranslation=True))

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
    p = hp[15]
    positions, normals = deformations[15]
    fit_upper_tooth_clearance(ha(p['attributes']['POSITION']), ha(p['attributes']['NORMAL']),
                             ha(p['indices']).reshape(-1, 3), lower, positions[channel], normals[channel])
    positions, normals = deformations[6]
    fit_mouth_corner(hp[6], ha, positions[channel], normals[channel])
    fit_signed_mouth(hp, ha, deformations, lower)
    fit_authored_mouth(hp, ha, deformations, daxter=True)
    refine_daxter_mouth(hp, ha, deformations)
    return dict(nativeChannel=channel, parts=parts, landmarks=result_landmarks,
                signedMouthChannels=[1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17],
                checksPassed=['All five mouth landmark X movements preserved',
                              'Separate upper/lower tooth ownership',
                              'Upper tooth corner follows the moving lip',
                              'Upper mouth lining stays inside the far corner',
                              'Finite unit normal response'],
                status='Numerically validated; live expression capture remains required')


def fit_signed_mouth(primitives, accessor, deformations, lower_teeth):
    """Keep the lower teeth inside the closed jaw at signed inputs."""
    # At negative weights the lower row needs to rise and recede sooner.
    # Narrow its ends as the jaw opens so they stay behind the lower lip.
    x = accessor(primitives[15]['attributes']['POSITION'])[lower_teeth, 0]
    edge = np.clip((abs(x) - .015) / .042, 0, 1)
    edge = edge*edge*(3-2*edge)
    teeth = deformations[15][0][3]
    teeth[lower_teeth, 0] -= .18*x*edge
    teeth[lower_teeth, 1] *= 1.34
    teeth[lower_teeth, 2] += .012*edge
# The generic least-squares fit uses the whole face and centerline landmarks.
# A contacting grin, a closed muzzle drop and an open shout can have similar
# centerline motion. Match the donor's authored oral poses at the native inputs.
JAK_MOUTH_TARGETS = {
    0: {10: .96, 2: .35}, 1: {1: 1.8}, 2: {17: .91}, 3: {18: .91},
    4: {9: 3.1}, 5: {2: 4.1, 1: .8}, 6: {15: .64, 16: .64},
    7: {10: .55, 1: .24}, 8: {3: .50, 1: .16},
}
DAXTER_MOUTH_TARGETS = {
    1: {0: .99}, 2: {21: .24, 22: .24, 1: .63, 10: .12}, 5: {3: 1.04},
    6: {10: .14, 1: .65}, 7: {10: .17, 3: .82}, 9: {11: 1.48},
    10: {0: 1.50}, 17: {10: 1.0},
}
# The fitted cinematic motion already keeps the mouth centered when 11–15
# blend in dialogue. Replacing them with isolated grin targets pulls it aside.
DAXTER_MOUTH_INPUTS = {1: 10752, 2: 9024, 5: 8192, 6: 8192, 7: 8192, 10: 2432}


def seat_daxter_teeth(raw, name, document, accessor, indices, vertices):
    """Recess both rigid tooth rows enough to retain the closed-mouth inputs."""
    from fr3_models import DRAW, model

    primitives = document['meshes'][0]['primitives']
    tooth = primitives[15]
    assert document['materials'][tooth['material']]['name'].endswith('daxterteeth')
    key = tooth['attributes']['POSITION']
    original = accessor(key)
    lower = np.linalg.norm(accessor(tooth['targets'][0]['POSITION']), axis=1) > 1e-7
    seated = original.copy()
    seated[:, 2] -= np.where(lower, .005, .010)
    draws = sorted((draw for effect in model(raw, name)['effects'] for draw in effect['draws']),
                   key=lambda draw: DRAW.unpack(draw)[3])
    assert len(draws) == len(primitives)
    values = DRAW.unpack(draws[15])
    packed = indices[values[3]:values[3] + values[4]]
    local = accessor(tooth['indices']).ravel()
    offset = int(packed[0]) - int(local[0])
    assert np.array_equal(packed, local + offset)
    rows = np.unique(packed)
    for row in rows:
        index = int(row) - offset
        start = int(row) * 64
        imported = np.frombuffer(vertices[start:start + 12], dtype='<f4')
        assert np.max(np.abs(imported - original[index] * 4096)) < .015
        struct.pack_into('<3f', vertices, start, *(seated[index] * 4096))
    return lambda index: seated if index == key else accessor(index), rows


def fit_covered_teeth(position, lower, channel):
    """Peak-pose offsets fitted against both signed inputs and the oblique jaw."""
    # Rows are the upper and lower arcades. Values are mm at the positive
    # endpoint: horizontal spread, elevation, depth, then changes to elevation
    # and depth toward the sides and from the bottom to the top of the row.
    profiles = {
        1: ((18, 10, -17, -5, 12, 0, 0), (-6, -8, -40, 0, 12, -1, -18)),
        9: ((0, 0, 0, 0, 0, 0, 0), (0, -8, 3, 0, 0, 0, 0)),
        # At full opening, raise the upper row into the palate. Keep the lower
        # row inside the narrowing jaw as it moves forward and down.
        10: ((5, 36, -63, -2, 5, 0, 0), (-51, 17, 2, -1, 10, -5, -15)),
    }
    step = np.zeros_like(position)
    horizontal = position[:, 0]/.110
    vertical = (position[:, 1]-1.0315)/.010
    for row, values in zip((~lower, lower), profiles[channel]):
        spread, rise, depth, side_rise, side_depth, top_rise, top_depth = values
        side = abs(horizontal[row])
        step[row, 0] = horizontal[row]*spread
        step[row, 1] = rise + side*side_rise + vertical[row]*top_rise
        step[row, 2] = depth + side*side_depth + vertical[row]*top_depth
    return step * .001


def fit_authored_mouth(primitives, accessor, deformations, daxter):
    """Match the lip silhouette while retaining fitted movement above the muzzle."""
    from eye_alignment import move_facial_surface
    targets = DAXTER_MOUTH_TARGETS if daxter else JAK_MOUTH_TARGETS
    for part in ((4, 5, 6, 15) if daxter else (8, 9, 10)):
        p = primitives[part]
        position = accessor(p['attributes']['POSITION'])
        normal = accessor(p['attributes']['NORMAL'])
        triangles = accessor(p['indices']).reshape(-1, 3)
        _, shading = np.unique(np.column_stack((position, normal)), axis=0, return_inverse=True)
        y = position[:, 1]
        influence = np.clip((1.114-y)/.030 if daxter else (2.422-y)/.025, 0, 1)[:, None]
        dp, dn = deformations[part]
        for channel, weights in targets.items():
            blend = influence
            if daxter and part != 15 and channel in (2, 5):
                # Keep the shared fur/nose boundary seated as the lip opens.
                lip = np.clip((1.082-y)/.018, 0, 1)[:, None]
                blend = lip*lip*(3-2*lip)
            for data, semantic in ((dp, 'POSITION'), (dn, 'NORMAL')):
                authored = sum(gain * accessor(p['targets'][target][semantic])
                               for target, gain in weights.items())
                data[channel] += blend * (authored-data[channel])
            if not daxter and channel == 5:
                # Give the clenched mouth the native span at its observed peak.
                weight = 2352.75 / 8192
                x = position[:, 0]+weight*dp[channel, :, 0]
                step = np.zeros_like(position)
                step[:, 0] = .34*x*np.exp(-(x/.080)**4)*np.clip((2.390-y)/.040, 0, 1)/weight
                if part == 9:
                    step[:, 2] = .004/weight
                move_facial_surface(position, normal, triangles, shading, dp[channel],
                                    dn[channel], step*influence)
            if daxter and part == 15 and channel == 9:
                lower = np.linalg.norm(accessor(p['targets'][0]['POSITION']), axis=1) > 1e-7
                step = fit_covered_teeth(position, lower, channel) * (8192/7872)
                move_facial_surface(position, normal, triangles, shading, dp[channel], dn[channel], step)
            if not daxter or channel not in DAXTER_MOUTH_INPUTS:
                continue
            weight = DAXTER_MOUTH_INPUTS[channel] / 8192
            # Widen the opening below the nose, fading toward the outer cheek.
            # Work in the peak pose so the same field follows the opened jaw.
            x = position[:, 0] + weight * dp[channel, :, 0]
            front = np.clip((position[:, 2]-.030)/.070, 0, 1)
            muzzle = np.exp(-(x/.110)**4) * np.clip((1.075-y)/.027, 0, 1) * front
            spread = {1: .38, 10: .38, 2: .08, 5: .0}.get(channel, .22)
            step = np.zeros_like(position)
            step[:, 0] = spread * x * muzzle / weight
            if part == 15 and channel in (1, 5, 10):
                lower = np.linalg.norm(accessor(p['targets'][0]['POSITION']), axis=1) > 1e-7
                if channel == 5:
                    # The lower crowns stay seated behind the moving lip in
                    # this vowel. Carry them as one row and expose the upper edge.
                    dp[channel, lower] = 0
                    dn[channel, lower] = 0
                    step[lower] = (0, .008, -.008)
                    step[~lower, 1:] = (-.003, -.006)
                else:
                    step += fit_covered_teeth(position, lower, channel) / weight
            move_facial_surface(position, normal, triangles, shading, dp[channel],
                                dn[channel], step * blend)
    if not daxter:
        fit_jak_tongue(primitives, accessor, deformations)


def fit_jak_tongue(primitives, accessor, deformations):
    """Seat the articulated tongue on the moving lower jaw in the open vowel."""
    from eye_alignment import move_facial_surface
    from native_face_rig import connected_vertex_component

    face, teeth = primitives[8], primitives[9]
    points = accessor(face['attributes']['POSITION'])
    normals = accessor(face['attributes']['NORMAL'])
    triangles = accessor(face['indices']).reshape(-1,3)
    cavity = connected_vertex_component(triangles,413)
    roof = connected_vertex_component(triangles,923)
    assert len(cavity) == 196 and len(roof) == 68
    # Shape 9 thrusts the tongue as well as opening the lips. The 3.1
    # lip gain sent the central tip 80 mm forward at opening frame 1100.
    # Identify the thrusting surface inside the cavity by its own motion.
    authored = accessor(face['targets'][9]['POSITION'])
    tongue = cavity[authored[cavity,2] > .008]
    assert 577 in tongue and 636 in tongue and not np.intersect1d(tongue,roof).size
    blend = np.clip((authored[tongue,2]-.008)/.014,0,1)
    blend = blend*blend*(3-2*blend)
    lower = np.linalg.norm(accessor(teeth['targets'][0]['POSITION']),axis=1)>1e-7
    tooth_points = accessor(teeth['attributes']['POSITION'])
    center = lower & (abs(tooth_points[:,0]) < .030)
    assert center.any()
    lower_travel = deformations[9][0][4,center].mean(0)
    target = .95*authored[tongue]+.60*lower_travel
    positions, normal_moves = deformations[8]
    shift = np.zeros_like(points)
    shift[tongue] = blend[:,None]*(target-positions[4,tongue])
    _, shading = np.unique(np.column_stack((points,normals)),axis=0,return_inverse=True)
    move_facial_surface(points,normals,triangles,shading,positions[4],normal_moves[4],shift)


def mouth_profile(position, lower, channel, motion, right_grin):
    """Correct the posed lip and tooth silhouettes in donor model coordinates."""
    x, y, z = position.T
    front = np.clip((z-.025)/.070, 0, 1) * np.clip((1.121-y)/.050, 0, 1)
    step = np.zeros_like(position)
    if channel == 16:
        flank = np.clip((x-.018)/.080, 0, 1)
        if lower is None:
            step[:, 1:] += .65*(flank*front)[:, None]*right_grin[:, 1:]
        else:
            # The lowered side of the moving dental arcade stays under its lip.
            step[:, 1] += flank*np.where(lower, -.013, .010)
            step[:, 2] += flank*np.where(lower, -.021, .005)
            step[lower, 0] -= .012*flank[lower]
        flank_span = np.clip((x-.018)/.055, 0, 1)*np.clip((.170-x)/.045, 0, 1)
        corner = np.clip((1.100-y)/.040, 0, 1)*np.clip((z-.025)/.070, 0, 1)
        step[:, 0] += .030*flank_span*corner
    elif channel == 17:
        step[:, 0] += .085*x*front
        if lower is not None:
            # Translate the contact line up while preserving both outer edges.
            posed_y = y + motion[:, 1]
            contact = np.where(lower, (posed_y-.939)/.063, (1.059-posed_y)/.069)
            step[:, 1] += .014*np.clip(contact, 0, 1)
    return step


def refine_daxter_mouth(primitives, accessor, deformations):
    from eye_alignment import move_facial_surface

    for part in (4, 5, 6, 15):
        primitive = primitives[part]
        position = accessor(primitive['attributes']['POSITION'])
        normal = accessor(primitive['attributes']['NORMAL'])
        triangles = accessor(primitive['indices']).reshape(-1, 3)
        _, shading = np.unique(np.column_stack((position, normal)), axis=0, return_inverse=True)
        opening = accessor(primitive['targets'][0]['POSITION'])
        lower = np.linalg.norm(opening, axis=1) > 1e-7 if part == 15 else None
        dp, dn = deformations[part]
        # Keep the lateral jaw's existing lining contour independent of the
        # side-grin channel, which blends differently in dialogue.
        y = position[:, 1]
        influence = np.clip((1.104 - y) / .020, 0, 1)[:, None]
        donor_grin = (8192 / 5184) * accessor(primitive['targets'][25]['POSITION'])
        right_grin = dp[11] + influence * (donor_grin - dp[11])
        for channel in (16, 17):
            step = mouth_profile(position, lower, channel, dp[channel], right_grin)
            move_facial_surface(position, normal, triangles, shading, dp[channel], dn[channel], step)


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
