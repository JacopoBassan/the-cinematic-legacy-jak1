"""Fit the Jak II Samos hanging beard to its native Jak 1 controls.

Keep the cinematic topology, moustache and live eyes. Correct the central
hanging beard skin field and its small local sash clearance.
"""
import json
import struct

import numpy as np


def _components(indices):
    adjacent = {}
    for triangle in indices.reshape(-1, 3):
        for vertex in triangle:
            adjacent.setdefault(int(vertex), set()).update(int(x) for x in triangle)
    pending = set(adjacent)
    result = []
    while pending:
        todo = [min(pending)]
        pending.remove(todo[0])
        component = set(todo)
        while todo:
            for vertex in adjacent[todo.pop()] - component:
                component.add(vertex)
                pending.discard(vertex)
                todo.append(vertex)
        result.append(np.asarray(sorted(component)))
    return result


def _write(output, doc, binary, replacements):
    result = bytearray(binary)
    for index, values in replacements.items():
        entry = doc['accessors'][index]
        view = doc['bufferViews'][entry['bufferView']]
        assert view.get('byteStride', values.dtype.itemsize*values.shape[1]) == values.dtype.itemsize*values.shape[1]
        offset = view.get('byteOffset', 0)+entry.get('byteOffset', 0)
        result[offset:offset+values.nbytes] = values.tobytes()
        if entry['type'] == 'VEC3' and ('min' in entry or 'max' in entry):
            entry.update(min=values.min(0).tolist(), max=values.max(0).tolist())
    encoded = json.dumps(doc, separators=(',', ':')).encode()
    encoded += b' '*(-len(encoded) % 4)
    result += b'\0'*(-len(result) % 4)
    output.write_bytes(struct.pack('<III', 0x46546c67, 2, 28+len(encoded)+len(result))+
        struct.pack('<II', len(encoded), 0x4e4f534a)+encoded+
        struct.pack('<II', len(result), 0x004e4942)+result)


def _beard_clearance(primitive, accessor, anchors):
    """Fit a small outward clearance while retaining the authored face seam."""
    positions = accessor(primitive['attributes']['POSITION'])
    normals = accessor(primitive['attributes']['NORMAL'])
    indices = accessor(primitive['indices']).reshape(-1, 3)
    displacement = np.zeros_like(positions)
    # Two local samples on the inspected hanging shell; propagate the second
    # sample to its coincident cap index. These do not define model bounds.
    for point, distance, count in [
        ((-.13079361617565155, 1.391558289527893, .4031016528606415), .010, 1),
        ((-.18439756333827972, 1.4430180788040161, .3087587058544159), .0055, 2),
    ]:
        rows = np.flatnonzero(np.all(positions == np.asarray(point, dtype=positions.dtype), axis=1))
        assert len(rows) == count
        assert np.array_equal(normals[rows], np.broadcast_to(normals[rows[0]], normals[rows].shape))
        direction = normals[rows[0]] / np.linalg.norm(normals[rows[0]])
        displacement[rows] = direction * distance
    moved = np.flatnonzero(np.any(displacement != 0, axis=1))
    assert len(moved) == 3 and not np.intersect1d(moved, anchors).size
    fitted = positions + displacement
    assert np.array_equal(positions.min(0), fitted.min(0))
    assert np.array_equal(positions.max(0), fitted.max(0))
    # Pool area normals across exact smooth seams, then rotate the authored
    # normals by the resulting local surface change. Retain hard normal seams
    # and the face attachment. Apply the same construction to all morph targets.
    groups = {}
    membership = np.asarray([groups.setdefault((tuple(p), tuple(n)), len(groups))
                             for p, n in zip(positions, normals)])
    support = np.unique(indices[np.isin(indices, moved).any(1)])
    support = np.flatnonzero(np.isin(membership, membership[support]))
    support = np.setdiff1d(support, anchors)

    def geometric(values):
        triangles = values[indices]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        total = np.zeros((len(groups), 3), dtype=float)
        for column in range(3):
            np.add.at(total, membership[indices[:, column]], cross)
        length = np.linalg.norm(total, axis=1, keepdims=True)
        return total / np.maximum(length, 1e-30)

    def rotate(old_positions, new_positions, authored):
        before, after = geometric(old_positions), geometric(new_positions)
        a, b = before[membership[support]], after[membership[support]]
        axis = np.cross(a, b)
        cosine = (a * b).sum(1)
        assert cosine.min() > .9
        values = authored[support].astype(float)
        rotated = values + np.cross(axis, values) + np.cross(axis, np.cross(axis, values)) / (1 + cosine[:, None])
        rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-30)
        result = authored.copy()
        result[support] = rotated
        return result

    fitted_normals = rotate(positions, fitted, normals)
    replacements = {primitive['attributes']['POSITION']: fitted,
                    primitive['attributes']['NORMAL']: fitted_normals}
    for target in primitive['targets']:
        delta = accessor(target['POSITION'])
        old_delta_normal = accessor(target['NORMAL'])
        new_absolute = rotate(positions + delta, fitted + delta, normals + old_delta_normal)
        new_delta_normal = old_delta_normal.copy()
        new_delta_normal[support] = new_absolute[support] - fitted_normals[support]
        replacements[target['NORMAL']] = new_delta_normal
    return replacements, dict(positionVertices=len(moved), normalVertices=len(support),
        maximumRestDisplacementMeters=float(np.linalg.norm(displacement, axis=1).max()),
        morphNormalTargets=len(primitive['targets']), exactCoincidentDisplacement=True,
        faceAnchorsUnchanged=True, morphPositionsUnchanged=True, overallBoundsUnchanged=True,
        method='Local hanging-shell clearance with seam-consistent authored normal rotation')


def restore_native_samos_face(output, native_path):
    """Return explicit provenance for the narrowly selected accessory changes."""
    from retarget_glb import load
    from fix_faces import closest_surface
    low, _, la = load(native_path)
    high, binary, ha = load(output)
    names = [low['nodes'][i]['name'] for i in low['skins'][0]['joints']]
    assert [high['nodes'][i]['name'] for i in high['skins'][0]['joints']] == names
    assert np.array_equal(la(low['skins'][0]['inverseBindMatrices']), ha(high['skins'][0]['inverseBindMatrices']))
    lp, hp = low['meshes'][0]['primitives'], high['meshes'][0]['primitives']
    assert low['materials'][lp[2]['material']]['name'] == 'beardandsideburns'
    assert high['materials'][hp[6]['material']]['name'] == 'samos-hair-envmap'
    assert len(lp[2]['targets']) == 30 and len(hp[6]['targets']) == 33
    drivers = [names.index(n) for n in ['chest', 'head', 'beard_lip', 'beard']]
    native, cinematic = lp[2], hp[6]
    source_pos = la(native['attributes']['POSITION'])
    target_pos = ha(cinematic['attributes']['POSITION'])
    source_joint = la(native['attributes']['JOINTS_0'])
    source_weight = la(native['attributes']['WEIGHTS_0'])
    old_joint = ha(cinematic['attributes']['JOINTS_0'])
    old_weight = ha(cinematic['attributes']['WEIGHTS_0'])
    beard_drivers = drivers[-2:]

    def front_components(primitive, accessor, positions, joints, weights):
        selected = []
        for component in _components(accessor(primitive['indices'])):
            owns_beard = np.any(np.isin(joints[component], beard_drivers) & (weights[component] > 1e-7))
            if not owns_beard:
                continue
            assert positions[component, 2].min() > .06
            assert positions[component, 1].min() > .89 and positions[component, 1].max() < 1.77
            selected.append(component)
        return np.concatenate(selected), selected

    source_used, source_parts = front_components(native, la, source_pos, source_joint, source_weight)
    all_front, all_front_parts = front_components(cinematic, ha, target_pos, old_joint, old_weight)
    assert len(source_parts) == 3 and len(source_used) == 117
    assert len(all_front_parts) == 14 and len(all_front) == 454
    # The hanging shell has split indices at its cap. Include every front-hair
    # component connected through exact coincident positions, so the cap and
    # side/back surfaces receive the same field. The moustache is separate.
    target_parts = [part for part in all_front_parts if target_pos[part, 1].min() < 1.2]
    assert len(target_parts) == 6 and sum(map(len, target_parts)) == 223
    selected_keys = {tuple(position) for part in target_parts for position in target_pos[part]}
    pending_parts = [part for part in all_front_parts if not any(part is selected for selected in target_parts)]
    while True:
        connected = [part for part in pending_parts if any(tuple(position) in selected_keys for position in target_pos[part])]
        if not connected:
            break
        for part in connected:
            target_parts.append(part)
            selected_keys.update(tuple(position) for position in target_pos[part])
        pending_parts = [part for part in pending_parts if not any(part is selected for selected in connected)]
    target_used = np.concatenate(target_parts)
    assert len(target_parts) == 7 and len(target_used) == 240
    assert sorted(len(part) for part in target_parts) == [10, 17, 41, 42, 43, 43, 44]
    protected = np.setdiff1d(np.unique(ha(cinematic['indices'])), target_used)
    assert not any(tuple(position) in selected_keys for position in target_pos[protected])
    source_triangles = la(native['indices']).reshape(-1, 3)
    source_triangles = source_triangles[np.isin(source_triangles, source_used).all(1)]
    assert len(source_triangles) == 172
    nearest, barycentric, distance = closest_surface(target_pos[target_used], source_pos[source_triangles])
    assert distance.max() < .104, 'Cinematic front beard no longer matches the inspected native field'
    source_field = np.column_stack([np.where(source_joint == j, source_weight, 0).sum(1) for j in drivers])
    assert np.allclose(source_field[source_used].sum(1), 1, atol=1e-6)
    field = np.maximum(np.einsum('nij,ni->nj', source_field[source_triangles[nearest]], barycentric), 0)
    field[field < 1e-7] = 0
    assert np.allclose(field.sum(1), 1, atol=1e-6)
    # Keep Merc's strongest three influences and record the discarded mass.
    overflow = (field > 0).sum(1) > 3
    discarded = np.zeros(len(field))
    rows = np.flatnonzero(overflow)
    weakest = field[rows].argmin(1)
    discarded[rows] = field[rows, weakest]
    assert discarded.max() < .15, float(discarded.max())
    field[rows, weakest] = 0
    field /= field.sum(1, keepdims=True)
    joints, weights = old_joint.copy(), old_weight.copy()
    for row, vertex in enumerate(target_used):
        active = np.flatnonzero(field[row] > 0)
        joints[vertex] = 0
        weights[vertex] = 0
        joints[vertex, :len(active)] = np.asarray(drivers)[active]
        weights[vertex, :len(active)] = field[row, active]
    # Preserve the authored face attachment. These vertices share positions
    # and every position morph with the drawn face, and originally used head
    # alone. Including them in a beard-field projection opens the chin seam.
    face = hp[10]
    assert high['materials'][face['material']]['name'] == 'samos-face'
    face_used = np.unique(ha(face['indices']))
    face_pos = ha(face['attributes']['POSITION'])
    face_keys = {tuple(position) for position in face_pos[face_used]}
    head = names.index('head')
    anchors = np.asarray([vertex for vertex in target_used
        if tuple(target_pos[vertex]) in face_keys
        and np.where(old_joint[vertex] == head, old_weight[vertex], 0).sum() == 1.])
    assert len(anchors) == 20
    assert len(face['targets']) == len(cinematic['targets']) == 33
    for vertex in anchors:
        matching = face_used[np.all(face_pos[face_used] == target_pos[vertex], axis=1)]
        assert len(matching)
        for beard_target, face_target in zip(cinematic['targets'], face['targets']):
            assert np.array_equal(ha(beard_target['POSITION'])[vertex][None],
                                  ha(face_target['POSITION'])[matching]) or np.all(
                ha(beard_target['POSITION'])[vertex] == ha(face_target['POSITION'])[matching])
    joints[anchors], weights[anchors] = old_joint[anchors], old_weight[anchors]
    native_rows = ~np.isin(target_used, anchors)
    final_field = np.column_stack([np.where(joints[target_used] == j, weights[target_used], 0).sum(1) for j in drivers])
    assert np.allclose(weights.sum(1), old_weight.sum(1), atol=1e-6)
    assert ((weights > 0).sum(1) <= 3).all()
    unselected = np.setdiff1d(np.arange(len(joints)), target_used)
    assert np.array_equal(joints[unselected], old_joint[unselected])
    assert np.array_equal(weights[unselected], old_weight[unselected])
    replacements = {cinematic['attributes']['JOINTS_0']: joints,
                    cinematic['attributes']['WEIGHTS_0']: weights}
    # A visible bare-eye sliver is native behavior. Programmer-eye materials
    # also include separate goggle projection lenses; preserve both islands.
    eye_reports = []
    for target_part, source_part, material in [(7, 23, 'programmer_eye_left-envmap'), (8, 24, 'programmer_eye_right-envmap')]:
        p, q = hp[target_part], lp[source_part]
        assert high['materials'][p['material']]['name'] == low['materials'][q['material']]['name'] == material
        position = ha(p['attributes']['POSITION'])
        bare = [part for part in _components(ha(p['indices'])) if position[part, 2].max() < .1]
        native_all = la(q['attributes']['POSITION'])
        native_bare = [part for part in _components(la(q['indices'])) if native_all[part, 2].max() < .1]
        assert len(bare) == len(native_bare) == 1
        eye_used, native_used = bare[0], native_bare[0]
        assert len(eye_used) == 15 and len(native_used) == 5
        lens_used = np.setdiff1d(np.unique(ha(p['indices'])), eye_used)
        native_position = la(q['attributes']['POSITION'])[native_used]
        before_range = np.array([position[eye_used, 0].min(), position[eye_used, 0].max()])
        native_range = np.array([native_position[:, 0].min(), native_position[:, 0].max()])
        assert all(np.max(np.abs(la(t['POSITION'])[native_used])) == 0 for t in q['targets'])
        eye_reports.append(dict(part=target_part, material=material, bareEyeVertices=len(eye_used),
            untouchedLensVertices=len(lens_used), cinematicBareEyeWidth=before_range.tolist(),
            nativeBareEyeWidth=native_range.tolist(), geometryUnchanged=True,
            unchanged=['all bare-eye and goggle lens vertices', 'normals', 'authored morphs',
                       'UVs', 'topology', 'materials', 'head binding', 'native live gaze/blink path']))
    _write(output, high, binary, replacements)
    return dict(beard=dict(part=6, vertices=len(target_used), components=len(target_parts),
        nativeVertices=len(source_used), nativeTriangles=len(source_triangles),
        maxSourceDistanceMeters=float(distance.max()),
        nativeFieldVertices=int(native_rows.sum()), authoredFaceAnchorVertices=len(anchors),
        faceAnchorMorphTargetsExact=33,
        threeInfluenceProjectionVertices=int(overflow[native_rows].sum()), maxDiscardedWeight=float(discarded[native_rows].max()),
        meanDiscardedWeight=float(discarded[native_rows].mean()),
        selectionMethod='Hanging beard components closed over exact coincident cap and shell seams',
        sharedBoundaryVerticesWithProtectedHair=0,
        drivers={names[j]:int((final_field[:, i] > 1e-7).sum()) for i,j in enumerate(drivers)},
        authoredMorphsUnchanged=True, noBoneMotionAddedToMorphs=True,
        excluded=['moustache', 'eyebrows', 'forelock', 'sideburns', 'ponytail', 'log hair']),
        eyes=eye_reports,
        unchanged=['all prototype geometry, normals, UVs and authored morphs', 'all other skin accessors', 'native joint order', 'native bind matrices',
                   'all textures and image bytes', 'all material definitions', 'all triangle indices'],
        method='Native hanging-beard skin field with authored face attachments')


def fitted_samos_face_accessors(raw, name, document, accessor, indices, vertices):
    """Apply local clearance after native expression coefficients are fitted.

    Return an accessor overlay for the existing face attachment pass and update
    the same imported base rows. Native expression coefficients and position
    morph deltas remain those fitted from the authored cinematic prototype.
    """
    from fr3_models import DRAW, model
    primitives = document['meshes'][0]['primitives']
    assert name == 'sage-lod0' and len(primitives) == 27
    beard, face = primitives[6], primitives[10]
    assert document['materials'][beard['material']]['name'] == 'samos-hair-envmap'
    assert document['materials'][face['material']]['name'] == 'samos-face'
    positions = accessor(beard['attributes']['POSITION'])
    face_keys = set(map(tuple, accessor(face['attributes']['POSITION'])[np.unique(accessor(face['indices']))]))
    used = np.unique(accessor(beard['indices']))
    anchors = np.asarray([v for v in used if tuple(positions[v]) in face_keys])
    # Includes the20 hanging-shell anchors and30 protected moustache/face joins.
    assert len(anchors) == 50
    replacements, report = _beard_clearance(beard, accessor, anchors)
    draws = sorted([d for effect in model(raw, name)['effects'] for d in effect['draws']],
                   key=lambda draw: DRAW.unpack(draw)[3])
    assert len(draws) == len(primitives)
    values = DRAW.unpack(draws[6])
    global_indices = indices[values[3]:values[3] + values[4]]
    local_indices = accessor(beard['indices']).ravel()
    offset = int(global_indices[0]) - int(local_indices[0])
    assert np.array_equal(global_indices, local_indices + offset)
    for attribute, column, scale in [('POSITION', 0, 4096.), ('NORMAL', 16, 1.)]:
        original = accessor(beard['attributes'][attribute])
        fitted = replacements[beard['attributes'][attribute]]
        rows = np.flatnonzero(np.any(original != fitted, axis=1))
        report['packed' + attribute.title() + 'Rows'] = (rows + offset).tolist()
        for row in rows:
            start = (int(row) + offset) * 64 + column
            imported = np.frombuffer(vertices[start:start + 12], dtype='<f4')
            assert np.max(np.abs(imported - original[row] * scale)) < (.015 if column == 0 else 1e-6)
            struct.pack_into('<3f', vertices, start, *(fitted[row] * scale))
    report.update(stage='After native facial coefficient fitting',
                  nativeExpressionCoefficientsUnchanged=True)
    return lambda index: replacements[index] if index in replacements else accessor(index), report


def anchor_native_samos_socket_rims(native_primitives, native_accessor, document, accessor, deformations):
    """Keep the fixed native eye boundary attached under every facial channel."""
    primitives = document['meshes'][0]['primitives']
    assert len(primitives) == len(native_primitives) == 27
    names = [document['nodes'][i]['name'] for i in document['skins'][0]['joints']]
    head = names.index('head')
    face, native_face = primitives[10], native_primitives[1]
    position = accessor(face['attributes']['POSITION'])
    normals = accessor(face['attributes']['NORMAL'])
    indices = accessor(face['indices']).reshape(-1, 3)
    face_used = np.unique(indices)
    face_joints = accessor(face['attributes']['JOINTS_0'])
    face_weights = accessor(face['attributes']['WEIGHTS_0'])
    native_position = native_accessor(native_face['attributes']['POSITION'])
    native_used = np.unique(native_accessor(native_face['indices']))
    rim = []
    for part, native_part in [(7, 23), (8, 24)]:
        eye, native_eye = primitives[part], native_primitives[native_part]
        eye_positions = accessor(eye['attributes']['POSITION'])
        bare = [c for c in _components(accessor(eye['indices'])) if eye_positions[c, 2].max() < .1]
        native_eye_positions = native_accessor(native_eye['attributes']['POSITION'])
        original_bare = [c for c in _components(native_accessor(native_eye['indices']))
                         if native_eye_positions[c, 2].max() < .1]
        assert len(bare) == len(original_bare) == 1
        assert len(bare[0]) == 15 and len(original_bare[0]) == 5
        eye_joints = accessor(eye['attributes']['JOINTS_0'])
        eye_weights = accessor(eye['attributes']['WEIGHTS_0'])
        assert (np.where(eye_joints[bare[0]] == head, eye_weights[bare[0]], 0).sum(1) == 1).all()
        assert np.max(np.abs(deformations[part][0][:, bare[0]])) == 0
        matches = []
        for vertex in bare[0]:
            matches.extend(face_used[np.all(position[face_used] == eye_positions[vertex], axis=1)])
        matches = np.unique(matches)
        assert len(np.unique(position[matches], axis=0)) == 12
        assert (np.where(face_joints[matches] == head, face_weights[matches], 0).sum(1) == 1).all()
        rim.extend(matches)
        native_matches = np.asarray([v for v in native_used
            if any(np.array_equal(native_position[v], native_eye_positions[e]) for e in original_bare[0])])
        assert len(np.unique(native_position[native_matches], axis=0)) == 5
        for target in native_face['targets']:
            assert np.max(np.abs(native_accessor(target['POSITION'])[native_matches])) == 0
        for target in native_eye['targets']:
            assert np.max(np.abs(native_accessor(target['POSITION'])[original_bare[0]])) == 0
    rim = np.unique(rim)
    old_position, old_normal = deformations[10]
    assert old_position.shape[0] == old_normal.shape[0] == 30
    corrected = old_position.copy()
    corrected[:, rim] = 0
    moved = np.flatnonzero(np.any(corrected != old_position, axis=(0, 2)))
    assert len(moved) == 7
    # Preserve smooth normal seams while retaining authored hard edges. Only
    # triangles touched by the corrected rim rows contribute a normal update.
    groups = {}
    membership = np.asarray([groups.setdefault((tuple(p), tuple(n)), len(groups))
                             for p, n in zip(position, normals)])
    support = np.unique(indices[np.isin(indices, moved).any(1)])
    support = np.flatnonzero(np.isin(membership, membership[support]))

    def geometric(points):
        triangles = points[indices]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        total = np.zeros((len(groups), 3), dtype=float)
        for column in range(3):
            np.add.at(total, membership[indices[:, column]], cross)
        return total / np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-30)

    corrected_normal = old_normal.copy()
    channels = np.flatnonzero(np.any(corrected != old_position, axis=(1, 2)))
    rotations = []
    for channel in channels:
        before = geometric(position + old_position[channel])[membership[support]]
        after = geometric(position + corrected[channel])[membership[support]]
        axis = np.cross(before, after)
        cosine = (before * after).sum(1)
        assert cosine.min() > .5
        values = normals[support] + old_normal[channel, support]
        rotated = values + np.cross(axis, values) + np.cross(axis, np.cross(axis, values)) / (1 + cosine[:, None])
        rotated /= np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-30)
        corrected_normal[channel, support] = rotated - normals[support]
        rotations.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).max()))
    deformations[10] = corrected, corrected_normal
    assert np.max(np.abs(corrected[:, rim])) == 0
    outside = np.setdiff1d(np.arange(len(position)), moved)
    assert np.array_equal(corrected[:, outside], old_position[:, outside])
    return dict(facePart=10, rimIndexRows=rim.tolist(), rimUniquePositions=24,
        correctedPositionRows=moved.tolist(), correctedChannels=channels.tolist(),
        localNormalSupportRows=support.tolist(), maximumChannelNormalRotationDegrees=max(rotations),
        nativeOriginalRimPositionChannelsZero=True, sharedHeadBinding=True,
        allNativeBlendSocketPositionsCoincident=True,
        unchanged=['all base geometry and cap curvature', 'eye and goggle geometry',
                   'authored UVs', 'native pupil, gaze and blink paths',
                   'all coefficients', 'all other facial position channels'])


def authored_samos_eye_uvs(primitives, accessor, eyes):
    """Retain distinct authored bare-eye and magnifying-lens UV domains."""
    assert len(primitives) == 27 and eyes == {7: 0, 8: 1}
    result, reports = {}, []
    for part, side in eyes.items():
        p = primitives[part]
        positions = accessor(p['attributes']['POSITION'])
        uv = accessor(p['attributes']['TEXCOORD_0']).astype('<f4')
        indices = accessor(p['indices']).reshape(-1, 3)
        used = np.unique(indices)
        assert len(used) == (41 if side == 0 else 39)
        assert (np.median(positions[used, 0]) > 0) == (side == 0)
        bare = [component for component in _components(indices) if positions[component, 2].max() < .1]
        assert len(bare) == 1 and len(bare[0]) == 15
        lens = np.setdiff1d(used, bare[0])
        assert positions[lens, 2].min() > .16
        assert np.isfinite(uv).all() and uv.min() >= 0 and uv.max() <= 1
        domains = []
        for label, vertices in [('bare-eye', bare[0]), ('goggle-lens', lens)]:
            assert np.array_equal(uv[vertices].min(0), [0., 0.])
            assert np.array_equal(uv[vertices].max(0), [1., 1.])
            triangles = indices[np.isin(indices, vertices).all(1)]
            samples = []
            for sample in [(0.5, 0.5), (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75)]:
                target = np.r_[sample, 1.]
                covered = False
                for triangle in triangles:
                    matrix = np.vstack([uv[triangle].T, np.ones(3)])
                    bary = np.linalg.lstsq(matrix, target, rcond=None)[0]
                    if np.linalg.norm(matrix@bary-target) < 1e-6 and bary.min() >= -1e-6:
                        covered = True
                        break
                assert covered, (part, label, sample)
                samples.append(list(sample))
            domains.append(dict(name=label, vertices=len(vertices), uvBounds=[[0., 0.], [1., 1.]], compositorSamplesCovered=samples))
        result[part] = uv.copy()
        reports.append(dict(part=part, side=side, domains=domains))
    return result, dict(method='Separate authored cinematic Samos bare-eye and goggle-lens UV domains',
        eyes=reports, staticEyes=False,
        unchanged=['all eye geometry and normals', 'native live eye IDs', 'native gaze controls',
                   'iris and pupil animation', 'automatic blink', 'native eye expression geometry'])


def fit_samos_socket_colors(raw, prototype_path):
    """Neutralize only dark eye-socket RGB, retaining alpha and expression data."""
    from retarget_glb import load
    from fr3_models import Reader,DRAW,model,serialize
    from inspect_faces import buffers
    doc,_,access=load(prototype_path)
    prims=doc['meshes'][0]['primitives'];face=prims[10]
    if doc['materials'][face['material']]['name']!='samos-face':raise ValueError('Unexpected Samos face material')
    positions=access(face['attributes']['POSITION']);local=access(face['indices']).ravel()
    sage=model(raw,'sage-lod0');_,indices,vstart,vertexbytes=buffers(raw,'sage-lod0')
    draws=sorted([d for e in sage['effects'] for d in e['draws']],key=lambda d:DRAW.unpack(d)[3])
    if len(draws)!=len(prims):raise ValueError('Unexpected Samos draw count')
    fv=DRAW.unpack(draws[10]);gids=indices[fv[3]:fv[3]+fv[4]];offset=int(gids[0])-int(local[0])
    if not np.array_equal(gids,local+offset):raise ValueError('Samos face index mapping differs')
    vertexview=np.frombuffer(vertexbytes,'u1').reshape(-1,64);face_colors=vertexview[offset:offset+len(positions),56:60]
    mask=(positions[:,1]>1.75)&(positions[:,1]<1.84)&(positions[:,2]>.035)&(positions[:,2]<.10)&np.all(face_colors==[52,53,11,128],axis=1)
    ids=np.intersect1d(np.flatnonzero(mask),np.unique(local))
    if len(ids)!=17 or not (np.all(vertexview[offset+ids,60]==5) and np.all(vertexview[offset+ids,32:36].copy().view('<f4')==1)):
        raise ValueError('Unexpected Samos eye-socket color or binding')
    modified=bytearray(vertexbytes)
    for row in ids:
        start=(offset+int(row))*64+56;modified[start:start+3]=bytes([128,128,128])
    changed_effects=0
    for effect in sage['effects']:
        r=Reader(effect['mod']);draws=r.draws();r.draws();start=r.pos+8;r.vector(64)
        matches=[d for d in draws if tuple(DRAW.unpack(d)[:3])+tuple(DRAW.unpack(d)[4:])==tuple(fv[:3])+tuple(fv[4:])]
        if not matches:continue
        if len(matches)!=1:raise ValueError('Unexpected Samos modified face draw')
        mv=DRAW.unpack(matches[0]);mapped=indices[mv[3]:mv[3]+mv[4]]
        if len(mapped)!=len(local):raise ValueError('Samos modified face mapping differs')
        mapping={}
        for li,mi in zip(local,mapped):
            if int(li) in mapping and mapping[int(li)]!=int(mi):raise ValueError('Inconsistent Samos face mapping')
            mapping[int(li)]=int(mi)
        mod=bytearray(effect['mod'])
        for row in ids:
            point=start+mapping[int(row)]*64+56
            if mod[point:point+4]!=bytes([52,53,11,128]):raise ValueError('Unexpected modified eye-socket color')
            mod[point:point+3]=bytes([128,128,128])
        effect['mod']=bytes(mod);changed_effects+=1
    if changed_effects!=1:raise ValueError('Unexpected Samos modified draw count')
    result=raw[:vstart+8]+bytes(modified)+raw[vstart+8+len(modified):]
    existing=model(result,'sage-lod0');result=result[:existing['start']]+serialize(sage)+result[existing['end']:]
    return result,{'faceRows':ids.tolist(),'changedRgbVertices':len(ids),
                   'alphaUnchanged':True,'geometryNormalsTexturesAndMorphsUnchanged':True}


def correct_samos_socket_texture(texture):
    """Replace baked socket darkness with nearby skin while preserving alpha."""
    import hashlib
    expected = '40e536e5b9d52f2cb09e989acf6d04b1be6c1ce7d5da6e6b5f4ac91da3dbe890'
    if (texture['width'], texture['height'], hashlib.sha256(texture['pixels']).hexdigest()) != (128, 128, expected):
        raise ValueError('Unsupported Samos face texture for socket correction')
    source = np.frombuffer(texture['pixels'],dtype='u1').reshape(128,128,4).copy()
    xx,yy=np.meshgrid(np.arange(128),np.arange(128))
    radius=np.sqrt(((xx-68.)/10.5)**2+((yy-21.)/8.)**2)
    weight=np.clip((1-radius)/.30,0,1)
    skin=np.broadcast_to(source[32:33,:,:3],(128,128,3)).astype(float)
    result=source.copy()
    result[:,:,:3]=np.rint(source[:,:,:3]*(1-weight[:,:,None])+skin*weight[:,:,None]).clip(0,255).astype('u1')
    changed=np.any(result!=source,axis=2)
    if (int(changed.sum()) != 259 or not np.array_equal(result[:,:,3],source[:,:,3]) or
        changed[:12].any() or changed[30:].any() or changed[:,:56].any() or changed[:,80:].any()):
        raise ValueError('Unexpected Samos face texture correction scope')
    texture['pixels']=result.tobytes()
    return {'texture':'samos-face','changedPixels':int(changed.sum()),'eyeAndGoggleMaterialsUnchanged':True,
            'alphaUnchanged':True,'sourceSha256':expected,
            'resultSha256':hashlib.sha256(texture['pixels']).hexdigest()}
