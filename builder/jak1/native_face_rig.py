"""Retain native Jak 1 jaw and brow bones on the cinematic face.

The Jak II donor animates its face with morphs and binds its jaw and brows to
neckB. Jak 1 also animates three independent facial joints. Copy their native
skin fields onto the donor's unchanged surfaces, keeping its authored morphs.
"""
import json
import struct

import numpy as np


def distribute_head_weights(joints, weights, head, drivers, fractions):
    """Split only the existing head influence, preserving other attachments."""
    result_joints = joints.copy()
    result_weights = weights.copy()
    assert fractions.shape == (len(weights), len(drivers))
    assert np.isfinite(fractions).all() and (fractions >= -1e-7).all()
    assert (fractions.sum(1) <= 1+1e-6).all()
    for vertex in range(len(weights)):
        if not np.any(fractions[vertex] > 1e-7):
            continue
        row = {int(j): float(w) for j, w in zip(joints[vertex], weights[vertex]) if w > 1e-7}
        head_weight = row.get(head, 0)
        if not head_weight:
            continue
        remaining = max(0., 1-float(fractions[vertex].sum()))
        row[head] = head_weight*remaining
        for driver, fraction in zip(drivers, fractions[vertex]):
            if fraction > 1e-7:
                row[driver] = row.get(driver, 0)+head_weight*float(fraction)
        row = {j: w for j, w in row.items() if w > 1e-7}
        assert len(row) <= 3, ('Facial transfer exceeds native Merc influence count', vertex, row)
        assert abs(sum(row.values())-1) < 1e-6
        # Preserve the existing column order, adding only newly active drivers.
        order = list(dict.fromkeys([int(j) for j in joints[vertex] if int(j) in row]+list(row)))
        result_joints[vertex] = 0
        result_weights[vertex] = 0
        for column, joint in enumerate(order):
            result_joints[vertex, column] = joint
            result_weights[vertex, column] = row[joint]
    assert np.isfinite(result_weights).all()
    assert np.allclose(result_weights.sum(1), weights.sum(1), atol=1e-6)
    return result_joints, result_weights


def lower_mouth_ownership(closed_field, opened_field, opening_y, upper_y, lower_y):
    """Disambiguate touching lips using their authored opening response.

    Closed nearest-surface lookup can assign the lower opening rim to the upper
    lip. Only the correspondence lookup is posed open. The actual donor neutral
    mesh and every authored morph remain unchanged. The continuous response
    fades out toward the upper lip and mouth corners instead of making a hard
    geometric cut through the face.
    """
    assert upper_y > lower_y
    semantic = np.clip((upper_y-opening_y)/(upper_y-lower_y), 0, 1)
    return np.maximum(closed_field, opened_field*semantic)


def closest_upper_tooth_surface(points, triangles):
    """Resolve roundoff ties in native triangle order, independent of KDTree order."""
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 3
    assert triangles.ndim == 3 and triangles.shape[1:] == (3, 3) and len(triangles)
    assert np.isfinite(points).all() and np.isfinite(triangles).all()
    t = triangles[None, :, :, :]
    p = points[:, None, :]
    a = t[:, :, 0]
    u, v, q = t[:, :, 1]-a, t[:, :, 2]-a, p-a

    def dot(x, y):
        return (x[..., 0]*y[..., 0] + x[..., 1]*y[..., 1]) + x[..., 2]*y[..., 2]

    uu, uv, vv, qu, qv = dot(u, u), dot(u, v), dot(v, v), dot(q, u), dot(q, v)
    denominator = uu*vv-uv*uv
    safe = np.where(abs(denominator) > 1e-20, denominator, 1)
    b, c = (qu*vv-qv*uv)/safe, (qv*uu-qu*uv)/safe
    barycentric = np.stack([1-b-c, b, c], axis=-1)
    valid = (barycentric.min(axis=-1) >= 0) & (abs(denominator) > 1e-20)
    projected = ((barycentric[..., 0, None]*t[:, :, 0] +
                  barycentric[..., 1, None]*t[:, :, 1]) +
                 barycentric[..., 2, None]*t[:, :, 2])
    best = np.where(valid, dot(projected-p, projected-p), np.inf)
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        edge = t[:, :, j]-t[:, :, i]
        alpha = np.clip(dot(p-t[:, :, i], edge)/np.maximum(dot(edge, edge), 1e-20), 0, 1)
        point = t[:, :, i]+alpha[..., None]*edge
        distance_squared = dot(point-p, point-p)
        edge_barycentric = np.zeros_like(barycentric)
        edge_barycentric[..., i], edge_barycentric[..., j] = 1-alpha, alpha
        barycentric = np.where((distance_squared < best)[..., None],
                               edge_barycentric, barycentric)
        best = np.minimum(best, distance_squared)
    distances = np.sqrt(best)
    # 1e-12 metres covers float64 roundoff, far below float32 source precision.
    # Source order is canonical; a centroid-query order must not break the tie.
    tied = distances <= distances.min(axis=1)[:, None]+1e-12
    nearest = tied.argmax(axis=1)
    rows = np.arange(len(points))
    return nearest, barycentric[rows, nearest], distances[rows, nearest]


def connected_vertex_component(indices, seed):
    """Return only the indexed triangle component containing an inspected pole."""
    triangles = indices.reshape(-1, 3)
    assert seed in triangles and np.all(triangles >= 0)
    component = {int(seed)}
    while True:
        connected = set(int(v) for triangle in triangles
                        if component.intersection(triangle) for v in triangle)
        if connected == component:
            return np.array(sorted(component), dtype=indices.dtype)
        component = connected


def transferred_face_weights(lp, la, hp, ha, native_names):
    """Return replacement skin accessors and provenance-checked rig evidence."""
    from fix_faces import closest_surface
    assert native_names[6] == 'neckB'
    assert native_names[36:39] == ['mouth', 'browL', 'browR']
    assert len(lp[0]['targets']) == 14 and len(hp[8]['targets']) == 32
    poles = [('upper-lip', 0, 5445, 8, 626, [0, 2.3109, .3330], [0, 2.3170, .3334]),
             ('lower-lip', 0, 5409, 8, 730, [0, 2.2973, .3330], [0, 2.3051, .3266]),
             ('upper-teeth', 2, 5682, 9, 55, [0, 2.3415, .3229], [0, 2.3477, .3096]),
             ('lower-teeth', 2, 5677, 9, 8, [0, 2.2837, .3127], [0, 2.2915, .2943])]
    for label, li, lv, hi, hv, low_position, high_position in poles:
        assert lv in la(lp[li]['indices']) and hv in ha(hp[hi]['indices'])
        assert np.allclose(la(lp[li]['attributes']['POSITION'])[lv], low_position, atol=7e-5), label
        assert np.allclose(ha(hp[hi]['attributes']['POSITION'])[hv], high_position, atol=7e-5), label
    source = lp[0]
    source_indices = la(source['indices']).reshape(-1, 3)
    source_joints = la(source['attributes']['JOINTS_0'])
    source_weights = la(source['attributes']['WEIGHTS_0'])
    drivers = [36, 37, 38]
    field = np.column_stack([np.where(source_joints == joint, source_weights, 0).sum(1) for joint in drivers])
    source_triangles = la(source['attributes']['POSITION'])[source_indices]
    replacements = {}
    rows = []
    for part in [8, 10]:
        primitive = hp[part]
        positions = ha(primitive['attributes']['POSITION'])
        used = np.unique(ha(primitive['indices']))
        nearest, barycentric, distance = closest_surface(positions, source_triangles)
        fractions = np.einsum('nij,ni->nj', field[source_indices[nearest]], barycentric)
        fractions = np.maximum(fractions, 0)
        fractions[fractions < 1e-7] = 0
        surface_limits = np.full(len(positions), .05 if part == 8 else .07)
        component_report = {}
        # The upper/lower lip boundary is geometrically close in closed mouths.
        # Validate its semantic ownership rather than silently accepting a swap.
        if part == 8:
            assert fractions[626, 0] == 0 and abs(fractions[730, 0]-1) < 1e-6
            native_opening = la(source['targets'][4]['POSITION'])
            donor_opening = ha(primitive['targets'][0]['POSITION'])
            opening_scale = float(donor_opening[730, 1]/native_opening[5409, 1])
            assert 1.10 < opening_scale < 1.12
            opened_source = la(source['attributes']['POSITION'])+native_opening*opening_scale
            open_nearest, open_barycentric, _ = closest_surface(positions+donor_opening, opened_source[source_indices])
            opened_field = np.einsum('nij,ni->nj', field[source_indices[open_nearest]], open_barycentric)[:, 0]
            fractions[:, 0] = lower_mouth_ownership(fractions[:, 0], opened_field,
                donor_opening[:, 1], float(donor_opening[626, 1]), float(donor_opening[730, 1]))
            # These are lower inner opening-rim poles, not merely the visible
            # outer lower lip. Their closed positions touch the upper surface.
            assert fractions[726, 0] > .98 and min(fractions[724:726, 0]) > .85
            assert np.all(fractions[[626, 627, 633, 643], 0] == 0)
            # The donor cavity is disconnected from its outer face. Closed
            # nearest-face lookup can mistake the floor for the external chin,
            # and pull the nearly static roof down with the jaw. Use inspected
            # cavity triangles for the floor and keep the roof head-bound.
            cavity = connected_vertex_component(ha(primitive['indices']), 413)
            roof = connected_vertex_component(ha(primitive['indices']), 923)
            assert len(cavity) == 196 and len(roof) == 68
            assert not np.intersect1d(cavity, roof).size
            assert np.linalg.norm(donor_opening[roof], axis=1).max() < .0025
            before_cavity = fractions.copy()
            outside_cavity = np.setdiff1d(used, cavity)
            cavity_seam = (np.linalg.norm(positions[cavity, None]-
                positions[outside_cavity][None], axis=2) < 1e-7).any(1)
            assert int(cavity_seam.sum()) == 33
            outside_roof = np.setdiff1d(used, roof)
            roof_matches = np.linalg.norm(positions[roof, None]-
                positions[outside_roof][None], axis=2) < 1e-7
            roof_mates = np.unique(outside_roof[np.nonzero(roof_matches)[1]])
            assert int(roof_matches.any(1).sum()) == 13 and len(roof_mates) == 12
            assert np.isin(roof_mates, cavity[cavity_seam]).all()
            inner_triangles = source_indices[((source_indices >= 5466) &
                                               (source_indices <= 5489)).all(1)]
            assert len(inner_triangles) == 29 and len(np.unique(inner_triangles)) == 21
            inner_nearest, inner_barycentric, inner_distance = closest_surface(positions[cavity],
                la(source['attributes']['POSITION'])[inner_triangles])
            inner_jaw = np.einsum('ni,ni->n', field[inner_triangles[inner_nearest], 0],
                                   inner_barycentric)
            assert np.isfinite(inner_jaw).all() and (inner_jaw >= -1e-7).all()
            assert (inner_jaw <= 1+1e-6).all() and inner_distance.max() < .06
            # Preserve the outer lip seam. The roof's duplicated seam vertices
            # must instead share its head ownership, including their floor mates.
            floor = cavity[~cavity_seam]
            fractions[floor, 0] = np.clip(inner_jaw[~cavity_seam], 0, 1)
            distance[floor] = inner_distance[~cavity_seam]
            surface_limits[floor] = .06
            fractions[roof, 0] = 0
            fractions[roof_mates, 0] = 0
            outer_seam = cavity[cavity_seam & ~np.isin(cavity, roof_mates)]
            assert len(outer_seam) == 21
            assert np.array_equal(fractions[outer_seam], before_cavity[outer_seam])
            assert np.array_equal(fractions[:, 1:], before_cavity[:, 1:])
            untouched = np.setdiff1d(used, np.union1d(cavity, roof))
            assert np.array_equal(fractions[untouched], before_cavity[untouched])
            assert not fractions[roof, 0].any() and not fractions[roof_mates, 0].any()
            assert int((fractions[cavity, 0] > 1e-7).sum()) == 167
            assert np.allclose(fractions[[535, 555, 554, 561], 0],
                               [.89073, 1, 1, .89073], atol=1e-5)
            assert np.all(fractions[[580, 584, 588, 596], 0] == 1)
            component_report = dict(cavityVertices=196, headBoundRoofVertices=68,
                roofSeamVertices=13, roofSeamMates=12, retainedOuterSeamVertices=21,
                cavityJawDriverVertices=167, nativeInnerTriangles=29,
                maxInnerSourceSurfaceDistance=float(inner_distance.max()),
                cavityField='Native inner-mouth triangles only; outer seams preserved')
        original_joints = ha(primitive['attributes']['JOINTS_0'])
        original_weights = ha(primitive['attributes']['WEIGHTS_0'])
        joints, weights = distribute_head_weights(original_joints, original_weights, 6, drivers, fractions)
        changed = used[np.any(joints[used] != original_joints[used], axis=1) |
                       np.any(weights[used] != original_weights[used], axis=1)]
        if len(changed):
            # Part 10 also contains the donor's deeper tongue/interior, not
            # only the eyebrow material named by its texture. Its inspected
            # correspondence reaches 64.4 mm inside the original closed mouth.
            assert (distance[changed] < surface_limits[changed]).all(), 'Facial correspondence exceeds inspected source surface'
        replacements[part] = (joints, weights)
        rows.append(dict(part=part, changedVertices=len(changed),
                         maxSourceSurfaceDistance=float(distance[changed].max()) if len(changed) else 0,
                         driverVertices={native_names[j]: int(np.any((joints[used] == j) & (weights[used] > 1e-7), axis=1).sum()) for j in drivers},
                         **component_report))
    # Donor shape 0 identifies the separate lower strip. Keep its rigid jaw
    # binding, but retain the native upper strip's independent jaw response too.
    # Its top is head-bound while the lower edge and corners partly follow the
    # jaw. Binding the whole upper strip to the head hides it under the upper
    # lip in native bone-only opening poses. Restrict correspondence to native
    # UPPER tooth triangles so nearby lower teeth or lips cannot take ownership.
    part = 9
    primitive = hp[part]
    positions = ha(primitive['attributes']['POSITION'])
    used = np.unique(ha(primitive['indices']))
    lower = np.linalg.norm(ha(primitive['targets'][0]['POSITION']), axis=1) > 1e-7
    assert len(used) == 81 and int(lower[used].sum()) == 41
    assert lower[8] and not lower[55]
    assert np.isin(lower[ha(primitive['indices']).reshape(-1, 3)].sum(1), [0, 3]).all()
    upper = used[~lower[used]]
    native_teeth = lp[2]
    native_indices = la(native_teeth['indices'])
    native_triangles = native_indices.reshape(-1, 3)
    native_upper = connected_vertex_component(native_indices, 5682)
    native_lower = connected_vertex_component(native_indices, 5677)
    assert len(native_upper) == len(native_lower) == 10
    assert not np.intersect1d(native_upper, native_lower).size
    assert np.array_equal(np.union1d(native_upper, native_lower), np.unique(native_indices))
    upper_triangles = native_triangles[np.isin(native_triangles, native_upper).all(1)]
    assert len(upper_triangles) == 8 and 5698 in native_upper
    native_joints = la(native_teeth['attributes']['JOINTS_0'])
    native_weights = la(native_teeth['attributes']['WEIGHTS_0'])
    native_jaw = np.where(native_joints == 36, native_weights, 0).sum(1)
    assert native_jaw[5682] == 0 and abs(float(native_jaw[5698])-26/255) < 1e-7
    assert int((native_jaw[native_upper] > 1e-7).sum()) == 5
    assert .89 < float(native_jaw[native_upper].max()) < .9
    nearest, barycentric, distance = closest_upper_tooth_surface(positions[upper],
        la(native_teeth['attributes']['POSITION'])[upper_triangles])
    source_jaw = native_jaw[upper_triangles[nearest]].astype(np.float64)
    upper_jaw = ((source_jaw[:, 0]*barycentric[:, 0] +
                  source_jaw[:, 1]*barycentric[:, 1]) + source_jaw[:, 2]*barycentric[:, 2])
    assert np.isfinite(upper_jaw).all() and (upper_jaw >= -1e-7).all()
    assert distance.max() < .03, 'Upper tooth correspondence exceeds inspected source strip'
    assert int((upper_jaw > 1e-7).sum()) == 20 and upper_jaw.max() < .7
    fractions = np.zeros((len(lower), 3))
    fractions[lower, 0] = 1
    fractions[upper, 0] = np.maximum(upper_jaw, 0)
    assert fractions[55, 0] == 0 and fractions[67, 0] == 0
    assert abs(float(fractions[56, 0])-.10383182) < 5e-5
    assert abs(float(fractions[58, 0])-.10521807) < 5e-5
    joints, weights = distribute_head_weights(ha(primitive['attributes']['JOINTS_0']),
        ha(primitive['attributes']['WEIGHTS_0']), 6, drivers, fractions)
    replacements[part] = (joints, weights)
    rows.append(dict(part=part, changedVertices=int((fractions[used, 0] > 1e-7).sum()),
                     rigidLowerTeeth=True, lowerToothVertices=41, upperToothVertices=40,
                     upperJawDriverVertices=20, headOnlyUpperToothVertices=20,
                     upperSourceTriangles=len(upper_triangles),
                     maxUpperSourceSurfaceDistance=float(distance.max()),
                     upperJawFractionRange=[float(upper_jaw.min()), float(upper_jaw.max())],
                     upperToothField='Native upper tooth component only, barycentric jaw weights'))
    for label, _, _, hi, hv, _, _ in poles:
        joints, weights = replacements[hi]
        jaw = float(weights[hv][joints[hv] == 36].sum())
        assert abs(jaw-(1 if label.startswith('lower') else 0)) < 1e-6, label
    return replacements, dict(nativeDrivers={native_names[i]: i for i in drivers}, parts=rows,
        nativeSurfaceField='Native face primitive 0, barycentric skin weights',
        unchanged=['neutral positions', 'normals', 'UVs', 'topology', 'authored morph targets',
                   'native joint order and bind matrices', 'non-head influences',
                   'upper tooth top anchors', 'upper lip jaw ownership'],
        status='Native facial rig restored; matched live expression captures required')


def restore_native_face_rig(output, native_path):
    from retarget_glb import load
    ld, _, la = load(native_path)
    hd, binary, ha = load(output)
    native_names = [ld['nodes'][i]['name'] for i in ld['skins'][0]['joints']]
    assert [hd['nodes'][i]['name'] for i in hd['skins'][0]['joints']] == native_names
    assert np.array_equal(la(ld['skins'][0]['inverseBindMatrices']), ha(hd['skins'][0]['inverseBindMatrices']))
    replacements, report = transferred_face_weights(ld['meshes'][0]['primitives'], la,
        hd['meshes'][0]['primitives'], ha, native_names)
    write_skin_accessors(output, hd, binary, replacements)
    return report


def write_skin_accessors(output, hd, binary, replacements):
    """Keep the existing GLB layout, changing only the selected skin accessors."""
    result = bytearray(binary)
    for part, (joints, weights) in replacements.items():
        primitive = hd['meshes'][0]['primitives'][part]
        for attribute, values in [('JOINTS_0', joints), ('WEIGHTS_0', weights)]:
            index = primitive['attributes'][attribute]
            entry = hd['accessors'][index]
            view = hd['bufferViews'][entry['bufferView']]
            assert view.get('byteStride', values.dtype.itemsize*values.shape[1]) == values.dtype.itemsize*values.shape[1]
            offset = view.get('byteOffset', 0)+entry.get('byteOffset', 0)
            result[offset:offset+values.nbytes] = values.tobytes()
    encoded = json.dumps(hd, separators=(',', ':')).encode()
    encoded += b' '*(-len(encoded) % 4)
    result += b'\0'*(-len(result) % 4)
    output.write_bytes(struct.pack('<III', 0x46546c67, 2, 28+len(encoded)+len(result))+
        struct.pack('<II', len(encoded), 0x4e4f534a)+encoded+
        struct.pack('<II', len(result), 0x004e4942)+result)


def remap_material_joints(doc, accessor, material_name, mapping):
    """Remap active joints only on the named cloth material, leaving skin alone."""
    replacements = {}
    for part, primitive in enumerate(doc['meshes'][0]['primitives']):
        if doc['materials'][primitive['material']]['name'] != material_name:
            continue
        joints = accessor(primitive['attributes']['JOINTS_0'])
        weights = accessor(primitive['attributes']['WEIGHTS_0'])
        result = joints.copy()
        for source, destination in mapping.items():
            result[(joints == source) & (weights > 0)] = destination
        replacements[part] = (result, weights.copy())
    return replacements


def restore_native_pants_rig(output, native_path):
    """Retain Jak 1's independent trouser knees without changing body joints."""
    from retarget_glb import load
    ld, _, la = load(native_path)
    hd, binary, ha = load(output)
    native_names = [ld['nodes'][i]['name'] for i in ld['skins'][0]['joints']]
    assert [hd['nodes'][i]['name'] for i in hd['skins'][0]['joints']] == native_names
    assert np.array_equal(la(ld['skins'][0]['inverseBindMatrices']), ha(hd['skins'][0]['inverseBindMatrices']))
    mapping = {native_names.index(side+'knee'): native_names.index('pants'+side+'knee') for side in ['L', 'R']}
    native = next(p for p in ld['meshes'][0]['primitives'] if ld['materials'][p['material']]['name'] == 'bam-cotton-pants')
    native_joints = la(native['attributes']['JOINTS_0'])
    native_weights = la(native['attributes']['WEIGHTS_0'])
    for destination in mapping.values():
        assert np.any((native_joints == destination) & (native_weights > 0)), 'Native trouser driver missing'
    replacements = remap_material_joints(hd, ha, 'jak-orig-pants', mapping)
    assert list(replacements) == [22], 'Cinematic trousers material ownership changed'
    part = 22
    primitive = hd['meshes'][0]['primitives'][part]
    used = np.unique(ha(primitive['indices']))
    old = ha(primitive['attributes']['JOINTS_0'])
    new, weights = replacements[part]
    counts = {native_names[source]: int(np.any((old[used] == source) & (weights[used] > 0), axis=1).sum()) for source in mapping}
    assert counts == {'Lknee': 87, 'Rknee': 102}, 'Cinematic trouser influence layout changed'
    write_skin_accessors(output, hd, binary, replacements)
    return dict(material='jak-orig-pants', part=part,
        jointRemapping={native_names[a]: native_names[b] for a, b in mapping.items()},
        driverVertices=counts, changedVertices=int(np.any(new[used] != old[used], axis=1).sum()),
        unchanged=['thigh influences', 'skin legs', 'feet', 'all other materials',
                   'all geometry and morph targets', 'all skin weight values', 'native bind matrices'],
        reason='Native body knees and trouser knees animate independently during attacks')
