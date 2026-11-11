"""Restore Jak 1 Keira's open sandals from the user's native extraction.

Whole connected lower-leg surfaces are exchanged beneath the pant hems. Shared
hand, belt and body materials remain intact. No triangle is cut across a calf.
The two cinematic trouser cuffs are fitted to the native Jak 1 hem height.
"""
import base64
import copy
import json
import struct
import sys
from pathlib import Path

import numpy as np

RELEASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE))
sys.path.insert(0, str(RELEASE / 'stages'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retarget_glb import load
from native_io import buffers
from fr3_models import decompress, DRAW
from verify_import import read_model

NATIVE_LIMITS = {0: .750, 7: .210, 8: .331, 9: .293, 10: .369}
DONOR_LIMITS = {17: .401, 18: .739, 19: .328, 22: .177, 23: float('inf')}
LEG_JOINTS = {side + name for side in ('L', 'R') for name in
              ('thigh', 'knee', 'ankle', 'ball', 'toeStrap', 'footFlaps')}
DONOR_NATIVE_EFFECTS = [0, 2, 1, 0, 3, 2]


def components(doc, acc, primitive):
    indices = acc(primitive['indices']).ravel().reshape(-1, 3)
    points = acc(primitive['attributes']['POSITION'])
    parents = np.arange(len(indices))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for ti, triangle in enumerate(indices):
        for vertex in triangle:
            key = tuple(np.round(points[vertex], 5))
            if key in seen:
                parents[root(ti)] = root(seen[key])
            else:
                seen[key] = ti
    result = {}
    for i in range(len(indices)):
        result.setdefault(root(i), []).append(i)
    return indices, [np.asarray(t, dtype=int) for t in result.values()]


def select_lower_components(doc, acc, primitive, limit):
    triangles, parts = components(doc, acc, primitive)
    names = [doc['nodes'][i]['name'] for i in doc['skins'][0]['joints']]
    points = acc(primitive['attributes']['POSITION'])
    joints = acc(primitive['attributes']['JOINTS_0'])
    weights = acc(primitive['attributes']['WEIGHTS_0'])
    selected = []
    details = []
    for part in parts:
        used = np.unique(triangles[part])
        active = {names[int(j)] for j, w in zip(joints[used].ravel(), weights[used].ravel()) if w > 0}
        if not active <= LEG_JOINTS or float(points[used, 1].max()) > limit + 1e-5:
            continue
        selected.extend(part)
        details.append(dict(triangles=len(part), minimum=points[used].min(0).tolist(),
                            maximum=points[used].max(0).tolist(), joints=sorted(active)))
    return np.asarray(sorted(selected), dtype=int), details


def source_draws(path, model_name):
    model = read_model(decompress(path), model_name)
    rows = []
    for effect_index, effect in enumerate(model['effects']):
        for draw_index, draw in enumerate(effect['draws']):
            data = DRAW.unpack(draw)
            rows.append(dict(sourceEffectIndex=effect_index, sourceDrawIndex=draw_index,
                             sourceDrawMode=int(data[0]), sourceTextureIndex=int(data[1]),
                             sourceEyeId=int(data[2]), sourceHasEnvmap=bool(effect['tail'][8])))
    return rows


def save_glb(doc, binary, output):
    doc['buffers'] = [{'byteLength': len(binary)}]
    encoded = json.dumps(doc, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary += b'\0' * (-len(binary) % 4)
    output.write_bytes(struct.pack('<III', 0x46546c67, 2, 28 + len(encoded) + len(binary)) +
                       struct.pack('<II', len(encoded), 0x4e4f534a) + encoded +
                       struct.pack('<II', len(binary), 0x004e4942) + binary)


def cuff_fit_metadata(native, na, donor, da):
    """Use complete disconnected cuffs, rather than clipping pant triangles."""
    bounds = []
    for doc, acc, index, expected in ((native, na, 6, 132), (donor, da, 21, 116)):
        primitive = doc['meshes'][0]['primitives'][index]
        positions = acc(primitive['attributes']['POSITION'])
        triangles, parts = components(doc, acc, primitive)
        cuffs = [p for p in parts if positions[np.unique(triangles[p]), 1].max() < .8]
        assert len(cuffs) == 2 and sum(map(len, cuffs)) == expected
        vertices = np.unique(triangles[np.concatenate(cuffs)])
        low, high = positions[vertices, 1].min(), positions[vertices, 1].max()
        for side in (-1, 1):
            selected = vertices[positions[vertices, 0] * side > 0]
            assert len(selected)
            assert np.allclose([positions[selected, 1].min(), positions[selected, 1].max()],
                               [low, high], atol=1e-7)
        bounds.append((float(low), float(high), vertices))
    target_low, target_high, _ = bounds[0]
    source_low, source_high, vertices = bounds[1]
    return dict(sourcePrimitiveIndex=21, sourceMinimumY=source_low, sourceMaximumY=source_high,
                targetMinimumY=target_low, targetMaximumY=target_high,
                heightScale=(target_high-target_low)/(source_high-source_low),
                sourceVertexIndices=vertices.tolist(), fittedVertices=len(vertices), fittedTriangles=116,
                scope='Both complete cinematic cuffs fit to native hem/cuff-top; X/Z, UVs and topology retained')


def fit_cuff_values(values, attribute, source_vertices, fit):
    values = values.copy()
    mask = np.isin(source_vertices, fit['sourceVertexIndices'])
    if attribute == 'POSITION':
        values[mask, 1] = ((values[mask, 1]-fit['sourceMinimumY'])*fit['heightScale'] +
                           fit['targetMinimumY'])
    elif attribute == 'NORMAL':
        values[mask, 1] /= fit['heightScale']
        values[mask] /= np.maximum(np.linalg.norm(values[mask], axis=1, keepdims=True), 1e-20)
    return values


def restore_open_feet(output, low_path, high_path, native_fr3, donor_fr3):
    current, old_binary, ca = load(output)
    native, native_binary, na = load(low_path)
    donor, _, da = load(high_path)
    cuff_fit = cuff_fit_metadata(native, na, donor, da)
    donor_indices = [i for i in range(28) if i not in (0, 1, 27)]
    assert len(current['meshes'][0]['primitives']) == len(donor_indices)
    names = [current['nodes'][i]['name'] for i in current['skins'][0]['joints']]
    assert names == [native['nodes'][i]['name'] for i in native['skins'][0]['joints']]
    raw = decompress(native_fr3)
    _, _, _, source_vertices = buffers(raw, 'assistant-lod0')
    source_rgba = np.frombuffer(source_vertices, 'u1').reshape(-1, 64)[:, 56:60]
    source_positions = np.frombuffer(source_vertices, '<f4').reshape(-1, 16)[:, :3]
    native_draws = source_draws(native_fr3, 'assistant-lod0')
    donor_draws = source_draws(donor_fr3, 'keira-highres-lod0')
    doc = copy.deepcopy(current)
    doc['accessors'], doc['bufferViews'] = [], []
    doc['meshes'][0]['primitives'] = []
    binary = bytearray()
    def view(data):
        binary.extend(b'\0' * (-len(binary) % 4))
        index = len(doc['bufferViews'])
        doc['bufferViews'].append(dict(buffer=0, byteOffset=len(binary), byteLength=len(data)))
        binary.extend(data)
        return index
    def add(values, source):
        index = len(doc['accessors'])
        accessor = {k: copy.deepcopy(source[k]) for k in ('componentType', 'type', 'normalized') if k in source}
        accessor.update(bufferView=view(values.tobytes()), count=len(values))
        if accessor['type'] == 'VEC3':
            accessor.update(min=values.min(0).tolist(), max=values.max(0).tolist())
        doc['accessors'].append(accessor)
        return index
    for image in doc['images']:
        old = current['bufferViews'][image['bufferView']]
        start = old.get('byteOffset', 0)
        image['bufferView'] = view(old_binary[start:start + old['byteLength']])
    doc['skins'][0]['inverseBindMatrices'] = add(ca(current['skins'][0]['inverseBindMatrices']),
                                              current['accessors'][current['skins'][0]['inverseBindMatrices']])
    manifest = []
    removed = []
    appended = []
    def append_primitive(source_doc, accessor, primitive, selected, source_index, origin,
                         source_vertices_map=None, native_material=None):
        triangle_indices = accessor(primitive['indices']).ravel().reshape(-1, 3)
        used, inverse = np.unique(triangle_indices[selected].ravel(), return_inverse=True)
        source_vertex_indices = used if source_vertices_map is None else source_vertices_map[used]
        attributes = {}
        for key, ai in primitive['attributes'].items():
            values = accessor(ai)[used]
            info = source_doc['accessors'][ai]
            if origin == 'native' and key == 'COLOR_0':
                values = source_rgba[used].copy()
                info = dict(componentType=5121, type='VEC4', normalized=True)
            if origin == 'native' and key == 'POSITION':
                assert np.array_equal(values * 4096, source_positions[used])
            if origin == 'donor' and source_index == 21:
                values = fit_cuff_values(values, key, source_vertex_indices, cuff_fit)
            attributes[key] = add(values, info)
        targets = []
        for target in primitive.get('targets', []):
            result = {}
            for key, ai in target.items():
                values = accessor(ai)[used]
                if origin == 'native':
                    assert np.max(np.abs(values)) < 1e-9, 'Native foot carries an unexpected facial morph'
                if origin == 'donor' and source_index == 21:
                    assert np.max(np.abs(values)) < 1e-9, 'Trouser cuffs carry an unexpected morph'
                result[key] = add(values, source_doc['accessors'][ai])
            targets.append(result)
        assert len(targets) == 29
        material = primitive['material'] if native_material is None else native_material
        out_index = len(doc['meshes'][0]['primitives'])
        draw = (native_draws if origin == 'native' else donor_draws)[source_index]
        row = dict(outputPrimitiveIndex=out_index, sourceGame='jak1' if origin == 'native' else 'jak2',
                   origin=origin, sourcePrimitiveIndex=source_index,
                   sourceMaterialName=source_doc['materials'][primitive['material']]['name'],
                   sourceVertexIndices=source_vertex_indices.tolist(), sourceTriangleIndices=selected.tolist(),
                   triangles=len(selected), vertices=len(used), **draw)
        row['nativeEffectIndex'] = draw['sourceEffectIndex'] if origin == 'native' else DONOR_NATIVE_EFFECTS[draw['sourceEffectIndex']]
        row['reflectionGroup'] = origin + '-effect-' + str(draw['sourceEffectIndex'])
        if origin == 'donor' and source_index == 21:
            row['cuffFit'] = cuff_fit
        primitive_out = dict(attributes=attributes,
                             indices=add(inverse.astype('<u4').reshape(-1, 1), dict(componentType=5125, type='SCALAR')),
                             material=material, mode=4, targets=targets,
                             extras=dict(cinematicSource={k: row[k] for k in ('origin', 'sourceGame', 'sourcePrimitiveIndex', 'nativeEffectIndex')}))
        doc['meshes'][0]['primitives'].append(primitive_out)
        manifest.append(row)
    for current_index, source_index in enumerate(donor_indices):
        primitive = current['meshes'][0]['primitives'][current_index]
        count = len(ca(primitive['indices'])) // 3
        dropped = np.asarray([], dtype=int)
        if source_index in DONOR_LIMITS:
            dropped, detail = select_lower_components(current, ca, primitive, DONOR_LIMITS[source_index])
            removed.append(dict(sourcePrimitiveIndex=source_index, triangles=len(dropped), components=detail))
        retained = np.setdiff1d(np.arange(count), dropped)
        if len(retained):
            source_map = np.unique(da(donor['meshes'][0]['primitives'][source_index]['indices']).ravel())
            append_primitive(current, ca, primitive, retained, source_index, 'donor', source_map)
    texture_map, image_map, sampler_map = {}, {}, {}
    def native_texture(index):
        if index in texture_map:
            return texture_map[index]
        texture = copy.deepcopy(native['textures'][index])
        image_index = texture['source']
        if image_index not in image_map:
            image = copy.deepcopy(native['images'][image_index])
            if 'uri' in image:
                prefix, encoded = image.pop('uri').split(',', 1)
                assert prefix == 'data:image/png;base64'
                image['bufferView'] = view(base64.b64decode(encoded))
                image['mimeType'] = 'image/png'
            else:
                source = native['bufferViews'][image['bufferView']]
                start = source.get('byteOffset', 0)
                image['bufferView'] = view(native_binary[start:start + source['byteLength']])
            image_map[image_index] = len(doc['images'])
            doc['images'].append(image)
        texture['source'] = image_map[image_index]
        if 'sampler' in texture:
            si = texture['sampler']
            if si not in sampler_map:
                sampler_map[si] = len(doc['samplers'])
                doc['samplers'].append(copy.deepcopy(native['samplers'][si]))
            texture['sampler'] = sampler_map[si]
        texture_map[index] = len(doc['textures'])
        doc['textures'].append(texture)
        return texture_map[index]
    for source_index, limit in NATIVE_LIMITS.items():
        primitive = native['meshes'][0]['primitives'][source_index]
        selected, detail = select_lower_components(native, na, primitive, limit)
        assert len(selected)
        material = copy.deepcopy(native['materials'][primitive['material']])
        material['name'] = 'jak1-open-feet/' + material['name']
        for container in (material, material.get('pbrMetallicRoughness', {})):
            for key, value in container.items():
                if key.endswith('Texture') and isinstance(value, dict) and 'index' in value:
                    value['index'] = native_texture(value['index'])
        material_index = len(doc['materials'])
        doc['materials'].append(material)
        append_primitive(native, na, primitive, selected, source_index, 'native', native_material=material_index)
        appended.append(dict(sourcePrimitiveIndex=source_index, triangles=len(selected), components=detail))
    assert sum(r['triangles'] for r in removed) == 1296, removed
    assert sum(r['triangles'] for r in appended) == 790, appended
    assert sum(r['triangles'] for r in manifest) == 10022
    save_glb(doc, binary, output)
    verification = verify_open_feet(output, low_path, high_path, manifest)
    report = dict(status='Original Jak 1 open feet and sandals restored with native lower-leg components',
                  donorTrianglesRemoved=1296, nativeTrianglesAppended=790, totalTriangles=10022,
                  removedComponents=removed, appendedComponents=appended, primitives=manifest,
                  seam='Complete native calves end beneath the cinematic pant hems; no cut triangles or exposed ankle seam',
                  nativeFootMorphTargetsZero=True, nativeRigAndWeightsPreserved=True,
                  cuffFit=cuff_fit,
                  verification=verification)
    output.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    # Shortening the donor cuff must also recover the native cuff's articulation.
    # Its original thigh blend was authored for a much longer piece of cloth.
    from refit_keira_cuffs import refit
    refit(output, low_path)
    report = json.loads(output.with_suffix('.provenance.json').read_text())
    report['verification'] = verify_open_feet(output, low_path, high_path, manifest)
    report['verification']['nativeCuffArticulationVerified'] = True
    output.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def verify_open_feet(output, low_path, high_path, manifest):
    """Check the saved hybrid against both extracted sources, including every morph."""
    result, _, ra = load(output)
    native, _, na = load(low_path)
    donor, _, da = load(high_path)
    cuff_fit = cuff_fit_metadata(native, na, donor, da)
    native_names = [native['nodes'][i]['name'] for i in native['skins'][0]['joints']]
    # Jak 1 evaluates only merc-ctrl.num-joints transformq joints. Unused
    # trailing skeleton joints can exist in jgeo without ever being evaluated.
    # Stay within the original geometry's active range instead of treating the
    # full exported skin array as proof that a joint is usable at runtime.
    native_active_limit = max(int(na(p['attributes']['JOINTS_0'])[
        na(p['attributes']['WEIGHTS_0']) > 0].max()) for p in native['meshes'][0]['primitives'])
    assert native_names == [result['nodes'][i]['name'] for i in result['skins'][0]['joints']]
    assert np.array_equal(na(native['skins'][0]['inverseBindMatrices']),
                          ra(result['skins'][0]['inverseBindMatrices']))
    rotation = np.asarray([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype='<f4')
    primitives = result['meshes'][0]['primitives']
    assert len(primitives) == len(manifest) == 29
    vertices = triangles = 0
    maximum_active_joint = 0
    for row, primitive in zip(manifest, primitives):
        is_native = row['origin'] == 'native'
        source_doc, sa = (native, na) if is_native else (donor, da)
        source = source_doc['meshes'][0]['primitives'][row['sourcePrimitiveIndex']]
        used = np.asarray(row['sourceVertexIndices'], dtype=int)
        selected = np.asarray(row['sourceTriangleIndices'], dtype=int)
        source_triangles = sa(source['indices']).ravel().reshape(-1, 3)[selected]
        assert np.array_equal(used[ra(primitive['indices']).ravel()], source_triangles.ravel())
        rotate = not is_native and row['sourcePrimitiveIndex'] in (2, 3, 4, 5)
        for key in ('POSITION', 'NORMAL', 'TEXCOORD_0'):
            expected = sa(source['attributes'][key])[used]
            if key == 'NORMAL' and not is_native:
                expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-20)
            if rotate and key in ('POSITION', 'NORMAL'):
                expected = expected @ rotation.T
            if not is_native and row['sourcePrimitiveIndex'] == 21:
                expected = fit_cuff_values(expected, key, used, cuff_fit)
            assert np.array_equal(expected, ra(primitive['attributes'][key])), (row['outputPrimitiveIndex'], key)
        for source_target, result_target in zip(source['targets'], primitive['targets']):
            assert source_target.keys() == result_target.keys()
            for key in source_target:
                expected = sa(source_target[key])[used]
                if rotate:
                    expected = expected @ rotation.T
                assert np.array_equal(expected, ra(result_target[key]))
        assert len(primitive['targets']) == len(source['targets']) == 29
        weights = ra(primitive['attributes']['WEIGHTS_0'])
        joints = ra(primitive['attributes']['JOINTS_0'])
        assert np.isfinite(weights).all() and (weights >= 0).all()
        assert np.allclose(weights.sum(1), 1, atol=1e-6)
        assert ((weights > 0).sum(1) <= 3).all() and joints.max() < len(native_names)
        maximum_active_joint = max(maximum_active_joint, int(joints[weights > 0].max()))
        assert maximum_active_joint <= native_active_limit, 'Influence lies outside original native evaluated joint range'
        assert not np.any((joints == native_names.index('prejoint')) & (weights > 0))
        for name in ('LpantFlap', 'RpantFlap'):
            assert not np.any((joints == native_names.index(name)) & (weights > 0)), 'Unevaluated native tail joint used'
        if is_native:
            assert np.array_equal(weights, sa(source['attributes']['WEIGHTS_0'])[used])
            assert np.array_equal(joints, sa(source['attributes']['JOINTS_0'])[used])
        elif row['sourcePrimitiveIndex'] in DONOR_LIMITS:
            remaining, _ = select_lower_components(result, ra, primitive, DONOR_LIMITS[row['sourcePrimitiveIndex']])
            assert not len(remaining), 'An enclosed shoe or donor calf component remains'
        triangles += len(selected)
        vertices += len(used)
    assert triangles == 10022
    assert [r['sourcePrimitiveIndex'] for r in manifest if r['origin'] == 'native'] == [0, 7, 8, 9, 10]
    return dict(triangles=triangles, drawVertices=vertices, primitives=len(primitives), nativeJoints=len(native_names),
                authoredMorphTargets=29, nativeJointOrderAndBindMatrices=True,
                retainedGeometryUVTopologyAndMorphsExact=False, unchangedSurfacesUVTopologyAndMorphsExact=True,
                documentedNativeCuffFitVerified=True, nativePantFlapWeightDependencies=0,
                maxActiveJointIndex=maximum_active_joint, nativeMaxActiveJointIndex=native_active_limit,
                influencesWithinOriginalNativeJointRange=True,
                nativeFootRigAndWeightsExact=True,
                remainingEnclosedShoeComponents=0, uncontrolledRootAccessoryVertices=0)
