"""Transfer native cuff articulation to fitted cinematic cuffs without changing geometry."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retarget_glb import load


def nearest_triangle(point, triangles):
    """Exact closest point, considering triangle interior and all three edges."""
    a, b, c = triangles.transpose(1, 0, 2)
    ab, ac, ap = b-a, c-a, point-a
    d00, d01, d11 = (ab*ab).sum(1), (ab*ac).sum(1), (ac*ac).sum(1)
    d20, d21 = (ap*ab).sum(1), (ap*ac).sum(1)
    denominator = d00*d11-d01*d01
    valid = denominator > 1e-16
    v = np.divide(d11*d20-d01*d21, denominator, out=np.zeros_like(d00), where=valid)
    w = np.divide(d00*d21-d01*d20, denominator, out=np.zeros_like(d00), where=valid)
    bary = np.column_stack([1-v-w, v, w])
    distance = np.where(valid & (bary >= 0).all(1),
                        ((np.einsum('ni,nij->nj', bary, triangles)-point)**2).sum(1), np.inf)
    all_bary, all_distance = [bary], [distance]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, j]-triangles[:, i]
        length = (edge*edge).sum(1)
        t = np.clip(np.divide(((point-triangles[:, i])*edge).sum(1), length,
                              out=np.zeros_like(length), where=length > 1e-16), 0, 1)
        weights = np.zeros_like(bary)
        weights[:, i], weights[:, j] = 1-t, t
        all_bary.append(weights)
        all_distance.append(((triangles[:, i]+t[:, None]*edge-point)**2).sum(1))
    choices = np.asarray(all_distance)
    case, index = np.unravel_index(choices.argmin(), choices.shape)
    return int(index), all_bary[case][index], float(np.sqrt(choices[case, index]))


def verify_native_cuff_articulation(output, native_path):
    """Reject incomplete or altered cuff transfers before import."""
    doc, _, acc = load(output)
    native, _, na = load(native_path)
    manifest = json.loads(output.with_suffix('.provenance.json').read_text())
    row = next(r for r in manifest['primitives'] if r['origin'] == 'donor' and r['sourcePrimitiveIndex'] == 21)
    primitive = doc['meshes'][0]['primitives'][row['outputPrimitiveIndex']]
    records = row['nativeCuffArticulation']['verticesDetail']
    sources = np.asarray(row['sourceVertexIndices'])
    selected = set(np.where(np.isin(sources, row['cuffFit']['sourceVertexIndices']))[0].tolist())
    assert len(records) == len(selected) == 120, 'Incomplete native cuff articulation records'
    assert {r['vertex'] for r in records} == selected, 'Missing or duplicated cuff vertex record'
    positions = acc(primitive['attributes']['POSITION'])
    joints, weights = acc(primitive['attributes']['JOINTS_0']), acc(primitive['attributes']['WEIGHTS_0'])
    source = native['meshes'][0]['primitives'][6]
    source_positions = na(source['attributes']['POSITION']).astype(float)
    triangles = na(source['indices']).reshape(-1, 3)
    source_joints, source_weights = na(source['attributes']['JOINTS_0']), na(source['attributes']['WEIGHTS_0'])
    count = len(native['skins'][0]['joints'])
    maximum_error = 0.0
    for record in records:
        vi, ti = record['vertex'], record['nativeTriangle']
        assert record['sourceVertex'] == int(sources[vi]) and record['nativeSourcePrimitive'] == 6
        assert 0 <= ti < len(triangles)
        ids = triangles[ti]
        side = 'L' if positions[vi, 0] > 0 else 'R'
        sign = 1 if side == 'L' else -1
        assert record['side'] == side
        assert source_positions[ids, 1].max() < .8 and (source_positions[ids, 0]*sign > 0).all()
        barycentric = np.asarray(record['barycentric'])
        assert np.isfinite(barycentric).all() and (barycentric >= 0).all() and (barycentric <= 1).all()
        assert abs(barycentric.sum()-1) < 1e-7
        expected = np.zeros(count)
        actual = np.zeros(count)
        for vertex, amount in zip(ids, barycentric):
            np.add.at(expected, source_joints[vertex], source_weights[vertex]*amount)
        expected /= expected.sum()
        np.add.at(actual, joints[vi], weights[vi])
        maximum_error = max(maximum_error, float(np.abs(expected-actual).max()))
        assert np.allclose(expected, actual, atol=1e-6, rtol=0)
        assert np.count_nonzero(weights[vi] > 0) <= 3
    return dict(vertices=120, nativeWeightsReconstructed=True, maximumWeightError=maximum_error)


def refit(output, native_path):
    doc, binary, acc = load(output)
    native, _, na = load(native_path)
    manifest_path = output.with_suffix('.provenance.json')
    manifest = json.loads(manifest_path.read_text())
    row = next(r for r in manifest['primitives'] if r['origin'] == 'donor' and r['sourcePrimitiveIndex'] == 21)
    assert 'cuffFit' in row
    primitive = doc['meshes'][0]['primitives'][row['outputPrimitiveIndex']]
    positions = acc(primitive['attributes']['POSITION'])
    source_vertices = np.asarray(row['sourceVertexIndices'])
    selected = np.where(np.isin(source_vertices, row['cuffFit']['sourceVertexIndices']))[0]
    assert len(selected) == 120
    native_primitive = native['meshes'][0]['primitives'][6]
    native_positions = na(native_primitive['attributes']['POSITION']).astype(float)
    native_triangles = na(native_primitive['indices']).reshape(-1, 3).astype(int)
    native_joints = na(native_primitive['attributes']['JOINTS_0'])
    native_weights = na(native_primitive['attributes']['WEIGHTS_0'])
    names = [native['nodes'][i]['name'] for i in native['skins'][0]['joints']]
    assert names == [doc['nodes'][i]['name'] for i in doc['skins'][0]['joints']]
    joints, weights = acc(primitive['attributes']['JOINTS_0']).copy(), acc(primitive['attributes']['WEIGHTS_0']).copy()
    old_joints, old_weights = joints.copy(), weights.copy()
    diagnostics = []
    for vi in selected:
        side = 'L' if positions[vi, 0] > 0 else 'R'
        sign = 1 if side == 'L' else -1
        allowed = ((native_positions[native_triangles, 1].max(1) < .8) &
                   (native_positions[native_triangles, 0]*sign > 0).all(1))
        triangle_ids = np.where(allowed)[0]
        assert len(triangle_ids) == 66
        triangle_index, barycentric, distance = nearest_triangle(
            positions[vi].astype(float), native_positions[native_triangles[triangle_ids]])
        barycentric = np.clip(barycentric, 0, 1)
        barycentric /= barycentric.sum()
        ti = int(triangle_ids[triangle_index])
        totals = {}
        for vertex, amount in zip(native_triangles[ti], barycentric):
            for bone, weight in zip(native_joints[vertex], native_weights[vertex]):
                if weight > 0:
                    totals[int(bone)] = totals.get(int(bone), 0.0) + amount*float(weight)
        totals = {bone: weight for bone, weight in totals.items() if weight > 1e-8}
        assert set(totals) <= {names.index(side+'knee'), names.index(side+'thigh'),
                               names.index('lKneeFlap' if side == 'L' else 'rKneeFlap')}
        assert 1 <= len(totals) <= 3
        joints[vi], weights[vi] = 0, 0
        total = sum(totals.values())
        for column, (bone, weight) in enumerate(sorted(totals.items(), key=lambda v: -v[1])):
            joints[vi, column], weights[vi, column] = bone, weight/total
        diagnostics.append(dict(vertex=int(vi), sourceVertex=int(source_vertices[vi]),
                                nativeSourcePrimitive=6, side=side, nativeTriangle=ti,
                                barycentric=barycentric.tolist(),
                                nativeSurfaceDistanceMeters=distance))
    assert np.allclose(weights.sum(1), 1, atol=1e-6)
    assert ((weights > 0).sum(1) <= 3).all()
    unselected = np.setdiff1d(np.arange(len(joints)), selected)
    assert np.array_equal(joints[unselected], old_joints[unselected])
    assert np.array_equal(weights[unselected], old_weights[unselected])
    binary = bytearray(binary)
    for attribute, values in (('JOINTS_0', joints), ('WEIGHTS_0', weights)):
        accessor = doc['accessors'][primitive['attributes'][attribute]]
        view = doc['bufferViews'][accessor['bufferView']]
        assert 'byteStride' not in view
        start = view.get('byteOffset', 0)+accessor.get('byteOffset', 0)
        binary[start:start+values.nbytes] = values.tobytes()
    # Import the writer without requiring FR3 utilities at module import time.
    import os
    os.environ.setdefault('CINEMATIC_BUILD_WORK', str(output.parent))
    from keira_feet import save_glb
    report = dict(method='Closest native cuff triangle, barycentric native skin weights',
                  vertices=len(selected), geometryUVTopologyAndMorphsUnchanged=True,
                  maximumNativeSurfaceDistanceMeters=max(r['nativeSurfaceDistanceMeters'] for r in diagnostics),
                  meanNativeSurfaceDistanceMeters=float(np.mean([r['nativeSurfaceDistanceMeters'] for r in diagnostics])),
                  verticesDetail=diagnostics)
    primitive.setdefault('extras', {})['nativeCuffArticulation'] = True
    save_glb(doc, binary, output)
    result, _, ra = load(output)
    for pi, (old, new) in enumerate(zip(doc['meshes'][0]['primitives'], result['meshes'][0]['primitives'])):
        assert old.keys() == new.keys()
        for attribute in old['attributes']:
            if pi == row['outputPrimitiveIndex'] and attribute in ('JOINTS_0', 'WEIGHTS_0'):
                continue
            assert np.array_equal(acc(old['attributes'][attribute]), ra(new['attributes'][attribute]))
        assert np.array_equal(acc(old['indices']), ra(new['indices']))
        for old_target, new_target in zip(old['targets'], new['targets']):
            for attribute in old_target:
                assert np.array_equal(acc(old_target[attribute]), ra(new_target[attribute]))
    assert np.array_equal(joints, ra(result['meshes'][0]['primitives'][row['outputPrimitiveIndex']]['attributes']['JOINTS_0']))
    assert np.array_equal(weights, ra(result['meshes'][0]['primitives'][row['outputPrimitiveIndex']]['attributes']['WEIGHTS_0']))
    row['nativeCuffArticulation'] = report
    manifest['nativeCuffArticulation'] = report
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    report['verification'] = verify_native_cuff_articulation(output, native_path)
    report['outputSha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name('cuff-weight-transfer.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    from jak1_paths import DATA
    parser.add_argument('--native', type=Path, default=DATA/
                        'decompiler_out/jak1/levels/village1/assistant-lod0.glb')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(args.source.read_bytes())
    args.output.with_suffix('.provenance.json').write_bytes(args.source.with_suffix('.provenance.json').read_bytes())
    report = refit(args.output, args.native)
    print(json.dumps({k: v for k, v in report.items() if k != 'verticesDetail'}, indent=2))


if __name__ == '__main__':
    main()
