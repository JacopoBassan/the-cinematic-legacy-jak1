"""Fit concealed trouser intersections using locally extracted cinematic data."""
import copy
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from retarget_glb import load
from keira_feet import save_glb
from native_samos import _write

EXPECTED = {
    'POSITION': '881015d9a456be5c17d94484f667841ff94eacfeac1eb54faeb8e1d80ea8bbf6',
    'NORMAL': 'dd25d0ceb455ee16cd41da43ea385ba7c0a4bc90f0f3e84ed9c9619b0ecb0024',
    'TEXCOORD_0': '7ea63361bea3c3b1b09ccc4c5e085130feebaa1f58f0295c2b9ff1d42fe01ca8',
    'JOINTS_0': 'a9b27edc6b659e160a5ab48274c303dbe27202a1acbbf518b73aa9e9baddd82d',
    'WEIGHTS_0': '7500313beebb489426bf19e8d5de081c89c64d4d4b1871972363c9fea7337ec8',
    'indices': '159e374320de58bdf90dd93a40db098effe300a66f9b3a6b7a4aa137729fe6d4',
}
REMOVED_TRIANGLES = tuple(range(134, 140)) + tuple(range(343, 349))


def fit_cinematic_pants(output):
    output = Path(output)
    doc, binary, acc = load(output)
    primitive = doc['meshes'][0]['primitives'][22]
    assert doc['materials'][primitive['material']]['name'] == 'jak-orig-pants'
    assert primitive.get('mode', 4) == 4, 'Expected independent GLB triangles'
    attrs = primitive['attributes']
    for name, expected in EXPECTED.items():
        index = primitive['indices'] if name == 'indices' else attrs[name]
        assert hashlib.sha256(acc(index).tobytes()).hexdigest() == expected, ('Pants input differs', name)
    assert all(not np.any(acc(index)) for target in primitive['targets'] for index in target.values())
    pos, normal = acc(attrs['POSITION']), acc(attrs['NORMAL'])
    triangles = acc(primitive['indices']).reshape(-1, 3)
    assert len(pos) == 274 and len(triangles) == 412
    assert np.allclose(pos[84] * [-1, 1, 1], pos[215], atol=1e-7)
    assert np.allclose(normal[84] * [-1, 1, 1], normal[215], atol=1e-7)
    assert all(np.count_nonzero(np.all(pos == pos[row], axis=1)) == 1 for row in (84, 215))
    fitted = pos.copy()
    for row in (84, 215):
        fitted[row] -= normal[row] * .004
    assert np.array_equal(fitted.min(0), pos.min(0)) and np.array_equal(fitted.max(0), pos.max(0))
    groups = {}
    membership = np.asarray([groups.setdefault((tuple(p), tuple(n)), len(groups)) for p, n in zip(pos, normal)])
    support = np.unique(triangles[np.isin(triangles, [84, 215]).any(1)])
    support = np.flatnonzero(np.isin(membership, membership[support]))

    def geometric(points):
        tri = points[triangles]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        total = np.zeros((len(groups), 3), dtype=float)
        for column in range(3):
            np.add.at(total, membership[triangles[:, column]], cross)
        return total / np.maximum(np.linalg.norm(total, axis=1, keepdims=True), 1e-30)

    before = geometric(pos)[membership[support]]
    after = geometric(fitted)[membership[support]]
    axis = np.cross(before, after)
    cosine = (before * after).sum(1)
    assert cosine.min() > .95
    values = normal[support].astype(float)
    values = values + np.cross(axis, values) + np.cross(axis, np.cross(axis, values)) / (1 + cosine[:, None])
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-30)
    revised_normal = normal.copy()
    revised_normal[support] = values
    # Keep independent triangles and their winding. Do not reindex or collapse
    # strips, weld vertices, or modify skinning, UVs, textures, or morphs.
    retained = np.delete(triangles, REMOVED_TRIANGLES, axis=0).astype('<u4').reshape(-1, 1)
    assert len(retained) == 400 * 3
    with tempfile.TemporaryDirectory(prefix='pants-fit-', dir=output.parent) as temporary:
        intermediate = Path(temporary) / 'surface.glb'
        _write(intermediate, doc, binary, {attrs['POSITION']: fitted, attrs['NORMAL']: revised_normal})
        result, old_binary, _ = load(intermediate)
        out_primitive = result['meshes'][0]['primitives'][22]
        raw = bytearray(old_binary)
        raw.extend(b'\0' * (-len(raw) % 4))
        view = len(result['bufferViews'])
        result['bufferViews'].append(dict(buffer=0, byteOffset=len(raw), byteLength=retained.nbytes))
        raw.extend(retained.tobytes())
        entry = copy.deepcopy(result['accessors'][out_primitive['indices']])
        entry.update(bufferView=view, count=len(retained))
        entry.pop('byteOffset', None)
        out_primitive['indices'] = len(result['accessors'])
        result['accessors'].append(entry)
        save_glb(result, raw, output)
    return dict(sourcePrimitiveIndex=22, removedTriangles=list(REMOVED_TRIANGLES),
                originalTriangles=412, retainedTriangles=400, fittedVertices=[84, 215],
                inwardOffsetMeters=.004, normalSupportRows=support.tolist(),
                maxRestNormalRotationDegrees=float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).max()),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest())
