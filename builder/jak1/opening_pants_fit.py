"""Fit cinematic trousers beneath the tunic in sampled animated poses."""
import copy
import hashlib
from pathlib import Path

import numpy as np

from keira_feet import save_glb
from native_samos import _write
from retarget_glb import load

EXPECTED = {
    'POSITION': '9e0f98f7cbfe404b3e64cd2974b03727a97723c442b2711597f3ade12a4e6049',
    'NORMAL': '1f9a7a4d22c46c63bc5e7e554d5b2da69130e9f5b9ee2bd02dde6fe3e643a4a4',
    'TEXCOORD_0': '7ea63361bea3c3b1b09ccc4c5e085130feebaa1f58f0295c2b9ff1d42fe01ca8',
    'JOINTS_0': 'a9b27edc6b659e160a5ab48274c303dbe27202a1acbbf518b73aa9e9baddd82d',
    'WEIGHTS_0': '7500313beebb489426bf19e8d5de081c89c64d4d4b1871972363c9fea7337ec8',
    'indices': 'ef88a3c2c366a41f47c653d2715a96497fbf03b3ae976fe9e090ac3bab31ba00',
}
EXPECTED_WALK = dict(EXPECTED, POSITION='2abe5cca4d34ddc3a595cd8b1cc32fc53b64b8a518a1f153b47f6b97534cc078',
                     NORMAL='9c71ccb8277695bbe308e0aed1d17d65f4dc25a5a4ea41bd9700ded8f6d32140')


def _fit_pants(output, expected, rows, triangle_index, triangle, offset):
    output = Path(output)
    doc, binary, acc = load(output)
    primitive = doc['meshes'][0]['primitives'][22]
    assert doc['materials'][primitive['material']]['name'] == 'jak-orig-pants'
    assert primitive.get('mode', 4) == 4
    attrs = primitive['attributes']
    for name, digest in expected.items():
        index = primitive['indices'] if name == 'indices' else attrs[name]
        assert hashlib.sha256(acc(index).tobytes()).hexdigest() == digest, ('Pants input differs', name)
    assert all(not np.any(acc(index)) for target in primitive['targets'] for index in target.values())
    pos, normal = acc(attrs['POSITION']), acc(attrs['NORMAL'])
    triangles = acc(primitive['indices']).reshape(-1, 3)
    assert len(pos) == 274 and len(triangles) == 400 and triangles[triangle_index].tolist() == triangle
    assert np.allclose(pos[rows[0]] * [-1, 1, 1], pos[rows[1]], atol=1e-7)
    assert np.allclose(normal[rows[0]] * [-1, 1, 1], normal[rows[1]], atol=1e-7)
    assert all(np.count_nonzero(np.all(pos == pos[row], axis=1)) == 1 for row in rows)
    fitted = pos.copy()
    for row in rows:
        fitted[row] -= normal[row] * offset
    assert np.array_equal(fitted.min(0), pos.min(0)) and np.array_equal(fitted.max(0), pos.max(0))
    groups = {}
    membership = np.asarray([groups.setdefault((tuple(p), tuple(n)), len(groups)) for p, n in zip(pos, normal)])
    support = np.unique(triangles[np.isin(triangles, rows).any(1)])
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
    _write(output, doc, binary, {attrs['POSITION']: fitted, attrs['NORMAL']: revised_normal})
    return dict(sourcePrimitiveIndex=22, fittedVertices=rows, inwardOffsetMeters=offset,
                normalSupportRows=support.tolist(),
                maxRestNormalRotationDegrees=float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).max()),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def fit_opening_pants(output):
    return _fit_pants(output, EXPECTED, [92, 225], 138, [92, 83, 88], .004)


def fit_walking_pants(output):
    return _fit_pants(output, EXPECTED_WALK, [78, 208], 323, [208, 219, 170], .01)


EXPECTED_JUMP = dict(
    EXPECTED_WALK,
    POSITION='1891ff22d021b83855799a38fa63328e72a84d2b348cb5ee9c72c7fe94445d47',
    NORMAL='170d68432a87cd40f5f03b6afeb3f99860696b72d15cb2379fe2e3b56afc0309',
    COLOR_0='ddc6031e9af2c2c778237aa8767223fd97f36b87175f040e675c0261be2dbb17',
)


def fit_jump_pants(output):
    """Trim guarded, symmetric upper intersections after the walking fit."""
    output = Path(output)
    doc, binary, acc = load(output)
    primitive = doc['meshes'][0]['primitives'][22]
    assert doc['materials'][primitive['material']]['name'] == 'jak-orig-pants'
    assert primitive.get('mode', 4) == 4
    for name, expected in EXPECTED_JUMP.items():
        index = primitive['indices'] if name == 'indices' else primitive['attributes'][name]
        assert hashlib.sha256(acc(index).tobytes()).hexdigest() == expected, ('Pants input differs', name)
    assert all(not np.any(acc(index)) for target in primitive['targets'] for index in target.values())
    triangles = acc(primitive['indices']).reshape(-1, 3)
    removed = {134: [79, 80, 87], 135: [80, 88, 87], 335: [216, 218, 210], 336: [209, 216, 210]}
    assert len(triangles) == 400 and all(triangles[i].tolist() == vertices for i, vertices in removed.items())
    retained = np.delete(triangles, list(removed), axis=0).astype('<u4').reshape(-1, 1)
    assert len(retained) == 396 * 3
    raw = bytearray(binary)
    raw.extend(b'\0' * (-len(raw) % 4))
    view = len(doc['bufferViews'])
    doc['bufferViews'].append(dict(buffer=0, byteOffset=len(raw), byteLength=retained.nbytes))
    raw.extend(retained.tobytes())
    entry = copy.deepcopy(doc['accessors'][primitive['indices']])
    entry.update(bufferView=view, count=len(retained))
    entry.pop('byteOffset', None)
    primitive['indices'] = len(doc['accessors'])
    doc['accessors'].append(entry)
    save_glb(doc, raw, output)
    return dict(sourcePrimitiveIndex=22, removedTriangles=list(removed), originalTriangles=400,
                retainedTriangles=396, sha256=hashlib.sha256(output.read_bytes()).hexdigest())
