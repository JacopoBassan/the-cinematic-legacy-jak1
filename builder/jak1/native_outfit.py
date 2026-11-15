"""Apply locally extracted gameplay clothing textures to cinematic Jak."""
import base64
import copy
import json
import struct
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from retarget_glb import load

def nearest(points, tri):
    tree = cKDTree(tri.mean(1))
    _, near = tree.query(points, k=min(48, len(tri)))
    if near.ndim == 1:
        near = near[:, None]
    t = tri[near]
    p = points[:, None, :]
    a, b, c = (t[:, :, 0], t[:, :, 1], t[:, :, 2])
    ab = b - a
    ac = c - a
    ap = p - a
    dot = lambda x, y: np.sum(x * y, axis=-1)
    aa, cc, dd = (dot(ab, ab), dot(ab, ac), dot(ac, ac))
    det = aa * dd - cc * cc
    u = np.divide(dd * dot(ap, ab) - cc * dot(ap, ac), det, out=np.zeros_like(det), where=abs(det) > 1e-12)
    v = np.divide(aa * dot(ap, ac) - cc * dot(ap, ab), det, out=np.zeros_like(det), where=abs(det) > 1e-12)
    bary = np.stack([1 - u - v, u, v], axis=-1)
    q = a + u[:, :, None] * ab + v[:, :, None] * ac
    dist = np.where((bary >= 0).all(-1) & (abs(det) > 1e-12), dot(q - p, q - p), np.inf)
    for x, y, ix, iy in [(a, b, 0, 1), (a, c, 0, 2), (b, c, 1, 2)]:
        edge = y - x
        den = dot(edge, edge)
        f = np.clip(np.divide(dot(p - x, edge), den, out=np.zeros_like(den), where=den > 1e-12), 0, 1)
        q = x + f[:, :, None] * edge
        d = dot(q - p, q - p)
        better = d < dist
        edge_bary = np.zeros_like(bary)
        edge_bary[:, :, ix] = 1 - f
        edge_bary[:, :, iy] = f
        bary = np.where(better[:, :, None], edge_bary, bary)
        dist = np.minimum(dist, d)
    choice = dist.argmin(1)
    rr = np.arange(len(points))
    return (near[rr, choice], bary[rr, choice], np.sqrt(dist[rr, choice]))
MAPPING = {
    4: 'bam-armor-envmap',
    5: 'bam-armor-envmap',
    6: 'bam-armor-envmap',
    7: 'bam-armor-envmap',
    12: 'bam-cotton-wrap',
    13: 'bam-leather-belt',
    14: 'bam-leather-belt',
    15: 'bam-coat-wrinkles',
    18: 'bam-cotton-wrap',
    19: 'bam-leather-belt',
    20: 'bam-coat-wrinkles',
    21: 'bam-leather-belt',
    22: 'bam-cotton-pants',
    23: 'bam-target-16x16-envmap',
}

def make(native_path, high_path, output, report_path):
    low, low_bin, la = load(native_path)
    doc, high_bin, ha = load(high_path)
    binary = bytearray(high_bin)
    expected = [
        'jak-orig-hairplain-envmap',
        'jak-orig-hair-envmap',
        'programmer_eye_left-envmap',
        'programmer_eye_right-envmap',
        'jak-orig-armor-envmap',
        'jak-orig-clips-envmap',
        'jak-orig-goggles-envmap',
        'jak-orig-lenscover-envmap',
        'jak-orig-face',
        'jak-orig-teeth',
        'jak-orig-eyebrow',
        'charHOLD',
        'jak-orig-wraps',
        'jak-orig-belt',
        'jak-orig-leatherpouch',
        'jak-orig-jackettop',
        'jak-orig-arm',
        'jak-orig-finger',
        'jak-orig-handwraps',
        'jak-orig-strap',
        'jak-orig-skirt',
        'jak-orig-earflaps',
        'jak-orig-pants',
        'jak-orig-lens',
    ]
    parts = doc['meshes'][0]['primitives']
    assert [doc['materials'][p['material']]['name'] for p in parts] == expected, 'Unexpected cinematic Jak materials'
    changed_views = []
    for index in MAPPING:
        attributes = parts[index]['attributes']
        for name, component, kind in [('TEXCOORD_0', 5126, 'VEC2'), ('COLOR_0', 5121, 'VEC4')]:
            entry = doc['accessors'][attributes[name]]
            assert entry['componentType'] == component and entry['type'] == kind
            changed_views.append(entry['bufferView'])
    assert len(changed_views) == len(set(changed_views)), 'Clothing attributes share an output buffer'
    other_views = []
    for index, part in enumerate(parts):
        for name, accessor in part['attributes'].items():
            if index not in MAPPING or name not in ('TEXCOORD_0', 'COLOR_0'):
                other_views.append(doc['accessors'][accessor]['bufferView'])
        other_views.append(doc['accessors'][part['indices']]['bufferView'])
        for target in part.get('targets', []):
            other_views.extend((doc['accessors'][accessor]['bufferView'] for accessor in target.values()))
    assert not set(changed_views).intersection(other_views), 'Outfit buffers overlap unrelated data'
    assert not output.exists() and (not report_path.exists())
    native_parts = low['meshes'][0]['primitives']
    native_names = [low['materials'][p['material']]['name'] for p in native_parts]
    assert all((native_names.count(name) == 1 for name in set(MAPPING.values()))), 'Unexpected gameplay Jak materials'
    src = {low['materials'][p['material']]['name']: p for p in native_parts}
    # Keep native image names so the texture builder restores PS2 alpha.
    used_textures = {}
    rows = []

    def view(data):
        binary.extend(b'\x00' * (-len(binary) % 4))
        vi = len(doc['bufferViews'])
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(data)})
        binary.extend(data)
        return vi

    def replace(ai, array):
        entry = doc['accessors'][ai]
        v = doc['bufferViews'][entry['bufferView']]
        start = v.get('byteOffset', 0) + entry.get('byteOffset', 0)
        assert v.get('byteStride', array.strides[0]) == array.strides[0] and array.nbytes <= v['byteLength']
        binary[start:start + array.nbytes] = array.tobytes()
    for i, material in MAPPING.items():
        hp = doc['meshes'][0]['primitives'][i]
        lp = src[material]
        native_mat = low['materials'][lp['material']]
        texture_index = native_mat['pbrMetallicRoughness']['baseColorTexture']['index']
        tex = low['textures'][texture_index]
        native_image = low['images'][tex['source']]
        key = native_image['name']
        if key not in used_textures:
            im = copy.deepcopy(native_image)
            if 'uri' in im:
                prefix, encoded = im.pop('uri').split(',', 1)
                assert prefix == 'data:image/png;base64'
                data = base64.b64decode(encoded)
            else:
                old = low['bufferViews'][im['bufferView']]
                start = old.get('byteOffset', 0)
                data = low_bin[start:start + old['byteLength']]
            im.update(bufferView=view(data), mimeType='image/png', name=key)
            image = len(doc['images'])
            doc['images'].append(im)
            texture = copy.deepcopy(tex)
            texture['source'] = image
            texture['sampler'] = len(doc['samplers'])
            doc['samplers'].append(copy.deepcopy(low['samplers'][tex['sampler']]))
            used_textures[key] = len(doc['textures'])
            doc['textures'].append(texture)
        doc['materials'][hp['material']]['pbrMetallicRoughness']['baseColorTexture']['index'] = used_textures[key]
        faces = la(lp['indices']).ravel().reshape(-1, 3).astype(int)
        hf = ha(hp['indices']).ravel().reshape(-1, 3).astype(int)
        points = ha(hp['attributes']['POSITION']).astype(float)
        tri = la(lp['attributes']['POSITION'])[faces].astype(float)
        assert len(points) and len(tri) and np.isfinite(points).all() and np.isfinite(tri).all()
        chosen, bary, dist = nearest(points, tri)
        assert np.isfinite(dist).all() and np.max(dist) < 0.5, ('Gameplay clothing surface mismatch', i)
        raw_uv = np.einsum('ni,nij->nj', bary, la(lp['attributes']['TEXCOORD_0'])[faces[chosen]])
        color = np.einsum('ni,nij->nj', bary, la(lp['attributes']['COLOR_0'])[faces[chosen]])
        # Preserve the existing mesh while unwrapping repeated UVs per component.
        neighbors = [set() for _ in range(len(points))]
        for a, b, c in hf:
            neighbors[a].update((b, c))
            neighbors[b].update((a, c))
            neighbors[c].update((a, b))
        uv = np.empty_like(raw_uv)
        seen = np.zeros(len(points), bool)
        components = 0
        for seed in range(len(points)):
            if seen[seed]:
                continue
            components += 1
            seen[seed] = True
            uv[seed] = raw_uv[seed]
            queue = deque([seed])
            while queue:
                here = queue.popleft()
                for other in sorted(neighbors[here]):
                    if seen[other]:
                        continue
                    uv[other] = raw_uv[other] + np.round(uv[here] - raw_uv[other])
                    seen[other] = True
                    queue.append(other)
        edges = np.concatenate((hf[:, [0, 1]], hf[:, [1, 2]], hf[:, [2, 0]]))
        delta = np.abs(uv[edges[:, 0]] - uv[edges[:, 1]])
        replace(hp['attributes']['TEXCOORD_0'], np.ascontiguousarray(uv, dtype='<f4'))
        rgba = np.rint(np.clip(color, 0, 1) * 255).astype('u1')
        rgba[:, 3] = ha(hp['attributes']['COLOR_0'])[:, 3]
        replace(hp['attributes']['COLOR_0'], np.ascontiguousarray(rgba))
        rows.append(dict(primitive=i, name=doc['materials'][hp['material']]['name'], native=material, vertices=len(points), components=components, p95DistanceMeters=float(np.quantile(dist, 0.95)), maximumDistanceMeters=float(np.max(dist)), edgesWithUvJumpOverHalf=int((delta.max(1) > 0.5).sum()), edges=len(edges)))
    doc['buffers'] = [{'byteLength': len(binary)}]
    encoded = json.dumps(doc, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary += b'\x00' * (-len(binary) % 4)
    output.write_bytes(struct.pack('<III', 1179937895, 2, 28 + len(encoded) + len(binary)) + struct.pack('<II', len(encoded), 1313821514) + encoded + struct.pack('<II', len(binary), 5130562) + binary)
    report_path.write_text(json.dumps(rows, indent=2) + '\n')
    return rows
