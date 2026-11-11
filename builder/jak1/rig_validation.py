"""Verify cinematic surface provenance and the original gameplay rig."""
import hashlib
import numpy as np
from retarget_glb import load

def verify(low_path, high_path, output):
    low, _, la = load(low_path)
    high, _, ha = load(high_path)
    result, _, ra = load(output)
    native_names = [low['nodes'][i]['name'] for i in low['skins'][0]['joints']]
    result_names = [result['nodes'][i]['name'] for i in result['skins'][0]['joints']]
    assert native_names == result_names, 'Gameplay joint order changed'
    assert np.array_equal(la(low['skins'][0]['inverseBindMatrices']),
                          ra(result['skins'][0]['inverseBindMatrices']))
    originals = high['meshes'][0]['primitives']
    rebuilt = result['meshes'][0]['primitives']
    assert len(originals) == len(rebuilt)
    triangles = vertices = 0
    for source, dest in zip(originals, rebuilt):
        used, inverse = np.unique(ha(source['indices']).ravel(), return_inverse=True)
        assert np.array_equal(inverse, ra(dest['indices']).ravel())
        for attribute in ('POSITION', 'TEXCOORD_0'):
            assert np.array_equal(ha(source['attributes'][attribute])[used],
                                  ra(dest['attributes'][attribute])), attribute
        assert len(source.get('targets', [])) == len(dest.get('targets', []))
        for st, dt in zip(source.get('targets', []), dest.get('targets', [])):
            for attribute in st:
                assert np.array_equal(ha(st[attribute])[used], ra(dt[attribute]))
        weights = ra(dest['attributes']['WEIGHTS_0'])
        joints = ra(dest['attributes']['JOINTS_0'])
        assert np.isfinite(weights).all() and (weights >= 0).all()
        assert np.allclose(weights.sum(axis=1), 1, atol=1e-6)
        assert ((weights > 0).sum(axis=1) <= 3).all()
        assert joints.max() < len(native_names)
        triangles += len(inverse) // 3
        vertices += len(used)
    return dict(output=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                drawVertices=vertices, triangles=triangles, nativeJoints=len(native_names),
                authoredMorphTargets=max(len(p.get('targets', [])) for p in rebuilt),
                exactGeometryUVTopologyAndMorphs=True, nativeJointOrderAndBindMatrices=True)


