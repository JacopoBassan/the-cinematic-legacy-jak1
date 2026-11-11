"""Asset-free checks for independent native jaw and eyebrow skin drivers."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'builder'))
from jak1.native_face_rig import distribute_head_weights, remap_material_joints, lower_mouth_ownership, closest_upper_tooth_surface


class NativeFaceRigTests(unittest.TestCase):
    def test_upper_tooth_ties_use_source_order_but_keep_real_distance_differences(self):
        # The later triangle has a nearer centroid. That must not decide an
        # equal surface-distance tie, including roundoff-sized perturbations.
        for dtype in (np.float32, np.float64):
            for height, expected in [(2., 0), (np.nextafter(2., 1.), 0),
                                     (2.-1e-14, 0), (2.-1e-6, 1)]:
                triangles = np.array([[[-10, -10, 0], [10, -10, 0], [0, 10, 0]],
                                      [[-1, -1, height], [1, -1, height], [0, 1, height]]],
                                     dtype=dtype)
                nearest, barycentric, distance = closest_upper_tooth_surface(
                    np.array([[0, 0, 1]], dtype=dtype), triangles)
                self.assertEqual(int(nearest[0]), expected)
                np.testing.assert_allclose(barycentric.sum(axis=1), 1, atol=1e-15)
                projected = np.sum(triangles[nearest]*barycentric[..., None], axis=1)
                np.testing.assert_allclose(projected[0, :2], [0, 0], atol=1e-14)
                self.assertAlmostEqual(float(distance[0]), abs(float(projected[0, 2])-1), places=12)

    def test_upper_tooth_edge_projection_handles_a_degenerate_triangle(self):
        triangles = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=np.float32)
        nearest, barycentric, distance = closest_upper_tooth_surface(
            np.array([[.5, 1, 0], [-1, 0, 0], [3, 0, 0]]), triangles)
        projected = np.sum(triangles[nearest]*barycentric[..., None], axis=1)
        np.testing.assert_allclose(projected, [[.5, 0, 0], [0, 0, 0], [2, 0, 0]])
        np.testing.assert_allclose(distance, [1, 1, 1])
        self.assertTrue(np.isfinite(barycentric).all())

    def test_authored_opening_separates_lower_rim_from_touching_upper_lip(self):
        # A closed lookup incorrectly gives both touching rim vertices zero.
        # Their independent authored responses identify upper/lower ownership.
        closed = np.array([0., 0., 1., .2, 0.])
        opened = np.array([.9, 1., 1., .8, 0.])
        dy = np.array([.005, -.085, -.085, -.04, 0.])
        got = lower_mouth_ownership(closed, opened, dy, .005, -.085)
        np.testing.assert_allclose(got, [0., 1., 1., .4, 0.])
        self.assertTrue(np.all(got >= closed))
        self.assertTrue(np.all((got >= 0) & (got <= 1)))

    def test_jaw_and_brows_respond_independently_without_opening_upper_mouth(self):
        # Upper lip, lower lip, upper teeth, lower teeth, left/right brows.
        joints = np.tile([6, 0, 0, 0], (6, 1)).astype('u1')
        weights = np.tile([1., 0, 0, 0], (6, 1)).astype('<f4')
        field = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0],
                          [1, 0, 0], [0, .75, 0], [0, 0, .5]])
        got_joints, got_weights = distribute_head_weights(joints, weights, 6, [36, 37, 38], field)
        for row in [0, 2]:
            np.testing.assert_array_equal(got_joints[row], joints[row])
            np.testing.assert_array_equal(got_weights[row], weights[row])
        # Evaluate distinct animated bone translations. A static donor face
        # would produce zero for every row and fail this response assertion.
        translations = np.zeros((40, 3))
        translations[36] = [0, -.02, .01]
        translations[37] = [0, .004, 0]
        translations[38] = [0, -.003, 0]
        response = np.sum(translations[got_joints]*got_weights[:, :, None], axis=1)
        np.testing.assert_allclose(response, [[0, 0, 0], [0, -.02, .01], [0, 0, 0],
                                              [0, -.02, .01], [0, .003, 0], [0, -.0015, 0]], atol=1e-9)

    def test_existing_neck_blend_is_preserved_and_zero_field_is_byte_exact(self):
        joints = np.array([[5, 6, 0, 0], [5, 6, 0, 0]], dtype='u1')
        weights = np.array([[.25, .75, 0, 0], [.25, .75, 0, 0]], dtype='<f4')
        got_joints, got_weights = distribute_head_weights(joints, weights, 6, [36, 37, 38],
            np.array([[.8, 0, 0], [0, 0, 0]]))
        self.assertEqual(float(got_weights[0][got_joints[0] == 5].sum()), .25)
        self.assertAlmostEqual(float(got_weights[0][got_joints[0] == 36].sum()), .6)
        np.testing.assert_array_equal(got_joints[1], joints[1])
        np.testing.assert_array_equal(got_weights[1], weights[1])
        np.testing.assert_allclose(got_weights.sum(1), 1)
        self.assertTrue(((got_weights > 0).sum(1) <= 3).all())

    def test_unsupported_influence_overflow_and_invalid_field_are_rejected(self):
        joints = np.array([[4, 5, 6, 0]], dtype='u1')
        weights = np.array([[.2, .3, .5, 0]], dtype='<f4')
        for field in [np.array([[.5, 0, 0]]), np.array([[1., .1, 0]]), np.array([[-.1, 0, 0]])]:
            with self.assertRaises(AssertionError):
                distribute_head_weights(joints, weights, 6, [36, 37, 38], field)

    def test_independent_trouser_knees_remap_only_the_cloth_material(self):
        data = {}
        primitives = []
        names = ['jak-orig-pants', 'jak-orig-face', 'jak-orig-boots']
        for part in range(3):
            data[f'j{part}'] = np.array([[26, 27, 0, 0], [30, 31, 0, 0]], dtype='u1')
            data[f'w{part}'] = np.array([[.4, .6, 0, 0], [.3, .7, 0, 0]], dtype='<f4')
            primitives.append(dict(material=part, attributes={'JOINTS_0': f'j{part}', 'WEIGHTS_0': f'w{part}'}))
        doc = dict(materials=[dict(name=name) for name in names], meshes=[dict(primitives=primitives)])
        result = remap_material_joints(doc, data.__getitem__, 'jak-orig-pants', {27: 79, 31: 77})
        self.assertEqual(list(result), [0])
        np.testing.assert_array_equal(result[0][0], [[26, 79, 0, 0], [30, 77, 0, 0]])
        np.testing.assert_array_equal(result[0][1], data['w0'])
        # Face/legs and feet keep the original independently animated body knees.
        np.testing.assert_array_equal(data['j1'], [[26, 27, 0, 0], [30, 31, 0, 0]])
        np.testing.assert_array_equal(data['j2'], data['j1'])


if __name__ == '__main__':
    unittest.main()
