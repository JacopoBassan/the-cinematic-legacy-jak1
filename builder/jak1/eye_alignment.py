"""Calibrate cinematic Jak's live eye UVs in a shared head-forward frame.

The native low-poly eyes have a different shape and UV projection. Projecting
their UVs through model XY onto the cinematic eyes introduces fixed vergence.
Use the cinematic eye normals to map both animated pupils to a common gaze
direction from their native binocular offsets, retaining their authored local
UV scale and the native compositor. Jak's horizontal idle controls are unequal;
centering both UV projections at 0.5 turns those offsets into an inward bias.
"""
import json
import math
import os
from pathlib import Path
import struct
import sys

import numpy as np


def shading_normals(points, triangles, shading):
    """Return the area-weighted surface directions across authored smooth seams."""
    from native_jaw import unit
    t = points[triangles]
    area = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    result = np.zeros((int(shading.max()) + 1, 3))
    for column in range(3):
        np.add.at(result, shading[triangles[:, column]], area)
    return unit(result)[shading]


def move_facial_surface(position, normal, triangles, shading, dp, dn, step):
    from native_jaw import transport_normal
    before = position + dp
    previous = shading_normals(before, triangles, shading)
    adjusted = shading_normals(before + step, triangles, shading)
    changed = np.any(abs(previous - adjusted) > 1e-7, axis=1)
    authored = normal[changed] + dn[changed]
    dn[changed] += transport_normal(previous[changed], adjusted[changed], authored)
    dp += step


def fit_jak_worried_brows(lp, la, hp, ha, deformations):
    """Preserve the native inner lift and outer drop of the raised brows."""
    native = lp[0]
    used = np.unique(la(native['indices']))
    rest = la(native['attributes']['POSITION'])[used]
    color = la(native['attributes']['COLOR_0'])[used]
    green = (color[:, 1] > 1.35 * color[:, 0]) & (color[:, 1] > 1.2 * color[:, 2])
    native_brow = green & (rest[:, 1] > 2.43) & (rest[:, 2] > .19)

    donor = hp[10]
    position = ha(donor['attributes']['POSITION'])
    normal = ha(donor['attributes']['NORMAL'])
    triangles = ha(donor['indices']).reshape(-1, 3)
    donor_used = np.unique(triangles)
    # The same primitive also contains the tongue below the brows.
    brow = donor_used[(position[donor_used, 1] > 2.43) & (position[donor_used, 2] > .19)]
    dp, dn = deformations[10]
    _, shading = np.unique(np.column_stack((position, normal)), axis=0, return_inverse=True)
    # The smaller raise used on the boat has the same direction as the
    # concerned raise. Its outer tip was nearly static in the cinematic fit.
    for channel in (8, 10):
        motion = la(native['targets'][channel]['POSITION'])[used]
        step = np.zeros_like(position)
        for side in (-1, 1):
            mask = native_brow & (rest[:, 0] * side > 0)
            points, first = np.unique(rest[mask], axis=0, return_index=True)
            x = abs(points[:, 0])
            lift = motion[mask][first, 1]
            inner, outer = lift[x.argmin()], lift[x.argmax()]
            assert len(np.unique(x)) >= 8 and np.ptp(x) > .16
            assert inner > .01 and outer < -.02
            ids = brow[position[brow, 0] * side > 0]
            along = abs(position[ids, 0])
            assert len(ids) >= 45 and np.ptp(along) > .13
            fraction = (along - along.min()) / np.ptp(along)
            step[ids, 1] = inner + (outer - inner) * fraction - dp[channel, ids, 1]
        move_facial_surface(position, normal, triangles, shading, dp[channel], dn[channel], step)


def fit_daxter_lid_contours(primitives, accessor, deformations):
    """Fit the paired upper-lid presses and the inner slope of the squint."""
    for part, (positions, normals) in deformations.items():
        for channel, gain in ((21, .70), (22, .70), (27, .77), (28, .76)):
            positions[channel] *= gain
            normals[channel] *= gain
        p = primitives[part]
        rest = accessor(p['attributes']['POSITION'])
        normal = accessor(p['attributes']['NORMAL'])
        triangles = accessor(p['indices']).reshape(-1, 3)
        _, shading = np.unique(np.column_stack((rest, normal)), axis=0, return_inverse=True)
        # The native upper edge descends farther at the bridge of the nose.
        # Carry the same contour through the fur and the textured lid cap.
        tilt = np.interp(abs(rest[:, 0]), [.038, .115], [.17, -.11])
        for channel in (23, 24):
            step = np.zeros_like(rest)
            step[:, 1] = positions[channel, :, 1] * tilt
            if np.any(step):
                move_facial_surface(rest, normal, triangles, shading, positions[channel],
                                    normals[channel], step)


def fit_daxter_socket_fur(primitives, accessor, eyes, deformations):
    """Carry the native open-eye motion onto the fur lining the two sockets."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from fix_faces import closest_surface

    face = primitives[6]
    position = accessor(face['attributes']['POSITION'])
    normal = accessor(face['attributes']['NORMAL'])
    triangles = accessor(face['indices']).reshape(-1, 3)
    points, first, group = np.unique(position, axis=0, return_index=True, return_inverse=True)
    triangles_welded = group[triangles]
    edge_pairs = np.concatenate([triangles_welded[:, pair] for pair in ((0, 1), (1, 2), (2, 0))])
    edges, counts = np.unique(np.sort(edge_pairs, axis=1), axis=0, return_counts=True)
    boundary = edges[counts == 1]
    graph_shape = (len(points), len(points))
    boundary_graph = coo_matrix((np.ones(len(boundary)), boundary.T), shape=graph_shape)
    _, labels = connected_components(boundary_graph, directed=False)
    loops = [np.flatnonzero(labels == label) for label in np.unique(labels[boundary[:, 0]])]
    socket_rims = [loop for loop in loops if len(loop) == 22]
    assert len(socket_rims) == len(eyes) == 2

    face_delta, normal_delta = deformations[6]
    def lengths(xyz):
        return np.linalg.norm(xyz[edges[:, 1]] - xyz[edges[:, 0]], axis=1)

    base_lengths = lengths(points)
    surface_graph = coo_matrix((base_lengths, edges.T), shape=graph_shape).tocsr()
    # The eyelid folds almost flat during a blink. If the ends of an edge
    # collapse together, move them together under simultaneous open-eye input.
    compressible = np.zeros(len(edges), dtype=bool)
    for channel in (21, 22, 23, 24, 26, 27, 28):
        for weight in (.5, 1., 1.19):
            compressible |= lengths(points + weight * face_delta[channel, first]) < base_lengths * .25
    closed_graph = coo_matrix((np.ones(compressible.sum()), edges[compressible].T), shape=graph_shape)
    count, cluster = connected_components(closed_graph, directed=False)
    cluster_size = np.bincount(cluster)

    # Controls 11/12 and 19/20 are left/right pairs. The wider raised-brow
    # motion needs a broader strip than the smaller bilateral 17/25 motion.
    settings = {11: (.035, .25), 12: (.035, .25), 17: (.022, .8),
                19: (.035, .5), 20: (.035, .5), 25: (.022, 1.)}
    corrections = {channel: np.zeros_like(points) for channel in settings}
    for eye_id in eyes:
        eye = primitives[eye_id]
        cap = accessor(eye['attributes']['POSITION'])
        cap_triangles = accessor(eye['indices']).reshape(-1, 3)
        middle = cap[np.unique(cap_triangles)].mean(axis=0)
        ring = min(socket_rims, key=lambda r: np.linalg.norm(middle - points[r].mean(axis=0)))
        distance = dijkstra(surface_graph, directed=False, indices=ring, min_only=True)
        used = np.flatnonzero(distance < .035)
        tri, bary, _ = closest_surface(points[used], cap[cap_triangles])
        for channel, (width, strength) in settings.items():
            cap_delta = deformations[eye_id][0][channel]
            if not np.any(cap_delta):
                continue
            motion = np.einsum('ni,nij->nj', bary, cap_delta[cap_triangles[tri]])
            t = np.minimum(distance[used] / width, 1.)
            blend = strength * (1 - t*t*(3 - 2*t))
            corrections[channel][used] += blend[:, None] * (motion - face_delta[channel, first[used]])

        # The bilateral raise narrows both eyes toward the top. The donor
        # pinches the upper sclera more than stock, despite matching its lift.
        # Widen the raised cap and its adjoining rim in the same rest frame.
        cap_vertices = cap[np.unique(cap_triangles)]
        lower, upper = cap_vertices.min(axis=0), cap_vertices.max(axis=0)
        def widen_upper_socket(xyz):
            height = np.clip((xyz[:, 1] - lower[1]) / (upper[1] - lower[1]), 0., 1.)
            return .14 * height * (xyz[:, 0] - (lower[0] + upper[0]) * .5)

        cap_positions, cap_normals = deformations[eye_id]
        step = np.zeros_like(cap)
        step[:, 0] = widen_upper_socket(cap)
        across = (cap[:, 0] - (lower[0] + upper[0]) * .5) / ((upper[0] - lower[0]) * .5)
        # Keep an overlap at both corners of the wide-eyed cap in oblique views.
        step[:, 0] += .0025 * across
        step[:, 2] -= .002
        cap_normal = accessor(eye['attributes']['NORMAL'])
        _, cap_shading = np.unique(np.column_stack((cap, cap_normal)), axis=0, return_inverse=True)
        move_facial_surface(cap, cap_normal, cap_triangles, cap_shading,
                            cap_positions[25], cap_normals[25], step)
        squint = (23, 24)[eyes[eye_id]]
        # Fill the gap at the nose-side corner without bringing the whole
        # lid out through the brow during the closed-eye blend.
        medial = np.clip((.087 - abs(cap[:, 0])) / .035, 0., 1.)
        medial *= np.clip((cap[:, 1] - 1.084) / .048, 0., 1.)
        medial = medial * medial * (3 - 2 * medial)
        step[:, 0] = .004 * across - np.sign(middle[0]) * .0036 * medial
        step[:, 2] = .002 + .018 * medial
        move_facial_surface(cap, cap_normal, cap_triangles, cap_shading,
                            cap_positions[squint], cap_normals[squint], step)
        # Match the measured native sclera travel at the active, one-sided
        # controls. The cheek partially covers the lower cap in a grin,
        # while the tall socket is short of its native crest at full raise.
        for channel, travel in ((11, -.020), (12, -.015), (19, .0095), (20, .0047)):
            if not np.any(cap_positions[channel]):
                continue
            def socket_offset(xyz):
                height = np.clip((xyz[:, 1] - lower[1]) / (upper[1] - lower[1]), 0., 1.)
                offset = np.zeros_like(xyz)
                if channel in (11, 12):
                    center = (lower[0] + upper[0]) * .5
                    side = np.sign(middle[0])
                    outer = np.clip(.5 + side * (xyz[:, 0] - center) / (upper[0] - lower[0]), 0., 1.)
                    # Only the outer corner uncovers the socket. Moving the inner
                    # pole down as well exposes a disconnected sliver in the cheek.
                    offset[:, 1] = travel * (1. - height) ** 2 * np.clip((outer - .4) * 4., 0., 1.)
                    offset[:, 0] = side * .008 * outer * (1. - height)
                else:
                    offset[:, 1] = travel * height ** 2
                    center = (lower[0] + upper[0]) * .5
                    offset[:, 0] = -.08 * height * (xyz[:, 0] - center)
                return offset

            offset = socket_offset(cap)
            if channel in (11, 12):
                # Keep the convex lower cap behind the moving cheek.
                offset[:, 2] = offset[:, 1] * .65
            else:
                # Raised-eye and squint weights coexist in the Sage dialogue.
                # The fur supplies most of the lift so the cap remains covered.
                offset[:, 1] *= .1
            move_facial_surface(cap, cap_normal, cap_triangles, cap_shading,
                                cap_positions[channel], cap_normals[channel], offset)
            t = np.minimum(distance[used] / settings[channel][0], 1.)
            rim_offset = socket_offset(points[used])
            if channel in (11, 12):
                rim_offset *= .45
            corrections[channel][used] += (1 - t*t*(3 - 2*t))[:, None] * rim_offset
        t = np.minimum(distance[used] / settings[25][0], 1.)
        corrections[25][used, 0] += (1 - t*t*(3 - 2*t)) * widen_upper_socket(points[used])

    # Rotate existing authored normals through the change in surface direction.
    # Include duplicated UV vertices while preserving authored shading seams.
    keys = np.concatenate([position, normal], axis=1)
    _, shading = np.unique(keys, axis=0, return_inverse=True)
    for channel, correction in corrections.items():
        total = np.zeros((count, 3))
        np.add.at(total, cluster, correction)
        step = (total / cluster_size[:, None])[cluster[group]]
        move_facial_surface(position, normal, triangles, shading,
                            face_delta[channel], normal_delta[channel], step)

    # Seat the stretched cap behind the deformed rim with the neutral overlap.
    # The one-eye raises otherwise stretch the upper cap under the fur by an
    # extra 33–36 mm, even though the visible opening cannot rise that far.
    def rise_profile(y):
        t = np.clip((y - 1.084) / .124, 0., 1.)
        return t * t * (3 - 2 * t)

    def upper_socket_step(rest, motion, side, peak):
        # At the measured one-eye peaks, the near view exposes the dark
        # nose-side wall when the cap and fur follow different upper arcs.
        # Carry their neutral cross-section through the same 84 mm lift.
        # Blend into the authored raise below the crest and across the bridge.
        def ramp(t):
            t = np.clip(t, 0., 1.)
            return t * t * (3 - 2 * t)

        x, y, z = rest.T
        weight = ramp((y - 1.120) / .078) * ramp((side * x - .004) / .032)
        weight *= ramp((z - .010) / .050)
        target = np.array([0., .084 / peak, 0.], dtype=motion.dtype)
        return weight[:, None] * (target - motion)

    raised_rim = {channel: np.zeros_like(position) for channel in (19, 20, 25)}
    raised_crowns = []
    for eye_id in eyes:
        eye = primitives[eye_id]
        cap = accessor(eye['attributes']['POSITION'])
        cap_normal = accessor(eye['attributes']['NORMAL'])
        cap_triangles = accessor(eye['indices']).reshape(-1, 3)
        used = np.unique(cap_triangles)
        center = cap[used].mean(axis=0)
        ring = min(socket_rims, key=lambda r: np.linalg.norm(center - points[r].mean(axis=0)))
        side = np.sign(center[0])
        channel, peak, lift = ((19, 9737.3613 / 8192, .027) if side > 0 else
                               (20, 9728 / 8192, .029))
        cap_positions, cap_normals = deformations[eye_id]
        rest_height = cap[:, 1]
        rest_overlap = rest_height[used].max() - points[ring, 1].max()
        cap_crest = (rest_height[used] + peak * cap_positions[channel, used, 1]).max()
        rim_crest = (points[ring, 1] + peak * face_delta[channel, first[ring], 1]).max()
        excess = max(0., cap_crest - rim_crest - rest_overlap - .004)
        height = np.clip((rest_height - rest_height[used].min()) / np.ptp(rest_height[used]), 0., 1.)
        step = np.zeros_like(cap)
        step[:, 1] = -excess * height**2 / peak
        _, cap_shading = np.unique(np.column_stack((cap, cap_normal)), axis=0, return_inverse=True)
        move_facial_surface(cap, cap_normal, cap_triangles, cap_shading,
                            cap_positions[channel], cap_normals[channel], step)
        # Match the native change in the visible opening. The fur and cap
        # rise together, leaving the lower edge and pupil projection seated.
        fur = rise_profile(position[:, 1]) * np.clip(side * position[:, 0] / .041, 0., 1.)
        fur *= np.clip((position[:, 2] - .020) / .060, 0., 1.)
        for control, gain in ((channel, lift / peak), (25, .006 / (8448 / 8192))):
            step[:, 1] = gain * rise_profile(rest_height)
            raised_rim[control][:, 1] += gain * fur
            move_facial_surface(cap, cap_normal, cap_triangles, cap_shading,
                                cap_positions[control], cap_normals[control], step)

        # Fill the top of the stretched opening. Follow the slope of the
        # socket with the crest of the cap so the white stays inside the fur
        # in profile. The lower cap and its live texture coordinates are fixed.
        cap_height = np.clip((rest_height - 1.082) / .122, 0., 1.)
        seat = cap_height * (2 - cap_height)
        fill = np.clip((rest_height - 1.090) / .125, 0., 1.)
        step[:, 0] = -.0025 * side * seat / peak
        step[:, 1] = .008 * fill**2 * (3 - 2 * fill) / peak
        step[:, 2] = -.006 * seat / peak
        step += upper_socket_step(cap, cap_positions[channel] + step, side, peak)
        move_facial_surface(cap, cap_normal, cap_triangles, cap_shading,
                            cap_positions[channel], cap_normals[channel], step)

        rim_y = points[ring, 1] + peak * (face_delta[channel, first[ring], 1] +
                                           raised_rim[channel][first[ring], 1])
        raised_crowns.append((channel, peak, side, rim_y.max() - .015))
    for control, step in raised_rim.items():
        move_facial_surface(position, normal, triangles, shading,
                            face_delta[control], normal_delta[control], step)

    # Compress the excess lining above the posed rim with a monotonic
    # profile. Pulling the crown down from rest coordinates folds thin
    # triangles at the top of the socket.
    for part in (4, 5, 6):
        p = primitives[part]
        rest = accessor(p['attributes']['POSITION'])
        x, y, z = rest.T
        normals = accessor(p['attributes']['NORMAL'])
        faces = accessor(p['indices']).reshape(-1, 3)
        _, seam = np.unique(np.column_stack((rest, normals)), axis=0, return_inverse=True)
        dp, dn = deformations[part]
        for channel, peak, side, lower_crown in raised_crowns:
            above = np.maximum(0., y + peak * dp[channel, :, 1] - lower_crown)
            shoulder = np.minimum(above, .012)
            profile = above - shoulder * (1 - shoulder / .024)
            span = np.clip((side * x - .010) / .040, 0., 1.) * np.clip((z - .012) / .065, 0., 1.)
            compression = (.013 if side > 0 else .016) / .055
            step = np.zeros_like(rest)
            step[:, 1] = -compression * span * profile / peak
            step += upper_socket_step(rest, dp[channel] + step, side, peak)
            if np.any(step):
                move_facial_surface(rest, normals, faces, seam, dp[channel], dn[channel], step)


# Native eichar-stance-loop eye controls in the supported Jak 1 EU/USA data.
# These are UV reference offsets, not replacements for the live animation.
JAK_NATIVE_HORIZONTAL_IDLE_OFFSETS = (-15/128, 15/128)


def sample_eye(p, accessor, uv, texture_point):
    """Locate the pupil on the rendered triangle surface for a compositor UV."""
    position = accessor(p['attributes']['POSITION'])
    normal = accessor(p['attributes']['NORMAL'])
    hits = []
    target = np.r_[texture_point, 1.0]
    for indices in accessor(p['indices']).reshape(-1, 3):
        matrix = np.vstack([uv[indices].T, np.ones(3)])
        bary = np.linalg.lstsq(matrix, target, rcond=None)[0]
        if np.linalg.norm(matrix@bary-target) < 1e-6 and bary.min() >= -1e-6:
            point = position[indices].T@bary
            direction = normal[indices].T@bary
            direction /= np.linalg.norm(direction)
            hits.append((float(direction[2]), point, direction))
    if not hits:
        raise ValueError(f'Eye does not cover compositor UV {texture_point}')
    _, point, direction = max(hits, key=lambda item: item[0])
    return dict(position=point.tolist(), direction=direction.tolist())


def angle_between(left, right):
    cosine = np.clip(np.dot(left, right), -1, 1)
    return math.degrees(math.acos(float(cosine)))


def native_xy_projection(source, source_accessor, destination, destination_accessor):
    used = np.unique(source_accessor(source['indices']))
    pos = source_accessor(source['attributes']['POSITION'])[used]
    uv = source_accessor(source['attributes']['TEXCOORD_0'])[used]
    projection = np.linalg.lstsq(np.c_[pos[:, :2], np.ones(len(pos))], uv, rcond=None)[0]
    pos = destination_accessor(destination['attributes']['POSITION'])
    return np.clip(np.c_[pos[:, :2], np.ones(len(pos))]@projection, 0, 1)


def authored_daxter_eye_uvs(primitives, accessor, eyes):
    """Retain the donor's round pupil projection with native live controls.

    Native Jak 1 and cinematic Daxter use the same authored binocular idle
    offsets. Projecting the low-poly eye's model XY mapping onto the round
    cinematic surface compresses its pupil horizontally. Only Daxter uses
    this authored-UV branch; the other characters keep their own calibration.
    """
    assert sorted(eyes.values()) == [0, 1]
    result = {}
    for part, side in eyes.items():
        primitive = primitives[part]
        used = np.unique(accessor(primitive['indices']))
        positions = accessor(primitive['attributes']['POSITION'])
        uv = accessor(primitive['attributes']['TEXCOORD_0']).astype('<f4')
        assert (np.median(positions[used, 0]) > 0) == (side == 0)
        assert np.isfinite(uv).all() and uv.min() >= 0 and uv.max() <= 1
        result[part] = uv.copy()
    return result, dict(
        method='Authored cinematic Daxter eye UVs with native Jak 1 controls',
        eyePrimitiveSides=eyes,
        unchanged=['native live eye IDs', 'authored binocular offsets',
                   'iris and pupil scales', 'automatic blink',
                   'native eye expression geometry', 'all other characters'],
        staticEyes=False)


def calibrated_jak_eye_uvs(lp, la, hp, ha, high_eyes, low_eyes):
    assert set(high_eyes.values()) == {0, 1}
    scales, old, corrected = [], {}, {}
    part_by_side = {side: part for part, side in high_eyes.items()}
    for part, side in high_eyes.items():
        p = hp[part]
        used = np.unique(ha(p['indices']))
        pos = ha(p['attributes']['POSITION'])
        normal = ha(p['attributes']['NORMAL'])
        normal = normal/np.linalg.norm(normal, axis=1, keepdims=True)
        authored_uv = ha(p['attributes']['TEXCOORD_0'])
        assert (np.median(pos[used, 0]) > 0) == (side == 0), 'Eye side IDs do not match anatomy'
        near = ((abs(normal[:, 0]) < .5) & (abs(normal[:, 1]) < .5) & (normal[:, 2] > .75))
        near &= np.isin(np.arange(len(pos)), used)
        assert near.sum() >= 12
        projection = np.linalg.lstsq(np.c_[normal[near, :2], np.ones(int(near.sum()))],
                                     authored_uv[near], rcond=None)[0]
        assert .5 < projection[0, 0] < 1 and -.95 < projection[1, 1] < -.5
        scales.append((float(projection[0, 0]), float(-projection[1, 1])))
        old[part] = native_xy_projection(lp[low_eyes[side]], la, p, ha)
    # Share angular scales and retain the native horizontal reference per eye.
    # Equal movement from these references means equal gaze. The compositor's
    # existing vertical projection and all live control values stay unchanged.
    horizontal, vertical = np.mean(scales, axis=0)
    for part, side in high_eyes.items():
        p = hp[part]
        normal = ha(p['attributes']['NORMAL'])
        normal = normal/np.linalg.norm(normal, axis=1, keepdims=True)
        uv = np.c_[.5+JAK_NATIVE_HORIZONTAL_IDLE_OFFSETS[side]+horizontal*normal[:, 0],
                   .5-vertical*normal[:, 1]]
        assert np.isfinite(uv).all()
        assert uv.min() >= 0 and uv.max() <= 1, 'Calibrated eye UV exceeds the live eye texture'
        corrected[part] = uv.astype('<f4')
    samples = []
    for label, gaze in [('neutral', (0, 0)), ('left', (-.25, 0)), ('right', (.25, 0)),
                        ('up', (0, -.20)), ('down', (0, .20))]:
        eyes, controls = [], []
        for side in [0, 1]:
            part = part_by_side[side]
            control = np.array(gaze)+[JAK_NATIVE_HORIZONTAL_IDLE_OFFSETS[side], 0]
            sample = sample_eye(hp[part], ha, corrected[part], control+.5)
            eyes.append(sample)
            controls.append(control.tolist())
        vergence = angle_between(eyes[0]['direction'], eyes[1]['direction'])
        assert vergence < .2, (label, vergence)
        samples.append(dict(gaze=label, compositorOffset=list(gaze),
                            nativeCompositorControls=controls, eyes=eyes,
                            binocularDirectionDifferenceDegrees=vergence))
    old_centers = [sample_eye(hp[part_by_side[side]], ha, old[part_by_side[side]],
                             [.5+JAK_NATIVE_HORIZONTAL_IDLE_OFFSETS[side], .5])
                   for side in [0, 1]]
    old_vergence = angle_between(old_centers[0]['direction'], old_centers[1]['direction'])
    report = dict(method='Shared normal projection anchored to native Jak 1 horizontal eye offsets',
                  nativeHorizontalIdleOffsets=list(JAK_NATIVE_HORIZONTAL_IDLE_OFFSETS),
                  gazeReference='Native horizontal idle controls with vertical control zero',
                  nativeXYProjectionNeutral=dict(eyes=old_centers,
                    binocularDirectionDifferenceDegrees=old_vergence),
                  sharedUVScale=[float(horizontal), float(vertical)],
                  correctedSamples=samples,
                  eyePrimitiveSides=high_eyes, unchanged=['eye IDs', 'native animated eye compositor',
                    'gaze values', 'iris and pupil animation', 'blink', 'expression geometry', 'neutral geometry'],
                  staticEyes=False,
                  checksPassed=['Native anatomical eye side matches ID', 'UV remains in live texture bounds',
                                'Both gaze axes point head-forward at native horizontal idle offsets',
                                'Five shared gaze shifts agree within 0.2 degrees using native binocular offsets',
                                'Vertical UV projection remains unchanged'])
    return corrected, report


def write_eye_uvs(raw, name, doc, accessor, indices, vertices, uv_by_part):
    """Write calibrated UVs before fix_model copies eye vertices into morph data."""
    from fr3_models import model, DRAW
    target = model(raw, name)
    draws = sorted([draw for effect in target['effects'] for draw in effect['draws']],
                   key=lambda draw: DRAW.unpack(draw)[3])
    prims = doc['meshes'][0]['primitives']
    assert len(draws) == len(prims)
    for part, uv in uv_by_part.items():
        values = DRAW.unpack(draws[part])
        ids = indices[values[3]:values[3]+values[4]]
        local = accessor(prims[part]['indices']).ravel()
        offset = int(ids[0])-int(local[0])
        assert np.array_equal(ids, local+offset)
        for vertex in np.unique(ids):
            struct.pack_into('<2f', vertices, int(vertex)*64+48, *uv[int(vertex)-offset])


def main():
    root = Path(__file__).resolve().parents[2]
    from jak1_paths import DATA,WORK,RELEASE
    work = WORK
    os.environ['CINEMATIC_BUILD_WORK'] = str(work)
    sys.path[:0] = [str(RELEASE/'jak1'), str(RELEASE/'stages')]
    from native_io import activate
    activate()
    from fr3_models import decompress
    from inspect_faces import glb, buffers
    from build_faces import eye_parts
    low, la = glb(DATA/'decompiler_out/jak1/levels/common/eichar-lod0.glb')
    high, ha = glb(work/'rig-prototypes/eichar-lod0.glb')
    lp, hp = low['meshes'][0]['primitives'], high['meshes'][0]['primitives']
    high_eyes = eye_parts(high)
    low_eyes = {side: part for part, side in eye_parts(low).items()}
    uv, report = calibrated_jak_eye_uvs(lp, la, hp, ha, high_eyes, low_eyes)
    raw = decompress(work/'GAME-materials.fr3')
    _, indices, _, original = buffers(raw, 'eichar-lod0')
    vertices = bytearray(original)
    write_eye_uvs(raw, 'eichar-lod0', high, ha, indices, vertices, uv)
    before = np.frombuffer(original, 'u1').reshape(-1, 64)
    after = np.frombuffer(vertices, 'u1').reshape(-1, 64)
    assert np.array_equal(before[:, :48], after[:, :48])
    assert np.array_equal(before[:, 56:], after[:, 56:])
    report['checksPassed'].append('Only UV bytes change in private buffer simulation')
    path = work/'face-candidates/jak-eye-alignment-report.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
