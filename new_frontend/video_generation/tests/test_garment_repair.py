"""Anatomical core tests: synthetic upper/lower garments, not motion validation."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "overlay/ContourCraft"))
import repair_garment_lbs as repair


def fixture():
    v = np.array([[.8, 0, 0], [.9, 0, 0], [.8, .1, 0],
                  [.1, -.8, 0], [.2, -.8, 0], [.1, -.9, 0],
                  [0, .1, 0], [.1, .1, 0], [0, .2, 0]], dtype=np.float32)
    n = len(v)
    w = np.zeros((n, 55), np.float32)
    w[:3, 18], w[3:6, 4], w[6:, 3] = 1, 1, 1
    g = dict(faces=np.arange(n).reshape(-1, 3), rest_pos=v.copy(), node_type=np.zeros((n, 1)),
             gender="male", fixed_size_metadata=dict(source_betas=np.zeros(10), target_betas=np.zeros(10)),
             lbs=dict(v=v.copy(), lbs_weights=w.copy(), shapedirs=np.zeros((n, 3, 10)),
                      posedirs=np.zeros((486, n * 3))))
    j = np.zeros((55, 3))
    for chain, axis, sign in [([16, 18, 20], 0, 1), ([17, 19, 21], 0, -1),
                              ([1, 4, 7], 1, -1), ([2, 5, 8], 1, -1),
                              ([0, 3, 6, 9, 12], 1, 1)]:
        for i, idx in enumerate(chain):
            j[idx, axis] = sign * i * .25
    body_pd = np.ones((486, n * 3), np.float32) * .123
    return g, copy.deepcopy(g), v.copy(), g["faces"], j, w, body_pd


def execute(data, config, apply=True):
    return repair.repair_with_body(*data, config, apply)


class AnatomicalRepair(unittest.TestCase):
    def test_healthy_region_is_unchanged(self):
        data = fixture()
        result, report, _ = execute(data, {"regions": ["left_arm"]})
        self.assertTrue(report["gates_passed"])
        self.assertEqual(report["proposed_vertices"], 0)
        np.testing.assert_array_equal(result["lbs"]["v"], data[0]["lbs"]["v"])

    def test_restores_arm_and_preserves_other_regions_and_inputs(self):
        data = fixture()
        data[0]["lbs"]["v"][:3, 2] += .15
        before = copy.deepcopy(data[0])
        result, report, _ = execute(data, {"regions": ["left_arm"]})
        self.assertTrue(report["applied"])
        np.testing.assert_allclose(result["lbs"]["v"], data[1]["lbs"]["v"])
        for key in ("rest_pos", "faces", "node_type"):
            np.testing.assert_array_equal(result[key], before[key])
        np.testing.assert_array_equal(data[0]["lbs"]["v"], before["lbs"]["v"])
        np.testing.assert_allclose(result["lbs"]["posedirs"].reshape(486, -1, 3)[:, :3], .123)
        np.testing.assert_array_equal(result["lbs"]["posedirs"].reshape(486, -1, 3)[:, 3:], 0)

    def test_same_algorithm_restores_trouser_leg(self):
        data = fixture()
        data[0]["lbs"]["v"][3:6, 2] += .15
        result, report, _ = execute(data, {"regions": ["left_leg"]})
        self.assertTrue(report["applied"])
        np.testing.assert_allclose(result["lbs"]["v"], data[1]["lbs"]["v"])

    def test_same_algorithm_restores_torso(self):
        data = fixture()
        data[0]["lbs"]["v"][6:, 2] += .15
        result, report, _ = execute(data, {"regions": ["torso"]})
        self.assertTrue(report["applied"])

    def test_diagnose_does_not_change_dictionary(self):
        data = fixture()
        data[0]["lbs"]["v"][:3, 2] += .15
        result, report, _ = execute(data, {"regions": ["left_arm"]}, False)
        self.assertFalse(report["applied"])
        np.testing.assert_array_equal(result["lbs"]["v"], data[0]["lbs"]["v"])

    def test_global_rotation_translation_do_not_select_wrong_vertices(self):
        data = list(fixture())
        data[0]["lbs"]["v"][:3, 2] += .15
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        shift = np.array([.15, -.2, .1])
        for d in data[:2]:
            d["lbs"]["v"] = d["lbs"]["v"] @ rot.T + shift
        data[2] = data[2] @ rot.T + shift
        data[4] = data[4] @ rot.T + shift
        result, report, _ = execute(data, {"regions": ["left_arm"]})
        self.assertTrue(report["applied"])
        np.testing.assert_allclose(result["lbs"]["v"], data[1]["lbs"]["v"])

    def test_overlarge_displacement_rejects_entire_candidate(self):
        data = fixture()
        data[0]["lbs"]["v"][:3, 2] += 1
        result, report, _ = execute(data, {"regions": ["left_arm"]})
        self.assertFalse(report["gates_passed"])
        self.assertFalse(report["applied"])
        np.testing.assert_array_equal(result["lbs"]["v"], data[0]["lbs"]["v"])

    def test_binding_only_keeps_geometry(self):
        data = fixture()
        data[0]["lbs"]["lbs_weights"][:3] = 0
        data[0]["lbs"]["lbs_weights"][:3, 3] = 1
        result, report, _ = execute(data, {"mode": "rebind", "regions": ["left_arm"]})
        self.assertTrue(report["applied"])
        np.testing.assert_array_equal(result["lbs"]["v"], data[0]["lbs"]["v"])
        np.testing.assert_array_equal(result["lbs"]["lbs_weights"][:3], data[5][:3])

    def test_apply_requires_reviewed_regions(self):
        with self.assertRaisesRegex(ValueError, "explicitly reviewed"):
            execute(fixture(), {})

    def test_unknown_and_nonfinite_settings_are_rejected(self):
        for config in ({"regions": ["sleeve"]}, {"clearance_m": float("nan")},
                       {"max_edge_strain": -1}, {"magic": True}):
            with self.assertRaises(ValueError):
                repair.validate_profile(config)

    def test_wrong_reference_topology_is_rejected(self):
        data = fixture()
        data[1]["faces"][0] = [2, 1, 0]
        with self.assertRaisesRegex(ValueError, "topology"):
            execute(data, {"regions": ["left_arm"]})

    def test_surface_distance_gate_catches_wrong_target(self):
        data = list(fixture())
        data[2] += 10
        _, report, _ = execute(data, {"mode": "rebind", "regions": ["left_arm"]})
        self.assertFalse(report["gates_passed"])

    def test_large_edge_distortion_rejected_even_in_rebind_mode(self):
        data = fixture()
        data[0]["lbs"]["v"][0, 0] += .15
        _, report, _ = execute(data, {"mode": "rebind", "regions": ["left_arm"]})
        self.assertFalse(report["gates_passed"])
        self.assertGreater(report["max_edge_strain"], .35)

    def test_reference_triangle_reversal_is_rejected(self):
        data = fixture()
        data[0]["lbs"]["v"][[0, 1]] = data[0]["lbs"]["v"][[1, 0]]
        _, report, _ = execute(data, {"mode": "rebind", "regions": ["left_arm"], "max_edge_strain": 1})
        self.assertFalse(report["gates_passed"])
        self.assertGreater(report["flipped_triangles"], 0)


if __name__ == "__main__":
    unittest.main()
