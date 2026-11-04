import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "overlay/ContourCraft"))
from toolkit_common import Job, ToolkitError, load_paths
from render_one_video import _pipeline_cmd
import repair_sleeve_lbs as repair


def fixture():
    n = 6
    w = np.zeros((n, 55), np.float32)
    w[:, 18] = 1
    return dict(faces=np.array([[0, 1, 2], [3, 4, 5]]), rest_pos=np.zeros((n, 3)),
        node_type=np.zeros((n, 1)), gender="male", fixed_size_metadata=dict(
        source_betas=np.zeros(10), target_betas=np.zeros(10), target_gender="male"),
        lbs=dict(v=np.zeros((n, 3)), lbs_weights=w, shapedirs=np.zeros((n, 3, 10)),
                 posedirs=np.zeros((486, n * 3))))


class InputGuards(unittest.TestCase):
    def test_reference_topology_mismatch_is_rejected(self):
        g = fixture()
        ref = copy.deepcopy(g)
        ref["faces"][0] = [1, 0, 2]
        with self.assertRaisesRegex(ValueError, "topology"):
            repair.validate_garment(g, ref)

    def test_non_fixed_size_input_is_rejected(self):
        g = fixture()
        g["lbs"]["shapedirs"][0, 0, 0] = .1
        with self.assertRaisesRegex(ValueError, "zero garment shapedirs"):
            repair.validate_garment(g)

    def test_reference_betas_must_match(self):
        g, ref = fixture(), fixture()
        ref["reference_metadata"] = {"source_betas": np.ones(10)}
        with self.assertRaisesRegex(ValueError, "betas differ"):
            repair.validate_garment(g, ref)

    def test_nan_weights_are_rejected(self):
        g = fixture()
        g["lbs"]["lbs_weights"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            repair.validate_garment(g)

    def test_target_gender_cannot_be_silently_swapped(self):
        with self.assertRaisesRegex(ValueError, "gender must match"):
            repair.repair(fixture(), None, "/does/not/exist", gender="female", mode="radial")

    def test_output_cannot_overwrite_input_before_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.pkl"
            source.write_bytes(b"not even a pickle")
            with self.assertRaises(SystemExit):
                repair.main(["--input-garment-dict", str(source),
                    "--output-garment-dict", str(source), "--body-model-root", folder])
            self.assertEqual(source.read_bytes(), b"not even a pickle")


class ConfigAndCommand(unittest.TestCase):
    def config(self, folder, repairs=None):
        p = Path(folder)
        cfg = {k: str(p / k) for k in ("contourcraft_root", "gaugar_root", "contourcraft_python",
               "gaugar_python", "data_root", "library_root", "work_root")}
        if repairs is not None:
            cfg["sleeve_repairs"] = repairs
        path = p / "paths.json"
        path.write_text(json.dumps(cfg))
        (p / "contourcraft_root").mkdir(exist_ok=True)
        (p / "contourcraft_root/run_outfit_preview_pipeline.py").touch()
        return path

    def test_default_is_disabled(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = load_paths(self.config(folder))
            job = Job(("lower", "shirt"), "male", 175, 75, "35_01", 80, 25)
            catalog = {"garments": {g: {"source_gender": "male"} for g in job.garments},
                       "motions": {"35_01": {"file": "35.npz"}}}
            cmd = _pipeline_cmd(paths, catalog, job, "test", None)
            self.assertNotIn("--sleeve-repairs-json", cmd)

    def test_relative_reference_and_only_selected_garment_are_forwarded(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "reference.pkl"
            reference.write_bytes(b"reference data")
            cfg = {"shirt": {"reference_dict": "reference.pkl", "gender_modes": {"male": "restore-rebind"}},
                   "other_shirt": {"gender_modes": {"female": "radial"}}}
            paths = load_paths(self.config(folder, cfg))
            job = Job(("lower", "shirt"), "male", 175, 75, "35_01", 80, 25)
            catalog = {"garments": {g: {"source_gender": "male"} for g in job.garments},
                       "motions": {"35_01": {"file": "35.npz"}}}
            cmd = _pipeline_cmd(paths, catalog, job, "test", None)
            forwarded = json.loads(cmd[cmd.index("--sleeve-repairs-json") + 1])
            self.assertEqual(set(forwarded), {"shirt"})
            self.assertEqual(Path(forwarded["shirt"]["reference_dict"]), reference.resolve())
            self.assertEqual(forwarded["shirt"]["gender_modes"], {"male": "restore-rebind"})

    def test_restore_requires_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.config(folder, {"shirt": {"gender_modes": {"male": "restore-rebind"}}})
            with self.assertRaises(ToolkitError):
                load_paths(path)

    def test_generic_profile_is_resolved_and_forwarded(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            path = self.config(folder)
            (p / "ref.pkl").touch()
            (p / "upper.json").write_text('{"regions":["left_arm"]}')
            cfg = json.loads(path.read_text())
            cfg["garment_repairs"] = {"shirt": {"reference_dict": "ref.pkl",
                "profile": "upper.json", "genders": ["male"]}}
            path.write_text(json.dumps(cfg))
            paths = load_paths(path)
            job = Job(("lower", "shirt"), "male", 175, 75, "35_01", 80, 25)
            catalog = {"garments": {g: {"source_gender": "male"} for g in job.garments},
                       "motions": {"35_01": {"file": "35.npz"}}}
            cmd = _pipeline_cmd(paths, catalog, job, "test", None)
            forwarded = json.loads(cmd[cmd.index("--garment-repairs-json") + 1])
            self.assertEqual(Path(forwarded["shirt"]["profile"]), (p / "upper.json").resolve())
            self.assertNotIn("--sleeve-repairs-json", cmd)

    def test_generic_and_legacy_cannot_double_apply(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.config(folder, {"shirt": {"gender_modes": {"male": "radial"}}})
            cfg = json.loads(path.read_text())
            cfg["garment_repairs"] = {"shirt": {}}
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ToolkitError, "不可同時"):
                load_paths(path)


if __name__ == "__main__":
    unittest.main()
