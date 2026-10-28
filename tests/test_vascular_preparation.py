import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vascular_preparation", REPO / "scripts/prepare_vascular_pilot.py")
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


class GraphInspectionTests(unittest.TestCase):
    def test_disconnected_cycle_and_isolated_node(self):
        positions = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [.8, .8, .8]])
        _, edges, sp, stats = prep.inspect_graph(positions, [[1, 0], [1, 2], [2, 0]], np.array([64]*3), 1e-5)
        self.assertEqual((stats["beta0"], stats["beta1"], stats["isolated_nodes"]), (2, 1, 1))
        self.assertEqual(sp[0, 3], 4)
        self.assertEqual(sp[0, 2], 1)
        np.testing.assert_array_equal(edges, [[0, 1], [0, 2], [1, 2]])

    def test_empty_and_single_node(self):
        _, _, sp, stats = prep.inspect_graph(np.empty((0, 3)), [], np.array([64]*3), 1e-5)
        self.assertEqual(sp.shape, (0, 0))
        self.assertEqual((stats["beta0"], stats["beta1"]), (0, 0))
        _, _, _, stats = prep.inspect_graph([[0, 0, 0]], [], np.array([64]*3), 1e-5)
        self.assertEqual((stats["beta0"], stats["isolated_nodes"]), (1, 1))

    def test_rejects_corrupt_geometry_and_connectivity(self):
        cases = [([[1, 0, 0]], []), ([[float("nan"), 0, 0]], []),
                 ([[0, 0, 0]], [[0, 1]]), ([[0, 0, 0]], [[0, 0]]),
                 ([[0, 0, 0], [.1, 0, 0]], [[0, 1], [1, 0]])]
        for points, edges in cases:
            with self.subTest(points=points, edges=edges), self.assertRaises(ValueError):
                prep.inspect_graph(points, edges, np.array([64]*3), 1e-5)

    def test_shortest_paths_reorder_with_nodes(self):
        xyz = np.array([[0, 0, 0], [.1, 0, 0], [.2, 0, 0], [.3, 0, 0]])
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        _, _, sp, stats = prep.inspect_graph(xyz, edges, np.array([64]*3), 1e-5)
        order = np.array([2, 0, 3, 1])
        inverse = np.argsort(order)
        _, _, perm_sp, perm_stats = prep.inspect_graph(xyz[order], inverse[edges], np.array([64]*3), 1e-5)
        np.testing.assert_array_equal(perm_sp, sp[order][:, order])
        self.assertEqual(stats, perm_stats)

    def test_selection_matches_previous_project_hash_contract(self):
        rows = [{"sample_id": f"sample_{i:04d}"} for i in range(100)]
        expected = sorted(sorted(rows, key=lambda row: (
            hashlib.sha256(("364505\0train\0" + row["sample_id"]).encode()).digest(),
            row["sample_id"]))[:20], key=lambda row: row["sample_id"])
        chosen = prep.select_records(list(reversed(rows)), 20, seed=364505, split="train")
        self.assertEqual(chosen, expected)
        small = prep.select_records(rows, 3, seed=364505, split="train")
        self.assertTrue({x['sample_id'] for x in small} <= {x['sample_id'] for x in chosen})


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "new_patches_boundary"
        self.root.mkdir()
        self.config = json.loads((REPO / "configs/vascular/pilot_data.json").read_text())
        self.config.update(dataset_root=str(self.root), sample_counts={"train": 2, "val": 2, "test": 2},
                           expected_patient_counts={"train": 1, "val": 1, "test": 1})
        (self.base / "new_split.csv").write_text("patient_id,split\n1,train\n2,val\n3,test\n")
        self.write_generation_metadata()
        rows = []
        for patient, split in enumerate(prep.SPLITS, 1):
            for i in range(4):
                sample = f"sample_{patient:06d}_{i:04d}"
                rows.append(dict(sample_id=sample, patient_id=str(patient), split=split,
                                 start_d="0", start_h="0", start_w="0"))
                for folder, suffix in (("raw", "_data.nii.gz"), ("seg", "_seg.nii.gz"), ("vtp", "_graph.vtp")):
                    path = self.root / split / folder / f"{sample}{suffix}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("placeholder: the injected reader must not load image arrays")
        self.rows = rows
        self.write_index(rows)
        self.read_paths = []

    def write_generation_metadata(self):
        (self.root / "generation_config.json").write_text(json.dumps({
            "split_sha256": prep.fingerprint(self.base / "new_split.csv"), "patch_size": [64]*3}))
        (self.root / "generation_summary.json").write_text('{"complete": true}')

    def write_index(self, rows):
        prep.write_csv(self.root / "patch_index.csv", rows, list(rows[0]))

    def reader(self, path):
        self.assertNotIn("test", path.parts)
        self.read_paths.append(path)
        return np.array([[.1, .2, .3], [.2, .3, .4]], dtype=np.float32), np.array([[0, 1]])

    def test_full_export_preserves_geometry_and_reserves_test(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        out = self.base / "prepared"
        report = prep.prepare(self.config, out, reader=self.reader)
        self.assertEqual(len(self.read_paths), 4)
        self.assertFalse(report["test_graph_content_read"])
        self.assertEqual(report["minimum_node_capacity_train_val"], 2)
        self.assertTrue((out / "COMPLETED.json").is_file())
        self.assertFalse((out / "graphs/test.npz").exists())
        with (out / "manifests/test.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)
        with np.load(out / "graphs/train.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["node_offsets"], [0, 2, 4])
            np.testing.assert_array_equal(archive["edge_offsets"], [0, 1, 2])
            np.testing.assert_array_equal(archive["sp_offsets"], [0, 4, 8])
            np.testing.assert_allclose(archive["node_features"][0], [.1, .2, .3])
            np.testing.assert_array_equal(archive["sp_values"][:4].reshape(2, 2), [[0, 1], [1, 0]])
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        with self.assertRaises(FileExistsError):
            prep.prepare(self.config, out, reader=self.reader)

    def test_smoke_is_marked_and_test_manifest_count_is_preserved(self):
        report = prep.prepare(self.config, self.base / "smoke", smoke_count=1, reader=self.reader)
        self.assertEqual(report["selected_counts"], {"train": 1, "val": 1, "test": 2})
        self.assertTrue(report["smoke"])

    def test_missing_csv_is_recovered_with_exact_original_hash(self):
        original = self.base / "new_split.csv"
        original_bytes = original.read_bytes()
        original.unlink()
        report = prep.prepare(self.config, self.base / "recovered", reader=self.reader)
        self.assertEqual(report["patient_split_source"], "patch_index_recovery")
        self.assertIsNone(report["patient_split_file"])
        self.assertEqual(report["patient_split_sha256"], hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual((self.base / "recovered/patient_split.csv").read_bytes(), original_bytes)
        self.assertFalse(original.exists())
        self.assertFalse(report["test_graph_content_read"])

    def test_missing_csv_recovery_rejects_changed_assignments_or_hash(self):
        (self.base / "new_split.csv").unlink()
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["patient_id"] == "1":
                row["split"] = "val"
        self.write_index(rows)
        with self.assertRaisesRegex(ValueError, "do not reproduce"):
            prep.validate_inputs(self.config)
        self.write_index(self.rows)
        metadata = json.loads((self.root / "generation_config.json").read_text())
        metadata["split_sha256"] = "0" * 64
        (self.root / "generation_config.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "GRALE_PATIENT_SPLIT_FILE"):
            prep.validate_inputs(self.config)

    def test_recovery_rejects_patient_in_multiple_splits(self):
        (self.base / "new_split.csv").unlink()
        rows = copy.deepcopy(self.rows)
        rows[0]["split"] = "val"
        self.write_index(rows)
        with self.assertRaisesRegex(ValueError, "multiple patch-index splits"):
            prep.validate_inputs(self.config)

    def test_relocated_csv_auto_discovery_and_explicit_override(self):
        original = self.base / "new_split.csv"
        relocated = self.root / "saved_split.csv"
        original.rename(relocated)
        # A legacy split is present too, but its fingerprint must not be accepted.
        (self.base / "splits.csv").write_text("patient_id,split\n1,test\n2,val\n3,train\n")
        inputs = prep.validate_inputs(self.config)
        self.assertEqual(inputs["split_file"], relocated.resolve())
        self.assertEqual(inputs["split_source"], "csv")
        alternate = self.base / "metadata/patient_assignment.csv"
        alternate.parent.mkdir()
        relocated.rename(alternate)
        inputs = prep.validate_inputs(self.config, patient_split_file=alternate)
        self.assertEqual(inputs["split_file"], alternate.resolve())
        with self.assertRaisesRegex(FileNotFoundError, "Explicit patient split"):
            prep.validate_inputs(self.config, patient_split_file=self.base / "typo.csv")

    def test_cli_metadata_preflight_needs_no_output_and_does_not_read_graphs(self):
        (self.base / "new_split.csv").unlink()
        config = self.base / "config.json"
        config.write_text(json.dumps(self.config))
        result = subprocess.run([sys.executable, str(REPO / "scripts/prepare_vascular_pilot.py"),
            "--config", str(config), "--check-inputs"], capture_output=True, text=True, check=True)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["patient_split_source"], "patch_index_recovery")
        self.assertEqual(summary["patient_counts"], {"train": 1, "val": 1, "test": 1})
        self.assertFalse((self.base / "new_split.csv").exists())
        self.assertFalse((self.root / "patient_split.csv").exists())

    def test_rejects_split_mismatch_and_incomplete_generation(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["split"] = "test"
        self.write_index(rows)
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            prep.prepare(self.config, self.base / "bad", reader=self.reader)
        self.write_index(self.rows)
        (self.root / "generation_summary.json").write_text('{"complete": false}')
        with self.assertRaisesRegex(ValueError, "not marked complete"):
            prep.prepare(self.config, self.base / "bad", reader=self.reader)

    def test_rejects_wrong_split_fingerprint_or_insufficient_samples(self):
        (self.base / "new_split.csv").write_text("patient_id,split\n1,train\n2,test\n3,val\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            prep.prepare(self.config, self.base / "bad", reader=self.reader)
        (self.base / "new_split.csv").write_text("patient_id,split\n1,train\n2,val\n3,test\n")
        self.write_generation_metadata()
        self.config["sample_counts"]["train"] = 10
        with self.assertRaisesRegex(ValueError, "requested 10 patches"):
            prep.prepare(self.config, self.base / "bad", reader=self.reader)

    def test_refuses_source_directory_output(self):
        with self.assertRaisesRegex(ValueError, "outside the source"):
            prep.prepare(self.config, self.root / "export", reader=self.reader)


class VtkReaderTests(unittest.TestCase):
    def test_ascii_vtp_reader(self):
        try:
            import vtkmodules.vtkIOXML
        except ImportError:
            self.skipTest("VTK is not installed in this local test environment")
        payload = '''<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian"><PolyData>
<Piece NumberOfPoints="3" NumberOfVerts="0" NumberOfLines="1" NumberOfStrips="0" NumberOfPolys="0">
<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">0 0 0 .1 .2 .3 .5 .5 .5</DataArray></Points>
<Lines><DataArray type="Int64" Name="connectivity" format="ascii">0 1</DataArray>
<DataArray type="Int64" Name="offsets" format="ascii">2</DataArray></Lines>
</Piece></PolyData></VTKFile>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.vtp"
            path.write_text(payload)
            xyz, edges = prep.read_vtp(path)
            self.assertEqual(xyz.shape, (3, 3))
            np.testing.assert_array_equal(edges, [[0, 1]])
            np.testing.assert_allclose(xyz[2], [.5]*3)

    def test_binary_vtp_reader_and_full_cli_export(self):
        try:
            from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
            from vtkmodules.vtkCommonCore import vtkPoints
            from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
        except ImportError:
            self.skipTest("VTK is not installed in this local test environment")
        fixture = PreparationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        points = vtkPoints()
        points.InsertNextPoint(.1, .2, .3)
        points.InsertNextPoint(.3, .4, .5)
        cells = vtkCellArray()
        cells.InsertNextCell(2)
        cells.InsertCellPoint(0)
        cells.InsertCellPoint(1)
        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(cells)
        # Test VTP files stay intentionally invalid: the CLI must not read them.
        for split in ("train", "val"):
            for path in (fixture.root / split / "vtp").glob("*.vtp"):
                writer = vtkXMLPolyDataWriter()
                writer.SetFileName(str(path))
                writer.SetInputData(poly)
                writer.SetDataModeToAppended()
                self.assertEqual(writer.Write(), 1)
        config_file = fixture.base / "config.json"
        config_file.write_text(json.dumps(fixture.config))
        output = fixture.base / "cli-output"
        subprocess.run([sys.executable, str(REPO / "scripts/prepare_vascular_pilot.py"),
                        "--config", str(config_file), "--output-dir", str(output)],
                       capture_output=True, text=True, check=True)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["selected_counts"], {"train": 2, "val": 2, "test": 2})
        self.assertFalse(report["test_graph_content_read"])
        with np.load(output / "graphs/val.npz", allow_pickle=False) as data:
            np.testing.assert_allclose(data["node_features"][0], [.1, .2, .3])


class ClusterPathsTests(unittest.TestCase):
    def test_paths_use_scratch_and_reject_work_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            env = dict(os.environ, WORK=str(base / "work"), SCRATCH=str(base / "scratch"))
            env.pop("GRALE_VENV", None)
            env.pop("GRALE_ARCH", None)
            command = ['bash', '-c', 'source "$1"; printf "%s\\n" "$GRALE_VENV" "$PIP_CACHE_DIR" "$GRALE_DATASET_ROOT"', '_', str(REPO / 'cluster/jean_zay/paths.sh')]
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            self.assertEqual(lines[0], str(base / "scratch/venvs/grale-a100-torch230"))
            self.assertTrue(lines[1].startswith(str(base / "scratch")))
            env["GRALE_VENV"] = str(base / "work/venvs/grale")
            result = subprocess.run(['bash', '-c', 'source "$1"', '_', str(REPO / 'cluster/jean_zay/paths.sh')], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_submission_uses_exported_paths_and_correct_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "dataset"
            root.mkdir()
            (root / "patch_index.csv").touch()
            (base / "new_split.csv").touch()
            scratch = base / "scratch"
            python = scratch / "venvs/grale-a100-torch230/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 0\n")
            python.chmod(0o755)
            binaries = base / "bin"
            binaries.mkdir()
            sbatch = binaries / "sbatch"
            sbatch.write_text('#!/bin/sh\nprintf "%s\\n" "$GRALE_PREPARE_MODE" "$GRALE_PREPARE_OUTPUT" "$@"\n')
            sbatch.chmod(0o755)
            env = dict(os.environ, WORK=str(base / "work"), SCRATCH=str(scratch),
                       GRALE_ARCH="a100", GRALE_VENV=str(python.parents[1]),
                       GRALE_REPO_DIR=str(REPO), GRALE_DATASET_ROOT=str(root),
                       GRALE_PREPARED_DATA=str(scratch / "export"),
                       PATH=str(binaries) + os.pathsep + os.environ["PATH"])
            env.pop("GRALE_PREPARE_OUTPUT", None)
            env["GRALE_LOG_DIR"] = str(base / "logs")
            result = subprocess.run(['bash', str(REPO / 'cluster/jean_zay/submit_prepare.sh'), 'full'],
                                    env=env, capture_output=True, text=True, check=True)
            self.assertIn(str(scratch / "export"), result.stdout)
            self.assertIn("prepare_data_cpu.slurm", result.stdout)
            self.assertNotIn("--gres", result.stdout)
            result = subprocess.run(['bash', str(REPO / 'cluster/jean_zay/submit_environment_smoke.sh'), 'a100'],
                                    env=env, capture_output=True, text=True, check=True)
            self.assertIn("--account=vnc@a100", result.stdout)
            self.assertIn("--partition=gpu_p5", result.stdout)
            self.assertIn("--qos=qos_gpu_a100-dev", result.stdout)


if __name__ == "__main__":
    unittest.main()
