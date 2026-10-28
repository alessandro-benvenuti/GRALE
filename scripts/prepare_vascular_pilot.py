#!/usr/bin/env python3
"""Freeze MRI patch manifests and audit/export GT graphs on a CPU node.

No image/segmentation arrays are loaded. Test graph content is not read unless
--include-test-graphs is explicitly supplied. No GRALE/Lightning import is needed.
The exported ragged NPZ arrays are inputs for the forthcoming vascular adapter;
they cannot be passed to the upstream categorical HDF5 loader unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = ("sample_id", "patient_id", "split", "start_d", "start_h",
                   "start_w", "image_path", "segmentation_path", "graph_path")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patient_id(value):
    """Normalize numeric identifiers without losing non-numeric identities."""
    value = str(value).strip()
    return str(int(value)) if value.isdecimal() else value


def read_patients(path):
    result = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key, split = patient_id(row["patient_id"]), row["split"]
            if not key or split not in SPLITS or key in result:
                raise ValueError(f"Invalid/duplicate patient assignment: {row}")
            result[key] = split
    if not result:
        raise ValueError("Patient split file is empty")
    return result


def read_index(path, patients):
    rows = {split: [] for split in SPLITS}
    seen = set()
    indexed_patients = set()
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "patient_id", "split", "start_d", "start_h", "start_w"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Patch index requires columns {sorted(required)}")
        for original in reader:
            sample, split = original["sample_id"], original["split"]
            key = patient_id(original["patient_id"])
            if not sample or Path(sample).name != sample or sample in {".", ".."}:
                raise ValueError(f"Invalid sample identifier: {sample!r}")
            if sample in seen:
                raise ValueError(f"Duplicate patch identifier: {sample}")
            if split not in SPLITS or patients.get(key) != split:
                raise ValueError(f"Patch/patient split mismatch: {sample}, patient={key}")
            seen.add(sample)
            indexed_patients.add(key)
            # Do not copy GT-derived counts/intensities into selection records.
            rows[split].append({key_: original[key_] for key_ in required})
            rows[split][-1]["patient_id"] = key
    if indexed_patients != set(patients):
        raise ValueError("Patch index does not cover exactly the saved patient split")
    return rows


def canonical_split_bytes(patients):
    """Reproduce the generator's saved CSV, without generating assignments."""
    def sort_key(key):
        return (0, int(key)) if key.isdecimal() else (1, key)

    rows = ["patient_id,split"]
    rows.extend(f"{key},{patients[key]}" for key in sorted(patients, key=sort_key))
    return ("\n".join(rows) + "\n").encode("utf-8")


def patients_from_index(path):
    """Recover existing patient membership from provenance, not graph labels."""
    patients = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"patient_id", "split"} <= set(reader.fieldnames or []):
            raise ValueError("Patch index must contain patient_id and split")
        for row in reader:
            key, split = patient_id(row["patient_id"]), row["split"]
            if not key or split not in SPLITS:
                raise ValueError(f"Invalid patch-index patient assignment: {row}")
            if key in patients and patients[key] != split:
                raise ValueError(f"Patient {key} occurs in multiple patch-index splits")
            patients[key] = split
    if not patients:
        raise ValueError("Patch index has no patient assignments")
    return patients


def validate_inputs(config, *, dataset_root=None, patient_split_file=None):
    """Validate metadata on the login node too; do not read graph/image arrays."""
    root = Path(dataset_root or config["dataset_root"]).expanduser().resolve()
    if config["schema_version"] != 1 or config["coordinate_order"] != ["D", "H", "W"]:
        raise ValueError("Unsupported configuration schema or coordinate order")
    if config["coordinate_space"] != "normalized" or config["laplacian_features"]:
        raise ValueError("This pilot requires normalized coordinates and no Laplacian features")
    patch_size = np.asarray(config["patch_size"], dtype=np.int64)
    if patch_size.shape != (3,) or np.any(patch_size <= 0):
        raise ValueError("patch_size must contain three positive integers")
    index_file = (root / config["patch_index_file"]).resolve()
    generation, completion = root / "generation_config.json", root / "generation_summary.json"
    metadata = {}
    if generation.is_file():
        metadata = json.loads(generation.read_text())
        if metadata.get("patch_size") != patch_size.tolist():
            raise ValueError("Configured patch size differs from generation_config.json")
        if not metadata.get("split_sha256"):
            raise ValueError("generation_config.json has no split_sha256")
    elif config.get("require_generation_config", config.get("require_generation_metadata", True)):
        raise FileNotFoundError(f"Required generation configuration is absent: {generation}")
    summary = json.loads(completion.read_text())
    if summary.get("complete") is not True:
        raise ValueError("The source dataset is not marked complete in generation_summary.json")
    if (metadata.get("fingerprint") is not None and summary.get("fingerprint") is not None
            and metadata["fingerprint"] != summary["fingerprint"]):
        raise ValueError("Generation configuration and summary fingerprints differ")
    configured = config.get("patient_split_file")
    split_file = ((root / Path(patient_split_file).expanduser()).resolve() if patient_split_file
                  else (root / configured).resolve() if configured else None)
    split_source = "csv"
    if patient_split_file and not split_file.is_file():
        raise FileNotFoundError(f"Explicit patient split file does not exist: {split_file}")
    if split_file is not None and split_file.is_file():
        patients, split_bytes = read_patients(split_file), split_file.read_bytes()
    else:
        # Only accept nearby alternatives if they match the original byte hash.
        candidates = sorted(set(root.glob("*split*.csv")) | set(root.parent.glob("*split*.csv")))
        expected_hash = metadata.get("split_sha256")
        matches = [path for path in candidates if path.is_file() and fingerprint(path) == expected_hash]
        if matches:
            split_file = matches[0].resolve()
            patients, split_bytes = read_patients(split_file), split_file.read_bytes()
        else:
            split_file = None
            split_source = "patch_index_recovery"
            patients = patients_from_index(index_file)
            split_bytes = canonical_split_bytes(patients)
    split_hash = hashlib.sha256(split_bytes).hexdigest()
    if metadata and split_hash != metadata["split_sha256"]:
        if split_source == "patch_index_recovery":
            raise ValueError("Recovered assignments do not reproduce the generation split hash. "
                             "Locate the original CSV and set GRALE_PATIENT_SPLIT_FILE to its path; "
                             "do not regenerate the split.")
        raise ValueError("generation_config.json does not match the saved patient split")
    counts = dict(Counter(patients.values()))
    if counts != config["expected_patient_counts"]:
        raise ValueError(f"Unexpected patient split {counts}; expected {config['expected_patient_counts']}")
    index = read_index(index_file, patients)
    for split in SPLITS:
        count = int(config["sample_counts"][split])
        if count < 1 or len(index[split]) < count:
            raise ValueError(f"{split}: requested {count} patches, available {len(index[split])}")
    for key in ("completed_patients", "expected_patients"):
        if key in summary and summary[key] != len(patients):
            raise ValueError(f"generation_summary.json {key} differs from the indexed patient count")
    if "patches_indexed" in summary and summary["patches_indexed"] != sum(map(len, index.values())):
        raise ValueError("generation_summary.json patches_indexed differs from the patch index")
    provenance = {
        "generation_config_present": generation.is_file(),
        "original_split_hash_verified": bool(metadata),
        "generation_patch_size_verified": bool(metadata),
        "source_dataset_marked_complete": True,
        "patient_assignments_checked_against_patch_index": True,
    }
    return {"root": root, "patch_size": patch_size, "split_file": split_file,
            "split_source": split_source, "split_bytes": split_bytes, "split_sha256": split_hash,
            "index_file": index_file, "index": index, "generation": generation,
            "completion": completion, "patient_counts": counts, "provenance": provenance}


def input_summary(inputs):
    return {"dataset_root": str(inputs["root"]),
            "patient_split_source": inputs["split_source"],
            "patient_split_file": str(inputs["split_file"]) if inputs["split_file"] else None,
            "patient_split_sha256": inputs["split_sha256"],
            "patient_counts": inputs["patient_counts"],
            "provenance": inputs["provenance"],
            "available_patch_counts": {split: len(rows) for split, rows in inputs["index"].items()}}


def inspect_dataset(config, *, dataset_root=None):
    """Inventory the actual copy without requiring generator companion files."""
    root = Path(dataset_root or config["dataset_root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    index = (root / config["patch_index_file"]).resolve()
    result = {
        "dataset_root": str(root),
        "top_level_entries": sorted(path.name + ("/" if path.is_dir() else "") for path in root.iterdir()),
        "parent_metadata_files": sorted(str(path) for path in root.parent.iterdir()
                                        if path.is_file() and path.suffix.lower() in {".csv", ".json"}),
        "split_folders": {split: {folder: (root / split / folder).is_dir()
                                  for folder in ("raw", "seg", "vtp")} for split in SPLITS},
        "patch_index_file": str(index), "patch_index_exists": index.is_file(),
    }
    result["generation_metadata"] = {
        name: {"dataset_root": (root / name).is_file(), "parent": (root.parent / name).is_file()}
        for name in ("generation_config.json", "generation_summary.json")}
    result["split_files"] = {}
    for split in SPLITS:
        result["split_files"][split] = {}
        for folder, suffix in (("raw", "_data.nii.gz"), ("seg", "_seg.nii.gz"), ("vtp", "_graph.vtp")):
            directory = root / split / folder
            if not directory.is_dir():
                result["split_files"][split][folder] = {"exists": False}
                continue
            names = sorted(path.name for path in directory.iterdir() if path.is_file())
            result["split_files"][split][folder] = {
                "exists": True, "file_count": len(names), "expected_suffix": suffix,
                "matching_suffix_count": sum(name.endswith(suffix) for name in names),
                "examples": names[:3]}
    if index.is_file():
        with index.open(newline="") as handle:
            reader = csv.DictReader(handle)
            result["patch_index_columns"] = reader.fieldnames
            required = set(MANIFEST_FIELDS[:6])
            result["missing_required_index_columns"] = sorted(required - set(reader.fieldnames or []))
            patch_counts = Counter()
            examples = []
            for row in reader:
                patch_counts[row.get("split") or "<missing>"] += 1
                if len(examples) < 3:
                    examples.append({key: row.get(key) for key in MANIFEST_FIELDS[:6]})
            result["indexed_patch_counts"] = dict(patch_counts)
            result["patch_index_examples"] = examples
        try:
            result["indexed_patient_counts"] = dict(Counter(patients_from_index(index).values()))
        except (ValueError, KeyError) as error:
            result["patch_index_assignment_error"] = str(error)
    return result


def select_records(rows, count, *, seed, split):
    """Same SHA256 membership/order as GNBM select_capped_records."""
    if count < 1 or len(rows) < count:
        raise ValueError(f"{split}: requested {count} patches, available {len(rows)}")

    def rank(row):
        sample = row["sample_id"]
        payload = f"{int(seed)}\0{split}\0{sample}".encode("utf-8")
        return hashlib.sha256(payload).digest(), sample

    return sorted(sorted(rows, key=rank)[:count], key=lambda row: row["sample_id"])


def add_paths(root, records):
    result = []
    for original in records:
        row = dict(original)
        sample, split = row["sample_id"], row["split"]
        for key, directory, suffix in (
            ("image_path", "raw", "_data.nii.gz"),
            ("segmentation_path", "seg", "_seg.nii.gz"),
            ("graph_path", "vtp", "_graph.vtp"),
        ):
            path = root / split / directory / f"{sample}{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)
            row[key] = str(path)
        result.append(row)
    return result


def read_vtp(path):
    """Read compressed/binary or ASCII VTP using VTK, including isolated nodes."""
    try:
        from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
        from vtkmodules.util.numpy_support import vtk_to_numpy
    except ImportError as error:
        raise ImportError("VTP extraction needs vtk; run the Jean Zay setup script") from error
    reader = vtkXMLPolyDataReader()
    errors = []
    reader.AddObserver("ErrorEvent", lambda *_: errors.append(str(path)))
    reader.SetFileName(str(path))
    reader.Update()
    if errors or reader.GetErrorCode():
        raise ValueError(f"Cannot read VTP graph: {path}")
    graph = reader.GetOutput()
    points = graph.GetPoints()
    positions = (np.empty((0, 3), dtype=np.float32) if points is None else
                 np.asarray(vtk_to_numpy(points.GetData()), dtype=np.float32).copy())
    cells = graph.GetLines()
    offsets = np.asarray(vtk_to_numpy(cells.GetOffsetsArray()), dtype=np.int64)
    connectivity = np.asarray(vtk_to_numpy(cells.GetConnectivityArray()), dtype=np.int64)
    if (not np.all(np.diff(offsets) == 2) or connectivity.size != 2 * cells.GetNumberOfCells()
            or graph.GetNumberOfPolys() or graph.GetNumberOfStrips()):
        raise ValueError(f"Expected two-endpoint VTP line cells: {path}")
    return positions, connectivity.reshape(-1, 2).copy()


def inspect_graph(positions, edges, patch_size, tolerance):
    positions = np.asarray(positions, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n = len(positions)
    if positions.shape != (n, 3) or not np.isfinite(positions).all():
        raise ValueError("Graph coordinates must be finite [N,3]")
    upper = (np.asarray(patch_size, dtype=np.float64) - 1) / patch_size
    if np.any(positions < -tolerance) or np.any(positions > upper + tolerance):
        raise ValueError("Coordinates violate the normalized D/H/W index/axis_size convention")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Edge endpoint is outside the node array")
    edges = np.sort(edges, axis=1)
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("Self-loop found in source graph")
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError("Duplicate undirected edge found in source graph")
    if len(edges):
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    neighbors = [[] for _ in range(n)]
    for u, v in edges:
        neighbors[u].append(int(v))
        neighbors[v].append(int(u))
    degrees = np.asarray([len(row) for row in neighbors])
    # Finite unreachable sentinel n, exceeding every valid finite hop distance.
    sp = np.full((n, n), n, dtype=np.uint32)
    components = 0
    visited = set()
    for source in range(n):
        if source not in visited:
            components += 1
        sp[source, source] = 0
        queue = deque([source])
        while queue:
            u = queue.popleft()
            visited.add(u)
            for v in neighbors[u]:
                if sp[source, v] == n:
                    sp[source, v] = sp[source, u] + 1
                    queue.append(v)
    lengths = (np.linalg.norm((positions[edges[:, 0]] - positions[edges[:, 1]]) * patch_size,
                              axis=1) if len(edges) else np.empty(0))
    stats = {
        "nodes": n, "edges": len(edges), "beta0": components,
        "beta1": len(edges) - n + components,
        "endpoints": int(np.sum(degrees == 1)),
        "bifurcations": int(np.sum(degrees >= 3)),
        "isolated_nodes": int(np.sum(degrees == 0)),
        "max_degree": int(degrees.max()) if n else 0,
        "zero_length_edges": int(np.sum(lengths <= 1e-8)),
        "duplicate_positions": n - len(np.unique(positions, axis=0)),
    }
    return positions, edges, sp, stats


def summarize(records):
    result = {"graphs": len(records)}
    for key in ("nodes", "edges", "beta0", "beta1", "endpoints", "bifurcations", "isolated_nodes"):
        values = np.asarray([row[key] for row in records])
        result[key] = {"min": int(values.min()), "max": int(values.max()),
                       "mean": float(values.mean()), "p95": float(np.quantile(values, .95))}
    result["empty_graphs"] = sum(row["nodes"] == 0 for row in records)
    result["graphs_with_cycles"] = sum(row["beta1"] > 0 for row in records)
    result["graphs_with_zero_length_edges"] = sum(row["zero_length_edges"] > 0 for row in records)
    return result


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_archive(path, records, graphs):
    nodes, edges, sp = zip(*graphs)
    np.savez_compressed(
        path, schema_version=np.asarray(1),
        sample_ids=np.asarray([row["sample_id"] for row in records]),
        patient_ids=np.asarray([row["patient_id"] for row in records]),
        node_features=np.concatenate(nodes, axis=0),
        edge_index=np.concatenate(edges, axis=0),
        node_offsets=np.cumsum([0] + [len(x) for x in nodes], dtype=np.int64),
        edge_offsets=np.cumsum([0] + [len(x) for x in edges], dtype=np.int64),
        sp_values=np.concatenate([x.reshape(-1) for x in sp]),
        sp_offsets=np.cumsum([0] + [x.size for x in sp], dtype=np.int64),
    )


def prepare(config, output, *, dataset_root=None, patient_split_file=None, smoke_count=None, include_test=False,
            reader=read_vtp):
    inputs = validate_inputs(config, dataset_root=dataset_root, patient_split_file=patient_split_file)
    root, patch_size = inputs["root"], inputs["patch_size"]
    if smoke_count is not None and smoke_count < 1:
        raise ValueError("smoke_count must be positive")
    split_file, index_file = inputs["split_file"], inputs["index_file"]
    generation, completion = inputs["generation"], inputs["completion"]
    counts, index = inputs["patient_counts"], inputs["index"]
    selected = {}
    for split in SPLITS:
        count = int(config["sample_counts"][split])
        if smoke_count is not None and split != "test":
            count = min(count, smoke_count)
        selected[split] = add_paths(root, select_records(index[split], count,
            seed=config["selection_seed"], split=split))

    # No accidental replacement, source writes, or completed-output reuse.
    output = Path(output).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the source dataset")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    # Atomic creation also prevents two concurrent submissions sharing a target.
    output.mkdir(parents=True, exist_ok=False)
    effective = dict(config, dataset_root=str(root))
    if patient_split_file:
        effective["patient_split_file"] = str(split_file)
    (output / "resolved_config.json").write_text(json.dumps(effective, indent=2) + "\n")
    (output / "patient_split.csv").write_bytes(inputs["split_bytes"])
    for split in SPLITS:
        write_csv(output / "manifests" / f"{split}.csv", selected[split], MANIFEST_FIELDS)
    audited = list(SPLITS) if include_test else ["train", "val"]
    all_stats, summaries = [], {}
    archive_hashes = {}
    (output / "graphs").mkdir()
    for split in audited:
        graphs, stats_rows = [], []
        for i, row in enumerate(selected[split], 1):
            positions, edges = reader(Path(row["graph_path"]))
            try:
                positions, edges, sp, stats = inspect_graph(positions, edges, patch_size,
                                                           config["coordinate_tolerance"])
            except ValueError as error:
                raise ValueError(f"{row['sample_id']}: {error}") from error
            graphs.append((positions, edges, sp))
            stats_rows.append(dict(sample_id=row["sample_id"], split=split, **stats))
            if i == 1 or i % 250 == 0 or i == len(selected[split]):
                print(f"[{split}] {i}/{len(selected[split])} graphs", flush=True)
        all_stats.extend(stats_rows)
        summaries[split] = summarize(stats_rows)
        archive = output / "graphs" / f"{split}.npz"
        write_archive(archive, selected[split], graphs)
        archive_hashes[split] = fingerprint(archive)
    max_nodes = max(row["nodes"] for row in all_stats if row["split"] != "test")
    write_csv(output / "graph_statistics.csv", all_stats, list(all_stats[0]))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "dataset_root": str(root),
        "preparation_script_sha256": fingerprint(Path(__file__)),
        "patient_split_file": str(split_file) if split_file else None,
        "patient_split_source": inputs["split_source"],
        "provenance": inputs["provenance"],
        "patient_split_sha256": inputs["split_sha256"],
        "exported_patient_split_file": str(output / "patient_split.csv"),
        "patch_index_file": str(index_file), "patch_index_sha256": fingerprint(index_file),
        "generation_config_sha256": fingerprint(generation) if generation.is_file() else None,
        "generation_summary_sha256": fingerprint(completion) if completion.is_file() else None,
        "patient_counts": counts, "selection_seed": config["selection_seed"],
        "selected_counts": {key: len(value) for key, value in selected.items()},
        "selected_patient_counts": {key: len({r['patient_id'] for r in value}) for key, value in selected.items()},
        "manifest_sha256": {split: fingerprint(output / "manifests" / f"{split}.csv") for split in SPLITS},
        "archive_sha256": archive_hashes, "audited_splits": audited,
        "test_graph_content_read": include_test, "smoke": smoke_count is not None,
        "minimum_node_capacity_train_val": max_nodes, "statistics": summaries,
        "coordinate_order": config["coordinate_order"], "patch_size": patch_size.tolist(),
        "shortest_path_unreachable": "number_of_nodes_in_that_graph",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Vascular graph preparation report", "",
        f"Source: `{root}`", "",
        f"Patient split: `{split_file}`" if split_file else "Patient split: existing assignments recovered from patch_index.csv.", "",
        f"Original generation split hash verified: {inputs['provenance']['original_split_hash_verified']}", "",
        f"Patch size verified against generation configuration: {inputs['provenance']['generation_patch_size_verified']}", "",
        "Source generation summary marks the dataset complete.", "",
        f"Selected patches: {report['selected_counts']}", "",
        f"Patient counts in source split: {counts}", "",
        f"Audited graph content: {', '.join(audited)}", "",
        f"Minimum capacity required by train/validation: **{max_nodes} nodes**", "",
        "This is a smoke export." if smoke_count else "This is the full pilot export.", "",
        "| Split | Graphs | Mean nodes | Max nodes | Graphs with cycles | Empty graphs |",
        "|---|---:|---:|---:|---:|---:|"]
    for split, stats in summaries.items():
        lines.append(f"| {split} | {stats['graphs']} | {stats['nodes']['mean']:.2f} | "
                     f"{stats['nodes']['max']} | {stats['graphs_with_cycles']} | {stats['empty_graphs']} |")
    lines += ["", "Test patches are reserved by manifest. Their graph content is untouched."
              if not include_test else "Test graph content was explicitly exported.", "",
              "NPZ archives require the vascular adapter; the upstream categorical HDF5 loader cannot read them.", ""]
    (output / "report.md").write_text("\n".join(lines))
    (output / "COMPLETED.json").write_text(json.dumps({"report_sha256": fingerprint(output / "report.json")}) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/vascular/pilot_data.json")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--patient-split-file", type=Path, help="Override the location of the saved patient CSV")
    parser.add_argument("--check-inputs", action="store_true", help="Validate metadata and split recovery without writing output or reading graphs")
    parser.add_argument("--inspect-dataset", action="store_true", help="List available files, split directories and patch-index patient counts without requiring generator metadata")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-count", type=int)
    parser.add_argument("--include-test-graphs", action="store_true",
                        help="Read/export test GT only for an explicitly frozen evaluation protocol")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.inspect_dataset:
        print(json.dumps(inspect_dataset(config, dataset_root=args.dataset_root), indent=2))
        return 0
    if args.check_inputs:
        print(json.dumps(input_summary(validate_inputs(config, dataset_root=args.dataset_root,
                         patient_split_file=args.patient_split_file)), indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --check-inputs or --inspect-dataset is used")
    report = prepare(config, args.output_dir,
        dataset_root=args.dataset_root, patient_split_file=args.patient_split_file, smoke_count=args.smoke_count,
        include_test=args.include_test_graphs)
    print(json.dumps({key: report[key] for key in ("selected_counts", "audited_splits", "minimum_node_capacity_train_val")}, indent=2))
    print(f"Report: {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
