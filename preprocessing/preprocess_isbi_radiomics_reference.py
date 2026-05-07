import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def select_primary_payload(payload):
    if isinstance(payload, dict) and "results" in payload:
        payload = payload["results"]

    if isinstance(payload, list):
        if not payload:
            return {}
        return max(payload, key=get_nodule_score)

    return payload if isinstance(payload, dict) else {}


def get_nodule_score(payload):
    try:
        return float(payload.get("nodule", {}).get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def flatten_numeric_features(obj, prefix=""):
    flattened = {}
    if not isinstance(obj, dict):
        return flattened

    for key, value in obj.items():
        key = str(key)
        if key.startswith("diagnostics_"):
            continue

        feature_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_numeric_features(value, feature_name))
        elif isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
            flattened[feature_name] = float(value)

    return flattened


def extract_radiomics_features(payload):
    payload = select_primary_payload(payload)

    if "radiomics_3d" in payload:
        features = payload.get("radiomics_3d", {}).get("features", {})
        return flatten_numeric_features(features)

    if "features" in payload:
        return flatten_numeric_features(payload["features"])

    return flatten_numeric_features(payload)


def relative_scan_stem(raw_path, input_root):
    raw = raw_path.replace("\\", "/")
    root = input_root.replace("\\", "/").rstrip("/")
    if raw.startswith(root + "/"):
        raw = raw[len(root) + 1 :]
    return PurePosixPath(raw.lstrip("/"))


def radiomics_json_path(raw_path, input_root, radiomics_json_root):
    rel = relative_scan_stem(raw_path, input_root)
    return Path(radiomics_json_root) / Path(rel.as_posix() + ".json")


def tensor_output_path(raw_path, input_root, tensor_root, suffix):
    rel = relative_scan_stem(raw_path, input_root)
    return Path(tensor_root) / Path(rel.as_posix() + suffix)


def load_scan_index(index_path):
    records = []
    for patient in load_json(index_path):
        reg_no = patient["reg_no"]
        for step in patient.get("time_steps", []):
            records.append(
                {
                    "reg_no": reg_no,
                    "date": step["date"],
                    "file_path": step["file_path"],
                }
            )
    return records


def collect_feature_schema(records, input_root, radiomics_json_root):
    feature_names = set()
    missing = []

    for record in records:
        json_path = radiomics_json_path(record["file_path"], input_root, radiomics_json_root)
        if not json_path.exists():
            missing.append(str(json_path))
            continue
        features = extract_radiomics_features(load_json(json_path))
        feature_names.update(features)

    return sorted(feature_names), missing


def vectorize_features(features, schema):
    return torch.tensor([features.get(name, 0.0) for name in schema], dtype=torch.float32)


def save_radiomics_tensors(records, schema, input_root, radiomics_json_root, tensor_root, suffix):
    saved = 0
    skipped = []

    for record in records:
        json_path = radiomics_json_path(record["file_path"], input_root, radiomics_json_root)
        if not json_path.exists():
            skipped.append(str(json_path))
            continue

        features = extract_radiomics_features(load_json(json_path))
        vector = vectorize_features(features, schema)

        out_path = tensor_output_path(record["file_path"], input_root, tensor_root, suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vector, out_path)
        saved += 1

    return saved, skipped


def save_norm_params(tensor_root, suffix, norm_filename):
    vectors = []
    for path in Path(tensor_root).rglob(f"*{suffix}"):
        vectors.append(torch.load(path, weights_only=True))

    if not vectors:
        raise RuntimeError("No radiomics tensors were saved; cannot compute normalization parameters.")

    matrix = torch.stack(vectors)
    norm = {
        "mean": matrix.mean(dim=0),
        "std": matrix.std(dim=0, unbiased=False).clamp_min(1e-8),
    }
    norm_path = Path(tensor_root) / norm_filename
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(norm, norm_path)
    return norm_path


def write_reference_service_contract(output_path):
    contract = {
        "purpose": "Reference output schema expected from a local DICOM/NIfTI nodule radiomics service.",
        "notes": [
            "The service itself depends on local detector, DICOM conversion, and PyRadiomics code.",
            "This repository does not ship those private model files or hospital-specific converters.",
            "For each scan time point, save one JSON payload under radiomics_json_root using the same relative path as m1_m2.json.",
        ],
        "example_payload": {
            "job_id": "example-job",
            "results": [
                {
                    "software_version": "v0.1",
                    "space": "HU",
                    "nodule": {
                        "score": 0.91,
                        "label": 0.0,
                        "box_order_assumed": "(z0, y0, z1, y1, x0, x1)",
                        "box_zyx": {"z0": 10, "z1": 20, "y0": 80, "y1": 120, "x0": 90, "x1": 130},
                    },
                    "radiomics_3d": {
                        "binWidth": 25,
                        "features": {
                            "original_firstorder_Mean": -520.4,
                            "original_firstorder_Variance": 3120.8,
                            "original_shape_MeshVolume": 1420.0,
                        },
                    },
                }
            ],
        },
    }
    write_json(output_path, contract)


def run(args):
    if args.write_contract:
        write_reference_service_contract(args.contract_output)

    records = load_scan_index(args.index)
    schema, missing_for_schema = collect_feature_schema(records, args.input_root, args.radiomics_json_root)
    if not schema:
        raise RuntimeError(
            "No radiomics features were found. Check --radiomics-json-root and the relative paths in the index file."
        )

    saved, missing_for_tensor = save_radiomics_tensors(
        records=records,
        schema=schema,
        input_root=args.input_root,
        radiomics_json_root=args.radiomics_json_root,
        tensor_root=args.tensor_root,
        suffix=args.tensor_suffix,
    )
    norm_path = save_norm_params(args.tensor_root, args.tensor_suffix, args.norm_filename)
    write_json(args.schema_output, schema)

    missing = sorted(set(missing_for_schema + missing_for_tensor))
    if missing:
        write_json(args.missing_output, missing)

    print(f"Scan records in index: {len(records)}")
    print(f"Radiomics feature dimension: {len(schema)}")
    print(f"Saved radiomics tensors: {saved}")
    print(f"Saved normalization parameters: {norm_path}")
    print(f"Saved feature schema: {args.schema_output}")
    if missing:
        print(f"Missing radiomics JSON files: {len(missing)}; see {args.missing_output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reference preprocessing for converting local nodule radiomics JSON outputs "
            "into model-ready _radiomics.pt tensors."
        )
    )
    parser.add_argument("--index", default="dataset/m1_m2.json", help="Scan index JSON used by the training dataset.")
    parser.add_argument("--input-root", default="/data/share/hospital_2", help="Raw image root used in m1_m2.json.")
    parser.add_argument(
        "--radiomics-json-root",
        default="outputs/radiomics_json",
        help="Root containing one radiomics JSON per scan, mirroring paths relative to input-root.",
    )
    parser.add_argument("--tensor-root", default="data/tensors", help="Output tensor root used by train/config.json.")
    parser.add_argument("--tensor-suffix", default="_radiomics.pt", help="Output tensor suffix.")
    parser.add_argument("--norm-filename", default="m2_norm_params.pt", help="Normalization filename under tensor-root.")
    parser.add_argument("--schema-output", default="outputs/radiomics_feature_schema.json", help="Feature order JSON.")
    parser.add_argument("--missing-output", default="outputs/missing_radiomics_json.json", help="Missing input report.")
    parser.add_argument(
        "--write-contract",
        action="store_true",
        help="Also write a JSON document describing the expected local service output schema.",
    )
    parser.add_argument(
        "--contract-output",
        default="outputs/radiomics_service_contract.json",
        help="Output path for the local service schema contract.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
