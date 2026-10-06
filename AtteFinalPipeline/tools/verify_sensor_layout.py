from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from typing import List, Optional, TextIO, Tuple

KNOWN_DATASETS = [
    "ucihar",
    "hospital",
    "uschad",
    "capture24",
    "capture24_walmsley",
    "capture24_full",
    "shoaib",
    "mhealth",
    "mhealth_nonull",
    "pamap2",
    "opportunity",
    "skoda",
]


def _hr(char: str = "-", width: int = 78) -> str:
    return char * width


def _write_header(fh: TextIO, title: str) -> None:
    fh.write("\n")
    fh.write(_hr("=") + "\n")
    fh.write(f"  {title}\n")
    fh.write(_hr("=") + "\n")


def _write_channel_table(fh: TextIO, names: List[str]) -> None:
    if not names:
        fh.write("    (no channels found)\n")
        return
    for i, name in enumerate(names):
        fh.write(f"    [{i:>3d}] {name}\n")


def _find_sample_json(dataset_dir: str) -> Optional[str]:
    if not os.path.isdir(dataset_dir):
        return None
    candidates = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "*.json"))):
        base = os.path.basename(path)
        if base.startswith("prediction_summary"):
            continue
        if base.startswith("tracking_worker"):
            continue
        if base.startswith("progress"):
            continue
        candidates.append(path)
    return candidates[0] if candidates else None


def _extract_layout_from_json(path: str) -> Tuple[List[str], List[str], int, int]:
    with open(path, "r") as fh:
        entry = json.load(fh)
    meta = entry.get("analysis_metadata", {})
    sensor_channels = list(meta.get("sensor_channels", []))
    T = int(meta.get("num_timesteps", 0))
    raw = entry.get("raw_sensor_data", {})
    raw_keys = list(raw.keys())
    if raw_keys:
        first_series = raw[raw_keys[0]]
        T = len(first_series)
    D = len(sensor_channels)
    return (sensor_channels, raw_keys, T, D)


def _verify_one_dataset(fh: TextIO, dataset: str, root: str) -> bool:
    dataset_dir = os.path.join(root, dataset)
    sample_path = _find_sample_json(dataset_dir)
    _write_header(fh, f"DATASET: {dataset}")
    fh.write(f"  Output dir : {dataset_dir}\n")
    if sample_path is None:
        fh.write(f"  STATUS     : [SKIP] no per-sample JSONs found in this directory\n")
        return False
    fh.write(f"  Sample JSON: {os.path.basename(sample_path)}\n")
    sensor_channels, raw_keys, T, D = _extract_layout_from_json(sample_path)
    fh.write(f"  Channels D : {D}\n")
    fh.write(f"  Timesteps T: {T}\n")
    if sensor_channels == raw_keys:
        fh.write("  Order match: OK (metadata order == raw_sensor_data key order)\n")
    else:
        fh.write("  Order match: MISMATCH between metadata and raw_sensor_data\n")
        for i, (a, b) in enumerate(zip(sensor_channels, raw_keys)):
            if a != b:
                fh.write(f"    First divergence at index {i}:\n")
                fh.write(f"      metadata[{i}]        = {a!r}\n")
                fh.write(f"      raw_sensor_data[{i}] = {b!r}\n")
                break
        if len(sensor_channels) != len(raw_keys):
            fh.write(
                f"    Length mismatch: metadata={len(sensor_channels)}, raw_sensor_data={len(raw_keys)}\n"
            )
    fh.write("\n")
    fh.write("  Channel layout (from analysis_metadata.sensor_channels):\n")
    _write_channel_table(fh, sensor_channels)
    dupes = sorted({n for n in sensor_channels if sensor_channels.count(n) > 1})
    if dupes:
        fh.write("\n")
        fh.write(f"  WARNING: duplicate channel names detected: {dupes}\n")
    return True


def _import_get_sensor_config(source_script_path: str):
    if source_script_path is None:
        from AtteFinalPipeline.data.sensors import get_sensor_config

        return get_sensor_config
    import importlib.util

    spec = importlib.util.spec_from_file_location("xai_src", source_script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {source_script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_sensor_config


def _cross_check_with_source(
    fh: TextIO,
    dataset: str,
    sensor_channels_actual: List[str],
    D: int,
    get_sensor_config,
) -> None:
    expected_names, _ = get_sensor_config(D, dataset)
    fh.write("\n")
    fh.write("  Cross-check against get_sensor_config():\n")
    if expected_names == sensor_channels_actual:
        fh.write("    EXACT MATCH -- JSON layout equals source-script layout\n")
        return
    fh.write("    MISMATCH -- diffs (idx, expected, actual):\n")
    max_len = max(len(expected_names), len(sensor_channels_actual))
    diffs = 0
    for i in range(max_len):
        e = expected_names[i] if i < len(expected_names) else "<missing>"
        a = (
            sensor_channels_actual[i]
            if i < len(sensor_channels_actual)
            else "<missing>"
        )
        if e != a:
            fh.write(f"      [{i:>3d}]  expected={e!r:<30s}  actual={a!r}\n")
            diffs += 1
    if diffs == 0:
        fh.write("    (length differs but overlapping elements agree)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify sensor-channel layout in xai_output/ JSONs and write a plain-text report."
    )
    parser.add_argument(
        "--root",
        default="./xai_output",
        help="Root directory containing per-dataset subfolders (default: ./xai_output)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./sensor_layout_report.txt",
        help="Path for the output .txt report (default: ./sensor_layout_report.txt)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Only verify this one dataset (default: all known datasets)",
    )
    parser.add_argument(
        "--check-against-source",
        action="store_true",
        help="Also import get_sensor_config from the source script and diff its output against the JSON.",
    )
    parser.add_argument(
        "--source-script",
        default=None,
        help="Path to the script that defines get_sensor_config (used only with --check-against-source).",
    )
    args = parser.parse_args()
    if not os.path.isdir(args.root):
        print(f"ERROR: root directory does not exist: {args.root}", file=sys.stderr)
        sys.exit(1)
    datasets = [args.dataset] if args.dataset else KNOWN_DATASETS
    get_sensor_config = None
    if args.check_against_source:
        try:
            get_sensor_config = _import_get_sensor_config(args.source_script)
        except Exception as e:
            print(
                f"WARNING: could not import source script ({e}); skipping cross-check.",
                file=sys.stderr,
            )
            get_sensor_config = None
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    found_any = False
    with open(args.output, "w") as fh:
        fh.write(_hr("=") + "\n")
        fh.write("  Sensor-channel layout report\n")
        fh.write(f"  Source root: {os.path.abspath(args.root)}\n")
        if args.check_against_source and get_sensor_config is not None:
            fh.write(
                f"  Cross-check: ON (source = {args.source_script or 'AtteFinalPipeline.data.sensors'})\n"
            )
        else:
            fh.write("  Cross-check: OFF\n")
        fh.write(_hr("=") + "\n")
        for ds in datasets:
            ok = _verify_one_dataset(fh, ds, args.root)
            if ok:
                found_any = True
                if get_sensor_config is not None:
                    sample = _find_sample_json(os.path.join(args.root, ds))
                    if sample:
                        sensor_channels, _, _, D = _extract_layout_from_json(sample)
                        try:
                            _cross_check_with_source(
                                fh, ds, sensor_channels, D, get_sensor_config
                            )
                        except Exception as e:
                            fh.write(f"    (cross-check failed: {e})\n")
        fh.write("\n")
        fh.write(_hr("=") + "\n")
        if found_any:
            fh.write("  Done.\n")
        else:
            fh.write(
                f"  No JSONs found under {args.root}/. Have you run xai_full_analysis.py yet?\n"
            )
        fh.write(_hr("=") + "\n")
    if not found_any:
        print(
            f"No JSONs found under {args.root}/. Report written to {args.output} anyway.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"Report written: {args.output}")


if __name__ == "__main__":
    main()
