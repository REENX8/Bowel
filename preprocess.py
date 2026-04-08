"""
preprocess.py — DICOM folder → .npy pipeline
=============================================
Converts a directory of per-patient DICOM series into windowed float32 .npy
volumes ready for training or inference with last.py / firstbowel_injury_model.py.

Usage
-----
    python preprocess.py --input-dir /path/to/patients --output-dir /path/to/out

The input directory is expected to contain one sub-folder per patient, each
holding the DICOM files for that patient's abdominal CT series:

    input-dir/
        patient_001/
            IM-0001-0001.dcm
            IM-0001-0002.dcm
            ...
        patient_002/
            ...

Outputs
-------
    output-dir/
        patient_001.npy    # float32, shape (D, 256, 256), values in [0, 1]
        patient_002.npy
        ...
        labels.csv         # patient_id, bowel_injury — fill bowel_injury column manually
"""

import argparse
import os
import sys
import csv
from pathlib import Path

import numpy as np

try:
    import pydicom
except ImportError:
    sys.exit("pydicom is required. Run: pip install pydicom")

try:
    from scipy.ndimage import zoom
except ImportError:
    sys.exit("scipy is required. Run: pip install scipy")

# ── HU windowing constants (match last.py / training script) ──────────────────
WL = -175   # window level (centre)
WW = 425    # window width
HU_MIN = WL - WW / 2   # -387.5
HU_MAX = WL + WW / 2   #   37.5

# ── Target spatial size ───────────────────────────────────────────────────────
TARGET_H = 256
TARGET_W = 256


# ── Helpers ───────────────────────────────────────────────────────────────────

def window01(pixel_array: np.ndarray) -> np.ndarray:
    """Apply HU window and normalise to [0, 1]."""
    arr = pixel_array.astype(np.float32)
    arr = np.clip(arr, HU_MIN, HU_MAX)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
    return arr


def _sort_key(ds):
    """Return a reliable sort key for DICOM slice ordering."""
    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        pass
    try:
        return float(ds.SliceLocation)
    except Exception:
        pass
    return int(getattr(ds, "InstanceNumber", 0))


def load_dicom_volume(patient_dir: Path) -> np.ndarray | None:
    """
    Load all DICOM slices from *patient_dir*, sort them, apply HU windowing,
    resize to TARGET_H × TARGET_W, and return float32 array (D, H, W).
    Returns None if fewer than 2 valid DICOM files are found.
    """
    dcm_files = sorted(patient_dir.glob("**/*.dcm"))
    dcm_files += sorted(patient_dir.glob("**/*.DCM"))
    if not dcm_files:
        print(f"  Warning: No .dcm/.DCM files found in {patient_dir}")
        return None

    datasets = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False)
            if not hasattr(ds, "pixel_array"):
                continue
            datasets.append(ds)
        except Exception as e:
            print(f"  Warning: Skipping {f.name}: {e}")
            continue

    if len(datasets) < 2:
        return None

    datasets.sort(key=_sort_key)

    slices = []
    for ds in datasets:
        try:
            arr = ds.pixel_array.astype(np.float32)
            # Apply RescaleSlope / RescaleIntercept to convert to HU
            try:
                slope = float(getattr(ds, "RescaleSlope", 1))
            except (TypeError, ValueError):
                slope = 1.0
            try:
                intercept = float(getattr(ds, "RescaleIntercept", 0))
            except (TypeError, ValueError):
                intercept = 0.0
            arr = arr * slope + intercept
        except Exception:
            continue

        # Window and normalise
        arr2d = window01(arr)

        # Resize to target spatial size if needed
        h, w = arr2d.shape[:2]
        if h != TARGET_H or w != TARGET_W:
            zoom_h = TARGET_H / h
            zoom_w = TARGET_W / w
            arr2d = zoom(arr2d, (zoom_h, zoom_w), order=1)

        slices.append(arr2d.astype(np.float32))

    if len(slices) < 2:
        return None

    return np.stack(slices, axis=0)  # (D, H, W)


def preprocess_dataset(input_dir: Path, output_dir: Path, verbose: bool = True):
    """
    Iterate over patient sub-folders in *input_dir*, convert each to .npy,
    and write a labels.csv template in *output_dir*.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not patient_dirs:
        print(f"No sub-directories found in {input_dir}. Expected one folder per patient.")
        return

    n_total = len(patient_dirs)
    n_ok = 0
    n_fail = 0
    rows: list[dict] = []

    for i, pdir in enumerate(patient_dirs):
        patient_id = pdir.name
        out_path = output_dir / f"{patient_id}.npy"

        prefix = f"[{i + 1:>{len(str(n_total))}}/{n_total}]"

        if verbose:
            print(f"{prefix} {patient_id} … ", end="", flush=True)

        vol = load_dicom_volume(pdir)
        if vol is None:
            if verbose:
                print("SKIP (no valid slices)")
            n_fail += 1
            continue

        np.save(str(out_path), vol)
        if verbose:
            print(f"OK  shape={vol.shape}  → {out_path.name}")

        rows.append({"patient_id": patient_id, "bowel_injury": ""})
        n_ok += 1

    # Write labels template
    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "bowel_injury"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {n_ok} volumes saved, {n_fail} skipped.")
    print(f"Labels template written to: {labels_path}")
    print("Fill in the 'bowel_injury' column (0 = no injury, 1 = injury) before training.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert DICOM patient folders to windowed .npy volumes for training."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing one sub-folder per patient with DICOM files.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory where .npy volumes and labels.csv will be written.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-file output.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        sys.exit(f"Input directory does not exist: {input_dir}")

    preprocess_dataset(input_dir, output_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
