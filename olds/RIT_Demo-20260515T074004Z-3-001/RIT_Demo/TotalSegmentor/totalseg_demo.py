"""
=============================================================
TotalSegmentator – DICOM Series Reader + Segmentation Demo
=============================================================
Compatible:  Python 3.9  |  TotalSegmentator 2.3.0
Usage:
    python totalseg_demo.py --dicom_dir /path/to/dicom/series
=============================================================
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pydicom
import SimpleITK as sitk
import nibabel as nib

# ─────────────────────────────────────────────────────────────
# 1.  DICOM SERIES → NIfTI (in-memory or temp file)
# ─────────────────────────────────────────────────────────────

def load_dicom_series(dicom_dir: str) -> sitk.Image:
    """Read a DICOM series folder and return a SimpleITK image."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in: {dicom_dir}")

    print(f"[INFO] Found {len(series_ids)} series ID(s). Using: {series_ids[0]}")
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    print(f"[INFO] Volume size : {image.GetSize()}")
    print(f"[INFO] Spacing (mm): {image.GetSpacing()}")
    print(f"[INFO] Origin      : {image.GetOrigin()}")
    return image


def sitk_to_nifti(sitk_image: sitk.Image, out_path: str) -> None:
    """Write a SimpleITK image to a NIfTI file."""
    sitk.WriteImage(sitk_image, out_path)
    print(f"[INFO] Saved NIfTI → {out_path}")


# ─────────────────────────────────────────────────────────────
# 2.  RUN TOTALSEGMENTATOR
# ─────────────────────────────────────────────────────────────

def run_totalsegmentator(input_nifti: str, output_dir: str, task: str = "total") -> None:
    """
    Run TotalSegmentator via Python API with Windows-safe settings.

    Key Windows fix: set nnUNet_n_proc_DA=0 and use num_processes_segmentation_export=0
    to avoid the multiprocessing MemoryError that occurs when pickling large
    numpy arrays across worker processes on Windows.

    task options (TS 2.x):
        'total'           – 117 structures (fast model)
        'total_mr'        – same for MRI input
        'lung_vessels'    – pulmonary vessels
        'lung'            – lung lobes
        'body'            – body + tissue
        'cardiac_motion'  – 4-D cardiac (CT)
    """
    import subprocess
    import shutil

    os.makedirs(output_dir, exist_ok=True)

    # ── Critical Windows fix: disable nnUNet worker processes ──────────────
    # nnUNet spawns multiprocessing workers to export segmentations.
    # On Windows this causes MemoryError when pickling large arrays.
    # Setting these env vars to 0/1 forces single-process (safe) mode.
    os.environ["nnUNet_n_proc_DA"]  = "0"   # data augmentation workers
    os.environ["OMP_NUM_THREADS"]   = "1"   # OpenMP thread safety

    print(f"[INFO] Running TotalSegmentator")
    print(f"       input : {input_nifti}")
    print(f"       output: {output_dir}")
    print(f"       task  : {task}  |  fast=True  |  workers=0 (Windows-safe)")

    # ── Try Python API first ────────────────────────────────────────────────
    try:
        from totalsegmentator.python_api import totalsegmentator
        import nibabel as nib
        import inspect

        input_img = nib.load(input_nifti)

        # Build kwargs - only pass num_processes if the API accepts it
        # (older TS versions don't have this param)
        sig = inspect.signature(totalsegmentator)
        kwargs = dict(task=task, fast=True, verbose=True)
        if "num_processes" in sig.parameters:
            kwargs["num_processes"] = 0          # single-process export
        if "num_processes_export" in sig.parameters:
            kwargs["num_processes_export"] = 0

        totalsegmentator(input_img, output_dir, **kwargs)
        print(f"[INFO] Segmentation masks saved in: {output_dir}")
        return

    except ImportError:
        print("[WARN] Python API import failed – trying CLI.")
    except Exception as e:
        print(f"[WARN] Python API error: {e}")
        print("[WARN] Falling back to CLI with --nr_thr_saving 0")

    # ── CLI fallback with explicit single-thread saving ─────────────────────
    ts_exe = shutil.which("TotalSegmentator") or shutil.which("totalsegmentator")

    if ts_exe:
        cmd = [ts_exe, "-i", input_nifti, "-o", output_dir,
               "--task", task, "--fast",
               "--nr_thr_saving", "1",    # single thread for segment saving
               "--nr_thr_resamp", "1"]    # single thread for resampling
    else:
        # Last resort: inline python
        cmd = [
            sys.executable, "-c",
            "import os; os.environ['nnUNet_n_proc_DA']= '0'; "
            "from totalsegmentator.python_api import totalsegmentator; "
            "import nibabel as nib; "
            f"totalsegmentator(nib.load(r'{input_nifti}'), r'{output_dir}', "
            f"task='{task}', fast=True, verbose=True)"
        ]

    print(f"[INFO] CLI command: {chr(10)}  {chr(32).join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "TotalSegmentator CLI failed.\n"
            "Manual fix — run this in your venv:\n"
            f"  set nnUNet_n_proc_DA=0\n"
            f"  TotalSegmentator -i \"{input_nifti}\" "
            f"-o \"{output_dir}\" --task {task} --fast --nr_thr_saving 1"
        )
    print(f"[INFO] Segmentation masks saved in: {output_dir}")


# ─────────────────────────────────────────────────────────────
# 3.  VISUALISE RESULTS
# ─────────────────────────────────────────────────────────────

LABEL_COLORS = {
    # --- abdominal (task=total) ---
    "liver":                      (0.8, 0.2, 0.2, 0.6),
    "spleen":                     (0.2, 0.6, 0.8, 0.6),
    "left_kidney":                (0.2, 0.8, 0.3, 0.6),
    "right_kidney":               (0.1, 0.6, 0.2, 0.6),
    "aorta":                      (0.9, 0.7, 0.1, 0.7),
    "heart":                      (0.9, 0.2, 0.5, 0.6),
    # --- lung lobes (task=lung) ---
    "lung_upper_lobe_left":       (0.3, 0.7, 1.0, 0.6),
    "lung_lower_lobe_left":       (0.0, 0.4, 0.9, 0.6),
    "lung_upper_lobe_right":      (1.0, 0.6, 0.2, 0.6),
    "lung_middle_lobe_right":     (1.0, 0.8, 0.1, 0.6),
    "lung_lower_lobe_right":      (0.9, 0.3, 0.0, 0.6),
    # --- lung vessels (task=lung_vessels) ---
    "pulmonary_vein":             (0.6, 0.1, 0.9, 0.6),
    "pulmonary_artery":           (0.9, 0.1, 0.5, 0.6),
    # --- trachea / airway ---
    "trachea":                    (0.8, 0.9, 0.2, 0.7),
}


def overlay_segmentations(ct_nifti: str, seg_dir: str,
                           slice_idx: int = None,
                           out_png: str = "totalseg_overlay.png",
                           task: str = "total") -> None:
    """
    Overlay available segmentation masks on the mid-axial CT slice.
    """
    ct_img  = nib.load(ct_nifti)
    ct_data = ct_img.get_fdata()

    # ── Auto mid-slice if not specified ──────────────────────
    if slice_idx is None:
        slice_idx = ct_data.shape[2] // 2

    ct_slice = ct_data[:, :, slice_idx].T

    # Window / Level presets
    if "lung" in task:
        wl, ww = -600, 1500   # Lung window
    else:
        wl, ww = 40, 400      # Abdomen window
    vmin, vmax = wl - ww / 2, wl + ww / 2
    ct_slice = np.clip(ct_slice, vmin, vmax)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor="black")
    ax.imshow(ct_slice, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(f"TotalSegmentator overlay  –  slice {slice_idx}",
                 color="white", fontsize=14, pad=10)
    ax.axis("off")

    # ── Overlay each found mask ───────────────────────────────
    legend_patches = []
    for organ, rgba in LABEL_COLORS.items():
        mask_path = Path(seg_dir) / f"{organ}.nii.gz"
        if not mask_path.exists():
            mask_path = Path(seg_dir) / f"{organ}.nii"
        if not mask_path.exists():
            continue

        seg_img  = nib.load(str(mask_path))
        seg_data = seg_img.get_fdata()
        seg_slice = seg_data[:, :, slice_idx].T

        rgba_img = np.zeros((*seg_slice.shape, 4))
        mask = seg_slice > 0
        rgba_img[mask] = rgba
        ax.imshow(rgba_img, origin="lower", alpha=0.7)

        from matplotlib.patches import Patch
        legend_patches.append(Patch(facecolor=rgba[:3], label=organ.replace("_", " ").title()))

    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right",
                  fontsize=9, framealpha=0.6, facecolor="#111")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Overlay saved → {out_png}")


def print_volume_stats(seg_dir: str, voxel_spacing_mm=(1.5, 1.5, 1.5)) -> None:
    """Estimate organ volumes (cm³) from binary masks."""
    voxel_vol_ml = (voxel_spacing_mm[0] *
                    voxel_spacing_mm[1] *
                    voxel_spacing_mm[2]) / 1000.0   # mm³ → cm³

    print("\n" + "─" * 40)
    print(f"{'Organ':<25} {'Volume (cm³)':>12}")
    print("─" * 40)
    for nii_path in sorted(Path(seg_dir).glob("*.nii*")):
        organ = nii_path.name.replace(".nii.gz","").replace(".nii","")
        seg   = nib.load(str(nii_path)).get_fdata()
        vol   = seg.sum() * voxel_vol_ml
        if vol > 0:
            print(f"  {organ:<23} {vol:>10.1f}")
    print("─" * 40 + "\n")


# ─────────────────────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TotalSegmentator DICOM Demo")
    parser.add_argument("--dicom_dir", required=True,
                        help="Path to folder containing DICOM files")
    parser.add_argument("--task", default="total",
                        help="TotalSegmentator task (default: total)")
    parser.add_argument("--slice", type=int, default=None,
                        help="Axial slice index to visualise (default: mid)")
    parser.add_argument("--out_dir", default="totalseg_output",
                        help="Output directory for masks + PNG")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nifti_path = os.path.join(args.out_dir, "input_volume.nii.gz")
    seg_dir    = os.path.join(args.out_dir, "segmentations")
    overlay_png= os.path.join(args.out_dir, "totalseg_overlay.png")

    # Step 1 – Load DICOM
    sitk_img = load_dicom_series(args.dicom_dir)

    # Step 2 – Export NIfTI
    sitk_to_nifti(sitk_img, nifti_path)

    # Step 3 – Segment
    run_totalsegmentator(nifti_path, seg_dir, task=args.task)

    # Step 4 – Visualise
    overlay_segmentations(nifti_path, seg_dir,
                          slice_idx=args.slice,
                          out_png=overlay_png,
                          task=args.task)

    # Step 5 – Volume stats
    print_volume_stats(seg_dir)

    print("[✅] TotalSegmentator demo complete!")
    print(f"     Overlay: {overlay_png}")
    print(f"     Masks  : {seg_dir}/")


# ─────────────────────────────────────────────────────────────
# WINDOWS MULTIPROCESSING GUARD
# On Windows, multiprocessing requires the entry point to be
# protected by  if __name__ == "__main__"  to prevent recursive
# subprocess spawning (which causes MemoryError on large volumes).
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # freeze_support() is required when building a Windows .exe,
    # harmless otherwise – call it before anything that may fork.
    from multiprocessing import freeze_support
    freeze_support()
    main()
