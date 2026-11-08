"""
=============================================================
MedSAM – DICOM Series Segmentation Demo
=============================================================
Compatible:  Python 3.10  |  segment-anything 1.0  |  PyTorch 2.3
Model:       MedSAM (SAM ViT-B fine-tuned on medical imaging)

Checkpoint download:
  Option A – HuggingFace:
    huggingface-cli download bowang-lab/MedSAM medsam_vit_b.pth
  Option B – Google Drive:
    https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN

Usage:
    # Automatic whole-slice segmentation
    python medsam_demo.py --dicom_dir /path/to/dicom --checkpoint medsam_vit_b.pth

    # Bounding-box prompted segmentation
    python medsam_demo.py --dicom_dir /path/to/dicom --checkpoint medsam_vit_b.pth \
                          --bbox 100 80 350 300
=============================================================
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import SimpleITK as sitk

# ─────────────────────────────────────────────────────────────
# 1.  DICOM LOADER
# ─────────────────────────────────────────────────────────────

def load_dicom_series(dicom_dir: str):
    reader = sitk.ImageSeriesReader()
    ids    = reader.GetGDCMSeriesIDs(dicom_dir)
    if not ids:
        raise FileNotFoundError(f"No DICOM series in: {dicom_dir}")
    files  = reader.GetGDCMSeriesFileNames(dicom_dir, ids[0])
    reader.SetFileNames(files)
    image  = reader.Execute()
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    print(f"[INFO] Loaded volume shape (Z,Y,X): {volume.shape}")
    return volume, image.GetSpacing()


def preprocess_slice(arr2d: np.ndarray,
                     wl: float = 40, ww: float = 400,
                     target_size: int = 1024) -> np.ndarray:
    """
    Window/level → 0-255 uint8 → resize to 1024×1024 → RGB.
    MedSAM expects 1024×1024 RGB input.
    """
    lo   = wl - ww / 2
    hi   = wl + ww / 2
    arr  = np.clip(arr2d, lo, hi)
    arr  = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    img  = Image.fromarray(arr).convert("RGB")
    img  = img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(img)


# ─────────────────────────────────────────────────────────────
# 2.  MEDSAM MODEL LOADER
# ─────────────────────────────────────────────────────────────

def load_medsam(checkpoint_path: str, device: str = "cpu"):
    """
    Load MedSAM model (SAM ViT-B backbone + MedSAM weights).
    """
    import torch
    from segment_anything import sam_model_registry

    if not Path(checkpoint_path).exists():
        _try_download_checkpoint(checkpoint_path)

    print(f"[INFO] Loading MedSAM checkpoint: {checkpoint_path}")
    

    sam = sam_model_registry["vit_b"](checkpoint=None)

    # 2. Load checkpoint safely on CPU
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 3. Load weights into model
    sam.load_state_dict(state_dict)
    sam.to(device)
    sam.eval()
    print(f"[INFO] MedSAM loaded on: {device} ✅")
    return sam


def _try_download_checkpoint(out_path: str):
    """Attempt to download MedSAM checkpoint from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
        print("[INFO] Checkpoint not found locally. Downloading from HuggingFace …")
        local = hf_hub_download(
            repo_id="bowang-lab/MedSAM",
            filename="medsam_vit_b.pth",
            local_dir=str(Path(out_path).parent),
        )
        Path(local).rename(out_path)
        print(f"[INFO] Checkpoint saved → {out_path}")
    except Exception as e:
        sys.exit(
            f"[ERROR] Cannot download checkpoint: {e}\n"
            f"        Please download manually from:\n"
            f"          https://huggingface.co/bowang-lab/MedSAM\n"
            f"        and pass --checkpoint <path>"
        )


# ─────────────────────────────────────────────────────────────
# 3.  MEDSAM INFERENCE HELPER
# ─────────────────────────────────────────────────────────────

def medsam_inference(model, img_rgb_1024: np.ndarray,
                     box_1024: np.ndarray,
                     device: str = "cpu") -> np.ndarray:
    """
    Run MedSAM forward pass for a single bounding box prompt.

    Args:
        img_rgb_1024 : (1024, 1024, 3) uint8
        box_1024     : (4,) float  [x1, y1, x2, y2] in 1024-px space
        device       : torch device string

    Returns:
        mask         : (H, W) bool
    """
    import torch
    from segment_anything.utils.transforms import ResizeLongestSide

    transform    = ResizeLongestSide(1024)

    # ── Image embedding ──────────────────────────────────────
    img_tensor = torch.as_tensor(
        img_rgb_1024.transpose(2, 0, 1),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        image_embedding = model.image_encoder(img_tensor)

    # ── Box prompt ───────────────────────────────────────────
    box_torch = torch.as_tensor(
        box_1024[None, None, :],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None, boxes=box_torch, masks=None)

        low_res_masks, iou_pred = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )

    # ── Post-process → binary mask ───────────────────────────
    from segment_anything.utils.amg import remove_small_regions
    from torch.nn.functional import interpolate

    upscaled = interpolate(
        low_res_masks, size=(1024, 1024),
        mode="bilinear", align_corners=False)
    mask = (upscaled[0, 0] > model.mask_threshold).cpu().numpy()
    return mask


def whole_slice_boxes(h: int, w: int) -> list:
    """Return a list of organ-region bounding boxes (heuristic)."""
    # Simple fixed regions for demonstration; replace with detector output
    regions = {
        "liver"  : [int(w*0.35), int(h*0.45), int(w*0.85), int(h*0.85)],
        "spleen" : [int(w*0.05), int(h*0.45), int(w*0.35), int(h*0.75)],
        "whole"  : [10, 10, w-10, h-10],
    }
    return regions


# ─────────────────────────────────────────────────────────────
# 4.  BATCH 3-D SEGMENTATION
# ─────────────────────────────────────────────────────────────

def segment_volume(model,
                   volume: np.ndarray,
                   box_orig: np.ndarray,
                   orig_size: tuple,
                   device: str = "cpu",
                   stride: int = 5) -> np.ndarray:
    """
    Propagate a single 2-D bounding box prompt through the volume
    (one slice every `stride` frames) to build a 3-D mask.

    Returns binary mask of shape (Z, H, W) at original resolution.
    """
    Z, H, W = volume.shape
    mask_3d = np.zeros((Z, H, W), dtype=bool)

    # Scale box from original space to 1024-px space
    scale_x = 1024 / orig_size[1]
    scale_y = 1024 / orig_size[0]
    box_1024 = np.array([
        box_orig[0] * scale_x,
        box_orig[1] * scale_y,
        box_orig[2] * scale_x,
        box_orig[3] * scale_y,
    ])

    import cv2
    for z in range(0, Z, stride):
        img_1024 = preprocess_slice(volume[z])
        mask_1024 = medsam_inference(model, img_1024, box_1024, device)
        # Resize mask back to original slice size
        mask_orig = cv2.resize(
            mask_1024.astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_3d[z] = mask_orig
        if z % 20 == 0:
            print(f"  Slices processed: {z}/{Z}", end="\r")
    print()
    return mask_3d


# ─────────────────────────────────────────────────────────────
# 5.  VISUALISATION
# ─────────────────────────────────────────────────────────────

COLORS = {
    "liver":  [255, 80,  80],
    "spleen": [80,  160, 255],
    "whole":  [80,  255, 160],
}


def visualise_masks(volume: np.ndarray,
                    masks: dict,
                    slice_idx: int,
                    box: np.ndarray,
                    out_png: str = "medsam_result.png") -> None:
    """
    Overlay all segmentation masks on the chosen axial slice.
    """
    raw   = volume[slice_idx]
    lo, hi = raw.min(), raw.max()
    bg    = ((np.clip(raw, lo, hi) - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)

    # Resize to 512 for display
    disp = cv2.resize(bg_rgb, (512, 512))
    H, W = disp.shape[:2]

    fig, axes = plt.subplots(1, 1 + len(masks), figsize=(5*(1+len(masks)), 5),
                              facecolor="black")
    axes = np.atleast_1d(axes)

    # CT + box
    axes[0].imshow(disp)
    axes[0].set_title(f"CT  slice {slice_idx}", color="white")
    x1, y1, x2, y2 = box
    sx, sy = W / volume.shape[2], H / volume.shape[1]
    rect = Rectangle((x1*sx, y1*sy), (x2-x1)*sx, (y2-y1)*sy,
                     linewidth=2, edgecolor="yellow", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].axis("off")

    # Mask overlays
    for i, (label, mask_3d) in enumerate(masks.items()):
        mask_sl = mask_3d[slice_idx]
        mask_res= cv2.resize(mask_sl.astype(np.uint8), (W, H),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = disp.copy().astype(float)
        color   = COLORS.get(label, [200,200,200])
        overlay[mask_res] = 0.5 * overlay[mask_res] + 0.5 * np.array(color)
        overlay = overlay.clip(0,255).astype(np.uint8)
        axes[i+1].imshow(overlay)
        axes[i+1].set_title(f"MedSAM – {label}", color="white")
        axes[i+1].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Result saved → {out_png}")


def save_volume_stats(masks: dict, spacing: tuple) -> None:
    voxvol = spacing[0] * spacing[1] * spacing[2] / 1000.0  # cm³
    print(f"\n{'─'*42}")
    print(f"{'Region':<20} {'Voxels':>10} {'Volume cm³':>12}")
    print("─" * 42)
    for label, mask3d in masks.items():
        n = int(mask3d.sum())
        v = n * voxvol
        print(f"  {label:<18} {n:>10,} {v:>11.1f}")
    print("─" * 42 + "\n")


# ─────────────────────────────────────────────────────────────
# 6.  INTERACTIVE BBOX DRAWER (optional)
# ─────────────────────────────────────────────────────────────

def interactive_bbox(volume: np.ndarray, slice_idx: int):
    """
    Open an OpenCV window and let the user draw a bounding box
    with the mouse. Returns (x1, y1, x2, y2) in original resolution.
    """
    raw = volume[slice_idx]
    lo, hi = raw.min(), raw.max()
    bg  = ((np.clip(raw, lo, hi) - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    bg  = cv2.resize(bg, (512, 512))
    clone = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    roi = cv2.selectROI("Draw bbox (Enter=confirm, c=cancel)",
                         clone, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    if roi == (0, 0, 0, 0):
        return None
    x, y, w, h = roi
    # Scale back to original resolution
    oh, ow = volume.shape[1], volume.shape[2]
    sx, sy = ow / 512, oh / 512
    return [int(x*sx), int(y*sy), int((x+w)*sx), int((y+h)*sy)]


# ─────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MedSAM DICOM Segmentation Demo")
    parser.add_argument("--dicom_dir",   required=True,
                        help="DICOM series directory")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to medsam_vit_b.pth checkpoint")
    parser.add_argument("--slice",       type=int, default=None,
                        help="Axial slice index to visualise (default: mid)")
    parser.add_argument("--bbox",        type=int, nargs=4, default=None,
                        metavar=("X1","Y1","X2","Y2"),
                        help="Bounding box in original pixel coords")
    parser.add_argument("--interactive", action="store_true",
                        help="Draw bounding box interactively via OpenCV")
    parser.add_argument("--stride",      type=int, default=5,
                        help="Slice stride for 3-D propagation (default: 5)")
    parser.add_argument("--device",      default="cpu",
                        choices=["cpu","cuda","mps"],
                        help="Torch device (default: cpu)")
    parser.add_argument("--out_dir",     default="medsam_output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load DICOM ─────────────────────────────────────────
    volume, spacing = load_dicom_series(args.dicom_dir)
    Z, H, W = volume.shape
    mid_slice = args.slice if args.slice is not None else Z // 2

    # ── Load model ─────────────────────────────────────────
    model = load_medsam(args.checkpoint, device=args.device)

    # ── Bounding box ───────────────────────────────────────
    if args.interactive:
        box = interactive_bbox(volume, mid_slice)
        if box is None:
            print("[WARN] No box drawn – using whole-slice box.")
            box = [10, 10, W-10, H-10]
    elif args.bbox:
        box = args.bbox
    else:
        # Default: entire slice
        box = [10, 10, W-10, H-10]

    print(f"[INFO] Bounding box (orig): {box}")

    # ── Run MedSAM on one slice (quick demo) ───────────────
    print("[INFO] Segmenting representative slices …")
    img_1024  = preprocess_slice(volume[mid_slice])
    scale_x   = 1024 / W
    scale_y   = 1024 / H
    box_1024  = np.array([box[0]*scale_x, box[1]*scale_y,
                           box[2]*scale_x, box[3]*scale_y])
    mask_2d   = medsam_inference(model, img_1024, box_1024, args.device)

    # Expand to 3-D (propagate through volume)
    print("[INFO] Propagating mask through volume …")
    mask_3d = segment_volume(
        model, volume, np.array(box), (H, W),
        device=args.device, stride=args.stride)

    masks = {"selected_region": mask_3d}

    # ── Visualise ──────────────────────────────────────────
    out_png = os.path.join(args.out_dir, "medsam_result.png")
    visualise_masks(volume, masks, mid_slice, np.array(box), out_png)
    save_volume_stats(masks, spacing)

    print(f"[✅] MedSAM demo complete!  Output in: {args.out_dir}/")


if __name__ == "__main__":
    main()
