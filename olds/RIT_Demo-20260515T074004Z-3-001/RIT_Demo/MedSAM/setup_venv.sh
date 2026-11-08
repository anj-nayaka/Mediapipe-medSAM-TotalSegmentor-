#!/bin/bash
# ============================================================
# MedSAM Segmentation – Virtual Environment Setup
# Python 3.10 recommended
# ============================================================

python3.10 -m venv venv_medsam
source venv_medsam/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ MedSAM environment ready!"
echo ""
echo "📥 Download MedSAM checkpoint:"
echo "   huggingface-cli download bowang-lab/MedSAM medsam_vit_b.pth --local-dir ."
echo ""
echo "👉 Run: python medsam_demo.py --dicom_dir /path/to/dicom --checkpoint medsam_vit_b.pth"
