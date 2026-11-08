#!/bin/bash
# ============================================================
# TotalSegmentator Virtual Environment Setup
# Python 3.10 recommended
# ============================================================

python3.10 -m venv venv_totalseg
source venv_totalseg/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "✅ TotalSegmentator environment ready!"
echo "👉 Activate with: source venv_totalseg/bin/activate"
