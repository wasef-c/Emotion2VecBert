#!/bin/bash
# Simple batch merge script

set -e

echo "=========================================="
echo "🚀 SIMPLE MERGE - ALL DATASETS"
echo "=========================================="
echo ""

# MSPP only
echo "Merging MSPP..."
python simple_merge.py \
  --dataset MSPP \
  --output cairocode/MSPP_Audio_Text_Merged \
  --split train

echo ""
echo "✅ Done!"
