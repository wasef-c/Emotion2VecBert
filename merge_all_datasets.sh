#!/bin/bash
# Batch script to merge all feature datasets with their audio counterparts
# Run this once to create merged datasets on HuggingFace Hub

set -e  # Exit on error

echo "=========================================="
echo "🚀 MERGING ALL DATASETS"
echo "=========================================="
echo ""
echo "This will merge 5 datasets and upload to Hub:"
echo "  1. IEMO"
echo "  2. MSPI"
echo "  3. MSPP (large - may take a while)"
echo "  4. CMUMOSEI"
echo "  5. SAMSEMO"
echo ""
echo "Make sure you're logged into HuggingFace CLI:"
echo "  huggingface-cli login"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Function to clean up memory between runs
cleanup_memory() {
  echo ""
  echo "🧹 Cleaning up memory..."
  sleep 5  # Give system time to free memory
  # Clear Python cache
  find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  sync  # Flush file system buffers
  echo "💾 Memory cleanup complete"
}

# # 1. IEMO
# echo ""
# echo "=========================================="
# echo "1/5: Merging IEMO..."
# echo "=========================================="
# python merge_and_upload_dataset.py \
#   --dataset IEMO \
#   --feature-dataset cairocode/IEMO_Emotion2Vec_Text \
#   --output-name cairocode/IEMO_Audio_Text_Merged \
#   --splits train \
#   --batch-size 100 \
#   --memory-threshold 80

# # 2. MSPI
# echo ""
# echo "=========================================="
# echo "2/5: Merging MSPI..."
# echo "=========================================="
# python merge_and_upload_dataset.py \
#   --dataset MSPI \
#   --feature-dataset cairocode/MSPI_Emotion2Vec_Text \
#   --output-name cairocode/MSPI_Audio_Text_Merged \
#   --splits train \
#   --batch-size 100 \
#   --memory-threshold 80

#   --feature-dataset cairocode/MSPP_Emotion2Vec_Text ^
#   --wav-dataset cairocode/MSPP_WAV ^
#   --output-name cairocode/MSPP_Audio_Text_Merged ^
#   --splits train ^
#   --batch-size 50 ^
#   --memory-threshold 75
# if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

# REM 4. CMUMOSEI
# echo.
# echo ==========================================
# echo 4/5: Merging CMUMOSEI...
# echo ==========================================
# python merge_and_upload_dataset.py ^
#   --feature-dataset cairocode/CMU_MOSEI_EMOTION2VEC_4class_2 ^
#   --wav-dataset cairocode/CMUMOSEI_WAV ^
#   --output-name cairocode/CMUMOSEI_Audio_Text_Merged ^
#   --splits train ^
#   --batch-size 100 ^
#   --memory-threshold 80
# if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

# REM 5. SAMSEMO
# echo.
# echo ==========================================
# echo 5/5: Merging SAMSEMO...
# echo ==========================================
# python merge_and_upload_dataset.py ^
#   --feature-dataset cairocode/samsemo_emotion2vec_4_V2 ^
#   --wav-dataset cairocode/SAMSEMO_WAV ^
#   --output-name cairocode/SAMSEMO_Audio_Text_Merged ^
#   --splits train ^
#   --batch-size 100 ^
#   --memory-threshold 80
# if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%


# 3. MSPP (large dataset - smaller batch size)
echo ""
echo "=========================================="
echo "3/5: Merging MSPP (LARGE - this may take a while)..."
echo "=========================================="
python merge_and_upload_dataset.py \
  --feature-dataset cairocode/MSPP_Emotion2Vec_Text \
  --wav-dataset cairocode/MSPP_WAV \
  --output-name cairocode/MSPP_Audio_Text_Merged \
  --splits train \

cleanup_memory


# 4. CMUMOSEI
echo ""
echo "=========================================="
echo "4/5: Merging CMUMOSEI..."
echo "=========================================="
python merge_and_upload_dataset.py \
  --dataset CMUMOSEI \
  --feature-dataset cairocode/CMU_MOSEI_EMOTION2VEC_4class_2 \
  --output-name cairocode/CMUMOSEI_Audio_Text_Merged \
  --splits train \
  --batch-size 50 \
  --memory-threshold 75

cleanup_memory

# 5. SAMSEMO
echo ""
echo "=========================================="
echo "5/5: Merging SAMSEMO..."
echo "=========================================="
python merge_and_upload_dataset.py \
  --dataset SAMSEMO \
  --feature-dataset cairocode/samsemo_emotion2vec_4_V2 \
  --output-name cairocode/SAMSEMO_Audio_Text_Merged \
  --splits train \
  --batch-size 50 \
  --memory-threshold 75

cleanup_memory

echo ""
echo "=========================================="
echo "✅ ALL DATASETS MERGED AND UPLOADED!"
echo "=========================================="
echo ""
echo "Merged datasets available at:"
echo "  - https://huggingface.co/datasets/cairocode/IEMO_Audio_Text_Merged"
echo "  - https://huggingface.co/datasets/cairocode/MSPI_Audio_Text_Merged"
echo "  - https://huggingface.co/datasets/cairocode/MSPP_Audio_Text_Merged"
echo "  - https://huggingface.co/datasets/cairocode/CMUMOSEI_Audio_Text_Merged"
echo "  - https://huggingface.co/datasets/cairocode/SAMSEMO_Audio_Text_Merged"
echo ""
echo "You can now update main.py to use these merged datasets!"
