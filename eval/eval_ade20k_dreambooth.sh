#!/bin/bash

# DreamBooth Evaluation Script for ADE20K - Using Local Checkpoint
# Comparing against ControlNet segmentation performance

# Set path to your local checkpoint
export DREAMBOOTH_DIR="checkpoints/canny/dreambooth"  # Path to your DreamBooth checkpoint directory

# How many GPUs and processes you want to use for evaluation
export NUM_GPUS=1

# Guidance scale and inference steps (same as your ControlNet setup)
export SCALE=7.5
export NUM_STEPS=20

echo "=== DreamBooth ADE20K Evaluation with Local Checkpoint ==="
echo "Model: $DREAMBOOTH_DIR"
echo "Dataset: limingcv/Captioned_ADE20K" 
echo "GPUs: $NUM_GPUS"
echo "Scale: $SCALE, Steps: $NUM_STEPS"
echo "Task: Semantic Segmentation Quality Assessment"
echo "========================================================="

# Verify checkpoint exists
if [ ! -d "$DREAMBOOTH_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found at $DREAMBOOTH_DIR"
    echo "Make sure you have downloaded a DreamBooth checkpoint to this location"
    exit 1
fi

if [ ! -f "$DREAMBOOTH_DIR/model_index.json" ]; then
    echo "❌ Error: model_index.json not found. Make sure this is a valid diffusers checkpoint."
    exit 1
fi

echo "✅ Checkpoint verified: $DREAMBOOTH_DIR"

# Generate images for evaluation
# Using the same ADE20K dataset as your ControlNet but treating it as pure text-to-image
echo ""
echo "=== Starting Image Generation ==="
accelerate launch --main_process_port=23456 --num_processes=$NUM_GPUS eval/eval_plus.py \
    --task_name='text2img' \
    --dataset_name='limingcv/Captioned_ADE20K' \
    --dataset_split='validation' \
    --prompt_column='prompt' \
    --image_column='image' \
    --model_path=$DREAMBOOTH_DIR \
    --model='dreambooth' \
    --guidance_scale=$SCALE \
    --num_inference_steps=$NUM_STEPS \
    --condition_column='' \
    --resolution=512 \
    --batch_size=4

# Path to the generated images (following your existing pattern)
export DATA_DIR="work_dirs/eval_dirs/Captioned_ADE20K/validation/dreambooth_${SCALE}-${NUM_STEPS}"

echo ""
echo "=== Generation Complete ==="
echo "Generated images saved to: $DATA_DIR"
echo ""
echo "Directory structure:"
echo "- $DATA_DIR/images/group_*/: Individual generated images" 
echo "- $DATA_DIR/visualization/: Side-by-side comparison grids"
echo "- $DATA_DIR/annotations/: Ground truth segmentation annotations"

# Show generation statistics
if [ -d "$DATA_DIR" ]; then
    echo ""
    echo "=== Generation Statistics ==="
    TOTAL_VIZ=$(find $DATA_DIR/visualization -name "*.png" 2>/dev/null | wc -l)
    TOTAL_IMGS=$(find $DATA_DIR/images -name "*.png" 2>/dev/null | wc -l)
    echo "Total samples processed: $TOTAL_VIZ"
    echo "Total images generated: $TOTAL_IMGS"
    
    # Show first few generated images for quick check
    echo ""
    echo "Sample visualization files:"
    ls $DATA_DIR/visualization/*.png 2>/dev/null | head -3
fi

echo ""
echo "=== Running Semantic Segmentation Evaluation ==="
echo "Evaluating segmentation quality of DreamBooth generated images using mmseg..."
echo "This compares: DreamBooth generated images → Segment → Compare with ground truth segmentation"

# Evaluation with mmseg api (same as your ControlNet evaluation)
mim test mmseg mmlab/mmseg/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py \
    --checkpoint https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth \
    --gpus 1 \
    --launcher pytorch \
    --cfg-options test_dataloader.dataset.datasets.0.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.1.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.2.data_root="${DATA_DIR}" \
                  test_dataloader.dataset.datasets.3.data_root="${DATA_DIR}" \
                  work_dir="${DATA_DIR}"

echo ""
echo "=== Evaluation Complete ==="
echo "Segmentation evaluation results saved in: ${DATA_DIR}"

echo ""
echo "=== Results Comparison ==="
echo "ControlNet results:"
echo "  - checkpoints/ade20k_reward-model-UperNet-R50/checkpoint-5000/controlnet: work_dirs/eval_dirs/Captioned_ADE20K/validation/checkpoints_ade20k_reward-model-UperNet-R50_checkpoint-5000_controlnet_${SCALE}-${NUM_STEPS}"
echo ""
echo "DreamBooth results:"
echo "  - $DATA_DIR"
echo ""
echo "Both evaluations use the same mmseg evaluation pipeline for fair comparison:"
echo "- ControlNet: Text + Segmentation Mask → Image → Segment → Compare with GT"
echo "- DreamBooth: Text → Image → Segment → Compare with GT"
echo ""
echo "Key metrics to compare:"
echo "- mIoU (mean Intersection over Union)"
echo "- mAcc (mean Accuracy)" 
echo "- aAcc (overall Accuracy)"
echo ""
echo "Check the evaluation logs above for detailed metrics!"

# Optional: Summary of what this evaluation tells us
echo ""
echo "=== What This Evaluation Measures ==="
echo "1. ControlNet: How well can guided generation with segmentation masks produce semantically accurate images?"
echo "2. DreamBooth: How well can free text-to-image generation naturally produce images with good semantic structure?"
echo ""
echo "This comparison helps understand:"
echo "- The value of segmentation guidance vs. free generation"
echo "- Whether DreamBooth can implicitly learn semantic structure from text alone"
echo "- Quantitative quality differences between controlled and uncontrolled generation