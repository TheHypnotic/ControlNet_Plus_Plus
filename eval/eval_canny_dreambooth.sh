#!/bin/bash

# DreamBooth Evaluation Script - Using Local Checkpoint
# Your downloaded checkpoint structure looks perfect!

# Set path to your local checkpoint
export DREAMBOOTH_DIR="checkpoints/canny/dreambooth"  # Path to your downloaded checkpoint directory

export NUM_GPUS=1

# Guidance scale and inference steps (same as your ControlNet setup)
export SCALE=7.5
export NUM_STEPS=20

echo "=== DreamBooth Evaluation with Local Checkpoint ==="
echo "Model: $DREAMBOOTH_DIR"
echo "Dataset: limingcv/MultiGen-20M_canny_eval" 
echo "GPUs: $NUM_GPUS"
echo "Scale: $SCALE, Steps: $NUM_STEPS"
echo "=================================================="

# Verify checkpoint exists
if [ ! -d "$DREAMBOOTH_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found at $DREAMBOOTH_DIR"
    exit 1
fi

if [ ! -f "$DREAMBOOTH_DIR/model_index.json" ]; then
    echo "❌ Error: model_index.json not found. Make sure this is a valid diffusers checkpoint."
    exit 1
fi

echo "✅ Checkpoint verified: $DREAMBOOTH_DIR"

# Generate images for evaluation
# Using the same dataset as your ControlNet but treating it as pure text-to-image
accelerate launch --main_process_port=21156 --num_processes=$NUM_GPUS eval/eval_plus.py \
    --task_name='text2img' \
    --dataset_name='limingcv/MultiGen-20M_canny_eval' \
    --dataset_split='validation' \
    --prompt_column='text' \
    --image_column='image' \
    --model_path=$DREAMBOOTH_DIR \
    --model='dreambooth' \
    --guidance_scale=$SCALE \
    --num_inference_steps=$NUM_STEPS \
    --condition_column='' \
    --resolution=512 \
    --batch_size=4

# Path to the generated images (following your existing pattern)
export DATA_DIR="work_dirs/eval_dirs/MultiGen-20M_canny_eval/validation/dreambooth_${SCALE}-${NUM_STEPS}"

echo ""
echo "=== Generation Complete ==="
echo "Generated images saved to: $DATA_DIR"
echo ""
echo "Directory structure:"
echo "- $DATA_DIR/images/group_*/: Individual generated images" 
echo "- $DATA_DIR/visualization/: Side-by-side comparison grids"
echo "- $DATA_DIR/annotations/: Ground truth annotations (if any)"

# Optional: Show some statistics
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
echo "=== Running Evaluation Metrics ==="

# Option 1: If you want to evaluate canny edge quality of DreamBooth generated images
# (This compares: DreamBooth generated images → extract canny → compare with ground truth canny)
echo "Running canny edge evaluation on DreamBooth generated images..."
python3 eval/eval_edge.py --task canny --root_dir ${DATA_DIR} --num_processes ${NUM_GPUS}

# Option 2: For more comprehensive text-to-image evaluation, you might want:
# echo "For additional metrics, consider:"
# echo "1. FID score: python3 eval/eval_fid.py --generated_dir ${DATA_DIR}/images --real_dir /path/to/real/images"
# echo "2. CLIP score: python3 eval/eval_clip.py --generated_dir ${DATA_DIR}/images --prompts_file /path/to/prompts.txt"

echo ""
echo "=== Results Comparison ==="
echo "ControlNet results: work_dirs/eval_dirs/MultiGen-20M_canny_eval/validation/checkpoints_canny_controlnet_${SCALE}-${NUM_STEPS}"
echo "DreamBooth results: $DATA_DIR"
echo ""
echo "Both evaluations use the same canny edge evaluation script for fair comparison:"
echo "- ControlNet: Text + Canny → Image → Extract Canny → Compare with GT"
echo "- DreamBooth: Text → Image → Extract Canny → Compare with GT"