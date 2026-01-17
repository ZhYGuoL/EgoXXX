#!/bin/bash

set -e

VIDEO_DIR="./preprocessed_videos/my_video"
VIDEO_PATH="$VIDEO_DIR/exo.mp4"

echo "=========================================="
echo "EgoX Inference Preparation"
echo "=========================================="
echo ""

# Check if video exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Error: Video not found at $VIDEO_PATH"
    echo "   Please use the video preprocessor to prepare your video first:"
    echo "   python video_preprocessor.py"
    exit 1
fi

echo "✓ Video found: $VIDEO_PATH"
echo ""

# Create depth_maps directory
DEPTH_DIR="$VIDEO_DIR/depth_maps/my_video"
mkdir -p "$DEPTH_DIR"
echo "✓ Created depth_maps directory: $DEPTH_DIR"
echo ""

# Generate depth maps
echo "Generating depth maps..."
python generate_depth_maps.py \
    --video_path "$VIDEO_PATH" \
    --output_dir "$DEPTH_DIR"
echo ""

# Create meta.json
echo "Creating meta.json..."
python create_meta_json.py \
    --video_dir "$VIDEO_DIR" \
    --output "$VIDEO_DIR/meta.json" \
    --description "In-the-wild video for EgoX inference"
echo ""

# Show structure
echo "=========================================="
echo "Directory structure created:"
echo "=========================================="
tree "$VIDEO_DIR" 2>/dev/null || find "$VIDEO_DIR" -type f | head -20
echo ""

echo "✓ Ready for EgoX inference!"
echo ""
echo "Next step:"
echo "  python modal_app.py"
echo ""
echo "Or run inference locally (if you have the models):"
echo "  python infer.py \\"
echo "    --meta_data_file $VIDEO_DIR/meta.json \\"
echo "    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \\"
echo "    --lora_path ./checkpoints/EgoX/pytorch_lora_weights.safetensors \\"
echo "    --lora_rank 256 \\"
echo "    --out ./results \\"
echo "    --seed 42 \\"
echo "    --use_GGA \\"
echo "    --cos_sim_scaling_factor 3.0 \\"
echo "    --in_the_wild"
