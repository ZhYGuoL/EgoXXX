#!/bin/bash

set -e

echo "=========================================="
echo "EgoX Full Pipeline: ViPE + EgoPrior + EgoX"
echo "=========================================="
echo ""

VIDEO_DIR="./preprocessed_videos/my_video"
VIDEO_PATH="$VIDEO_DIR/exo.mp4"

# Step 0: Verify video
if [ ! -f "$VIDEO_PATH" ]; then
    echo "❌ Error: Preprocessed video not found"
    echo "   Run the video preprocessor first:"
    echo "   python video_preprocessor.py"
    exit 1
fi

echo "✓ Video found: $VIDEO_PATH"
echo ""

# Step 1: Run ViPE inference on Modal
echo "=========================================="
echo "STEP 1: ViPE Inference (Modal H100)"
echo "=========================================="
echo ""
echo "Starting ViPE inference on Modal H100..."
echo "This will:"
echo "  - Estimate depth maps"
echo "  - Compute camera poses"
echo "  - Generate 3D point clouds"
echo ""

modal run vipe_modal.py --video-path "$VIDEO_PATH" --output-dir "$VIDEO_DIR/vipe_output"

echo ""
echo "✓ Step 1 complete!"
echo ""

# Step 2: Manual annotation
echo "=========================================="
echo "STEP 2: Manual Ego Trajectory Annotation"
echo "=========================================="
echo ""
echo "⚠️  IMPORTANT: Manual annotation is CRITICAL for output quality!"
echo ""
echo "The ViPE results are now available at: vipe_results/my_video/"
echo ""
echo "To manually annotate ego trajectories:"
echo ""
echo "1. Install EgoPriorRenderer locally (one-time setup):"
echo "   cd EgoX-EgoPriorRenderer"
echo "   conda env create -f envs/base.yml"
echo "   conda activate egox-egoprior"
echo "   pip install -r envs/requirements.txt"
echo "   pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9' --no-build-isolation"
echo "   pip install git+https://github.com/microsoft/MoGe.git"
echo "   pip install --no-build-isolation -e ."
echo "   cd .."
echo ""
echo "2. Start the interactive visualization:"
echo "   vipe visualize vipe_results/my_video --ego_manual"
echo ""
echo "3. In the viewer, for each of 49 frames:"
echo "   - Position the RED frustum to align with head position"
echo "   - Note the ego_extrinsics values from the UI panel (top-right)"
echo ""
echo "4. Update meta.json with the annotated ego_extrinsics"
echo ""
echo "Press Enter to CONTINUE WITH AUTO-GENERATED TRAJECTORY (lower quality)..."
read -p ""

# Step 3: Ego Prior Rendering and Depth Conversion
echo ""
echo "=========================================="
echo "STEP 3: Ego Prior Rendering (Modal H100)"
echo "=========================================="
echo ""

# Create basic meta.json if doesn't exist
if [ ! -f "$VIDEO_DIR/meta.json" ]; then
    echo "Creating default meta.json..."
    python create_meta_json.py \
        --video_dir "$VIDEO_DIR" \
        --output "$VIDEO_DIR/meta.json"
fi

echo "Rendering ego prior and converting depths on Modal H100..."
echo "(This may take several minutes)"
echo ""

modal run vipe_modal.py --video-path "$VIDEO_PATH" --output-dir "$VIDEO_DIR"

echo ""
echo "✓ Step 3 complete!"
echo ""

# Verify structure
echo "=========================================="
echo "Final Directory Structure"
echo "=========================================="
echo ""
find "$VIDEO_DIR" -type f | sort | head -20
echo ""

# Step 4: Ready for inference
echo "=========================================="
echo "✓ Ready for EgoX Inference!"
echo "=========================================="
echo ""
echo "Run inference on Modal with H100:"
echo "  modal run modal_app.py"
echo ""
echo "Or locally (if models downloaded):"
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
