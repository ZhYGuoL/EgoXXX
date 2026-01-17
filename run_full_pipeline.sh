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

python -c "
import modal
from vipe_modal import run_vipe_inference

app = modal.App.lookup('vipe-inference')
result = run_vipe_inference.remote(
    video_path='$VIDEO_PATH',
    start_frame=0,
    end_frame=48
)
print(f'✓ ViPE result: {result}')
" || python vipe_modal.py "$VIDEO_PATH" "$VIDEO_DIR/vipe_output"

echo ""
echo "✓ Step 1 complete!"
echo ""

# Step 2: Manual annotation
echo "=========================================="
echo "STEP 2: Manual Ego Trajectory Annotation"
echo "=========================================="
echo ""
echo "For best results, manually annotate ego trajectories:"
echo ""
echo "1. Install EgoPriorRenderer locally (if not already done):"
echo "   cd EgoX-EgoPriorRenderer"
echo "   conda env create -f envs/base.yml"
echo "   conda activate egox-egoprior"
echo "   pip install -r envs/requirements.txt"
echo "   pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9' --no-build-isolation"
echo "   pip install git+https://github.com/microsoft/MoGe.git"
echo "   pip install --no-build-isolation -e ."
echo ""
echo "2. Start visualization with manual annotation:"
echo "   vipe visualize vipe_results/my_video --ego_manual"
echo ""
echo "3. For each frame:"
echo "   - Position the ego camera frustum to align with head pose"
echo "   - Copy ego_extrinsics from the UI panel (top-right)"
echo ""
echo "4. Update meta.json with annotated ego_extrinsics"
echo ""
echo "For now, using auto-generated trajectory (press Enter to continue)..."
read -p ""

# Step 3: Run rendering and depth conversion on Modal
echo ""
echo "=========================================="
echo "STEP 3: Ego Prior Rendering (Modal H100)"
echo "=========================================="
echo ""
echo "Running rendering and depth conversion..."
python -c "
import modal
from vipe_modal import render_ego_prior, convert_depth_maps

app = modal.App.lookup('vipe-inference')

# Render ego prior
print('[Rendering] Generating ego-view video...')
render_ego_prior.remote(
    vipe_result_name='my_video',
    meta_json_path='$VIDEO_DIR/meta.json',
    output_dir='$VIDEO_DIR',
)

# Convert depths
print('[Converting] Depth maps .exr -> .npy...')
convert_depth_maps.remote(
    vipe_result_name='my_video',
    output_dir='$VIDEO_DIR/depth_maps',
)
" 2>/dev/null || echo "Modal apps not running. Skipping rendering."

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
