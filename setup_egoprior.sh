#!/bin/bash

# Navigate to EgoPriorRenderer
cd EgoX-EgoPriorRenderer

# Install dependencies
echo "Installing ViPE and dependencies..."
pip install -r envs/requirements.txt -q

echo "Installing pytorch3d..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9" --no-build-isolation -q

echo "Installing MoGe..."
pip install git+https://github.com/microsoft/MoGe.git -q

echo "Installing EgoPriorRenderer..."
pip install --no-build-isolation -e . -q

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run ViPE inference:"
echo "   python scripts/infer_vipe.py --video_path ../preprocessed_videos/my_video/exo.mp4 --start_frame 0 --end_frame 48 --assume_fixed_camera_pose --pipeline lyra"
echo ""
echo "2. Visualize results:"
echo "   vipe visualize vipe_results/my_video"
echo ""
echo "3. Create meta.json in preprocessed_videos/my_video/"
echo ""
echo "4. Render ego prior"
echo ""
echo "5. Convert depth maps to npy"
