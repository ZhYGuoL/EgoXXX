# EgoX Complete Pipeline Guide

This guide walks through the full EgoX workflow from raw video to egocentric generation.

## Overview

```
Raw Video
    ↓
[1] Video Preprocessing (local) - crop to 784x448, select 49 frames
    ↓ (preprocessed_videos/my_video/exo.mp4)
[2] ViPE Inference (Modal H100) - estimate depth, camera poses
    ↓ (vipe_results/my_video/)
[3] Manual Ego Annotation (local) - interactively position ego camera
    ↓ (updated meta.json)
[4] Ego Prior Rendering (Modal H100) - render egocentric view
    ↓ (preprocessed_videos/my_video/ego_Prior.mp4)
[5] Depth Conversion (Modal H100) - .exr → .npy
    ↓ (preprocessed_videos/my_video/depth_maps/)
[6] EgoX Inference (Modal H100) - generate final video
    ↓
Output Egocentric Video
```

---

## Step 1: Video Preprocessing ✓ (Done)

Convert your raw video to the required format:
- Resolution: 784×448 (cropped, not resized)
- Frames: 49 frames exactly
- Fixed camera pose (your responsibility)

```bash
python video_preprocessor.py
```

**Output:**
```
preprocessed_videos/my_video/
└── exo.mp4 (784×448, 49 frames)
```

---

## Step 2: ViPE Inference (Modal H100)

Run monocular depth estimation and camera pose estimation:

```bash
python -c "
import modal
from vipe_modal import run_vipe_inference

result = run_vipe_inference.remote(
    video_path='./preprocessed_videos/my_video/exo.mp4',
    start_frame=0,
    end_frame=48
)
print(f'ViPE result: {result}')
"
```

Or use the full pipeline:
```bash
bash run_full_pipeline.sh
```

**What ViPE generates:**
- Depth maps (.exr format)
- Camera extrinsics (world-to-camera poses)
- Camera intrinsics (focal length, principal point)
- Point clouds for visualization

**Output:** `vipe_results/my_video/`

---

## Step 3: Manual Ego Trajectory Annotation ⚠️ (Critical for Quality)

For in-the-wild videos, you must manually annotate ego camera trajectories. This is what makes EgoX work properly.

### Setup (Local)

```bash
# 1. Install EgoPriorRenderer
cd EgoX-EgoPriorRenderer

conda env create -f envs/base.yml
conda activate egox-egoprior
pip install -r envs/requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9" --no-build-isolation
pip install git+https://github.com/microsoft/MoGe.git
pip install --no-build-isolation -e .

cd ..
```

### Annotation Workflow

```bash
# Start interactive visualization
vipe visualize vipe_results/my_video --ego_manual
```

This opens an interactive 3D viewer where you can:
1. **See the point cloud** - 3D reconstruction from the video
2. **See exocentric camera** - Green frustum showing the camera pose
3. **Position ego camera** - Drag the red frustum to align with head position for each frame
4. **Export ego trajectory** - Copy `ego_extrinsics` from UI panel

### Manual Annotation Steps

For each frame (0-48):

1. In the viewer, use mouse to rotate/zoom the 3D view
2. Position the **red frustum** (ego camera) to where the person's head would be
3. Make sure it aligns with the body direction/movement
4. Note the ego_extrinsics values shown in the top-right UI panel
5. Move to next frame

### Update meta.json

After annotation, update `preprocessed_videos/my_video/meta.json`:

```json
{
  "test_datasets": [
    {
      "exo_path": "./preprocessed_videos/my_video/exo.mp4",
      "ego_prior_path": "./preprocessed_videos/my_video/ego_Prior.mp4",
      "camera_intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "camera_extrinsics": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz]],
      "ego_intrinsics": [[150, 0, 255.5], [0, 150, 255.5], [0, 0, 1]],
      "ego_extrinsics": [
        [[r11, r12, r13, tx], ...],  // Frame 0
        [[r11, r12, r13, tx], ...],  // Frame 1
        ...                           // 49 total frames
      ]
    }
  ]
}
```

**Quality depends on manual annotation accuracy!** The ego trajectories guide how the 3D scene is rendered from the person's perspective.

---

## Step 4: Ego Prior Rendering (Modal H100)

Generate the egocentric view video using the annotated trajectory:

```bash
python -c "
import modal
from vipe_modal import render_ego_prior

render_ego_prior.remote(
    vipe_result_name='my_video',
    meta_json_path='./preprocessed_videos/my_video/meta.json',
    output_dir='./preprocessed_videos/my_video',
    point_size=5.0,
    start_frame=0,
    end_frame=48
)
"
```

**Output:**
```
preprocessed_videos/my_video/
└── ego_Prior.mp4 (egocentric rendered video)
```

---

## Step 5: Depth Map Conversion (Modal H100)

Convert depth maps from `.exr` to `.npy` format:

```bash
python -c "
import modal
from vipe_modal import convert_depth_maps

convert_depth_maps.remote(
    vipe_result_name='my_video',
    output_dir='./preprocessed_videos/my_video/depth_maps'
)
"
```

**Output:**
```
preprocessed_videos/my_video/depth_maps/
├── frame_000000.npy
├── frame_000001.npy
└── ... (49 total)
```

---

## Step 6: EgoX Inference (Modal H100)

Now run the EgoX model to generate egocentric video:

```bash
# Update modal_app.py to use your video:
# Line 357: meta_data_file="./preprocessed_videos/my_video/meta.json"

python modal_app.py
```

**Final Output:**
```
results/output.mp4 (generated egocentric video)
```

---

## Full Pipeline (Automated)

For a one-command setup:

```bash
bash run_full_pipeline.sh
```

This will:
1. Verify video preprocessing ✓
2. Run ViPE on Modal
3. Prompt you for manual annotation
4. Run rendering and depth conversion on Modal
5. Show final structure

---

## Important Notes

### For In-the-Wild Videos
- **Manual annotation is critical** for good quality output
- Ego trajectories guide the 3D-to-2D projection
- Spend time positioning the ego camera correctly
- Bad annotation = poor generated video

### Camera Parameters
- `camera_intrinsics`: Extracted from ViPE (estimated or provided)
- `camera_extrinsics`: Extracted from ViPE (3×4 pose matrix)
- `ego_intrinsics`: Project Aria specs (typical for egocentric videos)
- `ego_extrinsics`: **Manually annotated** (49 frames, one per frame)

### When to Use Placeholders
- If you can't manually annotate: `generate_depth_maps.py --no_model` creates placeholder depths
- Placeholder depths are constant/simple - quality will be lower
- For Ego-Exo4D dataset: automatic ego trajectories are available

### Requirements Met
- ✓ ViPE inference for depth estimation
- ✓ Camera pose optimization
- ✓ Point cloud generation
- ✓ Ego prior rendering with fish-eye distortion
- ✓ Depth map conversion for EgoX
- ✓ Manual annotation workflow for in-the-wild videos

---

## Example: Complete Workflow

```bash
# 1. Preprocess video (UI)
python video_preprocessor.py
# → Upload, crop, save to preprocessed_videos/my_video/exo.mp4

# 2. Run ViPE (Modal)
python -c "from vipe_modal import run_vipe_inference; run_vipe_inference.remote('./preprocessed_videos/my_video/exo.mp4')"

# 3. Manually annotate (Local)
cd EgoX-EgoPriorRenderer
conda activate egox-egoprior
vipe visualize ../vipe_results/my_video --ego_manual
# → Position ego camera for 49 frames, update meta.json

# 4. Render & Convert (Modal)
cd ..
python -c "from vipe_modal import render_ego_prior, convert_depth_maps; render_ego_prior.remote(...); convert_depth_maps.remote(...)"

# 5. Run EgoX (Modal)
python modal_app.py
# → Final egocentric video output!
```

---

## Troubleshooting

### ViPE fails on Modal
- Check GPU memory (H100 has 80GB, should be enough)
- Verify video format (mp4, h264)
- Check frame range (0-48 for 49 frames)

### Manual annotation seems wrong
- Use the visualization tool's preview to validate
- Align ego frustum with actual head position in video
- Check that movement looks natural

### Depth maps look strange
- This is expected for placeholder depths
- Use ViPE-generated depths for better quality
- Ensure depth_maps/ directory has correct file format (.npy)

### EgoX output quality is poor
- Most likely: bad ego trajectory annotation
- Try re-annotating with better alignment
- Check camera parameters are reasonable

---

## References

- **EgoX Paper**: [EgoX: Egocentric Video Generation from a Single Exocentric Video](https://arxiv.org/abs/2512.08269)
- **ViPE**: [Video Pose Engine](https://github.com/nv-tlabs/vipe)
- **EgoPriorRenderer**: [GitHub](https://github.com/kdh8156/EgoX-EgoPriorRenderer)
