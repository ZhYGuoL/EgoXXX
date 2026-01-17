#!/usr/bin/env python3
"""
Create a basic meta.json for EgoX inference from preprocessed videos.
This generates default camera parameters and ego trajectory for in-the-wild videos.
"""

import json
import argparse
from pathlib import Path
import numpy as np


def create_default_meta_json(video_dir, output_path=None, description=""):
    """
    Create a meta.json with default parameters for a preprocessed video.

    Args:
        video_dir: Path to the video directory (contains exo.mp4)
        output_path: Path to save meta.json (default: video_dir/meta.json)
        description: Optional description/prompt for the video
    """

    video_dir = Path(video_dir)
    if not output_path:
        output_path = video_dir / "meta.json"
    else:
        output_path = Path(output_path)

    exo_video_path = video_dir / "exo.mp4"
    if not exo_video_path.exists():
        raise FileNotFoundError(f"exo.mp4 not found in {video_dir}")

    # Default camera intrinsics (estimated from 448x1232 resolution)
    # Using typical smartphone-like focal lengths
    camera_intrinsics = [
        [634.47327, 0.0, 392.0],   # fx, 0, cx
        [0.0, 634.4733, 224.0],    # 0, fy, cy
        [0.0, 0.0, 1.0]            # 0, 0, 1
    ]

    # Default exocentric camera extrinsics (world to camera, identity = looking at origin)
    camera_extrinsics = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ]

    # Default ego camera intrinsics (Project Aria specs)
    ego_intrinsics = [
        [150.0, 0.0, 255.5],       # fx, 0, cx
        [0.0, 150.0, 255.5],       # 0, fy, cy
        [0.0, 0.0, 1.0]            # 0, 0, 1
    ]

    # Default ego extrinsics - 49 frames with slight movement
    # These should be manually annotated for best results
    ego_extrinsics = []
    for frame_idx in range(49):
        # Slight forward movement and rotation
        t = frame_idx / 48.0  # Normalized frame index (0 to 1)

        # Small translation (forward 0.5 units)
        tx = t * 0.5
        ty = 0.0
        tz = -2.0  # 2 units away from origin

        # Slight rotation (yaw)
        angle = t * 0.2  # 0.2 radians
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Rotation matrix (yaw around Y)
        extrinsic = [
            [cos_a, 0.0, sin_a, tx],
            [0.0, 1.0, 0.0, ty],
            [-sin_a, 0.0, cos_a, tz]
        ]
        ego_extrinsics.append(extrinsic)

    # Create meta.json structure
    meta_data = {
        "test_datasets": [
            {
                "exo_path": str(exo_video_path),
                "ego_prior_path": str(video_dir / "ego_Prior.mp4"),
                "prompt": description or "A video scene.",
                "camera_intrinsics": camera_intrinsics,
                "camera_extrinsics": camera_extrinsics,
                "ego_intrinsics": ego_intrinsics,
                "ego_extrinsics": ego_extrinsics
            }
        ]
    }

    # Save meta.json
    with open(output_path, 'w') as f:
        json.dump(meta_data, f, indent=2)

    print(f"✓ Created meta.json at: {output_path}")
    print(f"\n⚠️  IMPORTANT NOTES:")
    print(f"   1. Camera parameters are DEFAULTS - consider replacing with actual values")
    print(f"   2. Ego trajectory is estimated - for best results, manually annotate using ViPE")
    print(f"   3. Make sure depth_maps/ exists and contains frame_*.npy files")
    print(f"\n   Structure should be:")
    print(f"   {video_dir}/")
    print(f"   ├── exo.mp4")
    print(f"   ├── ego_Prior.mp4 (optional)")
    print(f"   ├── meta.json (just created)")
    print(f"   └── depth_maps/")
    print(f"       ├── frame_000.npy")
    print(f"       └── ...")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create meta.json for EgoX inference")
    parser.add_argument("--video_dir", required=True, help="Path to video directory")
    parser.add_argument("--output", help="Output path for meta.json")
    parser.add_argument("--description", default="", help="Video description/prompt")

    args = parser.parse_args()

    try:
        create_default_meta_json(args.video_dir, args.output, args.description)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
