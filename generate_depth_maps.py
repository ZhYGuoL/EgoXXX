#!/usr/bin/env python3
"""
Generate depth maps for preprocessed videos.
Uses a simple monocular depth estimator or creates placeholder depth maps.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys

try:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def generate_depth_maps_from_video(video_path, output_dir, use_model=True):
    """
    Generate depth maps from a video.

    Args:
        video_path: Path to video file
        output_dir: Directory to save depth maps
        use_model: If True, use depth estimation model; if False, create placeholder depth maps
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Load depth estimation model if available
    depth_model = None
    image_processor = None

    if use_model and HAS_TRANSFORMERS:
        try:
            print("Loading depth estimation model...")
            image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
            depth_model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            depth_model.to(device)
            depth_model.eval()
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"Warning: Could not load depth model: {e}")
            print("Will create placeholder depth maps instead")
            use_model = False
    elif use_model:
        print("Warning: transformers not installed. Creating placeholder depth maps.")
        use_model = False

    # Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_idx + 1}/{total_frames}...", end='\r')

        if use_model and depth_model is not None:
            # Depth estimation with model
            try:
                # Convert BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Prepare input
                inputs = image_processor(images=image, return_tensors="pt")

                with torch.no_grad():
                    outputs = depth_model(**inputs)
                    predicted_depth = outputs.predicted_depth

                # Resize to match frame dimensions
                depth = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False
                ).squeeze().cpu().numpy()

                # Normalize to 0-1 range
                depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
                depth = (depth * 10).astype(np.float32)  # Scale to reasonable depth values

            except Exception as e:
                print(f"\nWarning: Error processing frame {frame_idx}: {e}")
                depth = np.ones((height, width), dtype=np.float32) * 5.0
        else:
            # Create placeholder depth map (constant depth)
            depth = np.ones((height, width), dtype=np.float32) * 5.0

        # Save depth map
        output_path = output_dir / f"frame_{frame_idx:06d}.npy"
        np.save(output_path, depth)

        frame_idx += 1

    cap.release()
    print(f"\n✓ Generated {frame_idx} depth maps in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps from video")
    parser.add_argument("--video_path", required=True, help="Path to video file")
    parser.add_argument("--output_dir", required=True, help="Output directory for depth maps")
    parser.add_argument("--no_model", action="store_true", help="Use placeholder depth maps instead of model")

    args = parser.parse_args()

    try:
        generate_depth_maps_from_video(
            args.video_path,
            args.output_dir,
            use_model=not args.no_model
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
