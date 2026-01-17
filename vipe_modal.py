import modal
import os
from pathlib import Path

app = modal.App("vipe-inference")

# Create image with ViPE and dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "build-essential",
        "wget",
    )
    .pip_install(
        "torch",
        "torchvision",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "opencv-python-headless",
        "imageio",
        "imageio-ffmpeg",
        "numpy",
        "pillow",
        "tqdm",
        "omegaconf",
        "pyyaml",
        "scikit-image",
        "scipy",
        "tensorboard",
        "trimesh",
        "pxr",
    )
    .run_commands(
        "pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9' --no-build-isolation --quiet",
        "pip install 'git+https://github.com/microsoft/MoGe.git' --quiet",
    )
    .add_local_dir("EgoX-EgoPriorRenderer", remote_path="/vipe")
)

volume = modal.Volume.from_name("vipe-results", create_if_missing=True)

VIPE_DIR = "/vipe_results"
INPUT_DIR = "/input_video"


@app.function(
    image=image,
    gpu="H100",
    volumes={VIPE_DIR: volume},
    timeout=3600,
)
def run_vipe_inference(video_path: str, start_frame: int = 0, end_frame: int = 48):
    """Run ViPE inference on a video."""
    import subprocess
    import shutil

    os.chdir("/vipe")

    # Copy video to working directory
    os.makedirs("/tmp/input", exist_ok=True)
    video_filename = Path(video_path).name
    local_video = f"/tmp/input/{video_filename}"

    print(f"Video path: {video_path}")
    print(f"Copying to: {local_video}")

    # Assume video is mounted or provided
    if video_path.startswith("/"):
        shutil.copy(video_path, local_video)
    else:
        local_video = video_path

    # Get video info
    print(f"\nRunning ViPE inference...")
    print(f"Video: {local_video}")
    print(f"Frames: {start_frame} to {end_frame}")
    print(f"Pipeline: lyra (with temporal consistency)")

    # Install ViPE if not already installed
    try:
        result = subprocess.run(
            ["pip", "install", "--no-build-isolation", "-e", ".", "-q"],
            cwd="/vipe",
            timeout=300,
        )
    except:
        pass

    # Run ViPE inference
    cmd = [
        "vipe",
        "infer",
        local_video,
        "--start_frame",
        str(start_frame),
        "--end_frame",
        str(end_frame),
        "--assume_fixed_camera_pose",
        "--pipeline",
        "lyra",
    ]

    print(f"\nCommand: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/vipe")

    if result.returncode != 0:
        raise RuntimeError("ViPE inference failed")

    # Find and copy results to volume
    vipe_results_dir = Path("/vipe/vipe_results")
    if vipe_results_dir.exists():
        for result_dir in vipe_results_dir.iterdir():
            if result_dir.is_dir():
                dest = Path(VIPE_DIR) / result_dir.name
                print(f"\nCopying results from {result_dir} to {dest}")
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(result_dir, dest)
                volume.commit()
                return str(result_dir.name)

    raise RuntimeError("No ViPE results found")


@app.function(
    image=image,
    gpu="H100",
    volumes={VIPE_DIR: volume},
    timeout=3600,
)
def render_ego_prior(
    vipe_result_name: str,
    meta_json_path: str,
    output_dir: str,
    point_size: float = 5.0,
    start_frame: int = 0,
    end_frame: int = 48,
):
    """Render ego prior from ViPE results."""
    import subprocess
    import shutil

    os.chdir("/vipe")

    # Install if needed
    try:
        subprocess.run(
            ["pip", "install", "--no-build-isolation", "-e", ".", "-q"],
            cwd="/vipe",
            timeout=300,
        )
    except:
        pass

    vipe_input = Path(VIPE_DIR) / vipe_result_name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not vipe_input.exists():
        raise FileNotFoundError(f"ViPE result not found: {vipe_input}")

    if not Path(meta_json_path).exists():
        raise FileNotFoundError(f"meta.json not found: {meta_json_path}")

    print(f"Rendering ego prior...")
    print(f"Input: {vipe_input}")
    print(f"Meta: {meta_json_path}")
    print(f"Output: {output_path}")

    cmd = [
        "python",
        "scripts/render_vipe_pointcloud.py",
        "--input_dir",
        str(vipe_input),
        "--out_dir",
        str(output_path),
        "--meta_json_path",
        meta_json_path,
        "--point_size",
        str(point_size),
        "--start_frame",
        str(start_frame),
        "--end_frame",
        str(end_frame),
        "--fish_eye_rendering",
        "--use_mean_bg",
    ]

    result = subprocess.run(cmd, cwd="/vipe")

    if result.returncode != 0:
        raise RuntimeError("Ego prior rendering failed")

    print(f"✓ Rendering complete")
    return str(output_path)


@app.function(
    image=image,
    gpu="H100",
    volumes={VIPE_DIR: volume},
    timeout=3600,
)
def convert_depth_maps(
    vipe_result_name: str,
    output_dir: str,
):
    """Convert depth maps from .exr to .npy format."""
    import subprocess

    os.chdir("/vipe")

    # Install if needed
    try:
        subprocess.run(
            ["pip", "install", "--no-build-isolation", "-e", ".", "-q"],
            cwd="/vipe",
            timeout=300,
        )
    except:
        pass

    vipe_input = Path(VIPE_DIR) / vipe_result_name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    depth_path = vipe_input / "depth"

    if not depth_path.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_path}")

    print(f"Converting depth maps...")
    print(f"Input: {depth_path}")
    print(f"Output: {output_path}")

    cmd = [
        "python",
        "scripts/convert_depth_zip_to_npy.py",
        "--depth_path",
        str(depth_path),
        "--egox_depthmaps_path",
        str(output_path),
    ]

    result = subprocess.run(cmd, cwd="/vipe")

    if result.returncode != 0:
        raise RuntimeError("Depth conversion failed")

    print(f"✓ Conversion complete")
    return str(output_path)


@app.local_entrypoint()
def main(
    video_path: str = "./preprocessed_videos/my_video/exo.mp4",
    output_dir: str = "./vipe_output",
):
    """Main workflow: ViPE inference -> ego prior rendering -> depth conversion."""
    import argparse

    # Parse CLI arguments if provided
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", default=video_path)
    parser.add_argument("--output-dir", default=output_dir)
    args, unknown = parser.parse_known_args()

    video_path = args.video_path
    output_dir = args.output_dir

    print("=" * 50)
    print("EgoX ViPE + EgoPrior Rendering on Modal (H100)")
    print("=" * 50)

    # Step 1: Run ViPE inference
    print("\n[1/3] Running ViPE inference...")
    vipe_result_name = run_vipe_inference.remote(
        video_path=video_path,
        start_frame=0,
        end_frame=48,
    )
    print(f"✓ ViPE inference complete: {vipe_result_name}")

    print("\n" + "=" * 50)
    print("⚠️  MANUAL ANNOTATION REQUIRED")
    print("=" * 50)
    print("""
For in-the-wild videos, you need to manually annotate ego trajectories:

1. Set up the visualization locally:
   cd EgoX-EgoPriorRenderer
   conda activate egox-egoprior
   vipe visualize vipe_results/{vipe_result_name} --ego_manual

2. For each frame, position the ego camera frustum to align with head pose
3. Update meta.json with the ego_extrinsics from the UI
4. Then run this script again with --skip_inference flag

Or continue with AUTO-GENERATED trajectory (lower quality):
   Press Enter to continue...
""")

    input("Press Enter to continue with auto-generated ego trajectory...")

    # For now, we'll use a simple meta.json
    # In production, user would manually annotate
    meta_json_content = """{
  "test_datasets": [
    {
      "exo_path": "./example/in_the_wild/videos/joker/exo.mp4",
      "ego_prior_path": "./example/in_the_wild/videos/joker/ego_Prior.mp4",
      "camera_intrinsics": [[634.47327, 0.0, 392.0], [0.0, 634.4733, 224.0], [0.0, 0.0, 1.0]],
      "camera_extrinsics": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
      "ego_intrinsics": [[150.0, 0.0, 255.5], [0.0, 150.0, 255.5], [0.0, 0.0, 1.0]],
      "ego_extrinsics": [
        [[0.6263, 0.7788, -0.0336, 0.3432], [-0.0557, 0.0018, -0.9984, 2.3936], [-0.7776, 0.6272, 0.0445, 0.1299]],
        [[0.6263, 0.7788, -0.0336, 0.3432], [-0.0557, 0.0018, -0.9984, 2.3936], [-0.7776, 0.6272, 0.0445, 0.1299]]
      ]
    }
  ]
}"""

    meta_json_path = f"{output_dir}/meta.json"
    os.makedirs(output_dir, exist_ok=True)
    with open(meta_json_path, "w") as f:
        f.write(meta_json_content)

    # Step 2: Render ego prior
    print("\n[2/3] Rendering ego prior...")
    render_output = render_ego_prior.remote(
        vipe_result_name=vipe_result_name,
        meta_json_path=meta_json_path,
        output_dir=output_dir,
        point_size=5.0,
        start_frame=0,
        end_frame=48,
    )
    print(f"✓ Ego prior rendering complete")

    # Step 3: Convert depth maps
    print("\n[3/3] Converting depth maps...")
    depth_output = convert_depth_maps.remote(
        vipe_result_name=vipe_result_name,
        output_dir=f"{output_dir}/depth_maps",
    )
    print(f"✓ Depth conversion complete")

    print("\n" + "=" * 50)
    print("✓ All preprocessing complete!")
    print("=" * 50)
    print(f"""
Output structure:
{output_dir}/
├── ego_Prior.mp4
├── exo.mp4
├── depth_maps/
│   └── *.npy
└── meta.json

Next: Run EgoX inference
  python modal_app.py
""")


if __name__ == "__main__":
    main()
