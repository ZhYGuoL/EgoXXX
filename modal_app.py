import modal

app = modal.App("egox-inference")

# Volume for storing model weights
volume = modal.Volume.from_name("egox-models", create_if_missing=True)

MODELS_DIR = "/models"
APP_DIR = "/app"

# Create the image with all dependencies and copy local files
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "diffusers==0.34.0",
        "transformers==4.49.0",
        "accelerate==1.5.2",
        "sentencepiece",
        "peft==0.17.1",
        "decord",
        "opencv-python-headless",
        "imageio",
        "imageio-ffmpeg",
        "tyro",
        "ftfy",
        "huggingface_hub",
        "pydantic",
        "wandb",
    )
    .add_local_dir("core", remote_path=f"{APP_DIR}/core")
    .add_local_dir("example", remote_path=f"{APP_DIR}/example")
)


@app.function(
    image=image,
    gpu="H100",
    volumes={MODELS_DIR: volume},
    timeout=3600,
)
def download_models():
    """Download the pretrained models to the volume."""
    import os
    from huggingface_hub import snapshot_download

    pretrained_path = f"{MODELS_DIR}/Wan2.1-I2V-14B-480P-Diffusers"
    egox_path = f"{MODELS_DIR}/EgoX"

    if not os.path.exists(pretrained_path):
        print("Downloading Wan2.1-I2V-14B pretrained model...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            local_dir=pretrained_path,
        )
        print("Pretrained model downloaded!")

    if not os.path.exists(egox_path):
        print("Downloading EgoX LoRA weights...")
        snapshot_download(
            repo_id="DAVIAN-Robotics/EgoX",
            local_dir=egox_path,
            allow_patterns="*.safetensors",
        )
        print("EgoX weights downloaded!")

    volume.commit()
    return "Models downloaded successfully!"


@app.function(
    image=image,
    gpu="H100",
    volumes={MODELS_DIR: volume},
    timeout=3600,
)
def run_inference(
    meta_data_file: str = f"{APP_DIR}/example/in_the_wild/meta.json",
    seed: int = 42,
    idx: int = 0,
    use_gga: bool = True,
    cos_sim_scaling_factor: float = 3.0,
    in_the_wild: bool = True,
):
    """Run EgoX inference on Modal."""
    import os
    import sys
    sys.path.insert(0, APP_DIR)

    os.chdir(APP_DIR)

    import random
    import numpy as np
    import torch
    from pathlib import Path

    # Set seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from core.inference.wan import generate_video
    from core.finetune.datasets.utils import load_from_json_file, iproj_disp

    model_path = f"{MODELS_DIR}/Wan2.1-I2V-14B-480P-Diffusers"
    lora_path = f"{MODELS_DIR}/EgoX/pytorch_lora_weights.safetensors"
    output_dir = "/results"
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata
    meta_data = load_from_json_file(meta_data_file)
    meta_data = meta_data['test_datasets']

    prompts = []
    exo_videos = []
    ego_prior_videos = []
    depth_map_paths = []
    camera_intrinsics = []
    camera_extrinsics = []
    ego_extrinsics = []
    ego_intrinsics = []
    take_names = []

    for i, meta in enumerate(meta_data):
        # Fix paths to use APP_DIR
        exo_path = meta['exo_path'].replace("./example", f"{APP_DIR}/example")
        ego_prior_path = meta['ego_prior_path'].replace("./example", f"{APP_DIR}/example")

        exo_videos.append(exo_path)
        ego_prior_videos.append(ego_prior_path)
        prompts.append(meta['prompt'])
        take_name = exo_path.split('/')[-2]
        # depth_root should be the dataset root (e.g., /app/example/in_the_wild)
        depth_root = "/".join(exo_path.split('/')[:-3])  # Go up 3 levels from exo.mp4
        depth_map_paths.append(Path(os.path.join(depth_root, 'depth_maps', take_name)))
        camera_extrinsics.append(meta['camera_extrinsics'])
        camera_intrinsics.append(meta['camera_intrinsics'])
        ego_extrinsics.append(meta['ego_extrinsics'])
        ego_intrinsics.append(meta['ego_intrinsics'])
        take_names.append(take_name)

    # Load models
    dtype = torch.bfloat16
    transformer_path = os.path.join(model_path, 'transformer')

    from transformers import CLIPVisionModel
    from core.finetune.models.wan_i2v.custom_transformer import WanTransformer3DModel_GGA as WanTransformer3DModel
    from core.finetune.models.wan_i2v.sft_trainer import WanWidthConcatImageToVideoPipeline

    print("Loading transformer...")
    transformer = WanTransformer3DModel.from_pretrained(transformer_path, torch_dtype=dtype)

    print("Loading image encoder...")
    image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=torch.float32)

    print("Creating pipeline...")
    pipe = WanWidthConcatImageToVideoPipeline.from_pretrained(
        model_path, image_encoder=image_encoder, transformer=transformer, torch_dtype=dtype
    )

    print("Loading LoRA weights...")
    pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
    pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    pipe.to("cuda")

    # Process the specified index
    i = idx
    prompt, exo_video_path, ego_prior_video_path = prompts[i], exo_videos[i], ego_prior_videos[i]
    take_name = take_names[i]

    print(f"Processing: {take_name}")
    print(f"Exo video: {exo_video_path}")
    print(f"Ego prior: {ego_prior_video_path}")

    attn_maps = None
    attn_masks = None
    point_vecs_per_frame = None
    cam_rays = None

    if use_gga:
        import cv2

        depth_map_path = depth_map_paths[i]
        camera_intrinsic = camera_intrinsics[i]
        camera_extrinsic = camera_extrinsics[i]
        ego_extrinsic = ego_extrinsics[i]
        ego_intrinsic = ego_intrinsics[i]

        device = 'cpu'
        C, F, H, W = 16, 13, 56, 154
        exo_H, exo_W = H, W - H
        W = H

        depth_maps = []
        for depth_map_file in sorted(depth_map_path.glob("*.npy")):
            depth_map = np.load(depth_map_file)
            depth_maps.append(torch.from_numpy(depth_map).unsqueeze(0))
        depth_maps = torch.cat(depth_maps, dim=0)

        ego_intrinsic = torch.tensor(ego_intrinsic)
        ego_extrinsic = torch.tensor(ego_extrinsic)
        camera_extrinsic = torch.tensor(camera_extrinsic)
        camera_intrinsic_tensor = torch.tensor(camera_intrinsic)

        if ego_extrinsic.shape[1] == 3 and ego_extrinsic.shape[2] == 4:
            ego_extrinsic = torch.cat([ego_extrinsic, torch.tensor([[[0, 0, 0, 1]]], dtype=ego_extrinsic.dtype).expand(ego_extrinsic.shape[0], -1, -1)], dim=1)
        if camera_extrinsic.shape == (3, 4):
            camera_extrinsic = torch.cat([camera_extrinsic, torch.tensor([[0, 0, 0, 1]], dtype=ego_extrinsic.dtype)], dim=0)

        scale = 1/8
        scaled_intrinsic = ego_intrinsic.clone()
        scaled_intrinsic[0, 0] *= scale
        scaled_intrinsic[1, 1] *= scale
        scaled_intrinsic[0, 2] *= scale
        scaled_intrinsic[1, 2] *= scale

        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
        ones = torch.ones_like(xs)
        pixel_coords = torch.stack([xs, ys, ones], dim=-1).view(-1, 3).to(dtype=ego_intrinsic.dtype)

        pixel_coords_cv = pixel_coords[..., :2].cpu().numpy().reshape(-1, 1, 2).astype(np.float32)
        K = scaled_intrinsic.cpu().numpy().astype(np.float32)

        distortion_coeffs = np.array([[-0.02340373583137989,0.09388021379709244,-0.06088035926222801,0.0053304750472307205,0.003342868760228157,-0.0006356257363222539,0.0005087381578050554,-0.0004747129278257489,-0.0011330085108056664,-0.00025734835071489215,0.00009328465239377692,0.00009424977179151028]])
        D = distortion_coeffs.astype(np.float32)
        normalized_points = cv2.undistortPoints(pixel_coords_cv, K, D, R=np.eye(3), P=np.eye(3))

        normalized_points = torch.from_numpy(normalized_points).squeeze(1).to(device)
        ones = torch.ones_like(normalized_points[..., :1])
        cam_rays_fish = torch.cat([normalized_points, ones], dim=-1)
        cam_rays = cam_rays_fish / torch.norm(cam_rays_fish, dim=-1, keepdim=True)
        cam_rays = cam_rays @ ego_extrinsic[::4, :3, :3]
        cam_rays = cam_rays.view(F, H, W, 3)

        height, width = depth_maps.shape[1], depth_maps.shape[2]
        cx = width / 2.0
        cy = height / 2.0
        camera_intrinsic_scale_y = cy / camera_intrinsic_tensor[1,2]
        camera_intrinsic_scale_x = cx / camera_intrinsic_tensor[0,2]
        camera_intrinsic_tensor[0, 0] = camera_intrinsic_tensor[0, 0] * camera_intrinsic_scale_x
        camera_intrinsic_tensor[1, 1] = camera_intrinsic_tensor[1, 1] * camera_intrinsic_scale_y
        camera_intrinsic_tensor[0, 2] = cx
        camera_intrinsic_tensor[1, 2] = cy

        camera_intrinsic_arr = np.array([camera_intrinsic_tensor[0, 0], camera_intrinsic_tensor[1, 1], cx, cy])

        disp_v, disp_u = torch.meshgrid(
            torch.arange(depth_maps.shape[1], device=device).float(),
            torch.arange(depth_maps.shape[2], device=device).float(),
            indexing="ij",
        )
        disp = torch.ones_like(disp_v)
        pts, _, _ = iproj_disp(torch.from_numpy(camera_intrinsic_arr), disp.cpu(), disp_u.cpu(), disp_v.cpu())

        if isinstance(pts, torch.Tensor):
            pts = pts.to(device)
        else:
            pts = torch.from_numpy(pts).to(device).float()

        rays = pts[..., :3]
        rays = rays / rays[..., 2:3]
        rays = rays.unsqueeze(0).expand(depth_maps.size(0), -1, -1, -1)
        camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsic)

        pcd_camera = rays * depth_maps.unsqueeze(-1)
        point_map = pcd_camera.to(dtype=camera_extrinsics_c2w.dtype)
        point_map = torch.tensor(point_map)

        p_f, p_h, p_w, p_p = point_map.shape
        point_map_world = point_map.reshape(-1, 3)

        camera_extrinsics_c2w = torch.linalg.inv(camera_extrinsic)
        ones_point = torch.ones(point_map_world.shape[0], 1, device=point_map_world.device)
        point_map_world = torch.cat([point_map_world, ones_point], dim=-1)
        point_map_world = (camera_extrinsics_c2w @ point_map_world.T).T[...,:3]
        point_map = point_map_world.reshape(p_f, p_h, p_w, 3).permute(0, 3, 1, 2)

        point_map = point_map[:, :, (point_map.shape[2] - 448)//2:(point_map.shape[2] + 448)//2, (point_map.shape[3] - 784)//2:(point_map.shape[3] + 784)//2]
        point_map = torch.nn.functional.interpolate(point_map, size=(exo_H, exo_W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        ego_extrinsic_c2w = torch.linalg.inv(ego_extrinsic)

        cam_origins = ego_extrinsic_c2w[::4, :3, 3].unsqueeze(1).expand(-1, exo_H * exo_W, -1)
        cam_origins = cam_origins.view(F, exo_H, exo_W, 3)

        if point_map.size(0) != ego_extrinsic_c2w.size(0):
            min_size = min(point_map.size(0), ego_extrinsic_c2w.size(0))
            point_map = point_map[:min_size]

        point_vecs_per_frame = []
        for j in range(cam_origins.size(0)):
            point_vec = point_map[::4] - cam_origins[j].unsqueeze(0)
            point_vec = point_vec / torch.norm(point_vec, dim=-1, keepdim=True)
            point_vecs_per_frame.append(point_vec)
        point_vecs_per_frame = torch.stack(point_vecs_per_frame, dim=0)

        point_vecs = point_map[::4] - cam_origins
        point_vecs = point_vecs / torch.norm(point_vecs, dim=-1, keepdim=True)

        cam_rays = torch.rot90(cam_rays, k=-1, dims=[1, 2])

        attn_maps = torch.cat((point_vecs, cam_rays), dim=2)
        attn_masks = torch.cat((torch.ones_like(point_vecs), torch.zeros_like(cam_rays)), dim=2)

    output_path = os.path.join(output_dir, f'{take_name}.mp4')

    print(f"Generating video...")
    video = generate_video(
        prompt=prompt,
        exo_video_path=exo_video_path,
        ego_prior_video_path=ego_prior_video_path,
        output_path=output_path,
        num_frames=49,
        width=784+448,
        height=448,
        num_inference_steps=50,
        guidance_scale=5.0,
        fps=30,
        num_videos_per_prompt=1,
        seed=seed,
        attention_GGA=attn_maps.unsqueeze(0) if attn_maps is not None else None,
        attention_mask_GGA=attn_masks.unsqueeze(0) if attn_masks is not None else None,
        point_vecs_per_frame=point_vecs_per_frame,
        cam_rays=cam_rays,
        do_kv_cache=False,
        cos_sim_scaling_factor=cos_sim_scaling_factor,
        pipe=pipe,
    )

    print(f"Video saved to: {output_path}")

    # Read and return the video bytes
    with open(output_path, "rb") as f:
        video_bytes = f.read()

    return video_bytes


@app.local_entrypoint()
def main():
    """Main entrypoint to run the inference."""
    import os

    # First, download models if needed
    print("Checking/downloading models...")
    result = download_models.remote()
    print(result)

    # Run inference
    print("\nRunning inference...")
    video_bytes = run_inference.remote(
        meta_data_file=f"{APP_DIR}/example/in_the_wild/meta.json",
        seed=846514,
        idx=0,
        use_gga=True,
        cos_sim_scaling_factor=3.0,
        in_the_wild=True,
    )

    # Save the result locally
    output_path = "results/output.mp4"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(video_bytes)
    print(f"\nVideo saved to: {output_path}")
