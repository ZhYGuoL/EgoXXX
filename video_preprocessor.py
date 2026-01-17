import gradio as gr
import cv2
import numpy as np
import os
from pathlib import Path
import shutil

# Output directory for processed videos
OUTPUT_DIR = Path("./preprocessed_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

# EgoX model requirements
REQUIRED_FRAMES = 49
REQUIRED_HEIGHT = 448
REQUIRED_WIDTH = 784


def get_video_info(video_path):
    """Get video information."""
    if video_path is None:
        return None, None, None, None, None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    return fps, total_frames, width, height, duration


def update_video_info(video):
    """Update video info display when video is uploaded."""
    if video is None:
        return (
            "No video uploaded",
            gr.update(maximum=100, value=0),
            gr.update(maximum=100, value=100),
            gr.update(maximum=0, value=0),
            gr.update(maximum=0, value=0),
            None,
            0,
            0
        )

    fps, total_frames, width, height, duration = get_video_info(video)

    # Calculate max crop positions
    max_crop_x = max(0, width - REQUIRED_WIDTH)
    max_crop_y = max(0, height - REQUIRED_HEIGHT)

    info_text = f"""**Video Information:**
- Resolution: {width} x {height}
- Total Frames: {total_frames}
- FPS: {fps:.2f}
- Duration: {duration:.2f}s

**EgoX Requirements:**
- Crop Size: {REQUIRED_WIDTH} x {REQUIRED_HEIGHT}
- Frames: {REQUIRED_FRAMES}
- Fixed camera pose (user responsibility)

**Crop Range:**
- Horizontal: 0 to {max_crop_x} px
- Vertical: 0 to {max_crop_y} px
"""

    return (
        info_text,
        gr.update(maximum=total_frames - 1, value=0),
        gr.update(maximum=total_frames - 1, value=min(REQUIRED_FRAMES - 1, total_frames - 1)),
        gr.update(maximum=max_crop_x, value=max_crop_x // 2),  # Center horizontally
        gr.update(maximum=max_crop_y, value=max_crop_y // 2),  # Center vertically
        video,
        width,
        height
    )


def preview_frame_with_crop(video, frame_num, crop_x, crop_y):
    """Preview a specific frame with crop overlay."""
    if video is None:
        return None

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # Ensure crop coordinates are valid
    crop_x = int(min(crop_x, max(0, w - REQUIRED_WIDTH)))
    crop_y = int(min(crop_y, max(0, h - REQUIRED_HEIGHT)))

    # Draw crop rectangle overlay
    overlay = frame.copy()

    # Darken areas outside crop region
    mask = np.zeros_like(frame)
    mask[crop_y:crop_y + REQUIRED_HEIGHT, crop_x:crop_x + REQUIRED_WIDTH] = 1
    darkened = (frame * 0.4).astype(np.uint8)
    frame_with_overlay = np.where(mask == 1, frame, darkened)

    # Draw border around crop area
    cv2.rectangle(
        frame_with_overlay,
        (crop_x, crop_y),
        (crop_x + REQUIRED_WIDTH, crop_y + REQUIRED_HEIGHT),
        (0, 255, 0),
        3
    )

    # Add dimension text
    cv2.putText(
        frame_with_overlay,
        f"Crop: {REQUIRED_WIDTH}x{REQUIRED_HEIGHT}",
        (crop_x + 10, crop_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return frame_with_overlay


def preview_cropped_frame(video, frame_num, crop_x, crop_y):
    """Preview the actual cropped result."""
    if video is None:
        return None

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # Ensure crop coordinates are valid
    crop_x = int(min(crop_x, max(0, w - REQUIRED_WIDTH)))
    crop_y = int(min(crop_y, max(0, h - REQUIRED_HEIGHT)))

    # Crop the frame
    cropped = frame[crop_y:crop_y + REQUIRED_HEIGHT, crop_x:crop_x + REQUIRED_WIDTH]

    return cropped


def calculate_output_info(video, start_frame, end_frame, crop_x, crop_y):
    """Calculate and display output video information."""
    if video is None:
        return "Upload a video first"

    num_frames = end_frame - start_frame + 1

    status = ""
    if num_frames < REQUIRED_FRAMES:
        status = f"Warning: Selected {num_frames} frames, but model requires {REQUIRED_FRAMES} frames. Video will be padded by repeating the last frame."
    elif num_frames > REQUIRED_FRAMES:
        status = f"Warning: Selected {num_frames} frames, but model requires {REQUIRED_FRAMES} frames. Video will be uniformly sampled to {REQUIRED_FRAMES} frames."
    else:
        status = f"Perfect! Selected exactly {REQUIRED_FRAMES} frames."

    return f"""**Output Preview:**
- Frames: {start_frame} to {end_frame} ({num_frames} frames)
- Crop Position: ({int(crop_x)}, {int(crop_y)})
- Output Resolution: {REQUIRED_WIDTH} x {REQUIRED_HEIGHT}
- Final Frames: {REQUIRED_FRAMES}

{status}
"""


def process_video(video, start_frame, end_frame, crop_x, crop_y, output_name):
    """Process the video: clip, crop, and save."""
    if video is None:
        return "Please upload a video first", None

    if not output_name:
        output_name = "processed_video"

    # Sanitize output name
    output_name = "".join(c for c in output_name if c.isalnum() or c in "._- ")

    try:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ensure crop coordinates are valid
        crop_x = int(min(crop_x, max(0, width - REQUIRED_WIDTH)))
        crop_y = int(min(crop_y, max(0, height - REQUIRED_HEIGHT)))

        # Read and crop selected frames
        frames = []
        for frame_idx in range(int(start_frame), int(end_frame) + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Crop frame
                cropped = frame[crop_y:crop_y + REQUIRED_HEIGHT, crop_x:crop_x + REQUIRED_WIDTH]
                frames.append(cropped)
        cap.release()

        if len(frames) == 0:
            return "Error: Could not read any frames", None

        # Adjust to exactly REQUIRED_FRAMES
        if len(frames) < REQUIRED_FRAMES:
            # Pad by repeating last frame
            while len(frames) < REQUIRED_FRAMES:
                frames.append(frames[-1].copy())
        elif len(frames) > REQUIRED_FRAMES:
            # Uniformly sample frames
            indices = np.linspace(0, len(frames) - 1, REQUIRED_FRAMES, dtype=int)
            frames = [frames[i] for i in indices]

        # Create output directory structure
        video_dir = OUTPUT_DIR / output_name
        video_dir.mkdir(exist_ok=True)

        output_path = video_dir / "exo.mp4"

        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (REQUIRED_WIDTH, REQUIRED_HEIGHT))

        for frame in frames:
            out.write(frame)
        out.release()

        # Convert to h264 for better compatibility using ffmpeg
        temp_path = video_dir / "exo_temp.mp4"
        shutil.move(str(output_path), str(temp_path))

        ffmpeg_result = os.system(f'ffmpeg -y -i "{temp_path}" -c:v libx264 -preset fast -crf 18 "{output_path}" -loglevel quiet')

        if output_path.exists() and ffmpeg_result == 0:
            temp_path.unlink(missing_ok=True)
        else:
            # Fallback if ffmpeg fails
            shutil.move(str(temp_path), str(output_path))

        success_msg = f"""**Video Processed Successfully!**

Saved to: `{output_path}`

**Video Details:**
- Resolution: {REQUIRED_WIDTH} x {REQUIRED_HEIGHT}
- Frames: {REQUIRED_FRAMES}
- Crop Position: ({crop_x}, {crop_y})

**Next Steps:**
1. Use EgoX-EgoPriorRenderer to generate:
   - Depth maps
   - Camera parameters
   - Ego prior video

2. Create meta.json:
   ```bash
   python meta_init.py --folder_path {video_dir} --output_json {video_dir}/meta.json --overwrite
   ```

3. Generate captions:
   ```bash
   python caption.py --json_file {video_dir}/meta.json --output_json {video_dir}/meta.json --overwrite
   ```

4. Run inference on Modal (after completing EgoPriorRenderer steps)
"""

        return success_msg, str(output_path)

    except Exception as e:
        return f"Error processing video: {str(e)}", None


def create_ui():
    """Create the Gradio UI."""

    with gr.Blocks(title="EgoX Video Preprocessor") as app:
        gr.Markdown("""
        # EgoX Video Preprocessor

        Prepare your exocentric (third-person) video for EgoX inference.

        **Requirements:**
        - Fixed camera pose (no camera movement)
        - Output: 49 frames at 784x448 resolution (cropped, not resized)
        """)

        # Hidden state for video dimensions
        video_width = gr.State(0)
        video_height = gr.State(0)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 1. Upload Video")
                video_input = gr.Video(label="Upload your exocentric video")
                video_info = gr.Markdown("No video uploaded")

                gr.Markdown("### 2. Select Frame Range")
                with gr.Row():
                    start_frame = gr.Slider(
                        minimum=0, maximum=100, value=0, step=1,
                        label="Start Frame"
                    )
                    end_frame = gr.Slider(
                        minimum=0, maximum=100, value=48, step=1,
                        label="End Frame"
                    )

                gr.Markdown("### 3. Position Crop Area")
                gr.Markdown(f"*Drag sliders to position the {REQUIRED_WIDTH}x{REQUIRED_HEIGHT} crop region*")
                with gr.Row():
                    crop_x = gr.Slider(
                        minimum=0, maximum=1000, value=0, step=1,
                        label="Crop X Position (horizontal)"
                    )
                with gr.Row():
                    crop_y = gr.Slider(
                        minimum=0, maximum=1000, value=0, step=1,
                        label="Crop Y Position (vertical)"
                    )

                gr.Markdown("### 4. Output Settings")
                output_name = gr.Textbox(
                    label="Video Name (folder name)",
                    placeholder="my_video",
                    value="my_video"
                )

                output_info = gr.Markdown("Upload a video to see output preview")

                process_btn = gr.Button("Process & Save Video", variant="primary", size="lg")
                result_text = gr.Markdown("")

            with gr.Column(scale=1):
                # Preview section
                gr.Markdown("### Frame Preview with Crop Overlay")
                gr.Markdown("*Green box shows the crop region. Dark areas will be removed.*")

                preview_slider = gr.Slider(
                    minimum=0, maximum=100, value=0, step=1,
                    label="Preview Frame"
                )
                frame_preview = gr.Image(label="Frame with Crop Overlay", type="numpy")

                gr.Markdown("### Cropped Result Preview")
                cropped_preview = gr.Image(label="What will be saved", type="numpy")

                gr.Markdown("### Processed Video")
                output_video = gr.Video(label="Output Preview")

        # Event handlers
        video_input.change(
            update_video_info,
            inputs=[video_input],
            outputs=[video_info, start_frame, end_frame, crop_x, crop_y, output_video, video_width, video_height]
        )

        # Update preview when any parameter changes
        def update_all_previews(video, frame_num, cx, cy):
            overlay = preview_frame_with_crop(video, frame_num, cx, cy)
            cropped = preview_cropped_frame(video, frame_num, cx, cy)
            return overlay, cropped

        for component in [preview_slider, crop_x, crop_y]:
            component.change(
                update_all_previews,
                inputs=[video_input, preview_slider, crop_x, crop_y],
                outputs=[frame_preview, cropped_preview]
            )

        # Update preview slider range when video changes
        video_input.change(
            lambda v: gr.update(maximum=get_video_info(v)[1] - 1 if v else 100, value=0),
            inputs=[video_input],
            outputs=[preview_slider]
        )

        # Initial preview when video loads
        video_input.change(
            update_all_previews,
            inputs=[video_input, preview_slider, crop_x, crop_y],
            outputs=[frame_preview, cropped_preview]
        )

        # Update output info when parameters change
        for component in [start_frame, end_frame, crop_x, crop_y]:
            component.change(
                calculate_output_info,
                inputs=[video_input, start_frame, end_frame, crop_x, crop_y],
                outputs=[output_info]
            )

        video_input.change(
            calculate_output_info,
            inputs=[video_input, start_frame, end_frame, crop_x, crop_y],
            outputs=[output_info]
        )

        # Process button
        process_btn.click(
            process_video,
            inputs=[video_input, start_frame, end_frame, crop_x, crop_y, output_name],
            outputs=[result_text, output_video]
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False, theme=gr.themes.Soft())
