import gradio as gr
import cv2
import numpy as np
import os
from pathlib import Path
import tempfile
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
        return None, None, None, None

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
        return "No video uploaded", gr.update(maximum=100, value=0), gr.update(maximum=100, value=100), None

    fps, total_frames, width, height, duration = get_video_info(video)

    info_text = f"""**Video Information:**
- Resolution: {width} x {height}
- Total Frames: {total_frames}
- FPS: {fps:.2f}
- Duration: {duration:.2f}s

**EgoX Requirements:**
- Resolution: {REQUIRED_WIDTH} x {REQUIRED_HEIGHT}
- Frames: {REQUIRED_FRAMES}
- Fixed camera pose (user responsibility)
"""

    return (
        info_text,
        gr.update(maximum=total_frames - 1, value=0),
        gr.update(maximum=total_frames - 1, value=min(REQUIRED_FRAMES - 1, total_frames - 1)),
        video
    )


def preview_frame(video, frame_num):
    """Preview a specific frame from the video."""
    if video is None:
        return None

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None


def calculate_output_info(video, start_frame, end_frame, target_width, target_height):
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

    resolution_status = ""
    if target_width != REQUIRED_WIDTH or target_height != REQUIRED_HEIGHT:
        resolution_status = f"\nNote: Resolution {target_width}x{target_height} differs from recommended {REQUIRED_WIDTH}x{REQUIRED_HEIGHT}"
    else:
        resolution_status = f"\nResolution matches EgoX requirements!"

    return f"""**Output Preview:**
- Frames: {start_frame} to {end_frame} ({num_frames} frames)
- Output Resolution: {target_width} x {target_height}
- Final Frames: {REQUIRED_FRAMES}

{status}{resolution_status}
"""


def process_video(video, start_frame, end_frame, target_width, target_height, output_name):
    """Process the video: clip, resize, and save."""
    if video is None:
        return "Please upload a video first", None

    if not output_name:
        output_name = "processed_video"

    # Sanitize output name
    output_name = "".join(c for c in output_name if c.isalnum() or c in "._- ")

    try:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Read selected frames
        frames = []
        for frame_idx in range(int(start_frame), int(end_frame) + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Resize frame
                frame = cv2.resize(frame, (int(target_width), int(target_height)), interpolation=cv2.INTER_LANCZOS4)
                frames.append(frame)
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
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (int(target_width), int(target_height)))

        for frame in frames:
            out.write(frame)
        out.release()

        # Convert to h264 for better compatibility using ffmpeg
        temp_path = video_dir / "exo_temp.mp4"
        shutil.move(str(output_path), str(temp_path))

        os.system(f'ffmpeg -y -i "{temp_path}" -c:v libx264 -preset fast -crf 18 "{output_path}" -loglevel quiet')

        if output_path.exists():
            temp_path.unlink(missing_ok=True)
        else:
            # Fallback if ffmpeg fails
            shutil.move(str(temp_path), str(output_path))

        success_msg = f"""**Video Processed Successfully!**

Saved to: `{output_path}`

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
        - Output: 49 frames at 784x448 resolution
        """)

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

                gr.Markdown("### 3. Set Output Resolution")
                with gr.Row():
                    target_width = gr.Number(value=REQUIRED_WIDTH, label="Width", precision=0)
                    target_height = gr.Number(value=REQUIRED_HEIGHT, label="Height", precision=0)

                with gr.Row():
                    gr.Button("Use EgoX Default (784x448)").click(
                        lambda: (REQUIRED_WIDTH, REQUIRED_HEIGHT),
                        outputs=[target_width, target_height]
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
                gr.Markdown("### Frame Preview")

                preview_slider = gr.Slider(
                    minimum=0, maximum=100, value=0, step=1,
                    label="Preview Frame"
                )
                frame_preview = gr.Image(label="Frame Preview", type="numpy")

                gr.Markdown("### Processed Video")
                output_video = gr.Video(label="Output Preview")

        # Event handlers
        video_input.change(
            update_video_info,
            inputs=[video_input],
            outputs=[video_info, start_frame, end_frame, output_video]
        )

        # Update preview when slider changes
        preview_slider.change(
            preview_frame,
            inputs=[video_input, preview_slider],
            outputs=[frame_preview]
        )

        # Update preview slider range when video changes
        video_input.change(
            lambda v: gr.update(maximum=get_video_info(v)[1] - 1 if v else 100, value=0),
            inputs=[video_input],
            outputs=[preview_slider]
        )

        # Update output info when parameters change
        for component in [start_frame, end_frame, target_width, target_height]:
            component.change(
                calculate_output_info,
                inputs=[video_input, start_frame, end_frame, target_width, target_height],
                outputs=[output_info]
            )

        video_input.change(
            calculate_output_info,
            inputs=[video_input, start_frame, end_frame, target_width, target_height],
            outputs=[output_info]
        )

        # Process button
        process_btn.click(
            process_video,
            inputs=[video_input, start_frame, end_frame, target_width, target_height, output_name],
            outputs=[result_text, output_video]
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False, theme=gr.themes.Soft())
