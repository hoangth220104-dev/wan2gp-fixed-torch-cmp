"""Video generation logic"""

import os
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from PIL import Image

from shared.utils.utils import sanitize_file_name
from shared.utils.utils import process_images_multithread, get_default_workers, resample
from shared.utils.audio_video import (
    save_video as save_video_file,
    combine_and_concatenate_video_with_audio_tracks,
    write_wav_file,
)
from shared.attention import get_supported_attention_modes
from mmgp import offload

from ltx2_server.model_manager import ModelManager


def _load_source_video(
    input_video_path: str,
    width: int,
    height: int,
    fps: float,
    block_size: int = 16,
) -> Optional[torch.Tensor]:
    """Load and resize a source video like the base app's prefix-video path."""
    try:
        import decord

        decord.bridge.set_bridge("torch")
        reader = decord.VideoReader(input_video_path)
        source_fps = round(reader.get_avg_fps())
        frame_indices = resample(
            source_fps,
            len(reader),
            max_target_frames_count=0,
            target_fps=fps,
            start_target_frame=0,
        )
        if len(frame_indices) == 0:
            return None

        frames = reader.get_batch(frame_indices)
        if frames.ndim != 4 or frames.shape[0] == 0:
            return None

        target_height = (int(height) // block_size) * block_size
        target_width = (int(width) // block_size) * block_size

        def resize_frame(frame):
            arr = frame.cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr).resize(
                (target_width, target_height),
                resample=Image.Resampling.LANCZOS,
            )
            return torch.from_numpy(np.array(img, dtype=np.uint8))

        frames_list = [frame for frame in frames]
        frames_list = process_images_multithread(
            resize_frame,
            frames_list,
            "upsample",
            wrap_in_list=False,
            max_workers=get_default_workers(),
            in_place=True,
        )
        prefix_video = torch.stack(frames_list).permute(3, 0, 1, 2)
        return prefix_video.float().div_(127.5).sub_(1.0)
    except Exception as e:
        print(f"  Error loading video: {e}")
        return None


def generate_video(
    model_manager: ModelManager,
    prompt: str,
    negative_prompt: str = "",
    image_start_path: Optional[str] = None,
    image_end_path: Optional[str] = None,
    audio_guide_path: Optional[str] = None,
    input_video_path: Optional[str] = None,
    video_prompt_type: str = "",
    width: int = 768,
    height: int = 512,
    num_frames: int = 121,
    fps: float = 24.0,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    input_video_strength: float = 1.0,
    denoising_strength: float = 1.0,
    prefix_frames_count: int = 0,
    attention: Optional[str] = None,
    sliding_window_size: int = 481,
    sliding_window_overlap: int = 17,
    task_id: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> Tuple[Dict[str, Any], int, float]:
    """
    Generate video using LTX-2 model.
    
    Returns:
        (result_dict, seed, generation_time_seconds)
    """
    
    # Set defaults
    if num_inference_steps is None:
        num_inference_steps = model_manager.model_def.get("default_steps", 40)
    
    if guidance_scale is None:
        guidance_scale = 4.0
    
    if seed is None:
        seed = int(np.random.randint(0, 2**32 - 1))
    
    print(f"\n[{task_id}] Generation:")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames: {num_frames}")
    print(f"  Steps: {num_inference_steps}")
    print(f"  Guidance: {guidance_scale}")
    print(f"  Seed: {seed}")
    print(f"  Image Start: {image_start_path if image_start_path else 'None'}")
    print(f"  Image End: {image_end_path if image_end_path else 'None'}")
    print(f"  Input Video: {input_video_path if input_video_path else 'None'}")
    print(f"  Video Prompt Type: '{video_prompt_type}' (len={len(video_prompt_type)})")
    print(f"  Audio Guide: {audio_guide_path if audio_guide_path else 'None'}")
    print(f"  Video Strength: {input_video_strength}")
    print(f"  Denoising Strength: {denoising_strength}")
    print(f"  Prefix Frames: {prefix_frames_count}")

    # Load images if provided
    image_start = Image.open(image_start_path).convert("RGB") if image_start_path else None
    image_end = Image.open(image_end_path).convert("RGB") if image_end_path else None

    # Load input video if provided
    input_video = None
    if input_video_path:
        print(f"  Loading input video from: {input_video_path}")
        input_video = _load_source_video(input_video_path, width=width, height=height, fps=fps)
        if input_video is not None:
            frame_count = int(input_video.shape[1])
            video_h, video_w = int(input_video.shape[2]), int(input_video.shape[3])
            print(f"  Video loaded: {frame_count} frames, shape={input_video.shape}")
            print(f"  Video resolution: {video_w}x{video_h}, Target: {width}x{height}")
            if prefix_frames_count == 0:
                prefix_frames_count = frame_count
                print(f"  Auto-set prefix_frames_count to {prefix_frames_count} (all preprocessed frames)")
        else:
            print("  Warning: No frames extracted from video")
    
    if input_video is not None:
        print(f"  DEBUG: input_video shape = {input_video.shape}, dtype = {input_video.dtype}")
        print(f"  DEBUG: input_video value_range = [{input_video.min():.2f}, {input_video.max():.2f}]")
        print(f"  DEBUG: FPS: {fps}, num_frames: {num_frames}, expected_duration: {num_frames/fps:.1f}s")
    # Load audio if provided
    input_waveform = None
    input_waveform_sample_rate = None

    if audio_guide_path:
        import soundfile as sf
        import librosa
        print(f"  Loading audio from: {audio_guide_path}")
        audio_data, sr = librosa.load(audio_guide_path, sr=None, mono=False)
        if audio_data.ndim == 1:
            audio_data = audio_data[np.newaxis, :]
        input_waveform = audio_data
        input_waveform_sample_rate = sr
        print(f"  Audio loaded: shape={audio_data.shape}, sr={sr}")
    
    # Setup progress callback
    def callback(step, preview_latents, is_init_call=False, **kwargs):
        if is_init_call or step < 0:
            override_steps = kwargs.get('override_num_inference_steps')
            if override_steps is not None:
                total = int(override_steps.item()) if hasattr(override_steps, 'item') else int(override_steps)
                print(f"  Starting pass {kwargs.get('pass_no', 1)} ({total} steps)...")
            return
        
        # Update progress
        if progress_callback:
            progress = (step + 1) / num_inference_steps * 100
            progress_callback(step + 1, num_inference_steps, progress)
        
        print(f"\r  Step {step+1}/{num_inference_steps} ({(step + 1) / num_inference_steps * 100:.1f}%)", end="", flush=True)
    
    # Setup attention
    supported_modes = get_supported_attention_modes()
    attn_mode = attention or _select_attention(supported_modes)
    
    if attn_mode not in supported_modes:
        print(f"  Warning: '{attn_mode}' not supported, using 'sdpa'")
        attn_mode = "sdpa"
    
    print(f"  Attention: {attn_mode}")
    offload.shared_state["_attention"] = attn_mode
    
    # Generate
    print("\n  Generating...")
    print(f"  Passing to model:")
    print(f"    image_start: {'Yes (PIL Image)' if image_start is not None else 'None'}")
    print(f"    image_end: {'Yes (PIL Image)' if image_end is not None else 'None'}")
    print(f"    input_video: {'Yes' if input_video is not None else 'None'}")
    print(f"    input_waveform: {'Yes' if input_waveform is not None else 'None'}")
    print(f"    input_waveform_sample_rate: {input_waveform_sample_rate}")
    print(f"    input_video_strength: {input_video_strength}")
    print(f"    denoising_strength: {denoising_strength}")
    print(f"    prefix_frames_count: {prefix_frames_count}")
    print(f"    video_prompt_type to model: '{video_prompt_type}'")
    print(f"    'G' in video_prompt_type: {'G' in video_prompt_type}")
    print(f"    'V' in video_prompt_type: {'V' in video_prompt_type}")

    start_time = time.time()

    result = model_manager.generate(
        input_prompt=prompt,
        n_prompt=negative_prompt if negative_prompt else None,
        image_start=image_start,
        image_end=image_end,
        sampling_steps=num_inference_steps,
        guide_scale=guidance_scale,
        alt_guide_scale=1.0,
        input_video=input_video,
        prefix_frames_count=prefix_frames_count,
        frame_num=num_frames,
        height=height,
        width=width,
        fps=fps,
        seed=seed,
        callback=callback,
        VAE_tile_size=None,
        input_waveform=input_waveform,
        input_waveform_sample_rate=input_waveform_sample_rate,
        audio_scale=1.0,
        video_prompt_type=video_prompt_type,
        input_video_strength=input_video_strength,
        denoising_strength=denoising_strength,
    )
    
    gen_time = time.time() - start_time
    print(f"\n  Generation completed in {gen_time:.2f}s")
    
    return result, seed, gen_time


def save_video_result(
    result: Dict[str, Any],
    seed: int,
    output_dir: str = "output",
    prompt: Optional[str] = None,
    fps: float = 24.0,
) -> Tuple[str, str]:
    """
    Save generated video to disk, muxing audio if present.

    Returns:
        (absolute_path, filename)
    """

    os.makedirs(output_dir, exist_ok=True)

    if result is None:
        raise RuntimeError("Generation returned None")

    video_tensor = result.get("x")
    if video_tensor is None:
        raise RuntimeError("No video tensor in result")

    # Add batch dimension if needed [C,F,H,W] -> [1,C,F,H,W]
    if torch.is_tensor(video_tensor) and video_tensor.ndim == 4:
        video_tensor = video_tensor.unsqueeze(0)

    # Generate filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if prompt:
        prompt_preview = sanitize_file_name(prompt[:50])
        filename = f"{timestamp}_{prompt_preview}_seed{seed}.mp4"
    else:
        filename = f"{timestamp}_seed{seed}.mp4"

    if not filename.endswith(".mp4"):
        filename += ".mp4"

    output_path = os.path.join(output_dir, filename)

    # Check if result contains audio
    audio_data = result.get("audio")
    audio_sampling_rate = result.get("audio_sampling_rate", 44100)
    has_audio = audio_data is not None

    if has_audio:
        # Save video without audio first (temp file)
        print(f"  Saving video (temp): {output_path}")
        temp_path = output_path.rsplit('.', 1)[0] + "_tmp.mp4"
        save_video_file(
            tensor=video_tensor,
            save_file=temp_path,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
            codec_type='libx264_8',
            container='mp4'
        )

        # Save audio to temp wav file
        audio_temp_path = output_path.rsplit('.', 1)[0] + "_audio.wav"
        print(f"  Saving audio (temp): {audio_temp_path}")

        # Handle audio data format
        audio_np = audio_data
        if audio_np.ndim == 2:
            # If shape is (1, T) or (2, T), transpose to (T, 1) or (T, 2)
            if audio_np.shape[0] in (1, 2) and audio_np.shape[1] > audio_np.shape[0]:
                audio_np = audio_np.T

        write_wav_file(audio_temp_path, audio_np, audio_sampling_rate)

        # Mux audio into video
        print(f"  Muxing audio into video...")
        combine_and_concatenate_video_with_audio_tracks(
            save_path_tmp=output_path,
            video_path=temp_path,
            source_audio_tracks=[],
            new_audio_tracks=[audio_temp_path],
            source_audio_duration=0,
            audio_sampling_rate=audio_sampling_rate,
            new_audio_from_start=True,
            source_audio_metadata=None,
            audio_codec_key="aac_128",
            verbose=False,
        )

        # Clean up temp files
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        if Path(audio_temp_path).exists():
            Path(audio_temp_path).unlink()

        print(f"  Video with audio saved: {output_path}")
    else:
        # Save video without audio
        print(f"  Saving: {output_path}")
        output_path = save_video_file(
            tensor=video_tensor,
            save_file=output_path,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
            codec_type='libx264_8',
            container='mp4'
        )

    return output_path, filename


def _select_attention(supported_modes: list) -> str:
    """Select best available attention mode"""
    for mode in ["flash", "sage2", "sage", "sdpa"]:
        if mode in supported_modes:
            return mode
    return "sdpa"
