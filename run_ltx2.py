#!/usr/bin/env python3
"""
Standalone LTX-2 Text/Image to Video Generation Script

This script runs the LTX-2 model independently without the full wgp.py Gradio interface.
Supports text-to-video and image-to-video generation.

Usage:
    python run_ltx2.py --prompt "Your prompt here"
    python run_ltx2.py --prompt "Your prompt" --image_start path/to/image.jpg
    python run_ltx2.py --help
"""

import os
import sys
import argparse
import torch
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set environment variables BEFORE importing anything that might use CUDA
os.environ["GRADIO_LANG"] = "en"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable TensorFlow oneDNN optimizations
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

import warnings
warnings.filterwarnings('ignore')

from mmgp import offload, safetensors2
from shared.utils import files_locator as fl
from shared.utils.utils import sanitize_file_name


def parse_args():
    parser = argparse.ArgumentParser(description="LTX-2 Text/Image to Video Generation")
    
    # Model configuration
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="ltx2_22B",
        choices=["ltx2_19B", "ltx2_22B"],
        help="LTX-2 model variant to use (default: ltx2_22B)"
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Path to transformer checkpoint (overrides auto-detection)"
    )
    parser.add_argument(
        "--gemma_path",
        type=str,
        default=None,
        help="Path to Gemma text encoder (overrides auto-detection)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp8", "int8"],
        help="Model precision (default: bf16)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt (default: empty)"
    )
    parser.add_argument(
        "--image_start",
        type=str,
        default=None,
        help="Path to starting image (optional)"
    )
    parser.add_argument(
        "--image_end",
        type=str,
        default=None,
        help="Path to ending image (optional)"
    )
    parser.add_argument(
        "--audio_guide",
        type=str,
        default=None,
        help="Path to audio file for audio-guided generation (optional)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Output video width (default: 768, must be divisible by 64)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output video height (default: 512, must be divisible by 64)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=121,
        help="Number of frames (default: 121)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frames per second (default: 24.0)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of denoising steps (default: 40 for 19B, 30 for 22B)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="CFG guidance scale (default: 4.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: random)"
    )
    parser.add_argument(
        "--video_length",
        type=float,
        default=None,
        help="Video length in seconds (alternative to num_frames)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)"
    )
    parser.add_argument(
        "--save_audio",
        action="store_true",
        help="Save generated audio as separate WAV file"
    )
    
    # Performance options
    parser.add_argument(
        "--profile",
        type=int,
        default=-1,
        choices=[-1, 0, 1, 2, 3, 4],
        help="Memory profile (-1: auto, 0-4: manual)"
    )
    parser.add_argument(
        "--attention",
        type=str,
        default=None,
        choices=["sdpa", "sage", "sage2", "flash"],
        help="Attention mechanism (default: auto)"
    )
    parser.add_argument(
        "--vae_tile_size",
        type=int,
        default=None,
        help="VAE tile size for memory efficiency (default: auto)"
    )
    parser.add_argument(
        "--sliding_window_size",
        type=int,
        default=481,
        help="Sliding window size (default: 481)"
    )
    parser.add_argument(
        "--sliding_window_overlap",
        type=int,
        default=17,
        help="Sliding window overlap (default: 17)"
    )
    parser.add_argument(
        "--vram_safety_coefficient",
        type=float,
        default=0.0,
        help="VRAM safety coefficient (0.0-0.95, default: 0.85). Higher = safer but slower"
    )
    
    return parser.parse_args()


def load_ltx2_model(model_type, transformer_path=None, gemma_path=None, dtype="bf16", profile=-1, vram_safety_coefficient=0.0):
    """Load LTX-2 model and return the pipeline."""
    
    from models.ltx2 import ltx2_handler
    
    # Get the family_handler class
    family_handler = ltx2_handler.family_handler
    
    print(f"Loading LTX-2 model: {model_type}")
    print(f"Precision: {dtype}")
    
    # Get model definition
    model_def = family_handler.query_model_def(model_type, {})
    
    # Determine dtype
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16  # Default to bf16 for LTX-2
    
    vae_dtype = torch.float32
    
    # Resolve Gemma path
    if gemma_path is None:
        gemma_folder = model_def.get("text_encoder_folder", "gemma-3-12b-it-qat-q4_0-unquantized")
        gemma_path = gemma_folder
    
    # If gemma_path is a directory, find the safetensors file
    if os.path.isdir(gemma_path):
        # Look for safetensors file in the directory
        safetensors_files = list(Path(gemma_path).glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(
                f"No safetensors file found in {gemma_path}. "
                f"Please specify the full path to the Gemma safetensors file."
            )
        if len(safetensors_files) > 1:
            # Prefer the non-quanto version if multiple exist
            for f in safetensors_files:
                if "quanto" not in f.name.lower():
                    safetensors_files = [f]
                    break
            else:
                safetensors_files = [safetensors_files[0]]
        
        gemma_checkpoint_path = str(safetensors_files[0])
        print(f"Found Gemma checkpoint: {gemma_checkpoint_path}")
    else:
        if not os.path.exists(gemma_path):
            raise FileNotFoundError(
                f"Gemma path not found: {gemma_path}. Please specify --gemma_path"
            )
        gemma_checkpoint_path = gemma_path
    
    print(f"Gemma text encoder directory: {gemma_path}")
    print(f"Gemma checkpoint file: {gemma_checkpoint_path}")
    
    # Load checkpoint paths
    from models.ltx2.ltx2_handler import _resolve_multi_file_paths
    checkpoint_paths = _resolve_multi_file_paths(model_def, model_type)
    
    # Override transformer path if provided
    if transformer_path is not None:
        checkpoint_paths["transformer"] = [transformer_path] if isinstance(transformer_path, str) else transformer_path
    elif checkpoint_paths.get("transformer") is None:
        # Try to find transformer in models directory
        try:
            transformer_path = fl.locate_file(f"{model_type}.safetensors")
            if transformer_path:
                checkpoint_paths["transformer"] = [transformer_path]
        except:
            pass
    
    # Ensure we have a transformer path
    if checkpoint_paths.get("transformer") is None:
        raise FileNotFoundError(
            "Transformer checkpoint not found. Please specify --transformer_path"
        )
    
    print(f"Transformer: {checkpoint_paths['transformer']}")
    print(f"Gemma text encoder: {gemma_path}")
    
    # Load the model
    print("Initializing LTX-2 model...")
    start_time = time.time()
    
    from models.ltx2.ltx2 import LTX2
    
    ltx2_instance = LTX2(
        model_filename=checkpoint_paths["transformer"],
        model_type=model_type,
        base_model_type=model_type,
        model_def=model_def,
        dtype=torch_dtype,
        VAE_dtype=vae_dtype,
        text_encoder_filename=gemma_checkpoint_path,
        text_encoder_filepath=gemma_path,
        checkpoint_paths=checkpoint_paths,
    )
    
    # Build the pipeline dict for offloading
    pipe = {
        "transformer": ltx2_instance.model,
        "text_encoder": ltx2_instance.text_encoder,
        "text_embedding_projection": ltx2_instance.text_embedding_projection,
        "text_embeddings_connector": ltx2_instance.text_embeddings_connector,
        "vae": ltx2_instance.video_decoder,
        "video_encoder": ltx2_instance.video_encoder,
        "audio_encoder": ltx2_instance.audio_encoder,
        "audio_decoder": ltx2_instance.audio_decoder,
        "vocoder": ltx2_instance.vocoder,
        "spatial_upsampler": ltx2_instance.spatial_upsampler,
    }
    if ltx2_instance.model2 is not None:
        pipe["transformer2"] = ltx2_instance.model2
    
    # Setup offloading
    print("Configuring memory offloading...")
    vram_coefficient = min(0.95, vram_safety_coefficient) if vram_safety_coefficient > 0 else 0.85
    offload_obj = offload.profile(
        pipe,
        profile_no=profile if profile >= 0 else 2,  # Default to profile 2
        quantizeTransformer=False,
        vram_safety_coefficient=vram_coefficient,
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    return ltx2_instance, offload_obj, model_def


def generate_video(
    ltx2_instance,
    model_def,
    prompt,
    negative_prompt="",
    image_start=None,
    image_end=None,
    audio_guide=None,
    width=768,
    height=512,
    num_frames=121,
    fps=24.0,
    num_inference_steps=None,
    guidance_scale=None,
    seed=None,
    vae_tile_size=None,
    sliding_window_size=481,
    sliding_window_overlap=17,
    audio_scale=1.0,
    attention=None,
):
    """Generate video using LTX-2 model."""
    
    # Set defaults based on model type
    if num_inference_steps is None:
        num_inference_steps = 40  # Default for 19B
    
    if guidance_scale is None:
        guidance_scale = 4.0
    
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    
    print(f"\nGeneration Parameters:")
    print(f"  Prompt: {prompt}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames: {num_frames}")
    print(f"  FPS: {fps}")
    print(f"  Steps: {num_inference_steps}")
    print(f"  Guidance Scale: {guidance_scale}")
    print(f"  Seed: {seed}")
    
    # Load images if provided
    if image_start is not None:
        print(f"  Start Image: {image_start}")
        image_start = Image.open(image_start).convert("RGB")
    
    if image_end is not None:
        print(f"  End Image: {image_end}")
        image_end = Image.open(image_end).convert("RGB")
    
    # Load audio if provided
    input_waveform = None
    input_waveform_sample_rate = None
    if audio_guide is not None:
        print(f"  Audio Guide: {audio_guide}")
        import soundfile as sf
        import librosa
        
        audio_data, sr = librosa.load(audio_guide, sr=None, mono=False)
        if audio_data.ndim == 1:
            audio_data = audio_data[np.newaxis, :]
        input_waveform = audio_data
        input_waveform_sample_rate = sr
    
    # Setup callback for progress tracking
    def callback(step, preview_latents, is_init_call=False, **kwargs):
        # Handle initialization call vs step calls
        if is_init_call or step < 0:
            # Initialization call
            override_steps = kwargs.get('override_num_inference_steps')
            if override_steps is not None:
                if hasattr(override_steps, 'item'):
                    total_steps = int(override_steps.item())
                else:
                    total_steps = int(override_steps)
                print(f"\n  Starting pass {kwargs.get('pass_no', 1)} ({total_steps} steps)...")
            else:
                print(f"\n  Starting pass {kwargs.get('pass_no', 1)}...")
            return
        
        # Step call - use the num_inference_steps we know about
        total_steps = num_inference_steps
        progress = (step + 1) / total_steps * 100
        print(f"\r  Step {step+1}/{total_steps} ({progress:.1f}%)", end="", flush=True)
    
    # Setup attention mechanism
    from shared.attention import get_supported_attention_modes
    supported_attention_modes = get_supported_attention_modes()
    
    # Choose attention mode: prefer flash/sage if available, fallback to sdpa
    if attention is not None:
        attn_mode = attention
    elif "flash" in supported_attention_modes:
        attn_mode = "flash"
    elif "sage2" in supported_attention_modes:
        attn_mode = "sage2"
    elif "sage" in supported_attention_modes:
        attn_mode = "sage"
    else:
        attn_mode = "sdpa"
    
    if attn_mode not in supported_attention_modes:
        print(f"Warning: Attention mode '{attn_mode}' not supported, falling back to 'sdpa'")
        attn_mode = "sdpa"
    
    print(f"Using attention mode: {attn_mode}")
    offload.shared_state["_attention"] = attn_mode
    
    # Call generate
    print("\nStarting generation...")
    start_time = time.time()
    
    result = ltx2_instance.generate(
        input_prompt=prompt,
        n_prompt=negative_prompt if negative_prompt else None,
        image_start=image_start,
        image_end=image_end,
        sampling_steps=num_inference_steps,
        guide_scale=guidance_scale,
        alt_guide_scale=1.0,
        input_video=None,
        prefix_frames_count=0,
        frame_num=num_frames,
        height=height,
        width=width,
        fps=fps,
        seed=seed,
        callback=callback,
        VAE_tile_size=vae_tile_size,
        input_waveform=input_waveform,
        input_waveform_sample_rate=input_waveform_sample_rate,
        audio_scale=audio_scale,
        sliding_window_size=sliding_window_size,
        sliding_window_overlap=sliding_window_overlap,
    )
    
    gen_time = time.time() - start_time
    print(f"\n\nGeneration completed in {gen_time:.2f} seconds")
    
    return result, seed


def save_results(result, output_dir, output_filename=None, save_audio=False, seed=None, prompt=None, fps=24.0):
    """Save generated video and audio using the same save_video function as wgp.py."""

    from shared.utils.audio_video import (
        save_video as save_video_file,
        combine_and_concatenate_video_with_audio_tracks,
        write_wav_file,
    )

    os.makedirs(output_dir, exist_ok=True)

    if result is None:
        print("ERROR: Generation returned None")
        return None, None

    # Extract video tensor
    video_tensor = result.get("x")
    if video_tensor is None:
        print("ERROR: No video tensor in result")
        return None, None

    print(f"Video tensor shape (raw): {video_tensor.shape}")
    print(f"Video tensor dtype: {video_tensor.dtype}")

    # LTX-2 output format: [C, F, H, W]
    # save_video expects [B, C, F, H, W] (5D tensor)
    if torch.is_tensor(video_tensor):
        if video_tensor.ndim == 4:
            # Add batch dimension: [C, F, H, W] -> [1, C, F, H, W]
            video_tensor = video_tensor.unsqueeze(0)
            print(f"Video tensor shape (after unsqueeze): {video_tensor.shape}")

    # Generate filename
    if output_filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if prompt:
            prompt_preview = sanitize_file_name(prompt[:50])
            output_filename = f"ltx2_{timestamp}_{prompt_preview}_seed{seed}.mp4"
        else:
            output_filename = f"ltx2_{timestamp}_seed{seed}.mp4"

    if not output_filename.endswith(".mp4"):
        output_filename += ".mp4"

    output_path = os.path.join(output_dir, output_filename)

    # Check if result contains audio
    audio_data = result.get("audio")
    audio_sampling_rate = result.get("audio_sampling_rate", 44100)
    has_audio = audio_data is not None

    if has_audio:
        # Save video without audio first (temp file)
        temp_path = output_path.rsplit('.', 1)[0] + "_tmp.mp4"
        print(f"Saving video (temp): {temp_path}")
        try:
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
            print(f"Video saved successfully!")
        except Exception as e:
            print(f"ERROR saving video: {e}")
            import traceback
            traceback.print_exc()
            return None, None

        # Save audio to temp wav file
        audio_temp_path = output_path.rsplit('.', 1)[0] + "_audio.wav"
        print(f"Saving audio (temp): {audio_temp_path}")

        audio_np = audio_data
        if audio_np.ndim == 2:
            if audio_np.shape[0] in (1, 2) and audio_np.shape[1] > audio_np.shape[0]:
                audio_np = audio_np.T

        write_wav_file(audio_temp_path, audio_np, audio_sampling_rate)

        # Mux audio into video
        print(f"  Muxing audio into video...")
        try:
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
            print(f"Video with audio saved: {output_path}")
        except Exception as e:
            print(f"ERROR muxing audio: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to video-only
            output_path = temp_path

        # Clean up temp files
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        if Path(audio_temp_path).exists():
            Path(audio_temp_path).unlink()

    else:
        # Save video without audio
        print(f"Saving video to: {output_path}")
        try:
            saved_path = save_video_file(
                tensor=video_tensor,
                save_file=output_path,
                fps=fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
                codec_type='libx264_8',
                container='mp4'
            )
            print(f"Video saved successfully!")
            output_path = saved_path
        except Exception as e:
            print(f"ERROR saving video: {e}")
            import traceback
            traceback.print_exc()
            output_path = None

    # Save separate audio file if requested (in addition to muxed audio)
    audio_path = None
    if save_audio and result.get("audio") is not None:
        audio_np = result["audio"]
        sample_rate = result.get("audio_sampling_rate", 44100)

        if audio_np.ndim == 2:
            if audio_np.shape[0] in (1, 2) and audio_np.shape[1] > audio_np.shape[0]:
                audio_np = audio_np.T

        audio_filename = output_filename.replace(".mp4", "_audio.wav") if output_filename else "audio.wav"
        audio_path = os.path.join(output_dir, audio_filename)

        import soundfile as sf
        sf.write(audio_path, audio_np.T, sample_rate)
        print(f"Audio saved to: {audio_path}")

    return output_path, audio_path


def main():
    args = parse_args()
    
    # Adjust dimensions to be divisible by 64
    width = args.width
    height = args.height
    if width % 64 != 0:
        width = int(np.ceil(width / 64) * 64)
        print(f"Adjusted width to {width} (must be divisible by 64)")
    if height % 64 != 0:
        height = int(np.ceil(height / 64) * 64)
        print(f"Adjusted height to {height} (must be divisible by 64)")
    
    # Calculate num_frames from video_length if provided
    num_frames = args.num_frames
    if args.video_length is not None:
        num_frames = int(args.video_length * args.fps)
        # Ensure it follows LTX-2 frame constraints (17 + 8*n)
        if num_frames < 17:
            num_frames = 17
        num_frames = 17 + ((num_frames - 17) // 8) * 8
        print(f"Calculated num_frames from video length: {num_frames}")
    
    # Load model
    ltx2_instance, offload_obj, model_def = load_ltx2_model(
        model_type=args.model_type,
        transformer_path=args.transformer_path,
        gemma_path=args.gemma_path,
        dtype=args.dtype,
        profile=args.profile,
        vram_safety_coefficient=args.vram_safety_coefficient,
    )
    
    # Generate video
    result, seed = generate_video(
        ltx2_instance=ltx2_instance,
        model_def=model_def,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image_start=args.image_start,
        image_end=args.image_end,
        audio_guide=args.audio_guide,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        vae_tile_size=args.vae_tile_size,
        sliding_window_size=args.sliding_window_size,
        sliding_window_overlap=args.sliding_window_overlap,
        attention=args.attention,
    )
    
    # Save results
    video_path, audio_path = save_results(
        result=result,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        save_audio=args.save_audio,
        seed=seed,
        prompt=args.prompt,
    )
    
    if video_path:
        print(f"\n✓ Generation complete!")
        print(f"  Video: {video_path}")
        if audio_path:
            print(f"  Audio: {audio_path}")
    else:
        print("\n✗ Generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
