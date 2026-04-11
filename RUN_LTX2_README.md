# LTX-2 Standalone Script

This script allows you to run the LTX-2 text/image-to-video model independently from the main Gradio interface.

## Features

- Text-to-video generation
- Image-to-video generation (start/end frames)
- Audio-guided video generation
- Support for both LTX-2 19B and 22B variants
- Flexible precision modes (bf16, fp16, fp8, int8)
- Memory-efficient offloading profiles

## Prerequisites

Make sure you have installed all dependencies from the main project:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Text-to-Video

```bash
python run_ltx2.py --prompt "A beautiful sunset over the ocean with waves crashing on the shore"
```

### Image-to-Video

```bash
python run_ltx2.py --prompt "The person starts walking and looking around" --image_start path/to/image.jpg
```

### With Custom Parameters

```bash
python run_ltx2.py \
  --prompt "A spaceship landing on Mars" \
  --width 1024 \
  --height 576 \
  --num_frames 121 \
  --num_inference_steps 40 \
  --guidance_scale 4.0 \
  --seed 42 \
  --output_dir my_videos
```

### Audio-Guided Generation

```bash
python run_ltx2.py \
  --prompt "A musical performance" \
  --audio_guide path/to/audio.wav \
  --save_audio
```

### Using LTX-2 22B Model

```bash
python run_ltx2.py \
  --model_type ltx2_22B \
  --prompt "Your prompt here"
```

## Command-Line Arguments

### Model Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_type` | Model variant: `ltx2_19B` or `ltx2_22B` | `ltx2_19B` |
| `--transformer_path` | Path to transformer checkpoint | Auto-detect |
| `--gemma_path` | Path to Gemma text encoder | Auto-detect |
| `--dtype` | Model precision: `bf16`, `fp16`, `fp8`, `int8` | `bf16` |

### Generation Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--prompt` | Text prompt (required) | - |
| `--negative_prompt` | Negative prompt | Empty |
| `--image_start` | Path to starting image | None |
| `--image_end` | Path to ending image | None |
| `--audio_guide` | Path to audio file for audio-guided generation | None |
| `--width` | Output width (divisible by 64) | 768 |
| `--height` | Output height (divisible by 64) | 512 |
| `--num_frames` | Number of frames | 121 |
| `--fps` | Frames per second | 24.0 |
| `--num_inference_steps` | Denoising steps | 40 (19B) / 30 (22B) |
| `--guidance_scale` | CFG scale | 4.0 |
| `--seed` | Random seed | Random |
| `--video_length` | Video length in seconds (alternative to num_frames) | None |

### Output Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--output_dir` | Output directory | `output` |
| `--output_filename` | Output filename | Auto-generated |
| `--save_audio` | Save generated audio as WAV | False |

### Performance Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--profile` | Memory profile (-1: auto, 0-4: manual) | -1 |
| `--attention` | Attention mechanism: `sdpa`, `sage`, `sage2`, `flash` | Auto |
| `--vae_tile_size` | VAE tile size for memory efficiency | Auto |
| `--sliding_window_size` | Sliding window size | 481 |
| `--sliding_window_overlap` | Sliding window overlap | 17 |
| `--vram_safety_coefficient` | VRAM safety coefficient | 1.0 |

## Examples

### High-Quality Short Clip

```bash
python run_ltx2.py \
  --prompt "Close-up of a cat eating food, cinematic lighting" \
  --width 768 \
  --height 512 \
  --num_frames 49 \
  --num_inference_steps 50 \
  --guidance_scale 4.5 \
  --seed 12345
```

### Long Video with Sliding Window

```bash
python run_ltx2.py \
  --prompt "A journey through a futuristic city" \
  --width 1024 \
  --height 576 \
  --num_frames 241 \
  --sliding_window_size 481 \
  --sliding_window_overlap 17
```

### Image Animation

```bash
python run_ltx2.py \
  --prompt "The landscape comes alive with moving clouds and flowing water" \
  --image_start landscape.jpg \
  --width 1024 \
  --height 768 \
  --num_frames 121 \
  --seed 42
```

### Low VRAM Mode

```bash
python run_ltx2.py \
  --prompt "Your prompt" \
  --profile 3 \
  --vae_tile_size 256 \
  --width 512 \
  --height 512 \
  --num_frames 49
```

## Memory Profiles

The memory profiles control how the model is offloaded between CPU and GPU:

- **Profile 0**: Maximum VRAM usage, fastest
- **Profile 1**: High VRAM usage, very fast
- **Profile 2**: Balanced (default)
- **Profile 3**: Low VRAM usage, slower
- **Profile 4**: Minimum VRAM usage, slowest

For GPUs with < 8GB VRAM, use profile 3 or 4.

## Notes

1. **Dimensions**: Width and height must be divisible by 64. The script will auto-adjust if needed.

2. **Frame Count**: LTX-2 works best with frame counts following the formula: `17 + 8*n`. The script uses 121 frames by default (17 + 8*13).

3. **Audio Generation**: LTX-2 can generate audio synchronized with video. Use `--save_audio` to save it separately.

4. **Model Files**: The script expects model files to be in the standard locations as defined by the main project. Use `--transformer_path` and `--gemma_path` to specify custom locations.

## Troubleshooting

### Out of Memory Error

- Use a higher memory profile: `--profile 3` or `--profile 4`
- Reduce resolution: `--width 512 --height 512`
- Reduce frame count: `--num_frames 49`
- Enable VAE tiling: `--vae_tile_size 256`

### Model Not Found Error

Make sure you have downloaded the required model files. The script will auto-detect files in the standard locations, or you can specify paths manually:

```bash
python run_ltx2.py \
  --transformer_path /path/to/transformer.safetensors \
  --gemma_path /path/to/gemma_folder \
  --prompt "Your prompt"
```
