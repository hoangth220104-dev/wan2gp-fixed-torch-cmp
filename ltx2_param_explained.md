# LTX-2 Parameters Explained

This document explains all parameters available in the LTX-2 video generation system and their effects on video quality.

## Table of Contents
- [Core Generation Parameters](#core-generation-parameters)
- [Image/Video Conditioning](#imagevideo-conditioning)
- [Audio Parameters](#audio-parameters)
- [Advanced Guidance Parameters](#advanced-guidance-parameters)
- [Self-Refiner Parameters](#self-refiner-parameters)
- [Technical/System Parameters](#technicalsystem-parameters)
- [Control Video Types](#control-video-types)

---

## Core Generation Parameters

### `prompt`
- **Description**: Text description of the video to generate
- **Impact**: Primary determinant of video content. Detailed, specific prompts generally produce better results than vague ones
- **Quality Effect**: Higher detail in prompt → Higher content relevance

### `negative_prompt`
- **Description**: Things to AVOID in the generated video
- **Default**: Empty (uses built-in default negative prompt)
- **Impact**: Suppresses unwanted elements (artifacts, specific objects, styles)
- **Quality Effect**: Helps reduce artifacts and unwanted content when properly specified

### `width` / `height`
- **Description**: Output video resolution in pixels
- **Default**: 768 x 512
- **Constraints**: Must be divisible by 64
- **Impact**:
  - Higher resolution = More detail but more VRAM and slower generation
  - LTX-2 internally processes at half resolution (latent space)
  - Final output is upscaled 2x by spatial upsampler
- **Quality Effect**:
  - Moderate resolution (768x512 to 1024x768) often optimal for quality/performance
  - Very high resolution may lose fine details due to tiling

### `num_frames`
- **Description**: Number of frames in the output video
- **Default**: 121
- **Minimum**: 17 frames
- **Constraints**: Should be divisible by 8 for optimal performance
- **Impact**: Longer videos = more memory and time
- **Quality Effect**: More frames allow longer scenes but individual frame quality may vary across longer sequences

### `fps` (frames per second)
- **Description**: Playback speed of the video
- **Default**: 24.0
- **Impact**: Determines how smooth the video appears
- **Quality Effect**: 24fps is standard cinema; higher fps = smoother but may show less quality per frame

### `num_inference_steps`
- **Description**: Number of denoising iterations
- **Default**:
  - 40 for ltx2_19B model
  - 30 for ltx2_22B model
- **Impact**:
  - More steps = cleaner, more refined output
  - Diminishing returns beyond ~40-50 steps
  - Each step adds ~2-3% generation time
- **Quality Effect**:
  - Low (10-20): Fast but noisy/artifact-prone
  - Medium (25-35): Good balance (recommended for most use cases)
  - High (40-50): Maximum quality, longer generation time

### `guidance_scale` (CFG Scale)
- **Description**: Classifier-Free Guidance strength
- **Default**: 4.0 (ltx2_22B), 3.0 (ltx2_19B)
- **Range**: Typically 1.0 - 10.0
- **Impact**:
  - Higher = Stronger adherence to prompt, more contrast
  - Lower = More creative/varied, less prompt accuracy
- **Quality Effect**:
  - Too low (1-2): Video may not follow prompt well
  - Optimal (3-5): Good prompt following with natural look
  - Too high (7+): Over-saturated, overly contrasted, artificial look

### `seed`
- **Description**: Random seed for reproducibility
- **Default**: Random (0-2^32)
- **Impact**: Same seed + same parameters = identical output
- **Quality Effect**: Different seeds produce different variations; some seeds may work better for specific prompts

---

## Image/Video Conditioning

### `image_start`
- **Description**: Starting image/frame for the video
- **Format**: PIL Image or file path
- **Impact**: Sets the initial frame; generated video will start from this image
- **Quality Effect**: Higher quality input image → better starting frame. The model encodes this into latent space.

### `image_end`
- **Description**: Ending image/frame for the video
- **Format**: PIL Image or file path
- **Impact**: Guides the video to end at this image (interpolation/interpolation)
- **Quality Effect**: Works best with similar lighting/composition to `image_start`. Strong guidance creates smoother transitions.

### `input_video` / `input_video_strength`
- **Description**: Source video to continue from/modify
- **Range**: 0.0 - 1.0 (for strength)
- **Default Strength**: 1.0
- **Impact**:
  - Strength 1.0: Full influence from source
  - Lower strength: More deviation from source, more creative freedom
- **Quality Effect**:
  - High strength: Preserves source style/motion closely
  - Low strength: Model has more freedom but may drift from source

### `prefix_frames_count`
- **Description**: Number of frames to use from input video
- **Default**: 0
- **Impact**: Specifies how many frames of the input video to maintain

---

## Audio Parameters

### `audio_guide` / `audio_guide_path`
- **Description**: Audio file to guide video generation
- **Format**: WAV, MP3, or other audio formats
- **Impact**: Motion in video correlates with audio features (volume, rhythm, etc.)
- **Quality Effect**: Synchronizes visual motion with audio beats/intensity

### `audio_scale` / `input_waveform_sample_rate`
- **Description**: Audio influence strength and sample rate
- **Default Scale**: 1.0
- **Default Sample Rate**: Auto-detected from audio file
- **Impact**:
  - Higher audio_scale = stronger correlation between audio and video motion
  - Sample rate affects audio feature extraction quality
- **Quality Effect**: Proper audio sync creates more immersive, natural-looking videos

### `audio_cfg_guidance_scale`
- **Description**: Guidance scale specifically for audio conditioning
- **Default**: Same as `guidance_scale`
- **Impact**: Controls how strongly audio influences the generation

### `video_prompt_type`
- **Description**: Type of audio-guided generation
- **Options**:
  - `""`: No audio guidance (default)
  - `"A"`: Audio-guided video generation
  - `"K"`: Control video + audio track (distilled pipeline only)

---

## Advanced Guidance Parameters

### `alt_guidance_scale`
- **Description**: Modality guidance scale for multi-modal generation
- **Default**: 1.0
- **Impact**: Scales the auxiliary guidance signal
- **Quality Effect**: Higher values increase contrast between video/audio modalities

### `alt_scale`
- **Description**: Guidance rescaling factor
- **Default**: 0.0 (two-stage), 0.7 (dev settings)
- **Range**: 0.0 - 1.0
- **Impact**: Rescales the guidance to prevent over-sharpening
- **Quality Effect**: Prevents over-saturated artifacts when properly tuned

### `cfg_star_switch`
- **Description**: Enable CFG* (star) rescaling guidance
- **Options**:
  - `0`: Off (default for distilled)
  - `1`: On
- **Impact**: Alternative guidance method that rescales predictions
- **Quality Effect**: Can improve prompt adherence and reduce artifacts

### `apg_switch`
- **Description**: Enable Adaptive Projected Guidance
- **Options**:
  - `0`: Off (default)
  - `1`: On
- **Impact**: Adaptively adjusts guidance during denoising
- **Quality Effect**: Can improve temporal consistency

---

## Perturbation Parameters

### `perturbation_switch`
- **Description**: Perturbation strategy for efficiency
- **Options**:
  - `0`: Off (default)
  - `1`: Skip Layer Guidance
  - `2`: Skip Self Attention
- **Impact**:
  - 1: Skips guidance computation on specific layers (faster)
  - 2: Skips self-attention on certain layers (different quality trade-off)
- **Quality Effect**: Trade-off between speed and quality

### `perturbation_layers`
- **Description**: Which layers to apply perturbation to
- **Default**:
  - 28 for ltx2_22B
  - 29 for ltx2_19B
- **Range**: 0-48
- **Impact**: Specifies layer indices for perturbation
- **Quality Effect**: Affects which aspects of the video are optimized

### `perturbation_start` / `perturbation_end`
- **Description**: Timing range for perturbation (as percentage of generation)
- **Default**: 0.0 - 1.0 (full range)
- **Impact**: When in the generation process perturbation is active
- **Quality Effect**: Affects how early/late the optimization kicks in

---

## Self-Refiner Parameters

The Self-Refiner is a quality enhancement module that refines uncertain regions.

### `self_refiner_setting`
- **Description**: Enable self-refiner
- **Options**:
  - `0`: Off
  - `1`: On
- **Default**: 1 (enabled)

### `self_refiner_plan`
- **Description**: Refinement plan as "stage:steps" pairs
- **Default**: "2-8:3" (stage 2, refine with 3 iterations)
- **Format**: "S-T:steps,S-T:steps,..."
- **Impact**: Defines which stages and how aggressively to refine
- **Quality Effect**: More refinement iterations = cleaner output but slower

### `self_refiner_f_uncertainty`
- **Description**: Uncertainty threshold for refinement
- **Default**: 0.1
- **Range**: 0.0 - 1.0
- **Impact**:
  - Lower = More regions refined (aggressive)
  - Higher = Only high-uncertainty regions refined
- **Quality Effect**: Balances refinement thoroughness vs artifacts

### `self_refiner_certain_percentage`
- **Description**: Percentage of pixels considered "certain"
- **Default**: 0.999
- **Range**: 0.0 - 1.0
- **Impact**: Higher = Less aggressive refinement
- **Quality Effect**: Very high values may skip needed refinement

---

## Technical/System Parameters

### `sliding_window_size`
- **Description**: Number of frames processed in sliding window
- **Default**: 481
- **Range**: 5 - 501 (must be odd)
- **Impact**: Affects temporal consistency and memory usage
- **Quality Effect**: Larger windows = better temporal coherence but more memory

### `sliding_window_overlap`
- **Description**: Overlap between adjacent windows
- **Default**: 17
- **Range**: 1 - 97
- **Impact**: Blending between window segments
- **Quality Effect**: More overlap = smoother transitions, but slower

### `VAE_tile_size`
- **Description**: Tile size for VAE operations
- **Default**: Auto-calculated based on VRAM
- **Impact**: Larger tiles = faster but more memory
- **Quality Effect**: Too small may cause seams; too large may OOM

### `attention`
- **Description**: Attention mechanism to use
- **Options**: `flash`, `sage2`, `sage`, `sdpa` (default fallback)
- **Impact**:
  - `flash`: Fastest if hardware supports it
  - `sage2`: SageAttention with improved precision
  - `sdpa`: Standard PyTorch attention
- **Quality Effect**: `sdpa` is most accurate; others trade some precision for speed

### `tiling_config`
- **Description**: Spatial/temporal tiling for memory management
- **Auto-calculated** from VAE_tile_size and fps
- **Impact**: Breaks video into tiles for processing
- **Quality Effect**: Proper tiling prevents OOM; poor tiling causes artifacts

---

## Control Video Types

### `video_prompt_type` Letters

| Letter | Name | Description | Pipeline |
|--------|------|-------------|----------|
| `V` | Continue Video | Continue from input video | Both |
| `L` | Continue Last | Continue last frame | Both |
| `G` | Guided | Use guided mode (enables denoising_strength) | Both |
| `P` | Pose Transfer | Transfer human motion | Distilled |
| `D` | Depth Transfer | Transfer depth map | Distilled |
| `E` | Canny Transfer | Transfer edge detection | Distilled |
| `K` | Keyframe Injection | Inject keyframes | Both |

### `masking_strength`
- **Description**: Strength of mask-based editing
- **Range**: 0.0 - 1.0
- **Impact**: How much to preserve masked regions
- **Quality Effect**: Higher = better preservation of masked content

### `denoising_strength`
- **Description**: How much to denoise/modify
- **Range**: 0.0 - 1.0
- **Default**: 1.0
- **Impact**:
  - 1.0 = Full generation
  - Lower = More preservation of input characteristics
- **Quality Effect**: Only active when `G` is in `video_prompt_type`

---

## LoRA Parameters

### `loras_slists`
- **Description**: List of LoRA adapters to apply
- **Format**: Dictionary with LoRA paths and strength multipliers
- **Impact**: Modifies model behavior/style without full retraining
- **Quality Effect**:
  - Style LoRAs: Apply artistic styles
  - Motion LoRAs: Modify motion characteristics
  - Quality LoRAs: Enhance specific aspects

### `lora_multiplier`
- **Description**: Global strength multiplier for all active LoRAs
- **Range**: 0.0 - 5.0
- **Default**: 1.0
- **Impact**: Scales all LoRA effects uniformly
- **Quality Effect**: Too high may distort; too low may be invisible

---

## Quality Optimization Guidelines

### Recommended Settings by Use Case

**High Quality (Slow)**
```
num_inference_steps: 40-50
guidance_scale: 3.5-4.5
sliding_window_overlap: 25-33
self_refiner_setting: 1
self_refiner_plan: "2-8:3"
```

**Balanced (Recommended)**
```
num_inference_steps: 30-40
guidance_scale: 3.0-4.0
sliding_window_overlap: 17
self_refiner_setting: 1
self_refiner_plan: "2-8:3"
```

**Fast (Lower Quality)**
```
num_inference_steps: 20-25
guidance_scale: 3.0
sliding_window_overlap: 9
self_refiner_setting: 0
```

### Resolution Recommendations

| Goal | Resolution | Frames |
|------|------------|--------|
| Preview/Test | 512x384 | 33-65 |
| Standard | 768x512 | 81-121 |
| High Quality | 1024x768 | 81-121 |
| Cinematic | 1280x768+ | 81-161 |

### Troubleshooting Quality Issues

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| Temporal flickering | Low sliding_window_overlap | Increase overlap to 25+ |
| Artifacts/Noise | Low inference_steps | Increase to 35+ |
| Doesn't follow prompt | Low guidance_scale | Increase to 4.0-5.0 |
| Over-saturated artificial | High guidance_scale | Decrease to 3.0-3.5 |
| Blurry output | High denoising_strength with input | Lower to 0.7-0.8 |
| Bad transitions | Different image_start/image_end styles | Match lighting/composition |

---

## Model Variants

### `ltx2_19B`
- **Parameters**: 19 billion
- **Default Steps**: 40
- **Best For**: Balanced quality/speed
- **Perturbation Layers**: 29

### `ltx2_22B` (Default)
- **Parameters**: 22 billion
- **Default Steps**: 30
- **Best For**: Higher quality with fewer steps
- **Perturbation Layers**: 28

---

## Pipeline Types

### Two-Stage Pipeline (Default)
1. **Stage 1**: Generate video at target resolution with CFG guidance
2. **Stage 2**: Upsample 2x and refine with distilled LoRA
- Uses negative prompts
- Supports full guidance options

### Distilled Pipeline
- Uses pre-computed distilled LoRA
- Faster generation (fewer effective steps)
- Limited guidance options (no negative prompts)
- Supports IC-LoRA (Identity Consistency) for control video


# Video Prompt Type Modes

  Flag: V
  Name: Continue Video
  Description: Uses the uploaded input video as the source/starting point for
    generation. The model extends or modifies the provided video.
  ────────────────────────────────────────
  Flag: L
  Name: Continue Last Video
  Description: Uses the previously generated video as the source. Useful for
    multi-pass extensions where you want to chain generations together.
  ────────────────────────────────────────
  Flag: G
  Name: Guided Denoising
  Description: Enables guided denoising with denoising_strength control. When
    enabled, allows you to control when denoising starts (e.g., 0.5 means
    denoising begins at 50% of inference steps). Without G, denoising defaults
  to
     full strength.

  These flags can be combined:
  - VG = Continue Video + Guided Denoising
  - VL = Continue Video + Continue Last Video
  - VLG = All three combined
