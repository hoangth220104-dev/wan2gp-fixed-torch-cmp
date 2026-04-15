"""LTX-2 model loading and management"""

import os
import time
import torch
from pathlib import Path
from typing import Optional, Tuple, Any

from models.ltx2 import ltx2_handler
from models.ltx2.ltx2 import LTX2
from models.ltx2.ltx2_handler import _resolve_multi_file_paths
from models.ltx2.ltx2 import _attach_lora_preprocessor
from shared.utils import files_locator as fl
from mmgp import offload

from ltx2_server.config import ServerConfig
from ltx2_server.lora_manager import lora_manager


class ModelManager:
    """Manages LTX-2 model lifecycle"""
    
    def __init__(self):
        self.ltx2_instance: Optional[LTX2] = None
        self.offload_obj: Optional[Any] = None
        self.model_type: Optional[str] = None
        self.model_def: Optional[dict] = None
        self.transformer = None  # Reference to transformer for LoRA loading
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.ltx2_instance is not None
    
    @property
    def lora_manager(self):
        """Get LoRA manager instance"""
        return lora_manager
    
    def load(self, config: ServerConfig) -> None:
        """Load LTX-2 model into memory"""
        if self.is_loaded:
            raise RuntimeError("Model already loaded. Unload first before reloading.")
        
        print(f"Loading LTX-2 model: {config.model_type}")
        start_time = time.time()
        
        self.ltx2_instance, self.offload_obj, self.model_def = _load_ltx2_model(config)
        self.model_type = config.model_type
        
        # Get transformer reference for LoRA loading
        self.transformer = self.ltx2_instance.model
        
        # Initialize LoRA manager
        lora_manager.initialize(self.transformer, config.model_type)
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds")
    
    def unload(self) -> None:
        """Unload model and free resources"""
        if self.offload_obj:
            self.offload_obj.unload_all()
            self.offload_obj = None
        
        self.ltx2_instance = None
        self.model_type = None
        self.model_def = None
        print("Model unloaded")
    
    def generate(self, **kwargs):
        """Wrapper for LTX-2 generate method"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return self.ltx2_instance.generate(**kwargs)


def _load_ltx2_model(
    config: ServerConfig,
) -> Tuple[LTX2, Any, dict]:
    """Load LTX-2 model and return (instance, offload_obj, model_def)"""
    
    model_type = config.model_type
    profile = config.profile
    vram_safety_coefficient = config.vram_safety_coefficient
    transformer_path = config.transformer_path
    gemma_path = config.gemma_path
    
    family_handler = ltx2_handler.family_handler
    model_def = family_handler.query_model_def(model_type, {})

    torch_dtype = torch.bfloat16
    vae_dtype = torch.float32

    # Resolve Gemma path
    if gemma_path and gemma_path.strip():
        # Use explicitly configured path
        final_gemma_path = gemma_path
        if not os.path.exists(final_gemma_path):
            raise FileNotFoundError(f"Configured Gemma path not found: {final_gemma_path}")
        gemma_checkpoint_path = final_gemma_path
        print(f"  Gemma (configured): {final_gemma_path}")
    else:
        # Auto-detect from model definition
        auto_path = model_def.get("text_encoder_folder", "gemma-3-12b-it-qat-q4_0-unquantized")
        final_gemma_path = auto_path
        
        if os.path.isdir(final_gemma_path):
            safetensors_files = list(Path(final_gemma_path).glob("*.safetensors"))
            if not safetensors_files:
                raise FileNotFoundError(f"No safetensors file found in {final_gemma_path}")
            if len(safetensors_files) > 1:
                for f in safetensors_files:
                    if "quanto" not in f.name.lower():
                        safetensors_files = [f]
                        break
                else:
                    safetensors_files = [safetensors_files[0]]
            gemma_checkpoint_path = str(safetensors_files[0])
            print(f"  Gemma (auto-detected): {final_gemma_path}")
            print(f"  Checkpoint: {gemma_checkpoint_path}")
        else:
            if not os.path.exists(final_gemma_path):
                raise FileNotFoundError(f"Gemma path not found: {final_gemma_path}")
            gemma_checkpoint_path = final_gemma_path
            print(f"  Gemma (auto-detected): {final_gemma_path}")

    print(f"  Gemma: {final_gemma_path}")
    print(f"  Checkpoint: {gemma_checkpoint_path}")

    # Resolve checkpoint paths
    checkpoint_paths = _resolve_multi_file_paths(model_def, model_type)

    # Override transformer path if explicitly configured
    if transformer_path and transformer_path.strip():
        checkpoint_paths["transformer"] = [transformer_path]
        print(f"  Transformer (configured): {transformer_path}")
    elif checkpoint_paths.get("transformer") is None:
        # Try auto-detection
        try:
            auto_transformer_path = fl.locate_file(f"{model_type}.safetensors")
            if auto_transformer_path:
                checkpoint_paths["transformer"] = [auto_transformer_path]
                print(f"  Transformer (auto-detected): {auto_transformer_path}")
        except Exception:
            pass

    if checkpoint_paths.get("transformer") is None:
        raise FileNotFoundError(f"Transformer checkpoint not found for {model_type}. Please set transformer_path in config.")

    print(f"  Transformer: {checkpoint_paths['transformer']}")
    
    # Initialize model
    print("Initializing LTX-2 pipeline...")
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
    
    # Build pipeline dict for offloading
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
    offload_obj = offload.profile(
        pipe,
        profile_no=profile if profile >= 0 else 2,
        quantizeTransformer=False,
        vram_safety_coefficient=vram_safety_coefficient,
    )
    
    return ltx2_instance, offload_obj, model_def
