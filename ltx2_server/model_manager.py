"""LTX-2 model loading and management"""

import os
import time
import torch
from pathlib import Path
from typing import Optional, Tuple, Any

from models.ltx2 import ltx2_handler
from models.ltx2.ltx2 import LTX2
from models.ltx2.ltx2_handler import _resolve_multi_file_paths
from shared.utils import files_locator as fl
from mmgp import offload

from ltx2_server.config import ServerConfig


class ModelManager:
    """Manages LTX-2 model lifecycle"""
    
    def __init__(self):
        self.ltx2_instance: Optional[LTX2] = None
        self.offload_obj: Optional[Any] = None
        self.model_type: Optional[str] = None
        self.model_def: Optional[dict] = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.ltx2_instance is not None
    
    def load(self, config: ServerConfig) -> None:
        """Load LTX-2 model into memory"""
        if self.is_loaded:
            raise RuntimeError("Model already loaded. Unload first before reloading.")
        
        print(f"Loading LTX-2 model: {config.model_type}")
        start_time = time.time()
        
        self.ltx2_instance, self.offload_obj, self.model_def = _load_ltx2_model(
            model_type=config.model_type,
            profile=config.profile,
            vram_safety_coefficient=config.vram_safety_coefficient,
        )
        self.model_type = config.model_type
        
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
    model_type: str,
    profile: int = -1,
    vram_safety_coefficient: float = 0.85,
) -> Tuple[LTX2, Any, dict]:
    """Load LTX-2 model and return (instance, offload_obj, model_def)"""
    
    family_handler = ltx2_handler.family_handler
    model_def = family_handler.query_model_def(model_type, {})
    
    torch_dtype = torch.bfloat16
    vae_dtype = torch.float32
    
    # Resolve Gemma path
    gemma_path = model_def.get("text_encoder_folder", "gemma-3-12b-it-qat-q4_0-unquantized")
    
    if os.path.isdir(gemma_path):
        safetensors_files = list(Path(gemma_path).glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors file found in {gemma_path}")
        if len(safetensors_files) > 1:
            for f in safetensors_files:
                if "quanto" not in f.name.lower():
                    safetensors_files = [f]
                    break
            else:
                safetensors_files = [safetensors_files[0]]
        gemma_checkpoint_path = str(safetensors_files[0])
    else:
        if not os.path.exists(gemma_path):
            raise FileNotFoundError(f"Gemma path not found: {gemma_path}")
        gemma_checkpoint_path = gemma_path
    
    print(f"  Gemma: {gemma_path}")
    print(f"  Checkpoint: {gemma_checkpoint_path}")
    
    # Resolve checkpoint paths
    checkpoint_paths = _resolve_multi_file_paths(model_def, model_type)
    
    if checkpoint_paths.get("transformer") is None:
        try:
            transformer_path = fl.locate_file(f"{model_type}.safetensors")
            if transformer_path:
                checkpoint_paths["transformer"] = [transformer_path]
        except Exception:
            pass
    
    if checkpoint_paths.get("transformer") is None:
        raise FileNotFoundError(f"Transformer checkpoint not found for {model_type}")
    
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
