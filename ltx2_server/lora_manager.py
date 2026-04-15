"""LoRA weight management for LTX-2 models"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from mmgp import offload


class LoRAManager:
    """Manages LoRA weights for LTX-2 model"""
    
    def __init__(self):
        self.loaded_loras: Dict[str, dict] = {}  # lora_path -> metadata
        self.active_loras: List[str] = []  # List of active LoRA paths
        self.lora_multipliers: Dict[str, float] = {}  # lora_path -> multiplier
        self.transformer = None
        self.is_initialized = False
    
    def initialize(self, transformer: torch.nn.Module, model_type: str) -> None:
        """Initialize LoRA manager with transformer reference"""
        self.transformer = transformer
        self.model_type = model_type
        self.is_initialized = True
        
        # Attach LoRA preprocessor if not already present
        if not hasattr(transformer, 'preprocess_loras'):
            from models.ltx2.ltx2 import _attach_lora_preprocessor
            _attach_lora_preprocessor(transformer)
        
        # Create a preprocessor function with correct signature for load_loras_into_model
        # It needs to be callable as preprocess_sd(sd) but internally calls transformer.preprocess_loras(model_type, sd)
        def preprocess_wrapper(sd: dict) -> dict:
            return transformer.preprocess_loras(model_type, sd)
        
        self.preprocess_sd = preprocess_wrapper
        
        print("LoRA Manager initialized")
    
    def load_lora(
        self,
        lora_path: str,
        multiplier: float = 1.0,
        activate: bool = True
    ) -> bool:
        """
        Load a LoRA weight file into the model
        
        Args:
            lora_path: Path to LoRA safetensors file
            multiplier: LoRA strength multiplier (0.0-2.0+)
            activate: Whether to activate the LoRA immediately
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("LoRA Manager not initialized. Call initialize() first.")
        
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")
        
        try:
            print(f"Loading LoRA: {lora_path} (multiplier={multiplier})")
            
            # Load LoRA into model with proper preprocessor
            offload.load_loras_into_model(
                self.transformer,
                [lora_path],
                activate_all_loras=activate,
                check_only=False,
                preprocess_sd=self.preprocess_sd,
                split_linear_modules_map=getattr(self.transformer, 'split_linear_modules_map', None)
            )
            
            # Track loaded LoRA
            self.loaded_loras[lora_path] = {
                'path': lora_path,
                'multiplier': multiplier,
                'active': activate,
                'filename': Path(lora_path).name
            }
            
            if activate and lora_path not in self.active_loras:
                self.active_loras.append(lora_path)
            
            self.lora_multipliers[lora_path] = multiplier
            
            print(f"✓ LoRA loaded successfully: {Path(lora_path).name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load LoRA {lora_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def set_lora_multiplier(self, lora_path: str, multiplier: float) -> None:
        """Update multiplier for a loaded LoRA"""
        if lora_path not in self.loaded_loras:
            raise KeyError(f"LoRA not loaded: {lora_path}")
        
        self.lora_multipliers[lora_path] = multiplier
        self.loaded_loras[lora_path]['multiplier'] = multiplier
        print(f"Updated LoRA multiplier for {Path(lora_path).name}: {multiplier}")
    
    def activate_lora(self, lora_path: str) -> None:
        """Activate a loaded LoRA"""
        if lora_path not in self.loaded_loras:
            raise KeyError(f"LoRA not loaded: {lora_path}")
        
        self.loaded_loras[lora_path]['active'] = True
        if lora_path not in self.active_loras:
            self.active_loras.append(lora_path)
        
        print(f"Activated LoRA: {Path(lora_path).name}")
    
    def deactivate_lora(self, lora_path: str) -> None:
        """Deactivate a loaded LoRA without unloading"""
        if lora_path not in self.loaded_loras:
            raise KeyError(f"LoRA not loaded: {lora_path}")
        
        self.loaded_loras[lora_path]['active'] = False
        if lora_path in self.active_loras:
            self.active_loras.remove(lora_path)
        
        print(f"Deactivated LoRA: {Path(lora_path).name}")
    
    def unload_lora(self, lora_path: str) -> bool:
        """Unload a LoRA from the model"""
        if lora_path not in self.loaded_loras:
            return False
        
        try:
            # Note: mmgp doesn't support unloading LoRAs easily
            # We can only deactivate, not fully unload without reloading model
            self.deactivate_lora(lora_path)
            del self.loaded_loras[lora_path]
            if lora_path in self.lora_multipliers:
                del self.lora_multipliers[lora_path]
            
            print(f"Unloaded LoRA: {Path(lora_path).name}")
            return True
        except Exception as e:
            print(f"Failed to unload LoRA: {e}")
            return False
    
    def get_loaded_loras(self) -> List[dict]:
        """Get list of loaded LoRAs with their status"""
        return [
            {
                'path': info['path'],
                'filename': info['filename'],
                'multiplier': info['multiplier'],
                'active': info['active']
            }
            for info in self.loaded_loras.values()
        ]
    
    def get_active_loras(self) -> List[str]:
        """Get list of active LoRA paths"""
        return self.active_loras.copy()
    
    def clear_all(self) -> None:
        """Deactivate all LoRAs (doesn't unload weights)"""
        for lora_path in list(self.active_loras):
            self.deactivate_lora(lora_path)
        print("All LoRAs deactivated")
    
    def load_loras_from_directory(
        self,
        directory: str,
        default_multiplier: float = 1.0,
        activate: bool = True
    ) -> List[str]:
        """
        Load all LoRA files from a directory
        
        Args:
            directory: Path to directory containing .safetensors LoRA files
            default_multiplier: Default strength for all LoRAs
            activate: Whether to activate loaded LoRAs
            
        Returns:
            List of successfully loaded LoRA paths
        """
        if not self.is_initialized:
            raise RuntimeError("LoRA Manager not initialized. Call initialize() first.")
        
        if not os.path.isdir(directory):
            print(f"LoRA directory not found: {directory}")
            return []
        
        # Find all .safetensors files
        lora_files = list(Path(directory).glob("*.safetensors"))
        if not lora_files:
            print(f"No .safetensors files found in: {directory}")
            return []
        
        print(f"\nLoading {len(lora_files)} LoRA(s) from: {directory}")
        loaded = []
        
        for lora_file in sorted(lora_files):
            lora_path = str(lora_file)
            try:
                success = self.load_lora(
                    lora_path=lora_path,
                    multiplier=default_multiplier,
                    activate=activate
                )
                if success:
                    loaded.append(lora_path)
            except Exception as e:
                print(f"✗ Failed to load {lora_file.name}: {e}")
        
        print(f"✓ Successfully loaded {len(loaded)}/{len(lora_files)} LoRA(s)\n")
        return loaded


# Global LoRA manager instance
lora_manager = LoRAManager()
