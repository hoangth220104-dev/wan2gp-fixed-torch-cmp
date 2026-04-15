#!/usr/bin/env python3
"""
LTX-2 FastAPI Server - CLI Entry Point

Usage:
    python ltx2_server.py --model_type ltx2_22B --port 8000
    uvicorn ltx2_server.main:create_app --factory --host 0.0.0.0 --port 8000
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn
from ltx2_server.config import ServerConfig
from ltx2_server.main import create_app


def parse_args():
    parser = argparse.ArgumentParser(
        description="LTX-2 FastAPI Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings
  python ltx2_server.py

  # Use custom model paths
  python ltx2_server.py --transformer_path /path/to/ltx2_22B.safetensors --gemma_path /path/to/gemma.safetensors

  # Use ltx2_19B model on port 8001
  python ltx2_server.py --model_type ltx2_19B --port 8001

  # Development mode with auto-reload
  python ltx2_server.py --reload

  # Use with uvicorn directly
  uvicorn "ltx2_server.main:create_app" --factory --host 0.0.0.0 --port 8000
        """,
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="ltx2_22B",
        choices=["ltx2_19B", "ltx2_22B"],
        help="LTX-2 model variant (default: ltx2_22B)",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="",
        help="Path to transformer safetensors file (overrides auto-detection)",
    )
    parser.add_argument(
        "--gemma_path",
        type=str,
        default="",
        help="Path to Gemma text encoder safetensors file (overrides auto-detection)",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="",
        help="Directory containing LoRA .safetensors files (auto-loaded on startup)",
    )
    parser.add_argument(
        "--profile",
        type=int,
        default=-1,
        choices=[-1, 0, 1, 2, 3, 4],
        help="Memory profile (-1: auto, 0-4: manual)",
    )
    parser.add_argument(
        "--vram_safety_coefficient",
        type=float,
        default=0.85,
        help="VRAM safety coefficient 0.0-0.95 (default: 0.85)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for generated videos (default: output)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, only 1 supported)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.workers > 1:
        print("Warning: Only 1 worker is supported. LTX-2 uses too much VRAM for multiple workers.")
        args.workers = 1
    
    # Create config
    config = ServerConfig(
        model_type=args.model_type,
        transformer_path=args.transformer_path,
        gemma_path=args.gemma_path,
        lora_dir=args.lora_dir,
        profile=args.profile,
        vram_safety_coefficient=args.vram_safety_coefficient,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    
    # Set config in environment so uvicorn factory can access it
    from ltx2_server.main import set_config
    set_config(config)
    
    print(f"\nStarting LTX-2 FastAPI Server")
    print(f"  Model: {args.model_type}")
    print(f"  Profile: {args.profile}")
    print(f"  VRAM Safety: {args.vram_safety_coefficient}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Server: http://{args.host}:{args.port}")
    print(f"  API Docs: http://{args.host}:{args.port}/docs")
    print(f"  ReDoc: http://{args.host}:{args.port}/redoc\n")
    
    # Start uvicorn
    uvicorn.run(
        "ltx2_server.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
