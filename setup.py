#!/usr/bin/env python3
"""
Setup script for little-scripts monorepo
Handles Poetry dependency installation and PyTorch nightly setup
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors while showing real-time output"""
    print(f"🔄 {description}...")
    try:
        # Use subprocess.run without capture_output to show real-time progress
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Command: {cmd}")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Setup little-scripts monorepo")
    parser.add_argument(
        "--project",
        choices=["colnomic", "panoptic", "all"],
        default="all",
        help="Which project to install dependencies for",
    )
    parser.add_argument(
        "--skip-pytorch", action="store_true", help="Skip PyTorch nightly installation"
    )

    args = parser.parse_args()

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ pyproject.toml not found. Run this script from the project root.")
        sys.exit(1)

    print("🚀 Setting up little-scripts monorepo...")

    # Install Poetry dependencies
    if args.project == "all":
        poetry_cmd = "poetry install --with dev,colnomic,panoptic"
    elif args.project == "colnomic":
        poetry_cmd = "poetry install --with dev,colnomic"
    elif args.project == "panoptic":
        poetry_cmd = "poetry install --with dev,panoptic"

    run_command(poetry_cmd, "Installing Poetry dependencies")

    # Install PyTorch nightly
    if not args.skip_pytorch:
        pytorch_cmd = "poetry run pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"
        run_command(pytorch_cmd, "Installing PyTorch nightly")

    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the environment: poetry shell")
    print("2. Run a project:")
    print("   - ColPali RAG: poetry run python colnomic_qdrant_rag/main.py")
    print("   - Panoptic Seg: poetry run python eomt_panoptic_seg/app.py")


if __name__ == "__main__":
    main()
