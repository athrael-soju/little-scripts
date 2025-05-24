#!/usr/bin/env python3
"""Setup script for the Python monorepo using UV."""

import subprocess
import sys
from pathlib import Path


def run_command(command, cwd=None):
    """Run a command and handle errors."""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error running command: {command}")
            print(f"   Output: {result.stderr}")
            return False
        else:
            print(f"✅ {command}")
            return True
    except Exception as e:
        print(f"❌ Exception running command: {command}")
        print(f"   Error: {e}")
        return False


def check_uv_installed():
    """Check if uv is installed, and install it if not."""
    try:
        result = subprocess.run("uv --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ UV is already installed: {result.stdout.strip()}")
            return True
    except Exception:
        pass
    
    print("📦 UV not found, installing...")
    # Install uv using pip
    if not run_command(f"{sys.executable} -m pip install uv"):
        print("❌ Failed to install uv. Please install it manually:")
        print("   pip install uv")
        print("   or visit: https://github.com/astral-sh/uv")
        return False
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up Python monorepo with UV...")
    
    # Check and install uv if needed
    if not check_uv_installed():
        return False
    
    # Get the root directory
    root_dir = Path(__file__).parent.parent
    
    # Install requirements using uv
    print("\n📦 Installing common requirements with UV...")
    if not run_command("uv pip install -r requirements.txt", cwd=root_dir):
        return False
    
    # Install development dependencies using uv
    print("\n🛠️  Installing development dependencies with UV...")
    if not run_command("uv pip install -e .[dev]", cwd=root_dir):
        return False
    
    # Install packages in development mode using uv
    packages = ["packages/utils", "packages/core"]
    for package in packages:
        print(f"\n📚 Installing {package} in development mode with UV...")
        if not run_command(f"uv pip install -e {package}", cwd=root_dir):
            return False
    
    # Install apps in development mode using uv
    apps = ["apps/web-app", "apps/cli-tool"]
    for app in apps:
        print(f"\n🚀 Installing {app} in development mode with UV...")
        if not run_command(f"uv pip install -e {app}", cwd=root_dir):
            return False
    
    print("\n🎉 Setup completed successfully with UV!")
    print("\nNext steps:")
    print("  • Run tests: python -m pytest tests/")
    print("  • Start web app: cd apps/web-app && python -m web_app.main")
    print("  • Use CLI tool: task-cli --help")
    print("\nUV commands for future use:")
    print("  • Install package: uv pip install <package>")
    print("  • Add dependency: uv add <package>")
    print("  • Create lock file: uv lock")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 