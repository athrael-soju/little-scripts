#!/usr/bin/env python3
"""
Simple launcher for the Audio RAG Gradio UI
"""

import os
import sys
from pathlib import Path


def main():
    """Launch the Gradio UI"""
    # Ensure we're in the right directory
    current_dir = Path(__file__).parent
    os.chdir(current_dir)

    # Check if environment variables are set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set")
        print(
            "   You can set it in the UI or export it: export OPENAI_API_KEY=your_key_here"
        )
        print()

    # Import and run the UI
    try:
        print("🔄 Importing gradio_ui module...")
        from gradio_ui import main as run_ui

        print("✅ Import successful!")

        print("🚀 Starting Audio RAG UI...")
        print("📝 The interface will open in your browser automatically")
        print("🔗 You can also access it at: http://localhost:7860")
        print()

        run_ui()
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        print("🔧 Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting UI: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
