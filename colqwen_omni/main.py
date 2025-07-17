#!/usr/bin/env python3
"""
Gradio UI for Audio RAG with ColQwen2_5Omni Model

This provides an intuitive web interface for:
1. Adding YouTube video links and processing them
2. Asking questions about the audio content
3. Getting text and audio responses
"""

import os
import gradio as gr
import tempfile
import base64
from typing import Optional, Tuple

from app import AudioRAG


class AudioRAGUI:
    """Gradio UI wrapper for AudioRAG system"""

    def __init__(self):
        self.rag_system: Optional[AudioRAG] = None
        self.is_model_loaded = False
        self.is_video_processed = False
        self.current_video_url = ""

    def initialize_system(self, api_key: str, api_model: str) -> str:
        """Initialize the AudioRAG system with API credentials"""
        try:
            if not api_key:
                return "❌ Please provide your OpenAI API key"
            if not api_model:
                api_model = "gpt-4o-mini-audio-preview"  # Default model

            self.rag_system = AudioRAG(api_key, api_model)
            return "✅ System initialized successfully"

        except Exception as e:
            return f"❌ Error initializing system: {str(e)}"

    def load_model(self) -> str:
        """Load the ColQwen2_5Omni model"""
        if not self.rag_system:
            return "❌ Please initialize the system first"

        try:
            self.rag_system.load_model()
            self.is_model_loaded = True
            return "✅ Model loaded successfully! You can now process videos."

        except Exception as e:
            return f"❌ Error loading model: {str(e)}"

    def process_video(self, video_url: str, chunk_length: int) -> str:
        """Process a YouTube video URL"""
        if not self.is_model_loaded:
            return "❌ Please load the model first"

        if not video_url:
            return "❌ Please provide a YouTube video URL"

        try:
            # Download audio
            status = "📥 Downloading audio from YouTube..."
            yield status

            audio_path = self.rag_system.download_youtube_audio(video_url)
            if not audio_path:
                yield "❌ Failed to download audio from YouTube"
                return

            # Process audio into chunks
            status = "🔄 Processing audio into chunks..."
            yield status

            audio_chunks = self.rag_system.chunk_audio(audio_path, chunk_length)

            # Create embeddings
            status = "🧠 Creating embeddings (this may take a while)..."
            yield status

            self.rag_system.create_embeddings(audio_chunks)

            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

            self.is_video_processed = True
            self.current_video_url = video_url

            final_status = f"✅ Video processed successfully!\n📊 Created {len(audio_chunks)} audio chunks\n🎯 Ready for questions!"
            yield final_status

        except Exception as e:
            yield f"❌ Error processing video: {str(e)}"

    def answer_question(self, question: str, num_chunks: int) -> Tuple[str, str, str]:
        """Answer a question about the processed video"""
        if not self.is_video_processed:
            return "❌ Please process a video first", "", ""

        if not question:
            return "❌ Please enter a question", "", ""

        try:
            response = self.rag_system.answer_query(question, k=num_chunks)
            wav_bytes = base64.b64decode(response["audio_data"])

            # Create a truly temporary file (in the system temp directory)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                audio_path = tmp.name

            answer_text = response["answer_text"]
            chunk_info = f"📋 Used audio chunks: {response['used_chunks']}"
            return answer_text, audio_path, chunk_info

        except Exception as e:
            return f"❌ Error answering question: {str(e)}", None, ""

    def get_processing_status(self) -> str:
        """Get current processing status"""
        if not self.rag_system:
            return "🔴 System not initialized"
        elif not self.is_model_loaded:
            return "🟡 Model not loaded"
        elif not self.is_video_processed:
            return "🟡 No video processed"
        else:
            return f"🟢 Ready - Video: {self.current_video_url[:50]}..."


def create_interface():
    """Create the Gradio interface"""

    ui = AudioRAGUI()

    with gr.Blocks(
        title="Audio RAG with ColQwen2.5-Omni", theme=gr.themes.Soft()
    ) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎵 Audio RAG with ColQwen2.5-Omni</h1>
            <p>Process YouTube videos and ask questions about their audio content</p>
        </div>
        """)

        with gr.Tab("🔧 Setup"):
            gr.Markdown("### Step 1: Initialize System")
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-...",
                    type="password",
                    value=os.getenv("OPENAI_API_KEY", ""),
                )
                api_model_input = gr.Textbox(
                    label="OpenAI Model",
                    placeholder="gpt-4o-mini-audio-preview",
                    value=os.getenv("OPENAI_MODEL", "gpt-4o-mini-audio-preview"),
                )

            init_btn = gr.Button("🚀 Initialize System", variant="primary")
            init_status = gr.Textbox(label="Initialization Status", interactive=False)

            gr.Markdown("### Step 2: Load Model")
            load_model_btn = gr.Button(
                "📥 Load ColQwen2.5-Omni Model", variant="secondary"
            )
            model_status = gr.Textbox(label="Model Status", interactive=False)

        with gr.Tab("🎥 Process Video"):
            gr.Markdown("### Add and Process YouTube Video")

            with gr.Row():
                video_url = gr.Textbox(
                    label="YouTube Video URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    scale=3,
                )
                chunk_length = gr.Slider(
                    label="Chunk Length (seconds)",
                    minimum=10,
                    maximum=120,
                    value=30,
                    step=10,
                    scale=1,
                )

            process_btn = gr.Button("🔄 Process Video", variant="primary")
            process_status = gr.Textbox(label="Processing Status", interactive=False)

        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("### Ask Questions About the Video")

            status_display = gr.Textbox(label="System Status", interactive=False)

            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What is the main topic discussed in the video?",
                    scale=3,
                )
                num_chunks_slider = gr.Slider(
                    label="Number of Audio Chunks to Use",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    scale=1,
                )

            ask_btn = gr.Button("🎯 Ask Question", variant="primary")

            with gr.Row():
                with gr.Column():
                    answer_text = gr.Textbox(
                        label="Text Answer", lines=5, interactive=False
                    )
                    chunk_info = gr.Textbox(
                        label="Source Information", interactive=False
                    )

                with gr.Column():
                    audio_response = gr.Audio(label="Audio Response", interactive=False)

        with gr.Tab("ℹ️ Help"):
            gr.Markdown("""
            ## How to Use This Interface
            
            ### 1. Setup (First Time Only)
            - Enter your OpenAI API key
            - Click "Initialize System"
            - Click "Load ColQwen2.5-Omni Model" (this may take a few minutes)
            
            ### 2. Process Video
            - Paste a YouTube video URL
            - Adjust chunk length if needed (30 seconds is usually good)
            - Click "Process Video" and wait for completion
            
            ### 3. Ask Questions
            - Enter your question about the video content
            - Adjust the number of audio chunks to use for context
            - Click "Ask Question" to get both text and audio responses
            
            ### Tips
            - The system works best with videos that have clear speech
            - Longer videos will take more time to process
            - You can ask multiple questions about the same video
            - The audio response will be played automatically
            
            ### Supported Video Sources
            - YouTube videos
            - Most video formats supported by yt-dlp
            """)

        # Event handlers
        def update_status_after_init(api_key, api_model):
            init_result = ui.initialize_system(api_key, api_model)
            status_result = ui.get_processing_status()
            return init_result, status_result

        def update_status_after_load():
            load_result = ui.load_model()
            status_result = ui.get_processing_status()
            return load_result, status_result

        def update_status_after_process(video_url, chunk_length):
            # Process video is a generator, so we need to handle it differently
            for status in ui.process_video(video_url, chunk_length):
                updated_status = ui.get_processing_status()
                yield status, updated_status

        init_btn.click(
            update_status_after_init,
            inputs=[api_key_input, api_model_input],
            outputs=[init_status, status_display],
        )

        load_model_btn.click(
            update_status_after_load, outputs=[model_status, status_display]
        )

        process_btn.click(
            update_status_after_process,
            inputs=[video_url, chunk_length],
            outputs=[process_status, status_display],
        )

        ask_btn.click(
            ui.answer_question,
            inputs=[question_input, num_chunks_slider],
            outputs=[answer_text, audio_response, chunk_info],
        )

        # Initialize status on load
        demo.load(ui.get_processing_status, outputs=[status_display])

    return demo


def main():
    """Launch the Gradio interface"""
    demo = create_interface()
    demo.launch(share=False, server_name="localhost", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
