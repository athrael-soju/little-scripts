#!/usr/bin/env python3
"""
Audio RAG with ColQwen2_5Omni Model

This script demonstrates how to:
1. Download YouTube videos and extract audio
2. Process audio into chunks
3. Create embeddings using ColQwen2_5Omni
4. Query the audio corpus and get answers using OpenAI's API

Prerequisites:
- Install system dependencies: poppler-utils, ffmpeg
  - Ubuntu/Debian: sudo apt-get install poppler-utils ffmpeg
  - macOS: brew install poppler ffmpeg
  - Windows: Download and install from official sources

- Install Python dependencies: pip install -r requirements.txt
"""

import os
import io
import base64
import torch
import numpy as np
from typing import List, Optional

# Audio/Video processing
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.io.wavfile import write
from moviepy import VideoFileClip
import subprocess

# ML and AI
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5Omni, ColQwen2_5OmniProcessor
from tqdm import tqdm
from torch.utils.data import DataLoader

# OpenAI API
from openai import OpenAI

# Display utilities (for Jupyter/IPython environments)
try:
    from IPython.display import Audio, display, Video

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    print("IPython not available - display functions disabled")


class AudioRAG:
    """Audio RAG system using ColQwen2_5Omni model"""

    def __init__(self, openai_api_key: str, openai_model: str):
        """Initialize the Audio RAG system

        Args:
            openai_api_key: OpenAI API key for GPT-4 audio
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.openai_model = openai_model
        self.model = None
        self.processor = None
        self.audio_embeddings = []
        self.audio_chunks = []

    def load_model(self):
        """Load the ColQwen2_5Omni model"""
        print("Loading ColQwen2_5Omni model...")
        self.model = ColQwen2_5Omni.from_pretrained(
            "vidore/colqwen-omni-v0.1",
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            attn_implementation="flash_attention_2"
            if is_flash_attn_2_available()
            else None,
        ).eval()
        self.processor = ColQwen2_5OmniProcessor.from_pretrained(
            "manu/colqwen-omni-v0.1"
        )
        print("Model loaded successfully!")

    def download_youtube_audio(self, url: str, output_path: str = "audio.wav"):
        """Download YouTube video and extract audio

        Args:
            url: YouTube video URL
            output_path: Path to save the audio file
        """
        cmd = [
            "yt-dlp",
            url,
            "--extract-audio",
            "--audio-format",
            "wav",
            "-o",
            output_path.replace(".wav", ".%(ext)s"),
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Successfully downloaded audio from {url}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Error downloading video: {e}")
            print(f"Error output: {e.stderr}")
            return None

    def convert_video_to_audio(self, video_path: str, output_path: str = "audio.wav"):
        """Convert video file to audio

        Args:
            video_path: Path to video file
            output_path: Path to save the audio file
        """
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(output_path)
            video.close()
            print(f"Successfully converted {video_path} to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error converting video to audio: {e}")
            return None

    def chunk_audio(
        self, audio_path: str, chunk_length_seconds: int = 30
    ) -> List[np.ndarray]:
        """Split audio into chunks

        Args:
            audio_path: Path to audio file
            chunk_length_seconds: Length of each chunk in seconds

        Returns:
            List of audio chunks as numpy arrays
        """
        print(f"Loading and chunking audio: {audio_path}")

        # Load audio
        audio = AudioSegment.from_wav(audio_path)

        # Set parameters
        target_rate = 16000
        chunk_length_ms = chunk_length_seconds * 1000

        audio_chunks = []

        # Split and process each chunk
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i : i + chunk_length_ms]

            # Convert to mono and resample
            chunk = chunk.set_channels(1)
            chunk = chunk.set_frame_rate(target_rate)

            # Convert to numpy array
            buf = io.BytesIO()
            chunk.export(buf, format="wav")
            buf.seek(0)

            rate, data = wavfile.read(buf)
            audio_chunks.append(data)

        print(f"Created {len(audio_chunks)} audio chunks")
        self.audio_chunks = audio_chunks
        return audio_chunks

    def create_embeddings(
        self, audio_chunks: Optional[List[np.ndarray]] = None, batch_size: int = 4
    ):
        """Create embeddings for audio chunks

        Args:
            audio_chunks: List of audio chunks (uses self.audio_chunks if None)
            batch_size: Batch size for processing
        """
        if audio_chunks is None:
            audio_chunks = self.audio_chunks

        if not audio_chunks:
            raise ValueError("No audio chunks provided")

        print(f"Creating embeddings for {len(audio_chunks)} audio chunks...")

        # Process audio chunks in batches
        dataloader = DataLoader(
            dataset=audio_chunks,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_audio(x),
        )

        embeddings = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

        self.audio_embeddings = embeddings
        print(f"Created {len(embeddings)} embeddings")
        return embeddings

    def query_audio(self, query: str, k: int = 10) -> List[int]:
        """Query the audio corpus and return top-k results

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of indices of top-k audio chunks
        """
        if not self.audio_embeddings:
            raise ValueError("No embeddings available. Run create_embeddings first.")

        # Ensure k doesn't exceed the number of available audio chunks
        k = min(k, len(self.audio_embeddings))

        if k == 0:
            raise ValueError("No audio chunks available for querying")

        # Process query
        batch_queries = self.processor.process_queries([query]).to(self.model.device)

        # Get query embeddings
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)

        # Score against corpus
        scores = self.processor.score_multi_vector(
            query_embeddings, self.audio_embeddings
        )

        # Get top-k results
        top_k_indices = scores[0].topk(k).indices.tolist()
        return top_k_indices

    def audio_to_base64(self, audio_data: np.ndarray, rate: int = 16000) -> str:
        """Convert audio data to base64 string

        Args:
            audio_data: Audio data as numpy array
            rate: Sample rate

        Returns:
            Base64 encoded audio string
        """
        buf = io.BytesIO()
        write(buf, rate, audio_data)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def answer_query(self, query: str, k: int = 5) -> dict:
        """Answer a query using the audio corpus and OpenAI API

        Args:
            query: Query string
            k: Number of audio chunks to use for context

        Returns:
            Dictionary with answer and audio response
        """
        # Get relevant audio chunks
        top_indices = self.query_audio(query, k)

        # Prepare content for OpenAI API
        content = [
            {
                "type": "text",
                "text": f"Answer the query using the audio files. Say which ones were used to answer. Query: {query}",
            }
        ]

        # Add audio chunks to content
        for i in top_indices:
            content.extend(
                [
                    {"type": "text", "text": f"The following is audio chunk # {i}."},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": self.audio_to_base64(self.audio_chunks[i]),
                            "format": "wav",
                        },
                    },
                ]
            )

        # Get response from OpenAI
        completion = self.client.chat.completions.create(
            model=self.openai_model,
            modalities=["text", "audio"],
            audio={"voice": "ballad", "format": "wav"},
            messages=[{"role": "user", "content": content}],
        )

        # Extract response
        response = {
            "query": query,
            "answer_text": completion.choices[0].message.audio.transcript,
            "audio_data": completion.choices[0].message.audio.data,
            "used_chunks": top_indices,
        }

        return response

    def save_audio_response(self, response: dict, filename: str = "response.wav"):
        """Save audio response to file

        Args:
            response: Response dictionary from answer_query
            filename: Output filename
        """
        wav_bytes = base64.b64decode(response["audio_data"])
        with open(filename, "wb") as f:
            f.write(wav_bytes)
        print(f"Audio response saved to {filename}")

    def display_audio_chunk(self, chunk_index: int):
        """Display an audio chunk (if in Jupyter environment)

        Args:
            chunk_index: Index of the chunk to display
        """
        if IPYTHON_AVAILABLE and chunk_index < len(self.audio_chunks):
            display(Audio(self.audio_chunks[chunk_index], autoplay=False, rate=16000))
        else:
            print(f"Cannot display audio chunk {chunk_index}")


def main():
    """Example usage of the AudioRAG system"""
    # Initialize system
    api_key = os.getenv("OPENAI_API_KEY")
    api_model = os.getenv("OPENAI_MODEL")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return

    rag = AudioRAG(api_key, api_model)

    # Load model
    rag.load_model()

    # Download and process audio
    youtube_url = "https://www.youtube.com/watch?v=lsbcN9-jU1Y"
    audio_path = rag.download_youtube_audio(youtube_url)

    if audio_path:
        # Process audio
        audio_chunks = rag.chunk_audio(audio_path)
        rag.create_embeddings(audio_chunks)

        # Query the system
        query = "Was Hannibal well liked by his men?"
        response = rag.answer_query(query)

        print(f"Query: {response['query']}")
        print(f"Answer: {response['answer_text']}")
        print(f"Used chunks: {response['used_chunks']}")

        # Save audio response
        rag.save_audio_response(response)

        # Display audio chunk if in Jupyter
        if IPYTHON_AVAILABLE:
            rag.display_audio_chunk(response["used_chunks"][0])


if __name__ == "__main__":
    main()
