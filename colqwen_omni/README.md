# Audio RAG with ColQwen2_5Omni

A Retrieval-Augmented Generation (RAG) system for audio content using ColQwen2_5Omni model and OpenAI's GPT-4 Audio API.

## Overview

This project demonstrates how to build an Audio RAG system that can:
- Download audio from YouTube videos
- Process audio into searchable chunks
- Create embeddings using ColQwen2_5Omni multimodal model
- Query audio content using natural language
- Generate answers using OpenAI's GPT-4 with audio understanding

## Features

- **🖥️ Web Interface**: Intuitive Gradio UI for easy interaction
- **🎵 Audio Processing**: Download YouTube videos and convert to audio
- **📝 Text Queries**: Search audio content using natural language queries
- **🤖 AI-Powered Answers**: Get responses from GPT-4 with audio understanding
- **🔊 Audio Responses**: Receive answers as both text and audio
- **⚡ Efficient Search**: Fast semantic search through audio embeddings
- **📊 Batch Processing**: Process multiple audio chunks efficiently

## Quick Start

1. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install poppler-utils ffmpeg
   
   # macOS
   brew install poppler ffmpeg
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export OPENAI_MODEL="gpt-4o-audio-preview"  # Optional
   ```

4. **Run the system**:
   
   **Option A: Web Interface (Recommended)**
   ```bash
   python run_ui.py
   ```
   This will open a web interface at `http://localhost:7860` where you can:
   - Set up your API keys
   - Process YouTube videos
   - Ask questions and get audio responses
   
   **Option B: Command Line**
   ```bash
   python app.py
   ```

## Web Interface (Gradio UI)

The easiest way to use this system is through the web interface:

### Starting the UI

```bash
python run_ui.py
```

The interface will open automatically in your browser at `http://localhost:7860`.

### Using the Interface

#### 1. Setup Tab
- **Initialize System**: Enter your OpenAI API key and model name
- **Load Model**: Download and load the ColQwen2.5-Omni model (takes a few minutes)

#### 2. Process Video Tab
- **Add YouTube URL**: Paste any YouTube video link
- **Adjust Settings**: Set chunk length (30 seconds recommended)
- **Process**: Click to download and process the video

#### 3. Ask Questions Tab
- **Enter Questions**: Type your question about the video content
- **Get Answers**: Receive both text and audio responses
- **Adjust Context**: Control how many audio chunks to use

### UI Features

- **📊 Real-time Status**: See processing progress and system status
- **🎵 Audio Playback**: Play audio responses directly in the browser
- **⚙️ Configurable Settings**: Adjust chunk length and search parameters
- **📝 Help Section**: Built-in documentation and tips
- **🔄 Batch Processing**: Process multiple videos without reloading the model

## Command Line Usage

### Basic Example

```python
from app import AudioRAG

# Initialize the system
rag = AudioRAG(api_key="your-openai-api-key")
rag.load_model()

# Process audio from YouTube
audio_path = rag.download_youtube_audio("https://www.youtube.com/watch?v=VIDEO_ID")
audio_chunks = rag.chunk_audio(audio_path)
rag.create_embeddings(audio_chunks)

# Query the system
response = rag.answer_query("What was discussed about artificial intelligence?")
print(f"Answer: {response['answer_text']}")

# Save audio response
rag.save_audio_response(response, "answer.wav")
```

### Advanced Usage

```python
# Process local video file
audio_path = rag.convert_video_to_audio("my_video.mp4")

# Custom chunk size (default is 30 seconds)
audio_chunks = rag.chunk_audio(audio_path, chunk_length_seconds=60)

# Custom batch size for embedding creation
rag.create_embeddings(audio_chunks, batch_size=2)

# Get more results
top_chunks = rag.query_audio("machine learning", k=10)
```

## Project Structure

```
colqwen_omni/
├── app.py               # Main AudioRAG class
├── main.py              # Gradio web interface
├── run_ui.py            # UI launcher script
├── requirements.txt     # Python dependencies
├── setup.md             # Detailed setup instructions
├── README.md            # This file
└── Practical_3_AudioRAG.ipynb  # Original Jupyter notebook
```

## Requirements

### System Requirements
- Python 3.8+
- FFmpeg (for audio processing)
- Poppler (for PDF processing, if needed)
- CUDA-compatible GPU (recommended) or CPU

### Python Dependencies
See `requirements.txt` for the complete list. Key dependencies include:
- torch
- transformers
- openai
- colpali-engine
- moviepy
- pydub
- yt-dlp
- gradio (for web interface)

## How It Works

1. **Audio Acquisition**: Downloads YouTube videos or processes local files
2. **Audio Processing**: Splits audio into 30-second chunks and resamples to 16kHz
3. **Embedding Generation**: Uses ColQwen2_5Omni to create embeddings for each chunk
4. **Query Processing**: Converts text queries to embeddings and finds similar audio chunks
5. **Answer Generation**: Sends relevant audio chunks to GPT-4 for question answering
6. **Response**: Returns both text and audio responses

## Performance Tips

- Use GPU for faster model inference
- Adjust chunk size based on your content (30s default)
- Process audio in batches to optimize memory usage
- Cache embeddings for repeated queries

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `create_embeddings()`
2. **FFmpeg Not Found**: Ensure FFmpeg is installed and in PATH
3. **Model Loading Issues**: Check GPU memory or use CPU mode
4. **YouTube Download Fails**: Update yt-dlp: `pip install --upgrade yt-dlp`

### GPU Requirements

- ColQwen2_5Omni requires significant GPU memory
- CPU mode available but slower
- Flash attention recommended for faster inference

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the little-scripts collection. See the main repository for license information.

## Acknowledgments

- [ColPali](https://github.com/illuin-tech/colpali) for the multimodal model
- [OpenAI](https://openai.com) for GPT-4 Audio API
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube downloading

## Support

For detailed setup instructions, see `setup.md`.
For issues and questions, please open an issue in the main repository. 