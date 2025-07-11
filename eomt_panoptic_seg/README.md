# EOMT Panoptic Segmentation App

A [Gradio-based web application](https://huggingface.co/spaces/Athrael/eomt) for interactive panoptic segmentation using the **EOMT (Encoder-only Mask Transformer)** model - a minimalist approach that repurposes a plain Vision Transformer (ViT) for image segmentation.

## 🚀 Features

- **Interactive Web Interface**: User-friendly Gradio interface for uploading and processing images
- **Multiple Visualization Types**:
  - Segmentation mask with color-coded segments (with error handling for empty masks)
  - Overlay visualization on original image with transparency control
  - Contour detection with distinct color coding for each segment
  - Individual instance masks in grid layout with segment statistics
  - Edge detection with improved boundary highlighting
  - Segment isolation showing largest segment with detailed information
  - Boundary density heatmap with gradient magnitude visualization
- **Real-time Processing**: Fast inference with CUDA support and proper GPU memory handling
- **Sample Images**: Built-in sample images with intuitive button selection
- **Detailed Analytics**: Comprehensive segment statistics and visual analysis
- **Enhanced Visualizations**: Improved color coding, titles, and statistical information for better analysis

## 🧠 Model

This application uses the **EOMT (Encoder-only Mask Transformer)** model, as presented in the CVPR 2025 highlight paper ["Your ViT is Secretly an Image Segmentation Model"](https://www.tue-mps.org/eomt/):

- **Model ID**: `tue-mps/coco_panoptic_eomt_large_640`
- **Task**: Panoptic Segmentation
- **Dataset**: COCO Panoptic
- **Input Size**: 640x640 (automatically resized)
- **Architecture**: Plain Vision Transformer (ViT) repurposed for segmentation
- **Key Innovation**: No adapters, no decoders - just the ViT encoding image patches and segmentation queries as tokens
- **Performance**: Up to 4× faster than complex methods while maintaining state-of-the-art accuracy

## 🛠️ Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/athrael.soju/little-scripts.git
   cd little-scripts/eomt_panoptic_seg
   ```

2. **Install dependencies**:

   **For standard setup:**
   ```bash
   uv pip install -r requirements.txt
   ```

   **For newer NVIDIA GPUs (e.g., RTX 5090):**
   ```bash
   # Install PyTorch nightly for latest CUDA support
   uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

   # Then install remaining dependencies
   uv pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended for faster inference)
- Minimum 4GB RAM
- Internet connection (for first-time model download)
- `uv` package manager (optional, for newer GPU support)

### Dependencies
- **PyTorch**: Latest stable version (or nightly for newer GPUs like RTX 5090)
- **Transformers**: Latest development version from GitHub
- **Gradio**: For the web interface
- **OpenCV**: For image processing and contour detection
- **Matplotlib**: For visualization
- **NumPy & PIL**: For image handling
- **SciPy**: For advanced image processing operations

## 🎯 Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to the provided URL (usually `http://localhost:7860`)

3. **Upload an image** or select from sample images using the clickable buttons

4. **Choose visualization type**:
   - **Mask**: Color-coded segmentation mask with error handling for empty results
   - **Overlay**: Transparent mask overlay on original image with optimized transparency
   - **Contours**: Segment boundaries with distinct colors for each segment
   - **Instance Masks**: Individual instance masks in a 3×3 grid showing segment IDs and pixel counts
   - **Edge Detection**: Improved boundary highlighting with RGBA overlay technique
   - **Segment Isolation**: Largest segment isolated with detailed statistics
   - **Heatmap**: Boundary density visualization with gradient magnitude and proper labeling

5. **View results** with interactive segment analysis and high-quality visualizations

## 🔧 Configuration

The application automatically downloads the EOMT model on first run. Model files are cached locally by Hugging Face Transformers.

### Model Configuration
- **Model**: `tue-mps/coco_panoptic_eomt_large_640`
- **Image Processor**: Auto-configured for the model
- **Inference Mode**: PyTorch inference mode for optimal performance
- **CUDA Support**: Automatic GPU detection and proper tensor handling

## 📊 Features Detail

### Visualization Types

1. **Mask View**: Clean segmentation mask with color-coded segments using matplotlib's tab20 colormap, includes error handling for empty masks
2. **Overlay View**: Weighted combination of original image and segmentation mask with optimized transparency
3. **Contours View**: Precise segment boundaries with distinct colors for each segment using OpenCV and matplotlib colormaps
4. **Instance Masks View**: Grid layout showing individual segments with segment IDs and pixel counts for detailed inspection
5. **Edge Detection View**: Enhanced boundary detection using RGBA overlay technique with Canny edge detection
6. **Segment Isolation View**: Focus on the largest segment with detailed statistics (segment ID and pixel count)
7. **Heatmap View**: Boundary density analysis with gradient magnitude visualization and proper colorbar labeling

### Interactive Features

- **Sample Images**: Pre-loaded COCO dataset images with large clickable thumbnails
- **Upload Support**: Drag-and-drop or click to upload custom images
- **Real-time Processing**: Fast inference with proper GPU memory management
- **High-Quality Output**: All visualizations rendered at 150 DPI for crisp results
- **Enhanced Information**: Detailed segment statistics, pixel counts, and gradient analysis

## 🎨 Visualization Examples

The application provides multiple ways to visualize panoptic segmentation results:

- **Color-coded segments** using matplotlib's tab20 colormap for clear differentiation
- **Statistical analysis** showing segment distribution and coverage metrics
- **Contour detection** with OpenCV for precise boundary identification
- **Overlay techniques** with adjustable transparency for better visual understanding
- **Heatmap analysis** for boundary density and gradient visualization
- **Individual segment isolation** for detailed examination

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Tensor Errors**:
   - The app automatically handles CUDA tensor to CPU conversion with robust error handling
   - Ensure proper PyTorch installation with CUDA support
   - Check GPU memory availability

2. **Model Download Issues**:
   - Ensure stable internet connection
   - Check Hugging Face Hub access
   - Verify sufficient disk space

3. **Memory Issues**:
   - Reduce image size before upload
   - Close other applications
   - Consider using CPU inference for large images

4. **Performance Issues**:
   - Use GPU if available
   - Reduce image resolution
   - Check system resources

5. **PyTorch Installation Issues**:
   - For newer GPUs (RTX 5090, etc.), use PyTorch nightly builds
   - Install `uv` for better package management: `pip install uv`
   - Check CUDA compatibility with your GPU
   - Verify transformers installation from GitHub source

## 📁 Project Structure

```
eomt_panoptic_seg/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 📄 License

Open source - feel free to use and modify as needed.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This application requires a GPU for optimal performance. CPU inference is supported but will be significantly slower. For newer NVIDIA GPUs, use PyTorch nightly builds for best compatibility. The app includes proper CUDA tensor handling to prevent memory-related errors.

**Research**: This implementation is based on the CVPR 2025 paper ["Your ViT is Secretly an Image Segmentation Model"](https://www.tue-mps.org/eomt/) by Kerssies et al., demonstrating that plain Vision Transformers can achieve state-of-the-art segmentation performance.
