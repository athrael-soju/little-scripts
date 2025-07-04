import io

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, EomtForUniversalSegmentation

# Load model globally to avoid reloading
print("Loading model...")
model_id = "tue-mps/coco_panoptic_eomt_large_640"
processor = AutoImageProcessor.from_pretrained(model_id)
model = EomtForUniversalSegmentation.from_pretrained(model_id)

# Check for CUDA availability and move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded successfully on {device}!")


def run_inference(image):
    """Run panoptic segmentation inference"""
    inputs = processor(images=image, return_tensors="pt")

    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = [(image.height, image.width)]
    preds = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=target_sizes
    )

    return preds[0]


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array, handling CUDA tensors"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor


def visualize_mask(image, segmentation_mask):
    """Show segmentation mask only"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Segmentation mask - convert tensor to numpy
    mask_np = tensor_to_numpy(segmentation_mask)

    if mask_np.max() > 0:
        ax.imshow(mask_np, cmap="tab20")
    else:
        # If no segments, show a blank image with text
        ax.imshow(np.zeros_like(mask_np), cmap="gray")
        ax.text(
            0.5,
            0.5,
            "No segments detected",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="red",
            weight="bold",
        )

    ax.axis("off")

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def visualize_overlay(image, segmentation_mask):
    """Show segmentation overlay on original image"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Original image
    ax.imshow(image)

    # Overlay segmentation mask with transparency
    mask_np = tensor_to_numpy(segmentation_mask)
    ax.imshow(mask_np, cmap="tab20", alpha=0.6)

    ax.axis("off")

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def visualize_contours(image, segmentation_mask):
    """Show contours of segments on original image"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Original image
    ax.imshow(image)

    # Convert mask to numpy and find contours
    mask_np = tensor_to_numpy(segmentation_mask).astype(np.uint8)

    # Find unique segments
    unique_segments = np.unique(mask_np)

    # Create a colormap for distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_segments)))

    # Draw contours for each segment
    for i, segment_id in enumerate(unique_segments):
        if segment_id == 0:  # Skip background
            continue

        # Create binary mask for this segment
        binary_mask = (mask_np == segment_id).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw contours with unique color for each segment
        for contour in contours:
            if len(contour) > 2:  # Only draw if contour has enough points
                contour = contour.reshape(-1, 2)
                ax.plot(
                    contour[:, 0],
                    contour[:, 1],
                    color=colors[i % len(colors)],
                    linewidth=2,
                    alpha=0.8,
                )

    ax.axis("off")

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def visualize_instance_masks(image, segmentation_mask):
    """Show individual instance masks in a grid"""
    mask_np = tensor_to_numpy(segmentation_mask)
    unique_segments, counts = np.unique(mask_np, return_counts=True)

    # Get top 9 segments by size (excluding background)
    non_bg_indices = unique_segments != 0
    if np.any(non_bg_indices):
        top_segments = unique_segments[non_bg_indices][
            np.argsort(counts[non_bg_indices])[-9:]
        ]
        top_counts = counts[non_bg_indices][np.argsort(counts[non_bg_indices])[-9:]]
    else:
        top_segments = []
        top_counts = []

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, (segment_id, count) in enumerate(zip(top_segments, top_counts)):
        binary_mask = (mask_np == segment_id).astype(float)
        axes[i].imshow(binary_mask, cmap="Blues")
        axes[i].set_title(
            f"Segment {segment_id}\nPixels: {count}", fontsize=10, weight="bold"
        )
        axes[i].axis("off")

    # Fill empty subplots with informative text
    for i in range(len(top_segments), 9):
        axes[i].axis("off")
        if i == 0 and len(top_segments) == 0:
            axes[i].text(
                0.5,
                0.5,
                "No segments\ndetected",
                transform=axes[i].transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="red",
                weight="bold",
            )

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def visualize_edges(image, segmentation_mask):
    """Show edge detection on segmentation boundaries"""
    mask_np = tensor_to_numpy(segmentation_mask)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Original image
    ax.imshow(image)

    # Edge detection on mask
    if mask_np.max() > 0:  # Check if mask has any segments
        edges = cv2.Canny((mask_np * 255 / mask_np.max()).astype(np.uint8), 50, 150)

        # Create a colored edge overlay
        edge_overlay = np.zeros((*edges.shape, 4))  # RGBA
        edge_overlay[edges > 0] = [1, 1, 0, 1]  # Yellow with full alpha

        # Overlay edges
        ax.imshow(edge_overlay)
    else:
        # If no segments, just show the original image
        ax.text(
            0.5,
            0.5,
            "No segments detected",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="red",
            weight="bold",
        )

    ax.axis("off")

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def visualize_segment_isolation(image, segmentation_mask):
    """Show the largest segment isolated from the rest"""
    mask_np = tensor_to_numpy(segmentation_mask)
    unique_segments, counts = np.unique(mask_np, return_counts=True)

    # Find largest segment (excluding background)
    non_bg_indices = unique_segments != 0
    if np.any(non_bg_indices):
        largest_segment = unique_segments[non_bg_indices][
            np.argmax(counts[non_bg_indices])
        ]
        largest_count = counts[non_bg_indices][np.argmax(counts[non_bg_indices])]
    else:
        largest_segment = unique_segments[np.argmax(counts)]
        largest_count = counts[np.argmax(counts)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Isolated segment
    isolated_mask = (mask_np == largest_segment).astype(float)

    if isolated_mask.max() > 0:
        ax.imshow(isolated_mask, cmap="Reds")
        ax.set_title(
            f"Largest Segment (ID: {largest_segment}, Pixels: {largest_count})",
            fontsize=14,
            weight="bold",
            pad=20,
        )
    else:
        ax.imshow(np.zeros_like(isolated_mask), cmap="gray")
        ax.text(
            0.5,
            0.5,
            "No segments detected",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="red",
            weight="bold",
        )

    ax.axis("off")

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def visualize_heatmap(image, segmentation_mask):
    """Show boundary density heatmap"""
    mask_np = tensor_to_numpy(segmentation_mask)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if mask_np.max() > 0:
        # Calculate gradient magnitude for boundary detection
        gradient_magnitude = np.gradient(mask_np.astype(float))
        gradient_magnitude = np.sqrt(
            gradient_magnitude[0] ** 2 + gradient_magnitude[1] ** 2
        )

        # Boundary heatmap
        im = ax.imshow(gradient_magnitude, cmap="hot")
        ax.set_title("Boundary Density Heatmap", fontsize=14, weight="bold", pad=20)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Gradient Magnitude")
    else:
        # If no segments, show a blank heatmap
        ax.imshow(np.zeros_like(mask_np), cmap="hot")
        ax.text(
            0.5,
            0.5,
            "No segments detected",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="red",
            weight="bold",
        )
        ax.axis("off")

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def create_visualization(image, viz_type):
    """Create visualization based on selected type"""
    if image is None:
        return None

    try:
        # Run inference
        prediction = run_inference(image)
        segmentation_mask = prediction["segmentation"]

        if viz_type == "Mask":
            return visualize_mask(image, segmentation_mask)
        elif viz_type == "Overlay":
            return visualize_overlay(image, segmentation_mask)
        elif viz_type == "Contours":
            return visualize_contours(image, segmentation_mask)
        elif viz_type == "Instance Masks":
            return visualize_instance_masks(image, segmentation_mask)
        elif viz_type == "Edge Detection":
            return visualize_edges(image, segmentation_mask)
        elif viz_type == "Segment Isolation":
            return visualize_segment_isolation(image, segmentation_mask)
        elif viz_type == "Heatmap":
            return visualize_heatmap(image, segmentation_mask)
        else:
            # Default fallback
            return visualize_mask(image, segmentation_mask)

    except Exception as e:
        print(f"Error in visualization: {e}")
        # Return a simple error visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            f"Error during processing:\n{str(e)}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="red",
            weight="bold",
        )
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        plt.close()

        return Image.open(buf)


def load_sample_image(img_path):
    """Load a sample image from URL"""
    try:
        response = requests.get(img_path, stream=True)
        response.raise_for_status()
        return Image.open(response.raw)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="Panoptic Segmentation Visualizer", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🎨 Panoptic Segmentation Visualizer

        Upload an image and select a visualization type to see different ways of viewing the panoptic segmentation results.
        The model used is `tue-mps/coco_panoptic_eomt_large_640`.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="pil", height=400)

                viz_type = gr.Radio(
                    choices=[
                        "Mask",
                        "Overlay",
                        "Contours",
                        "Instance Masks",
                        "Edge Detection",
                        "Segment Isolation",
                        "Heatmap",
                    ],
                    label="Visualization Type",
                    value="Mask",
                    info="Choose how to visualize the segmentation results",
                )

                process_btn = gr.Button(
                    "🚀 Process Image", variant="primary", size="lg"
                )

                gr.Markdown("""
                ### Visualization Types:
                - **Mask**: Segmentation mask with color-coded segments
                - **Overlay**: Transparent segmentation overlay on original image
                - **Contours**: Segment boundaries outlined on original image
                - **Instance Masks**: Individual instance masks in a grid (top 9 by size)
                - **Edge Detection**: Segmentation boundaries highlighted in yellow
                - **Segment Isolation**: Shows the largest segment isolated from the rest
                - **Heatmap**: Boundary density visualization with color mapping
                """)

            with gr.Column(scale=2):
                output_image = gr.Image(
                    label="Segmentation Result", type="pil", height=600
                )

        process_btn.click(
            fn=create_visualization,
            inputs=[image_input, viz_type],
            outputs=output_image,
        )

        # Sample images with thumbnails
        gr.Markdown("### 📸 Try with sample images:")

        sample_images = [
            ("http://images.cocodataset.org/val2017/000000039769.jpg", "Cats on Couch"),
            ("http://images.cocodataset.org/val2017/000000397133.jpg", "Street Scene"),
            ("http://images.cocodataset.org/val2017/000000037777.jpg", "Living Room"),
            (
                "http://images.cocodataset.org/val2017/000000174482.jpg",
                "Person with Laptop",
            ),
            ("http://images.cocodataset.org/val2017/000000000785.jpg", "Dining Table"),
        ]

        def create_thumbnail_gallery():
            """Create a gallery of clickable thumbnails"""
            gallery_images = []
            for img_url, img_name in sample_images:
                try:
                    img = load_sample_image(img_url)
                    if img:
                        # Resize to thumbnail while maintaining aspect ratio
                        img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                        gallery_images.append((img, img_name))
                except Exception as e:
                    print(f"Failed to load {img_name}: {e}")
                    continue
            return gallery_images

        with gr.Row():
            thumbnail_gallery = gr.Gallery(
                value=create_thumbnail_gallery(),
                label="Sample Images",
                show_label=True,
                elem_id="thumbnail_gallery",
                columns=5,
                rows=1,
                object_fit="contain",
                height=200,
                allow_preview=False,
            )

        def select_from_gallery(evt: gr.SelectData):
            """Handle gallery selection"""
            selected_idx = evt.index
            if selected_idx < len(sample_images):
                img_url, _ = sample_images[selected_idx]
                return load_sample_image(img_url)
            return None

        thumbnail_gallery.select(select_from_gallery, outputs=image_input)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
