"""Gradio UI for Z-Image-Turbo application."""

import random
import gradio as gr
from generator import generate_image
from prompt_expander import prompt_enhance
from config import RES_CHOICES, RESOLUTION_SET, EXAMPLE_PROMPTS


def create_generate_handler(pipe, prompt_expander_instance):
    """
    Create a generate handler function with access to pipe and prompt_expander.

    Args:
        pipe: ZImagePipeline instance
        prompt_expander_instance: PromptExpander instance

    Returns:
        function: Generate handler function for Gradio
    """

    def generate(
        prompt,
        resolution="1024x1024 ( 1:1 )",
        seed=42,
        steps=9,
        shift=3.0,
        enhance=False,
        random_seed=True,
        gallery_images=None,
        progress=gr.Progress(track_tqdm=True),
    ):
        """
        Generate an image using the Z-Image model based on the provided prompt and settings.

        Args:
            prompt (str): Text prompt describing the desired image content
            resolution (str): Output resolution in format "WIDTHxHEIGHT ( RATIO )"
            seed (int): Seed for reproducible generation
            steps (int): Number of inference steps for the diffusion process
            shift (float): Time shift parameter for the flow matching scheduler
            enhance (bool): Whether to enhance the prompt using DashScope API
            random_seed (bool): Whether to generate a new random seed
            gallery_images (list): List of previously generated images to append to
            progress (gr.Progress): Gradio progress tracker

        Returns:
            tuple: (gallery_images, seed_str, seed_int)
        """
        if pipe is None:
            raise gr.Error("Model not loaded.")

        final_prompt = prompt

        if enhance:
            final_prompt, _ = prompt_enhance(prompt, True, prompt_expander_instance)
            print(f"Enhanced prompt: {final_prompt}")

        if random_seed:
            new_seed = random.randint(1, 1000000)
        else:
            new_seed = seed if seed != -1 else random.randint(1, 1000000)

        try:
            resolution_str = resolution.split(" ")[0]
        except Exception:
            resolution_str = "1024x1024"

        image = generate_image(
            pipe=pipe,
            prompt=final_prompt,
            resolution=resolution_str,
            seed=new_seed,
            guidance_scale=0.0,
            num_inference_steps=int(steps + 1),
            shift=shift,
        )

        if gallery_images is None:
            gallery_images = []
        gallery_images.append(image)

        return gallery_images, str(new_seed), int(new_seed)

    return generate


def update_res_choices(_res_cat):
    """
    Update resolution choices based on selected category.

    Args:
        _res_cat: Selected resolution category

    Returns:
        gr.update: Gradio update object with new choices
    """
    if str(_res_cat) in RES_CHOICES:
        res_choices = RES_CHOICES[str(_res_cat)]
    else:
        res_choices = RES_CHOICES["1024"]
    return gr.update(value=res_choices[0], choices=res_choices)


def create_ui(pipe, prompt_expander_instance):
    """
    Create and configure the Gradio UI.

    Args:
        pipe: ZImagePipeline instance
        prompt_expander_instance: PromptExpander instance

    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(title="Z-Image Demo") as demo:
        gr.Markdown(
            """<div align="center">

# Z-Image Generation Demo

[![GitHub](https://img.shields.io/badge/GitHub-Z--Image-181717?logo=github&logoColor=white)](https://github.com/Tongyi-MAI/Z-Image)

*An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer*

</div>"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt", lines=3, placeholder="Enter your prompt here..."
                )

                with gr.Row():
                    choices = [int(k) for k in RES_CHOICES.keys()]
                    res_cat = gr.Dropdown(
                        value=1024, choices=choices, label="Resolution Category"
                    )

                    initial_res_choices = RES_CHOICES["1024"]
                    resolution = gr.Dropdown(
                        value=initial_res_choices[0],
                        choices=RESOLUTION_SET,
                        label="Width x Height (Ratio)",
                    )

                with gr.Row():
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    random_seed = gr.Checkbox(label="Random Seed", value=True)

                with gr.Row():
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=100,
                        value=8,
                        step=1,
                        interactive=False,
                    )
                    shift = gr.Slider(
                        label="Time Shift",
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.1,
                    )

                generate_btn = gr.Button("Generate", variant="primary")

                gr.Markdown("### Example Prompts")
                gr.Examples(examples=EXAMPLE_PROMPTS, inputs=prompt_input, label=None)

            with gr.Column(scale=1):
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    columns=2,
                    rows=2,
                    height=600,
                    object_fit="contain",
                    format="png",
                    interactive=False,
                )
                used_seed = gr.Textbox(label="Seed Used", interactive=False)

        res_cat.change(update_res_choices, inputs=res_cat, outputs=resolution)

        # Enable enhance is disabled by default
        enable_enhance = gr.State(value=False)

        generate_handler = create_generate_handler(pipe, prompt_expander_instance)
        generate_btn.click(
            generate_handler,
            inputs=[
                prompt_input,
                resolution,
                seed,
                steps,
                shift,
                enable_enhance,
                random_seed,
                output_gallery,
            ],
            outputs=[output_gallery, used_seed, seed],
        )

    return demo
