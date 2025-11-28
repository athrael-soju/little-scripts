"""Gradio UI for Z-Image-Turbo application."""

import random
import gradio as gr
from generator import generate_image
from prompt_expander import prompt_enhance
from config import RES_CHOICES, RESOLUTION_SET, EXAMPLE_PROMPTS
from translations import TRANSLATIONS, LANGUAGES, get_text


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
        lang="en",
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
            lang (str): Current language code
            progress (gr.Progress): Gradio progress tracker

        Returns:
            tuple: (gallery_images, seed_str, seed_int)
        """
        if pipe is None:
            raise gr.Error(get_text(lang, "model_not_loaded"))

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


def update_ui_language(lang):
    """
    Update all UI components with the selected language.

    Args:
        lang (str): Language code

    Returns:
        tuple: Updated component properties
    """
    t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])

    # Get language-specific example prompts, fallback to English
    examples = EXAMPLE_PROMPTS.get(lang, EXAMPLE_PROMPTS["en"])

    # Return updates for all translatable components
    return (
        # Header markdown
        f"""<div align="center">

# {t['title']}

[![GitHub](https://img.shields.io/badge/GitHub-Z--Image-181717?logo=github&logoColor=white)](https://github.com/Tongyi-MAI/Z-Image)

*{t['subtitle']}*

</div>""",
        # Prompt input
        gr.update(label=t["prompt_label"], placeholder=t["prompt_placeholder"]),
        # Resolution category
        gr.update(label=t["resolution_category"]),
        # Resolution dropdown
        gr.update(label=t["resolution_label"]),
        # Seed
        gr.update(label=t["seed_label"]),
        # Random seed checkbox
        gr.update(label=t["random_seed"]),
        # Steps slider
        gr.update(label=t["steps_label"]),
        # Time shift slider
        gr.update(label=t["time_shift_label"]),
        # Generate button
        gr.update(value=t["generate_btn"]),
        # Example prompts header
        f"### {t['example_prompts']}",
        # Example dataset with language-specific prompts
        gr.update(samples=examples),
        # Output gallery
        gr.update(label=t["generated_images"]),
        # Seed used textbox
        gr.update(label=t["seed_used"]),
    )


def create_ui(pipe, prompt_expander_instance):
    """
    Create and configure the Gradio UI.

    Args:
        pipe: ZImagePipeline instance
        prompt_expander_instance: PromptExpander instance

    Returns:
        gr.Blocks: Configured Gradio interface
    """
    # Get initial translations
    initial_lang = "en"
    t = TRANSLATIONS[initial_lang]

    with gr.Blocks(title="Z-Image Demo") as demo:
        # Language selector at the top
        with gr.Row():
            with gr.Column(scale=4):
                pass  # Spacer
            with gr.Column(scale=1, min_width=150):
                lang_dropdown = gr.Dropdown(
                    value="en",
                    choices=list(LANGUAGES.keys()),
                    label=t["language_label"],
                    info="🌐",
                )

        # Store current language in state
        current_lang = gr.State(value="en")

        header_md = gr.Markdown(
            f"""<div align="center">

# {t['title']}

[![GitHub](https://img.shields.io/badge/GitHub-Z--Image-181717?logo=github&logoColor=white)](https://github.com/Tongyi-MAI/Z-Image)

*{t['subtitle']}*

</div>"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label=t["prompt_label"],
                    lines=3,
                    placeholder=t["prompt_placeholder"],
                )

                with gr.Row():
                    choices = [int(k) for k in RES_CHOICES.keys()]
                    res_cat = gr.Dropdown(
                        value=1024,
                        choices=choices,
                        label=t["resolution_category"],
                    )

                    initial_res_choices = RES_CHOICES["1024"]
                    resolution = gr.Dropdown(
                        value=initial_res_choices[0],
                        choices=RESOLUTION_SET,
                        label=t["resolution_label"],
                    )

                with gr.Row():
                    seed = gr.Number(label=t["seed_label"], value=42, precision=0)
                    random_seed = gr.Checkbox(label=t["random_seed"], value=True)

                with gr.Row():
                    steps = gr.Slider(
                        label=t["steps_label"],
                        minimum=1,
                        maximum=100,
                        value=8,
                        step=1,
                        interactive=False,
                    )
                    shift = gr.Slider(
                        label=t["time_shift_label"],
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.1,
                    )

                generate_btn = gr.Button(t["generate_btn"], variant="primary")

                example_header = gr.Markdown(f"### {t['example_prompts']}")
                example_dataset = gr.Dataset(
                    components=[prompt_input],
                    samples=EXAMPLE_PROMPTS["en"],
                    type="index",
                )

            with gr.Column(scale=1):
                output_gallery = gr.Gallery(
                    label=t["generated_images"],
                    columns=2,
                    rows=2,
                    height=600,
                    object_fit="contain",
                    format="png",
                    interactive=False,
                )
                used_seed = gr.Textbox(label=t["seed_used"], interactive=False)

        # Language change handler
        lang_dropdown.change(
            update_ui_language,
            inputs=[lang_dropdown],
            outputs=[
                header_md,
                prompt_input,
                res_cat,
                resolution,
                seed,
                random_seed,
                steps,
                shift,
                generate_btn,
                example_header,
                example_dataset,
                output_gallery,
                used_seed,
            ],
        )

        # Update current language state
        lang_dropdown.change(
            lambda x: x,
            inputs=[lang_dropdown],
            outputs=[current_lang],
        )

        res_cat.change(update_res_choices, inputs=res_cat, outputs=resolution)

        # Example dataset click handler
        def select_example(evt: gr.SelectData, lang):
            """Handle example selection based on current language."""
            examples = EXAMPLE_PROMPTS.get(lang, EXAMPLE_PROMPTS["en"])
            return examples[evt.index][0]

        example_dataset.select(
            select_example,
            inputs=[current_lang],
            outputs=[prompt_input],
        )

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
                current_lang,
            ],
            outputs=[output_gallery, used_seed, seed],
        )

    return demo
