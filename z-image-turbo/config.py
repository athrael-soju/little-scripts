"""Configuration management for Z-Image-Turbo application."""

import os


# ==================== Environment Variables ==================================
MODEL_PATH = os.environ.get("MODEL_PATH", "Tongyi-MAI/Z-Image-Turbo")
ENABLE_COMPILE = os.environ.get("ENABLE_COMPILE", "false").lower() == "true"
ENABLE_WARMUP = os.environ.get("ENABLE_WARMUP", "false").lower() == "true"
# =============================================================================


# ==================== Auto-detect Flash Attention ============================
def detect_attention_backend():
    """Auto-detect the best available attention backend."""
    backend = os.environ.get("ATTENTION_BACKEND", "auto")

    if backend != "auto":
        return backend

    # Try to detect Flash Attention 2
    try:
        import flash_attn  # noqa: F401

        print("Flash Attention detected and will be used")
        return "flash"  # Backend name is 'flash' not 'flash_attn'
    except ImportError:
        pass

    # Check if Flash Attention 3 is available
    try:
        import flash_attn_3  # noqa: F401

        print("Flash Attention 3 detected and will be used")
        return "_flash_3"  # Backend name for Flash Attention 3
    except ImportError:
        pass

    print("Flash Attention not found, using native attention backend")
    return "native"


ATTENTION_BACKEND = detect_attention_backend()
# =============================================================================


# ==================== Resolution Choices ======================================
RES_CHOICES = {
    "1024": [
        "1024x1024 ( 1:1 )",
        "1152x896 ( 9:7 )",
        "896x1152 ( 7:9 )",
        "1152x864 ( 4:3 )",
        "864x1152 ( 3:4 )",
        "1248x832 ( 3:2 )",
        "832x1248 ( 2:3 )",
        "1280x720 ( 16:9 )",
        "720x1280 ( 9:16 )",
        "1344x576 ( 21:9 )",
        "576x1344 ( 9:21 )",
    ],
    "1280": [
        "1280x1280 ( 1:1 )",
        "1440x1120 ( 9:7 )",
        "1120x1440 ( 7:9 )",
        "1472x1104 ( 4:3 )",
        "1104x1472 ( 3:4 )",
        "1536x1024 ( 3:2 )",
        "1024x1536 ( 2:3 )",
        "1600x896 ( 16:9 )",
        "896x1600 ( 9:16 )",
        "1680x720 ( 21:9 )",
        "720x1680 ( 9:21 )",
    ],
}

RESOLUTION_SET = []
for resolutions in RES_CHOICES.values():
    RESOLUTION_SET.extend(resolutions)
# =============================================================================


# ==================== Example Prompts =========================================
EXAMPLE_PROMPTS = [
    [
        "A gentleman and his poodle wearing matching outfits at a dog show, indoor lighting, with audience in the background."
    ],
    [
        "An atmospheric dark portrait of an elegant Chinese beauty in a dark room. A strong beam of light passes through a louver, casting a clear lightning-shaped shadow on her face, illuminating just one eye. High contrast, sharp boundary between light and dark, mysterious atmosphere, Leica camera tones."
    ],
    [
        "A medium-shot phone selfie of a young East Asian woman with long black hair taking a mirror selfie in a brightly lit elevator. She wears a black off-shoulder crop top with white flower patterns and dark jeans. Her head is slightly tilted, lips pursed in a kissing pose, very cute and playful. She holds a dark gray smartphone in her right hand, covering part of her face, with the rear camera lens facing the mirror."
    ],
    [
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp, bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (Xi'an Giant Wild Goose Pagoda), blurred colorful distant lights."
    ],
    [
        '''A vertical digital illustration depicting a serene and majestic Chinese landscape, rendered in a style reminiscent of traditional Shanshui painting but with a modern, clean aesthetic. The scene is dominated by towering, steep cliffs in various shades of blue and teal, which frame a central valley. In the distance, layers of mountains fade into a light blue and white mist, creating a strong sense of atmospheric perspective and depth. A calm, turquoise river flows through the center of the composition, with a small, traditional Chinese boat, possibly a sampan, navigating its waters. The boat has a bright yellow canopy and a red hull, and it leaves a gentle wake behind it. It carries several indistinct figures of people. Sparse vegetation, including green trees and some bare-branched trees, clings to the rocky ledges and peaks. The overall lighting is soft and diffused, casting a tranquil glow over the entire scene.'''
    ],
    [
        """A fictional movie poster for the English film 'The Taste of Memory'. The scene is set in a rustic 19th-century style kitchen. In the center of the frame, a middle-aged man with reddish-brown hair and a small mustache (played by actor Arthur Penhaligon) stands behind a wooden table. He wears a white shirt, black vest, and beige apron, looking at a lady while holding a large piece of raw red meat, with a wooden cutting board below. To his right, a black-haired woman with her hair in a high bun (played by actress Eleanor Vance) leans on the table, smiling gently at him. She wears a light-colored shirt and a long dress that is white on top and blue on the bottom. On the table, in addition to a cutting board with chopped scallions and shredded cabbage, there is a white ceramic plate, fresh herbs, and on the left, a wooden box with a cluster of dark grapes. The background is a rough gray-white plastered wall with a landscape painting hanging on it. On the far right countertop sits a vintage oil lamp."""
    ],
    [
        """A close-up square composition photograph with the main subject being a huge, bright green plant leaf, overlaid with text, giving it the appearance of a poster or magazine cover. The main subject is a thick, waxy-textured leaf that curves diagonally across the frame from the lower left to the upper right. Its surface is highly reflective, capturing a bright direct light source, forming a prominent highlight, with parallel fine veins visible beneath the bright surface. The background consists of other dark green leaves, slightly out of focus, creating a shallow depth of field effect that highlights the main foreground leaf. The overall style is realistic photography, with high contrast between the bright leaves and the dark shadowed background."""
    ],
]
# =============================================================================
