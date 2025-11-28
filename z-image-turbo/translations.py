"""Multi-language translations for Z-Image-Turbo UI."""

# Supported languages with their display names
LANGUAGES = {
    "en": "English",
    "zh": "中文",
}

# Translation dictionary
TRANSLATIONS = {
    "en": {
        "title": "Z-Image Generation Demo",
        "subtitle": "An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer",
        "prompt_label": "Prompt",
        "prompt_placeholder": "Enter your prompt here...",
        "resolution_category": "Resolution Category",
        "resolution_label": "Width x Height (Ratio)",
        "seed_label": "Seed",
        "random_seed": "Random Seed",
        "steps_label": "Steps",
        "time_shift_label": "Time Shift",
        "generate_btn": "Generate",
        "example_prompts": "Example Prompts",
        "generated_images": "Generated Images",
        "seed_used": "Seed Used",
        "language_label": "Language",
        "model_not_loaded": "Model not loaded.",
    },
    "zh": {
        "title": "Z-Image 图像生成演示",
        "subtitle": "基于单流扩散Transformer的高效图像生成基础模型",
        "prompt_label": "提示词",
        "prompt_placeholder": "在此输入您的提示词...",
        "resolution_category": "分辨率类别",
        "resolution_label": "宽 x 高 (比例)",
        "seed_label": "随机种子",
        "random_seed": "随机种子",
        "steps_label": "步数",
        "time_shift_label": "时间偏移",
        "generate_btn": "生成",
        "example_prompts": "示例提示词",
        "generated_images": "生成的图像",
        "seed_used": "使用的种子",
        "language_label": "语言",
        "model_not_loaded": "模型未加载。",
    },
}


def get_text(lang: str, key: str) -> str:
    """
    Get translated text for a given language and key.

    Args:
        lang (str): Language code (e.g., 'en', 'zh', 'ko')
        key (str): Translation key

    Returns:
        str: Translated text, falls back to English if not found
    """
    if lang not in TRANSLATIONS:
        lang = "en"
    return TRANSLATIONS[lang].get(key, TRANSLATIONS["en"].get(key, key))


def get_language_choices() -> list:
    """
    Get list of language choices for dropdown.

    Returns:
        list: List of (code, display_name) tuples
    """
    return [(code, name) for code, name in LANGUAGES.items()]
