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
# Language-specific example prompts
EXAMPLE_PROMPTS = {
    "en": [
        [
            "A gentleman and his poodle wearing matching outfits at a dog show, indoor lighting, with audience in the background."
        ],
        [
            "An atmospheric dark portrait of an elegant Chinese beauty in a dark room. A strong beam of light passes through a louver, casting a clear lightning-shaped shadow on her face, illuminating just one eye. High contrast, sharp boundary between light and dark, mysterious atmosphere, Leica camera tones."
        ],
        [
            "A medium-shot phone selfie of a young East Asian woman with long black hair taking a mirror selfle in a brightly lit elevator. She wears a black off-shoulder crop top with white flower patterns and dark jeans. Her head is slightly tilted, lips pursed in a kissing pose, very cute and playful. She holds a dark gray smartphone in her right hand, covering part of her face, with the rear camera lens facing the mirror."
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
    ],
    "zh": [
        [
            "一位绅士和他的贵宾犬在狗展上穿着配套的服装,室内照明,背景中有观众。"
        ],
        [
            "一幅优雅的中国美女在黑暗房间中的氛围暗黑肖像。一束强烈的光线穿过百叶窗,在她的脸上投下清晰的闪电状阴影,只照亮一只眼睛。高对比度,明暗边界分明,神秘氛围,徕卡相机色调。"
        ],
        [
            "一张中景手机自拍照,一位长黑发的年轻东亚女性在明亮的电梯里拍镜子自拍。她穿着黑色露肩短上衣,上面有白色花朵图案,配深色牛仔裤。她的头微微倾斜,嘴唇噘起做亲吻姿势,非常可爱俏皮。她右手拿着深灰色智能手机,遮住部分脸,后置摄像头镜头面向镜子。"
        ],
        [
            "身穿红色汉服的年轻中国女子,刺绣精美。妆容完美,红色花卉额饰。精致高髻,金色凤凰头饰,红花,珠饰。手持圆形折扇,上面有仙女、树木、鸟。霓虹闪电灯,明亮的黄色光芒,照在伸出的左手掌上方。柔和照明的户外夜景背景,轮廓分明的多层宝塔(西安大雁塔),模糊的彩色远处灯光。"
        ],
        [
            '''一幅垂直数字插图,描绘了宁静壮丽的中国山水,以传统山水画风格呈现,但具有现代、简洁的美学。场景由高耸陡峭的悬崖主导,呈现各种深浅的蓝色和青绿色,框出中央山谷。远处,层层山峦消失在浅蓝色和白色雾霭中,营造出强烈的大气透视和深度感。一条平静的绿松石色河流穿过构图中心,一艘小型中国传统船只,可能是舢板,在水面航行。船有明亮的黄色船篷和红色船身,身后留下轻柔的波纹。船上载着几个模糊的人影。稀疏的植被,包括绿树和一些光秃树枝的树木,附着在岩石壁架和山峰上。整体照明柔和弥漫,为整个场景投下宁静的光辉。'''
        ],
        [
            """一张虚构的英文电影《记忆的味道》的电影海报。场景设置在一个质朴的19世纪风格厨房。画面中央,一位棕红色头发、留着小胡子的中年男子(由演员亚瑟·潘哈里根饰演)站在木桌后面。他穿着白衬衫、黑色背心和米色围裙,看着一位女士,手里拿着一大块生红肉,下面有一块木制砧板。在他右侧,一位黑发高髻的女子(由女演员埃莉诺·万斯饰演)倚在桌上,温柔地对他微笑。她穿着浅色衬衫和上白下蓝的长裙。桌上除了放有切好的葱和卷心菜丝的砧板外,还有一个白色瓷盘、新鲜香草,左边有一个装着一串深色葡萄的木盒。背景是粗糙的灰白色粉刷墙壁,上面挂着一幅风景画。最右边的台面上放着一盏老式油灯。"""
        ],
        [
            """一张近景方形构图照片,主体是一片巨大、明亮的绿色植物叶子,覆盖着文字,使其看起来像海报或杂志封面。主体是一片厚实、蜡质纹理的叶子,从左下角到右上角对角弯曲穿过画面。其表面高度反射,捕捉到明亮的直射光源,形成突出的高光,平行的细脉纹在明亮表面下可见。背景由其他深绿色叶子组成,略微失焦,创造出浅景深效果,突出前景主叶。整体风格是写实摄影,明亮叶子和黑暗阴影背景之间有高对比度。"""
        ],
    ],
}
# =============================================================================
