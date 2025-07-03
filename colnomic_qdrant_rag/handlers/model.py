import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available


class ModelHandler:
    """Handles the ColPali model and processing."""

    def __init__(self, model_name, processor_name):
        self.model_name = model_name
        self.processor_name = processor_name
        self.model = None
        self.processor = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def setup(self):
        """Initialize ColPali model and processor if not already loaded."""
        if self.model and self.processor:
            print("Model already loaded.")
            return

        print("Loading ColPali model...")
        self.model = ColQwen2_5.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2"
            if is_flash_attn_2_available()
            else None,
        )
        self.processor = ColQwen2_5_Processor.from_pretrained(self.model_name)

    def get_image_embeddings(self, images):
        """Generate embeddings for a batch of images."""
        with torch.no_grad():
            batch_images = self.processor.process_images(images).to(self.device)
            return self.model(**batch_images)

    def get_query_embedding(self, query_text):
        """Generate embedding for a text query."""
        with torch.no_grad():
            batch_query = self.processor.process_queries([query_text]).to(self.device)
            return self.model(**batch_query)
