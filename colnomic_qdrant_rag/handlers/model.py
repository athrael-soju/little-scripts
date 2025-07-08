import config
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

    def _get_patches(self, image_size):
        """Get the number of patches for an image."""
        # ColQwen2_5_Processor requires spatial_merge_size parameter
        # Using default value of 2 which is standard for ColQwen models
        return self.processor.get_n_patches(image_size, spatial_merge_size=2)

    def _generate_pooled_embeddings(self, image_embeddings, processed_images, images):
        """Generate mean-pooled embeddings by rows and columns (ColQwen optimization)."""
        pooled_by_rows_batch = []
        pooled_by_columns_batch = []

        for image_embedding, tokenized_image, image in zip(
            image_embeddings, processed_images.input_ids, images
        ):
            x_patches, y_patches = self._get_patches(image.size)
            image_tokens_mask = tokenized_image == self.processor.image_token_id
            image_tokens = image_embedding[image_tokens_mask].view(
                x_patches, y_patches, self.model.dim
            )

            # Mean pooling by rows and columns
            pooled_by_rows = torch.mean(image_tokens, dim=0)
            pooled_by_columns = torch.mean(image_tokens, dim=1)

            # Get prefix and postfix tokens
            image_token_idxs = torch.nonzero(image_tokens_mask.int(), as_tuple=False)
            first_image_token_idx = image_token_idxs[0].cpu().item()
            last_image_token_idx = image_token_idxs[-1].cpu().item()
            prefix_tokens = image_embedding[:first_image_token_idx]
            postfix_tokens = image_embedding[last_image_token_idx + 1 :]

            # Concatenate prefix + pooled + postfix
            pooled_by_rows = torch.cat(
                (prefix_tokens, pooled_by_rows, postfix_tokens), dim=0
            )
            pooled_by_columns = torch.cat(
                (prefix_tokens, pooled_by_columns, postfix_tokens), dim=0
            )

            pooled_by_rows_batch.append(pooled_by_rows.cpu().float().numpy().tolist())
            pooled_by_columns_batch.append(
                pooled_by_columns.cpu().float().numpy().tolist()
            )

        return pooled_by_rows_batch, pooled_by_columns_batch

    def get_image_embeddings(self, images):
        """Generate embeddings for a batch of images."""
        with torch.no_grad():
            batch_images = self.processor.process_images(images).to(self.device)
            image_embeddings = self.model(**batch_images)

            if config.ENABLE_RERANKING_OPTIMIZATION:
                # Generate both original and pooled embeddings
                pooled_by_rows, pooled_by_columns = self._generate_pooled_embeddings(
                    image_embeddings, batch_images, images
                )
                return {
                    "original": image_embeddings,
                    "pooled_rows": pooled_by_rows,
                    "pooled_columns": pooled_by_columns,
                }
            else:
                # Return original embeddings only (backward compatibility)
                return image_embeddings

    def get_query_embedding(self, query_text):
        """Generate embedding for a text query."""
        with torch.no_grad():
            batch_query = self.processor.process_queries([query_text]).to(self.device)
            query_embeddings = self.model(**batch_query)

            if config.ENABLE_RERANKING_OPTIMIZATION:
                # For queries, we need the embedding as numpy array for all vector types
                query_embedding_np = query_embeddings.cpu().float().numpy()
                return {
                    "original": query_embedding_np,
                    "pooled_rows": query_embedding_np,  # Same embedding for all query types
                    "pooled_columns": query_embedding_np,
                }
            else:
                # Return original embeddings only (backward compatibility)
                return query_embeddings
