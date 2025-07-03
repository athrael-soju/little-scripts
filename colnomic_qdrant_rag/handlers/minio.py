import config
import io
import json
from minio import Minio
from minio.error import S3Error


class MinioHandler:
    """Handles all interactions with the MinIO object store."""

    def __init__(self):
        """Initializes the MinIO client."""
        try:
            self.client = Minio(
                config.MINIO_ENDPOINT,
                access_key=config.MINIO_ACCESS_KEY,
                secret_key=config.MINIO_SECRET_KEY,
                secure=config.MINIO_USE_SSL,
            )
            self.bucket_name = config.MINIO_BUCKET
        except Exception as e:
            raise ConnectionError(f"Failed to initialize MinIO client: {e}")

    def ensure_bucket_exists(self):
        """Checks if the bucket exists, creates it if it doesn't, and ensures it's public."""
        try:
            found = self.client.bucket_exists(self.bucket_name)
            if not found:
                # print(f"Bucket '{self.bucket_name}' not found. Creating it...")
                self.client.make_bucket(self.bucket_name)
                # print(f"Bucket '{self.bucket_name}' created successfully.")
                self.set_public_policy()
            else:
                # print(f"Bucket '{self.bucket_name}' already exists.")
                # For existing buckets, we can also ensure the policy is set.
                # This is useful if the policy was ever changed manually.
                self.set_public_policy()

        except S3Error as e:
            raise ConnectionError(
                f"Failed to check or create bucket '{self.bucket_name}': {e}"
            )

    def set_public_policy(self):
        """Sets a public read-only policy on the bucket."""
        try:
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": ["*"]},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{self.bucket_name}/*"],
                    },
                ],
            }
            self.client.set_bucket_policy(self.bucket_name, json.dumps(policy))
            # print(f"Public read policy set for bucket '{self.bucket_name}'.")
        except S3Error as e:
            print(
                f"Warning: Could not set public policy for bucket '{self.bucket_name}': {e}. Please ensure the MinIO user has 's3:SetBucketPolicy' permissions."
            )

    def upload_image(self, image_name: str, image_data: bytes) -> str:
        """
        Uploads an image to MinIO and returns its public URL.

        Args:
            image_name: The name for the image file in the bucket.
            image_data: The raw image data in bytes.

        Returns:
            The public URL of the uploaded image.
        """
        try:
            image_stream = io.BytesIO(image_data)
            self.client.put_object(
                self.bucket_name,
                image_name,
                image_stream,
                length=len(image_data),
                content_type="image/png",  # Assuming PNG, can be made more generic
            )

            # Construct the public URL
            protocol = "https" if config.MINIO_USE_SSL else "http"
            url = (
                f"{protocol}://{config.MINIO_ENDPOINT}/{self.bucket_name}/{image_name}"
            )
            return url

        except S3Error as e:
            raise IOError(f"Failed to upload image '{image_name}' to MinIO: {e}")
