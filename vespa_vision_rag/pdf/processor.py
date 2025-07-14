import os
import tempfile
from io import BytesIO
from typing import List, Tuple, Dict, Any
from PIL import Image
import requests
from pdf2image import convert_from_path
from pypdf import PdfReader
import base64

from config import Config


class PDFProcessor:
    """Handler class for PDF processing operations"""
    
    def __init__(self):
        """Initialize the PDF processor"""
        self.temp_dir = tempfile.mkdtemp()
    
    def download_pdf(self, url: str) -> BytesIO:
        """
        Download a PDF from a URL
        
        Args:
            url: URL of the PDF to download
            
        Returns:
            BytesIO object containing the PDF content
            
        Raises:
            Exception: If download fails
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BytesIO(response.content)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download PDF from {url}: {str(e)}")
    
    def extract_pdf_content(self, pdf_url: str) -> Tuple[List[Image.Image], List[str]]:
        """
        Extract images and text from a PDF
        
        Args:
            pdf_url: URL of the PDF to process
            
        Returns:
            Tuple of (images, page_texts) where images are PIL Images and page_texts are strings
        """
        # Download the PDF
        pdf_file = self.download_pdf(pdf_url)
        
        # Create temporary file for pdf2image
        temp_file = os.path.join(self.temp_dir, "temp.pdf")
        
        try:
            # Save PDF to temporary file
            with open(temp_file, "wb") as f:
                f.write(pdf_file.read())
            
            # Extract text using pypdf
            reader = PdfReader(temp_file)
            page_texts = []
            for page_number in range(len(reader.pages)):
                page = reader.pages[page_number]
                text = page.extract_text()
                page_texts.append(text)
            
            # Convert to images using pdf2image
            images = convert_from_path(temp_file)
            
            # Validate that we have the same number of pages
            if len(images) != len(page_texts):
                raise ValueError(f"Mismatch: {len(images)} images vs {len(page_texts)} text pages")
            
            return images, page_texts
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def process_multiple_pdfs(self, pdf_configs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process multiple PDFs and extract their content
        
        Args:
            pdf_configs: List of dictionaries with 'title' and 'url' keys
            
        Returns:
            List of dictionaries with 'title', 'url', 'images', and 'texts' keys
        """
        processed_pdfs = []
        
        for pdf_config in pdf_configs:
            print(f"Processing PDF: {pdf_config['title']}")
            
            try:
                images, texts = self.extract_pdf_content(pdf_config['url'])
                
                processed_pdf = {
                    'title': pdf_config['title'],
                    'url': pdf_config['url'],
                    'images': images,
                    'texts': texts
                }
                
                processed_pdfs.append(processed_pdf)
                print(f"Successfully processed {len(images)} pages")
                
            except Exception as e:
                print(f"Error processing PDF {pdf_config['title']}: {str(e)}")
                continue
        
        return processed_pdfs
    
    def resize_image(self, image: Image.Image, max_height: int = None) -> Image.Image:
        """
        Resize an image to a maximum height while maintaining aspect ratio
        
        Args:
            image: PIL Image to resize
            max_height: Maximum height in pixels. Defaults to Config.MAX_IMAGE_HEIGHT
            
        Returns:
            Resized PIL Image
        """
        max_height = max_height or Config.MAX_IMAGE_HEIGHT
        
        width, height = image.size
        if height > max_height:
            ratio = max_height / height
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return image.resize((new_width, new_height))
        
        return image
    
    def image_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """
        Convert a PIL Image to base64 string
        
        Args:
            image: PIL Image to convert
            format: Image format (JPEG, PNG, etc.)
            
        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        image.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def cleanup(self):
        """Clean up temporary directory"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup() 