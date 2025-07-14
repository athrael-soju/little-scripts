from typing import List, Dict, Any, Optional
from PIL import Image
from vespa.io import VespaQueryResponse


class DisplayUtils:
    """Utilities for displaying query results and images"""
    
    def __init__(self):
        """Initialize display utilities"""
        pass
    
    def display_query_results(self, query: str, response: VespaQueryResponse, hits: int = 5) -> None:
        """
        Display query results in a formatted way
        
        Args:
            query: The original query text
            response: Vespa query response
            hits: Number of hits to display
        """
        try:
            # Try to import IPython display for notebook environments
            from IPython.display import display, HTML
            
            query_time = response.json.get("timing", {}).get("searchtime", -1)
            query_time = round(query_time, 2)
            count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
            
            html_content = f"<h3>Query text: '{query}', query time {query_time}s, count={count}, top results:</h3>"
            
            for i, hit in enumerate(response.hits[:hits]):
                title = hit["fields"]["title"]
                url = hit["fields"]["url"]
                page = hit["fields"]["page_number"]
                image = hit["fields"]["image"]
                score = hit["relevance"]
                
                html_content += f"<h4>PDF Result {i + 1}</h4>"
                html_content += f'<p><strong>Title:</strong> <a href="{url}">{title}</a>, page {page+1} with score {score:.2f}</p>'
                html_content += f'<img src="data:image/png;base64,{image}" style="max-width:100%;">'
            
            display(HTML(html_content))
            
        except ImportError:
            # Fallback to console display if IPython is not available
            self._display_console_results(query, response, hits)
    
    def _display_console_results(self, query: str, response: VespaQueryResponse, hits: int = 5) -> None:
        """
        Display query results in console format
        
        Args:
            query: The original query text
            response: Vespa query response
            hits: Number of hits to display
        """
        query_time = response.json.get("timing", {}).get("searchtime", -1)
        query_time = round(query_time, 2)
        count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
        
        print(f"\n=== Query Results ===")
        print(f"Query: '{query}'")
        print(f"Query time: {query_time}s")
        print(f"Total count: {count}")
        print(f"Showing top {min(hits, len(response.hits))} results:\n")
        
        for i, hit in enumerate(response.hits[:hits]):
            title = hit["fields"]["title"]
            url = hit["fields"]["url"]
            page = hit["fields"]["page_number"]
            score = hit["relevance"]
            
            print(f"Result {i + 1}:")
            print(f"  Title: {title}")
            print(f"  URL: {url}")
            print(f"  Page: {page + 1}")
            print(f"  Score: {score:.2f}")
            print(f"  Image: [Base64 data available]")
            print()
    
    def display_image(self, image: Image.Image, title: str = None) -> None:
        """
        Display a PIL image
        
        Args:
            image: PIL Image to display
            title: Optional title for the image
        """
        try:
            from IPython.display import display
            
            if title:
                print(f"=== {title} ===")
            
            display(image)
            
        except ImportError:
            print(f"Image display not available in console mode")
            if title:
                print(f"Image: {title}")
            print(f"Image size: {image.size}")
            print(f"Image mode: {image.mode}")
    
    def display_text_content(self, text: str, title: str = None, max_length: int = 1000) -> None:
        """
        Display text content with optional truncation
        
        Args:
            text: Text to display
            title: Optional title for the text
            max_length: Maximum length to display
        """
        if title:
            print(f"\n=== {title} ===")
        
        if len(text) > max_length:
            print(f"{text[:max_length]}...")
            print(f"\n[Text truncated - showing first {max_length} characters of {len(text)} total]")
        else:
            print(text)
    
    def display_pdf_info(self, pdf_info: Dict[str, Any]) -> None:
        """
        Display information about a processed PDF
        
        Args:
            pdf_info: Dictionary containing PDF information
        """
        print(f"\n=== PDF Information ===")
        print(f"Title: {pdf_info.get('title', 'Unknown')}")
        print(f"URL: {pdf_info.get('url', 'Unknown')}")
        print(f"Pages: {len(pdf_info.get('images', []))}")
        print(f"Text pages: {len(pdf_info.get('texts', []))}")
        
        if 'embeddings' in pdf_info:
            print(f"Embeddings: {len(pdf_info['embeddings'])}")
    
    def display_stats(self, stats: Dict[str, Any], title: str = "Statistics") -> None:
        """
        Display statistics in a formatted way
        
        Args:
            stats: Dictionary of statistics to display
            title: Title for the statistics section
        """
        print(f"\n=== {title} ===")
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    
    def display_embeddings_info(self, embeddings: List, title: str = "Embeddings Info") -> None:
        """
        Display information about embeddings
        
        Args:
            embeddings: List of embeddings
            title: Title for the embeddings section
        """
        print(f"\n=== {title} ===")
        print(f"Total embeddings: {len(embeddings)}")
        
        if embeddings:
            first_embedding = embeddings[0]
            if hasattr(first_embedding, 'shape'):
                print(f"First embedding shape: {first_embedding.shape}")
            elif hasattr(first_embedding, '__len__'):
                print(f"First embedding length: {len(first_embedding)}")
            
            # Show type information
            print(f"Embedding type: {type(first_embedding)}")
    
    def create_results_summary(self, query: str, response: VespaQueryResponse) -> Dict[str, Any]:
        """
        Create a summary of query results
        
        Args:
            query: The original query text
            response: Vespa query response
            
        Returns:
            Dictionary with results summary
        """
        query_time = response.json.get("timing", {}).get("searchtime", -1)
        count = response.json.get("root", {}).get("fields", {}).get("totalCount", 0)
        
        summary = {
            "query": query,
            "query_time": round(query_time, 2),
            "total_count": count,
            "results_returned": len(response.hits),
            "top_score": response.hits[0]["relevance"] if response.hits else 0,
            "results": []
        }
        
        for hit in response.hits:
            result = {
                "title": hit["fields"]["title"],
                "url": hit["fields"]["url"],
                "page": hit["fields"]["page_number"],
                "score": hit["relevance"]
            }
            summary["results"].append(result)
        
        return summary 