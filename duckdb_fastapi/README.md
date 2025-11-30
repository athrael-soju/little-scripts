# DuckDB Analytics FastAPI Service

A lightweight FastAPI service providing columnar analytics storage for OCR data using DuckDB. Designed to work seamlessly with document processing pipelines, offering efficient storage and querying of structured OCR results.

## Overview

This service provides a high-performance analytical database for OCR document processing workflows. Using DuckDB's columnar storage engine, it efficiently stores and queries document metadata, page content, and structured regions extracted from documents.

## Features

- **Document Management**: Store and retrieve document metadata with versioning
- **Page-Level Storage**: Efficient storage of OCR results with full text and markdown
- **Region Tracking**: Store structured bounding boxes and content for document regions
- **Full-Text Search**: Fast text search across all indexed documents
- **Custom Queries**: Execute read-only SQL queries for analytics
- **Health Monitoring**: Built-in health checks and database statistics
- **Maintenance API**: Initialize, clear, or reset database storage
- **Request Tracking**: Automatic request ID injection and timing middleware

## API Endpoints

### Core Information

- `GET /` - Service information and version
- `GET /health` - Health check with database status
- `GET /info` - Database information and table counts
- `GET /stats` - Aggregate statistics (documents, pages, regions)

### Document Management

- `POST /documents/check` - Check if a document exists
- `POST /documents/store` - Store document metadata
- `POST /documents/store/batch` - Batch store multiple documents
- `GET /ocr/documents` - List all indexed documents (paginated)
- `GET /ocr/documents/{filename}` - Get document information
- `DELETE /ocr/documents/{filename}` - Delete all data for a document

### OCR Data Storage

- `POST /ocr/store` - Store OCR data for a single page
- `POST /ocr/store/batch` - Store multiple OCR pages in batch
- `GET /ocr/pages/{filename}/{page_number}` - Get complete page data
- `GET /ocr/pages/{filename}/{page_number}/regions` - Get only regions (optimized for retrieval)

### Search and Queries

- `POST /search/text` - Full-text search across OCR content
- `POST /query` - Execute custom read-only SQL queries

### Maintenance

- `POST /maintenance/initialize` - Ensure database schema is ready
- `POST /maintenance/clear` - Delete all data while keeping schema
- `POST /maintenance/delete` - Reset database (drop all tables)

## Installation

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)

### Local Setup

1. **Navigate to the project directory:**

   ```bash
   cd duckdb_fastapi
   ```

2. **Create environment file:**

   ```bash
   cp .env.example .env
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the service:**

   ```bash
   python main.py
   ```

   The service will be available at `http://0.0.0.0:8300`

### Docker Deployment

1. **Build and run with Docker Compose:**

   ```bash
   docker compose up -d
   ```

2. **Check logs:**

   ```bash
   docker compose logs -f
   ```

3. **Stop the service:**

   ```bash
   docker compose down
   ```

## Configuration

Configure the service via environment variables or `.env` file:

```bash
# Database Configuration
DUCKDB_DATABASE_PATH=/app/data/ocr_data.duckdb  # Path to DuckDB file

# API Server
DUCKDB_API_HOST=0.0.0.0
DUCKDB_API_PORT=8300

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## Database Schema

The service automatically creates three main tables:

### documents

Stores document metadata and versioning information.

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    document_id VARCHAR NOT NULL,      -- UUID for the document
    filename VARCHAR NOT NULL,          -- Original filename
    file_size_bytes BIGINT,            -- File size
    total_pages INTEGER,               -- Total page count
    first_indexed TIMESTAMP,           -- First indexing time
    last_indexed TIMESTAMP             -- Last update time
);
```

### pages

Stores page-level OCR results with full text and markdown.

```sql
CREATE TABLE pages (
    id INTEGER PRIMARY KEY,
    document_id VARCHAR NOT NULL,       -- Document UUID
    page_id VARCHAR NOT NULL,           -- Page UUID
    filename VARCHAR NOT NULL,          -- Document filename
    page_number INTEGER NOT NULL,       -- Page number (0-indexed)
    page_width_px INTEGER,             -- Page width in pixels
    page_height_px INTEGER,            -- Page height in pixels
    image_url VARCHAR,                 -- MinIO URL to page image
    text TEXT,                         -- Extracted plain text
    markdown TEXT,                     -- Formatted markdown
    storage_url VARCHAR,               -- MinIO URL to full JSON
    extracted_at TIMESTAMP,            -- OCR extraction time
    created_at TIMESTAMP               -- Database insertion time
);
```

### regions

Stores structured regions with bounding boxes.

```sql
CREATE TABLE regions (
    id INTEGER PRIMARY KEY,
    page_id VARCHAR NOT NULL,           -- Page UUID (foreign key)
    region_id VARCHAR,                  -- Region identifier
    label VARCHAR,                      -- Region type (text, table, figure, image)
    bbox_x1 INTEGER,                   -- Bounding box coordinates
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    content TEXT                        -- Region content or image URL
);
```

## Usage Examples

### Store Document Metadata

```bash
curl -X POST "http://localhost:8300/documents/store" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "report.pdf",
    "file_size_bytes": 1048576,
    "total_pages": 10
  }'
```

### Store OCR Page Data

```bash
curl -X POST "http://localhost:8300/ocr/store" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "page_id": "650e8400-e29b-41d4-a716-446655440001",
    "filename": "report.pdf",
    "page_number": 0,
    "page_width_px": 2480,
    "page_height_px": 3508,
    "text": "Full text extracted from page...",
    "markdown": "# Title\n\nContent...",
    "regions": [
      {
        "id": "region-1",
        "label": "text",
        "bbox": [100, 200, 800, 400],
        "content": "Text content in this region"
      }
    ],
    "extracted_at": "2025-11-30T12:00:00Z"
  }'
```

### Search Text

```bash
curl -X POST "http://localhost:8300/search/text?q=financial&limit=10"
```

### List Documents

```bash
curl "http://localhost:8300/ocr/documents?limit=50&offset=0"
```

### Get Page Regions (Optimized)

```bash
curl "http://localhost:8300/ocr/pages/report.pdf/0/regions"
```

### Execute Custom Query

```bash
curl -X POST "http://localhost:8300/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SELECT filename, COUNT(*) as page_count FROM pages GROUP BY filename",
    "limit": 100
  }'
```

### Get Statistics

```bash
curl "http://localhost:8300/stats"
```

### Check Document Existence

```bash
curl -X POST "http://localhost:8300/documents/check" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "report.pdf",
    "file_size_bytes": 1048576,
    "total_pages": 10
  }'
```

## Architecture

### Project Structure

```
duckdb_fastapi/
├── app/
│   ├── api/
│   │   └── routes.py              # API endpoint handlers
│   ├── core/
│   │   ├── config.py              # Configuration management
│   │   ├── database.py            # DuckDB connection manager
│   │   └── logging.py             # Logging setup
│   ├── middleware/
│   │   ├── request_id.py          # Request ID injection
│   │   └── timing.py              # Response time logging
│   ├── models/
│   │   └── schemas.py             # Pydantic models
│   ├── services/
│   │   └── duckdb_service.py      # Database operations
│   └── main.py                    # FastAPI app factory
├── main.py                        # Service entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── entrypoint.sh
```

### Key Components

- **DatabaseManager**: Manages DuckDB connections and schema initialization
- **DuckDBAnalyticsService**: Encapsulates all database queries and operations
- **Pydantic Models**: Type-safe request/response schemas
- **Middleware**: Request tracking and performance monitoring

## Performance Optimization

### Query Performance

DuckDB provides excellent analytical query performance out of the box:

- **Columnar storage**: Efficient for analytical queries
- **Compression**: Automatic data compression
- **Parallel execution**: Multi-threaded query processing

### Best Practices

1. **Use region endpoint**: For retrieval tasks, use `/ocr/pages/{filename}/{page_number}/regions` instead of the full page endpoint
2. **Batch operations**: Use batch endpoints for inserting multiple records
3. **Indexed queries**: DuckDB automatically optimizes common query patterns
4. **Limit results**: Always specify reasonable limits for large result sets

### Storage Efficiency

- DuckDB uses efficient columnar compression
- Typical compression ratio: 3-5x for text content
- File-based storage: Easy backup and migration

## Security

### Query Safety

The service implements multiple security layers for custom queries:

- **Read-only queries**: Only `SELECT` statements allowed
- **Keyword filtering**: Blocks `DROP`, `DELETE`, `INSERT`, `UPDATE`, etc.
- **Comment stripping**: Removes SQL comments to prevent bypass attempts
- **Multiple statement prevention**: Blocks queries with multiple semicolons
- **Length limits**: Queries limited to 100,000 characters
- **Automatic LIMIT**: Adds `LIMIT` clause if not present

### Safe Query Examples

```sql
-- ✓ Allowed
SELECT filename, page_number, text FROM pages WHERE filename = 'report.pdf';

-- ✓ Allowed
SELECT COUNT(*) FROM regions WHERE label = 'table';

-- ✗ Blocked (not SELECT)
UPDATE pages SET text = 'modified';

-- ✗ Blocked (dangerous keyword)
SELECT * FROM pages; DROP TABLE regions;
```

## Integration Examples

### Python Client

```python
import requests
import json

class DuckDBClient:
    def __init__(self, base_url="http://localhost:8300"):
        self.base_url = base_url

    def store_page(self, page_data):
        """Store OCR page data."""
        response = requests.post(
            f"{self.base_url}/ocr/store",
            json=page_data
        )
        return response.json()

    def search_text(self, query, limit=50):
        """Search for text across documents."""
        response = requests.post(
            f"{self.base_url}/search/text",
            params={"q": query, "limit": limit}
        )
        return response.json()

    def get_page_regions(self, filename, page_number):
        """Get regions for a specific page."""
        response = requests.get(
            f"{self.base_url}/ocr/pages/{filename}/{page_number}/regions"
        )
        return response.json()

    def run_query(self, sql_query, limit=1000):
        """Execute custom SQL query."""
        response = requests.post(
            f"{self.base_url}/query",
            json={"query": sql_query, "limit": limit}
        )
        return response.json()

# Usage
client = DuckDBClient()

# Search for documents
results = client.search_text("financial report")
print(f"Found {results['row_count']} matching pages")

# Get regions for retrieval
regions = client.get_page_regions("report.pdf", 0)
for region in regions:
    print(f"{region['label']}: {region['content'][:50]}...")
```

### Integration with OCR Pipeline

```python
async def process_document(pdf_path, ocr_service, duckdb_service):
    """Process document and store in DuckDB."""
    import uuid
    from datetime import datetime

    # Generate document ID
    doc_id = str(uuid.uuid4())
    filename = os.path.basename(pdf_path)

    # Store document metadata
    await duckdb_service.store_document({
        "document_id": doc_id,
        "filename": filename,
        "file_size_bytes": os.path.getsize(pdf_path),
        "total_pages": count_pdf_pages(pdf_path)
    })

    # Process each page
    for page_num, page_image in enumerate(extract_pages(pdf_path)):
        # Run OCR
        ocr_result = await ocr_service.process_image(page_image)

        # Store in DuckDB
        page_data = {
            "document_id": doc_id,
            "page_id": str(uuid.uuid4()),
            "filename": filename,
            "page_number": page_num,
            "page_width_px": page_image.width,
            "page_height_px": page_image.height,
            "text": ocr_result["text"],
            "markdown": ocr_result["markdown"],
            "regions": ocr_result["regions"],
            "extracted_at": datetime.utcnow().isoformat()
        }

        await duckdb_service.store_page(page_data)
```

## API Documentation

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8300/docs`
- ReDoc: `http://localhost:8300/redoc`

## Health Checks

The `/health` endpoint provides service status:

```json
{
  "status": "healthy",
  "database": "connected",
  "version": "1.0.0"
}
```

Use for Docker health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8300/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

## Troubleshooting

### Database File Locked

If you get "database is locked" errors:

```bash
# Stop all services accessing the database
docker compose down

# Remove lock files
rm /app/data/ocr_data.duckdb.wal

# Restart service
docker compose up -d
```

### Storage Path Issues

Ensure the database directory exists and has proper permissions:

```bash
# Create directory
mkdir -p /app/data

# Set permissions (Docker)
chmod 755 /app/data
```

### Memory Issues

For large datasets, monitor DuckDB memory usage:

```python
# Limit memory usage
connection.execute("SET memory_limit='4GB'")
connection.execute("SET threads=4")
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .
```

## Backup and Migration

### Export Data

```bash
# Export to JSON
curl -X POST "http://localhost:8300/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT * FROM pages"}' > pages_backup.json

# Or copy the DuckDB file directly
cp /app/data/ocr_data.duckdb ./backup/
```

### Restore Data

```bash
# Simply replace the DuckDB file
cp ./backup/ocr_data.duckdb /app/data/

# Restart service
docker compose restart
```

## License

Open source - feel free to use and modify as needed.

## Related Projects

- [DeepSeek OCR](../deepseek-ocr/) - OCR service that can feed data into this analytics service
- [PaddleOCR-VL](../paddleocr_vl/) - Alternative OCR service
- [ColModernVBert FastAPI](../colmodernvbert_fastapi/) - Document embedding service

## Acknowledgments

Built with [DuckDB](https://duckdb.org/) - an in-process SQL OLAP database management system.
